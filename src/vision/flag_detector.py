"""
Detector de Bandeiras dos Shinpan (Pendência I7 — rótulos automáticos pelo próprio vídeo).

Quando 2 dos 3 shinpan levantam a bandeira da mesma cor, é ponto (vermelha = Aka, branca = Shiro).
Este módulo aprende a reconhecer a bandeira erguida e transforma isso em pontos no gabarito.

Fluxo:
1. Mineração: amostra os vídeos (1 quadro/s), detecta as pessoas com o modelo de pose local e recorta
   a região acima dos ombros de cada uma. Recortes com mancha vermelha/branca vão primeiro na fila.
2. Rotulagem (tela do Gabarito de Lutas): o usuário marca cada recorte como
   "vermelha" (erguida), "branca" (erguida), "ambas" (as duas erguidas = parada/yame) ou "nenhuma".
3. Treino: classificador YOLO (yolo11n-cls) iniciado dos pesos de pose locais, sem download.
   Validação separada por vídeo.
4. Eventos: em cada vídeo, classifica as pessoas em ~4 quadros/s, acompanha cada uma e marca a
   SUBIDA da bandeira; 2+ pessoas subindo a mesma cor em até 2 s = ponto. 2+ pessoas erguendo as DUAS
   bandeiras = parada (yame: fim da luta, consulta dos shinpan ou irregularidade), que nunca vira ponto.
5. Avaliação contra os gabaritos manuais (fórmula única de precisão/recall/F1) e, só se o F1 for alto,
   gravação dos pontos como automáticos no gabarito das lutas ainda sem pontos.

Uso via linha de comando (a partir da pasta Dev/):
    python -m src.vision.flag_detector mine "<pasta de vídeos>"
    python -m src.vision.flag_detector train
    python -m src.vision.flag_detector evaluate "<pasta de vídeos>"
    python -m src.vision.flag_detector apply "<pasta de vídeos>"
    python -m src.vision.flag_detector status
"""

import os
import json
import math
import random
import shutil
import datetime
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from src.utils import model_registry
from src.utils.logger_manager import log_event
from src.utils.test_isolation import is_test_mode, get_test_sandbox_dir

MODEL_FAMILY = "flag_detector"
CLASSES = ["vermelha", "branca", "nenhuma"]          # classes obrigatórias do modelo
STOP_CLASS = "ambas"                                  # as duas bandeiras erguidas = parada (yame)
LABEL_CLASSES = ["vermelha", "branca", STOP_CLASS, "nenhuma"]
COLOR_TO_KENSHI = {"vermelha": "AKA", "branca": "SHIRO"}
DEFAULT_DATASET_DIR = os.path.join("data", "flag_dataset")
CROP_SIZE = 128
CROP_UP_TORSOS = 2.4     # quanto o recorte sobe acima dos ombros (em troncos)
CROP_SIDE_TORSOS = 1.9   # quanto se abre para cada lado
MIN_LABELS = {"vermelha": 15, "branca": 15, "nenhuma": 30}
# A parada é opcional: só entra no treino com este mínimo (abaixo dele os recortes ficam guardados, fora do treino)
MIN_STOP_LABELS = 10
# Subida de cor até este intervalo de uma parada é o próprio gesto da parada, não ponto
STOP_GUARD_SEC = 1.5
# Atraso típico entre o golpe e a subida das bandeiras
FLAG_REACTION_SEC = 1.0
# Janela para casar uma subida de bandeiras com um ponto oficial: [ponto, ponto + 4 s]
MATCH_WINDOW_SEC = (-0.5, 4.0)
MIN_F1_TO_APPLY = 0.9
MIN_VIDEOS_TO_APPLY = 3
AUTO_ORIGIN = "automatico_bandeiras"

# Índices COCO-17 do modelo de pose
KP = {"nariz": 0, "ombro_e": 5, "ombro_d": 6, "quadril_e": 11, "quadril_d": 12}


# ------------------------------------------------------------------------------
# Geometria e heurística de cor
# ------------------------------------------------------------------------------

def crop_box(keypoints: np.ndarray, width: int, height: int) -> Optional[Tuple[int, int, int, int]]:
    """Região acima dos ombros, ao alcance do braço (onde a bandeira erguida aparece)."""
    try:
        ls, rs, lh, rh = (keypoints[KP[k]] for k in ("ombro_e", "ombro_d", "quadril_e", "quadril_d"))
    except Exception:
        return None
    pts = [ls, rs, lh, rh]
    if any(p is None or (p[0] <= 0 and p[1] <= 0) for p in pts):
        return None
    sx, sy = (ls[0] + rs[0]) / 2.0, (ls[1] + rs[1]) / 2.0
    hx, hy = (lh[0] + rh[0]) / 2.0, (lh[1] + rh[1]) / 2.0
    t = math.hypot(sx - hx, sy - hy)
    if t < 8:
        return None
    # Braço estendido (~1,3 tronco) + haste e pano da bandeira (~0,9 tronco): a bandeira erguida, inclusive
    # na diagonal, chega a ~2,3 troncos acima e ~1,8 para o lado do ombro. Antes (1,6 × 1,3) ela ficava de fora.
    x0, x1 = int(max(0, sx - CROP_SIDE_TORSOS * t)), int(min(width, sx + CROP_SIDE_TORSOS * t))
    y0, y1 = int(max(0, sy - CROP_UP_TORSOS * t)), int(min(height, sy + 0.3 * t))
    if x1 - x0 < 8 or y1 - y0 < 8:
        return None
    return x0, y0, x1, y1


def color_hint(crop_bgr: np.ndarray) -> Optional[str]:
    """Pista barata para ordenar a fila de rotulagem: mancha compacta vermelha/branca na metade de cima."""
    import cv2
    h = crop_bgr.shape[0]
    top = crop_bgr[: max(1, int(h * 0.6))]
    hsv = cv2.cvtColor(top, cv2.COLOR_BGR2HSV)
    masks = {
        "vermelha": ((hsv[..., 0] < 8) | (hsv[..., 0] > 172)) & (hsv[..., 1] > 130) & (hsv[..., 2] > 70),
        "branca": (hsv[..., 1] < 40) & (hsv[..., 2] > 190),
    }
    area = top.shape[0] * top.shape[1]
    for color, m in masks.items():
        n, _, stats, _ = cv2.connectedComponentsWithStats(m.astype(np.uint8), 8)
        for i in range(1, n):
            x, y, w, hh, a = stats[i]
            if 0.002 * area <= a <= 0.12 * area and x > 0 and y > 0 and x + w < top.shape[1] and y + hh < top.shape[0]:
                if max(w, hh) / max(1, min(w, hh)) <= 3.5 and a / (w * hh) >= 0.35:
                    return color
    return None


# ------------------------------------------------------------------------------
# Dataset de recortes
# ------------------------------------------------------------------------------

def get_dataset_dir() -> str:
    d = os.path.join(get_test_sandbox_dir(), "flag_dataset") if is_test_mode() else DEFAULT_DATASET_DIR
    os.makedirs(os.path.join(d, "crops"), exist_ok=True)
    return d


def _labels_path(dataset_dir: Optional[str] = None) -> str:
    return os.path.join(dataset_dir or get_dataset_dir(), "labels.json")


def load_dataset(dataset_dir: Optional[str] = None) -> Dict[str, Dict[str, Any]]:
    path = _labels_path(dataset_dir)
    if not os.path.isfile(path):
        return {}
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f).get("recortes", {})


def mined_videos(dataset_dir: Optional[str] = None) -> List[str]:
    """Vídeos já minerados (não são minerados de novo)."""
    path = _labels_path(dataset_dir)
    if not os.path.isfile(path):
        return []
    with open(path, "r", encoding="utf-8") as f:
        raw = json.load(f)
    return sorted(set(raw.get("videos_minerados", [])) | {it["video_name"] for it in raw.get("recortes", {}).values()})


def save_dataset(items: Dict[str, Dict[str, Any]], dataset_dir: Optional[str] = None,
                 mined: Optional[List[str]] = None) -> None:
    path = _labels_path(dataset_dir)
    from src.utils.file_utils import safe_write_json
    done = sorted(set(mined if mined is not None else mined_videos(dataset_dir)))
    safe_write_json(path, {"versao": 1, "classes": LABEL_CLASSES, "recortes": items, "videos_minerados": done}, indent=1)


SHORT_VIDEO_SEC = 60.0      # cortes curtos (golpes em dúvida das análises): amostra mais quadros
SHORT_VIDEO_FPS = 4.0


def mining_candidates(video_paths: List[str], dataset_dir: Optional[str] = None,
                      labels_path: Optional[str] = None) -> Tuple[List[str], Dict[str, int]]:
    """
    Vídeos que valem minerar: ainda não minerados e que podem ter bandeiras de shinpan (lutas). Treinos e
    exames marcados no Gabarito, e vídeos descartados, ficam de fora.
    """
    from src.engine import weak_labels as wl
    done = set(mined_videos(dataset_dir))
    gab = wl.load_match_labels(labels_path or wl.DEFAULT_LABELS_PATH)["videos"]
    out, skipped = [], {"ja_minerados": 0, "nao_luta": 0}
    for v in video_paths:
        name = os.path.basename(v)
        if name in done:
            skipped["ja_minerados"] += 1
        elif wl.is_entry_ignored(gab.get(name)):
            skipped["nao_luta"] += 1
        else:
            out.append(v)
    return out, skipped


def set_labels(labels: Dict[str, str], dataset_dir: Optional[str] = None) -> int:
    """Grava rótulos {id_recorte: classe}; 'pular' marca o recorte como visto sem rótulo."""
    items = load_dataset(dataset_dir)
    n = 0
    for crop_id, label in labels.items():
        if crop_id in items and (label in LABEL_CLASSES or label == "pular"):
            items[crop_id]["rotulo"] = label
            items[crop_id]["rotulado_em"] = datetime.datetime.now().isoformat(timespec="seconds")
            n += 1
    save_dataset(items, dataset_dir)
    return n


def dataset_stats(dataset_dir: Optional[str] = None) -> Dict[str, int]:
    items = load_dataset(dataset_dir)
    stats = {c: 0 for c in LABEL_CLASSES}
    stats.update({"pular": 0, "sem_rotulo": 0, "total": len(items)})
    for it in items.values():
        stats[it.get("rotulo") or "sem_rotulo"] = stats.get(it.get("rotulo") or "sem_rotulo", 0) + 1
    return stats


def next_to_label(limit: int = 12, dataset_dir: Optional[str] = None) -> List[Dict[str, Any]]:
    """Fila de rotulagem: recortes com pista de cor primeiro (mais chance de serem bandeiras)."""
    items = [dict(v, id=k) for k, v in load_dataset(dataset_dir).items() if not v.get("rotulo")]
    # Pista vermelha é a mais precisa; a branca tem mais ruído (camisetas, papéis, parede)
    hint_order = {"vermelha": 0, "branca": 1, None: 2}
    items.sort(key=lambda it: (hint_order.get(it.get("pista"), 2), it["video_name"], it["tempo_s"]))
    return items[:limit]


def _pose_model_path() -> str:
    for name in ("yolov26n-pose.pt", "yolo11n-pose.pt", "yolov8n-pose.pt"):
        path = os.path.join("models", name)
        if os.path.isfile(path):
            return path
    raise FileNotFoundError("Nenhum modelo de pose encontrado em models/.")


def _iter_people(pose_model, frame) -> List[np.ndarray]:
    res = pose_model.predict(frame, verbose=False, imgsz=960)[0]
    if res.keypoints is None or res.keypoints.xy is None:
        return []
    return [kp for kp in res.keypoints.xy.cpu().numpy()]


def mine_crops(video_paths: List[str], sample_fps: float = 1.0, negative_ratio: float = 0.15,
               max_per_video: int = 400, device: str = "0", dataset_dir: Optional[str] = None,
               seed: int = 7, progress: Optional[Any] = None) -> Dict[str, int]:
    """
    Extrai recortes dos vídeos para rotulagem. Recortes já existentes são mantidos e cada vídeo minerado é
    registrado (não é minerado de novo). Cortes curtos (< 60 s) são amostrados a 4 quadros/s.
    progress(i, total, nome) é chamado antes de cada vídeo.
    """
    import cv2
    from ultralytics import YOLO
    rng = random.Random(seed)
    ds_dir = dataset_dir or get_dataset_dir()
    items = load_dataset(ds_dir)
    done = set(mined_videos(ds_dir))
    pose = YOLO(_pose_model_path())
    pose.to("cuda" if device not in ("cpu", "") else "cpu")
    added = {}
    for vi, video in enumerate(video_paths):
        name = os.path.basename(video)
        if progress:
            progress(vi, len(video_paths), name)
        cap = cv2.VideoCapture(video)
        fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        rate = SHORT_VIDEO_FPS if total and total / fps < SHORT_VIDEO_SEC else sample_fps
        step = max(1, int(round(fps / rate)))
        count = 0
        fi = -1
        while count < max_per_video:
            fi += 1
            if fi >= total or not cap.grab():
                break
            if fi % step:
                continue  # leitura sequencial: só decodifica os quadros amostrados
            ok, frame = cap.retrieve()
            if not ok:
                break
            h, w = frame.shape[:2]
            for pi, kp in enumerate(_iter_people(pose, frame)):
                box = crop_box(kp, w, h)
                if box is None:
                    continue
                x0, y0, x1, y1 = box
                crop = frame[y0:y1, x0:x1]
                hint = color_hint(crop)
                if hint is None and rng.random() > negative_ratio:
                    continue
                crop_id = f"{os.path.splitext(name)[0][:40]}_{fi}_{pi}".replace(" ", "_")
                if crop_id in items:
                    continue
                path = os.path.join(ds_dir, "crops", f"{abs(hash(crop_id)) % 10**12}_{fi}_{pi}.jpg")
                cv2.imwrite(path, cv2.resize(crop, (CROP_SIZE, CROP_SIZE), interpolation=cv2.INTER_AREA))
                items[crop_id] = {"video_name": name, "tempo_s": round(fi / fps, 2), "caixa": [x0, y0, x1, y1],
                                  "arquivo": path, "pista": hint, "rotulo": None}
                count += 1
        cap.release()
        added[name] = count
        done.add(name)
        save_dataset(items, ds_dir, mined=sorted(done))
    return added


# ------------------------------------------------------------------------------
# Treino
# ------------------------------------------------------------------------------

def get_models_dir() -> str:
    return model_registry.get_models_dir(MODEL_FAMILY)


def training_classes(stats: Dict[str, int]) -> List[str]:
    """Classes do próximo treino: as obrigatórias, mais a parada quando já houver rótulos suficientes."""
    return CLASSES + ([STOP_CLASS] if stats.get(STOP_CLASS, 0) >= MIN_STOP_LABELS else [])


def split_by_video(items: Dict[str, Dict[str, Any]], val_fraction: float = 0.25, seed: int = 7,
                   classes: Optional[List[str]] = None) -> Tuple[List[str], List[str]]:
    """Separa treino e validação por vídeo (nenhum vídeo nos dois lados)."""
    classes = classes or CLASSES
    videos = sorted({it["video_name"] for it in items.values() if it.get("rotulo") in classes})
    rng = random.Random(seed)
    rng.shuffle(videos)
    n_val = max(1, int(round(len(videos) * val_fraction))) if len(videos) > 1 else 0
    val_videos = set(videos[:n_val])
    train_ids = [k for k, it in items.items() if it.get("rotulo") in classes and it["video_name"] not in val_videos]
    val_ids = [k for k, it in items.items() if it.get("rotulo") in classes and it["video_name"] in val_videos]
    return train_ids, val_ids


def check_enough_labels(stats: Dict[str, int]) -> Optional[str]:
    missing = [f"{c}: {stats.get(c, 0)}/{n}" for c, n in MIN_LABELS.items() if stats.get(c, 0) < n]
    return ("Rótulos insuficientes para treinar (" + ", ".join(missing) + ").") if missing else None


BALANCE_RATIO = 0.3     # no treino, cada classe passa a ter ≥ 30% do tamanho da maior (por repetição)
MAX_REPEATS = 20


def balance_repeats(counts: Dict[str, int]) -> Dict[str, int]:
    """Quantas vezes cada recorte de treino entra (1 = sem cópia) para nenhuma classe ficar pequena demais."""
    top = max(counts.values()) if counts else 0
    return {c: (1 if n == 0 else max(1, min(MAX_REPEATS, math.ceil(BALANCE_RATIO * top / n)))) for c, n in counts.items()}


def _epoch_info(trainer, started: float) -> Dict[str, Any]:
    """Andamento real de uma época do YOLO (só números que o próprio treino mede)."""
    import time
    epoch, total = int(trainer.epoch) + 1, int(trainer.epochs)
    elapsed = time.time() - started
    m = getattr(trainer, "metrics", None) or {}
    loss = None
    try:
        items = trainer.label_loss_items(trainer.tloss, prefix="train")
        loss = float(next(iter(items.values()))) if items else None
    except Exception:
        try:
            loss = float(trainer.tloss)
        except Exception:
            loss = None
    stopper = getattr(trainer, "stopper", None)
    return {
        "etapa": "treino", "epoca": epoch, "epocas": total, "decorrido_s": round(elapsed, 1),
        "restante_s": round(elapsed / epoch * (total - epoch), 1) if epoch else None,
        "perda_treino": None if loss is None else round(loss, 4),
        "perda_validacao": m.get("val/loss"), "acerto_validacao": m.get("metrics/accuracy_top1"),
        "paciencia": None if stopper is None else int(getattr(stopper, "patience", 0)),
    }


def train_flag_classifier(epochs: int = 40, device: str = "0", dataset_dir: Optional[str] = None,
                          progress: Optional[Any] = None) -> Dict[str, Any]:
    """
    Treina o classificador de bandeira erguida e salva a versão (ainda não ativa).
    progress(dict) recebe o andamento: preparação, cada época (perdas e acerto na validação) e a medição final.
    """
    import time
    from ultralytics import YOLO
    report = progress or (lambda info: None)
    ds_dir = dataset_dir or get_dataset_dir()
    items = load_dataset(ds_dir)
    stats = dataset_stats(ds_dir)
    problem = check_enough_labels(stats)
    if problem:
        raise ValueError(problem)
    classes = training_classes(stats)
    train_ids, val_ids = split_by_video(items, classes=classes)
    if not val_ids:
        raise ValueError("É preciso rotular recortes de pelo menos 2 vídeos (validação separada por vídeo).")

    count = lambda ids: {c: sum(1 for k in ids if items[k]["rotulo"] == c) for c in classes}  # noqa: E731
    report({"etapa": "preparando", "classes": classes, "treino": count(train_ids), "validacao": count(val_ids),
            "repeticoes": balance_repeats(count(train_ids)),
            "videos_validacao": sorted({items[k]["video_name"] for k in val_ids})})
    split_dir = os.path.join(ds_dir, "_split")
    shutil.rmtree(split_dir, ignore_errors=True)
    for split, ids in (("train", train_ids), ("val", val_ids)):
        for c in classes:
            os.makedirs(os.path.join(split_dir, split, c), exist_ok=True)
        for k in ids:
            it = items[k]
            shutil.copy(it["arquivo"], os.path.join(split_dir, split, it["rotulo"], os.path.basename(it["arquivo"])))
        if split == "train":
            # Equilíbrio: "nenhuma" é muito mais comum que bandeira erguida; sem isso o modelo aprende a responder
            # sempre "nenhuma". Só o TREINO recebe cópias (o YOLO as varia com aumento de dados); a validação
            # fica com os recortes reais, para a medida continuar honesta.
            reps = balance_repeats(count(ids))
            for k in ids:
                it = items[k]
                base, ext = os.path.splitext(os.path.basename(it["arquivo"]))
                for r in range(1, reps.get(it["rotulo"], 1)):
                    shutil.copy(it["arquivo"], os.path.join(split_dir, split, it["rotulo"], f"{base}_rep{r}{ext}"))

    os.environ.setdefault("YOLO_OFFLINE", "1")
    stamp = datetime.datetime.now().strftime("%Y%m%dT%H%M%S")
    # Caminho ABSOLUTO: com caminho relativo, o YOLO grava na pasta global "runs_dir" das configurações dele
    # (que pode ser de outro projeto) e o best.pt não é encontrado aqui.
    runs = os.path.abspath(os.path.join(get_models_dir(), "runs"))
    model = YOLO("yolo11n-cls.yaml").load(_pose_model_path())
    started = time.time()
    model.add_callback("on_fit_epoch_end", lambda trainer: report(_epoch_info(trainer, started)))
    model.train(data=os.path.abspath(split_dir), epochs=epochs, imgsz=CROP_SIZE, batch=32, device=device, amp=False,
                plots=False, project=runs, name=stamp, exist_ok=True, verbose=False, workers=0, patience=10)
    save_dir = str(getattr(getattr(model, "trainer", None), "save_dir", "") or os.path.join(runs, stamp))
    best = os.path.join(save_dir, "weights", "best.pt")
    if not os.path.isfile(best):
        raise FileNotFoundError(f"O treino terminou, mas os pesos não foram encontrados em {best}.")
    final = os.path.join(get_models_dir(), f"flag_cls_{stamp}.pt")
    shutil.copy(best, final)

    report({"etapa": "medindo", "recortes": len(val_ids)})
    val = evaluate_crops(final, [items[k] for k in val_ids])
    meta = {
        "modelo": final, "treinado_em": stamp, "classes": classes,
        "recortes": {"treino": len(train_ids), "validacao": len(val_ids)},
        "videos_validacao": sorted({items[k]["video_name"] for k in val_ids}),
        "validacao": val,
    }
    with open(final.replace(".pt", ".json"), "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2, ensure_ascii=False)
    return meta


def evaluate_crops(model_path: str, crops: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Precisão/recall por cor nos recortes de validação (fórmula única)."""
    from ultralytics import YOLO
    from src.engine.weak_labels import precision_recall_f1
    model = YOLO(model_path)
    preds = classify_crops(model, [c["arquivo"] for c in crops])
    out = {}
    present = {c["rotulo"] for c in crops}
    for color in ("vermelha", "branca", STOP_CLASS):
        if color == STOP_CLASS and color not in present:
            continue
        tp = sum(1 for c, p in zip(crops, preds) if p == color and c["rotulo"] == color)
        fp = sum(1 for c, p in zip(crops, preds) if p == color and c["rotulo"] != color)
        fn = sum(1 for c, p in zip(crops, preds) if p != color and c["rotulo"] == color)
        out[color] = precision_recall_f1(tp, fp, fn)
    out["acuracia"] = round(sum(1 for c, p in zip(crops, preds) if p == c["rotulo"]) / len(crops), 4) if crops else None
    return out


def classify_crops(model, images: List[Any]) -> List[str]:
    if not images:
        return []
    results = model.predict(images, imgsz=CROP_SIZE, verbose=False)
    names = model.names
    return [names[int(r.probs.top1)] for r in results]


# ------------------------------------------------------------------------------
# Eventos de bandeira num vídeo
# ------------------------------------------------------------------------------

def flag_events(samples: List[Tuple[float, List[Tuple[Tuple[float, float], float, str]]]],
                off_before: int = 2, on_after: int = 2, join_sec: float = 2.0,
                min_people: int = 2) -> Dict[str, List[Dict[str, Any]]]:
    """
    samples: [(tempo_s, [((x, y) do ombro, tamanho do tronco, classe), ...]), ...]
    Acompanha cada pessoa pela posição e marca a SUBIDA da bandeira (off_before amostras sem, on_after com).
    Duas ou mais pessoas subindo a mesma cor em até join_sec = um ponto. Duas ou mais erguendo as duas
    bandeiras = parada (yame); subida de cor perto de uma parada é o gesto da parada e não vira ponto.
    """
    tracks: List[Dict[str, Any]] = []
    raises: List[Tuple[float, str, int]] = []
    for si, (t_s, people) in enumerate(samples):
        used = set()
        for (pos, torso, cls) in people:
            best = None
            for k, tr in enumerate(tracks):
                if k in used or si - tr["last"] > 3:
                    continue
                d = math.hypot(tr["pos"][0] - pos[0], tr["pos"][1] - pos[1])
                if d < 0.7 * torso and (best is None or d < best[0]):
                    best = (d, k)
            if best is None:
                tracks.append({"pos": pos, "last": si, "hist": []})
                k = len(tracks) - 1
            else:
                k = best[1]
            used.add(k)
            tr = tracks[k]
            tr["pos"], tr["last"] = pos, si
            tr["hist"].append((t_s, cls))
            window = tr["hist"][-(off_before + on_after):]
            if len(window) == off_before + on_after:
                first = window[off_before][1]
                if (first in COLOR_TO_KENSHI or first == STOP_CLASS) and all(c == first for _, c in window[off_before:]) \
                        and all(c != first for _, c in window[:off_before]):
                    raises.append((window[off_before][0], first, k))
    events: List[Dict[str, Any]] = []
    for color in (*COLOR_TO_KENSHI, STOP_CLASS):
        group: List[Tuple[float, int]] = []
        for t_s, c, k in sorted(r for r in raises if r[1] == color):
            if group and t_s - group[0][0] > join_sec:
                if len({kk for _, kk in group}) >= min_people:
                    events.append({"cor": color, "bandeira_s": round(group[0][0], 2), "juizes": len({kk for _, kk in group})})
                group = []
            group.append((t_s, k))
        if group and len({kk for _, kk in group}) >= min_people:
            events.append({"cor": color, "bandeira_s": round(group[0][0], 2), "juizes": len({kk for _, kk in group})})
    stops = sorted(({"tipo": "parada", "bandeira_s": e["bandeira_s"], "juizes": e["juizes"]}
                    for e in events if e["cor"] == STOP_CLASS), key=lambda e: e["bandeira_s"])
    points = []
    for ev in events:
        if ev["cor"] == STOP_CLASS or any(abs(ev["bandeira_s"] - p["bandeira_s"]) <= STOP_GUARD_SEC for p in stops):
            continue
        ev["kenshi"] = COLOR_TO_KENSHI[ev["cor"]]
        ev["golpe_estimado_s"] = round(max(0.0, ev["bandeira_s"] - FLAG_REACTION_SEC), 2)
        points.append(ev)
    return {"pontos": sorted(points, key=lambda e: e["bandeira_s"]), "paradas": stops}


def raise_events(samples: List[Tuple[float, List[Tuple[Tuple[float, float], float, str]]]], **kw) -> List[Dict[str, Any]]:
    """Só os pontos (ver flag_events)."""
    return flag_events(samples, **kw)["pontos"]


def detect_flag_events(video_path: str, model_path: Optional[str] = None, sample_fps: float = 4.0,
                       device: str = "0") -> Dict[str, Any]:
    """Pontos indicados pelas bandeiras dos shinpan num vídeo."""
    import cv2
    from ultralytics import YOLO
    model_path = model_path or model_registry.get_active_model_path(MODEL_FAMILY)
    if not model_path:
        raise ValueError("Nenhum detector de bandeiras ativo. Treine e ative um modelo antes.")
    pose = YOLO(_pose_model_path())
    cls_model = YOLO(model_path)
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    step = max(1, int(round(fps / sample_fps)))
    samples = []
    fi = 0
    while fi < total:
        if not cap.grab():
            break
        if fi % step == 0:
            ok, frame = cap.retrieve()
            if not ok:
                break
            h, w = frame.shape[:2]
            people, crops = [], []
            for kp in _iter_people(pose, frame):
                box = crop_box(kp, w, h)
                if box is None:
                    continue
                x0, y0, x1, y1 = box
                crops.append(cv2.resize(frame[y0:y1, x0:x1], (CROP_SIZE, CROP_SIZE), interpolation=cv2.INTER_AREA))
                sx, sy = (kp[5][0] + kp[6][0]) / 2.0, (kp[5][1] + kp[6][1]) / 2.0
                hx, hy = (kp[11][0] + kp[12][0]) / 2.0, (kp[11][1] + kp[12][1]) / 2.0
                people.append(((float(sx), float(sy)), float(math.hypot(sx - hx, sy - hy))))
            classes = classify_crops(cls_model, crops)
            samples.append((fi / fps, [(p, t, c) for (p, t), c in zip(people, classes)]))
        fi += 1
    cap.release()
    ev = flag_events(samples)
    return {"video_name": os.path.basename(video_path), "modelo": model_path, "eventos": ev["pontos"],
            "paradas": ev["paradas"]}


# ------------------------------------------------------------------------------
# Avaliação contra o gabarito manual e gravação automática
# ------------------------------------------------------------------------------

def match_events_to_points(events: List[Dict[str, Any]], entry: Dict[str, Any]) -> Dict[str, int]:
    """Casa eventos de bandeira com pontos oficiais (mesmo kenshi, bandeira até 4 s após o golpe)."""
    from src.engine import weak_labels as wl
    points = [p for p in wl.parse_official_points(entry) if p["is_strike"]]
    used, tp = set(), 0
    for ev in events:
        kenshi = wl.normalize_kenshi(ev["kenshi"])
        for p in points:
            dt = ev["bandeira_s"] - p["time_sec"]
            if p["index"] not in used and p["kenshi_id"] in (kenshi, None) and MATCH_WINDOW_SEC[0] <= dt <= MATCH_WINDOW_SEC[1]:
                used.add(p["index"])
                tp += 1
                break
    return {"tp": tp, "fp": len(events) - tp, "fn": len(points) - tp}


def evaluate_against_gabarito(videos_dir: str, labels_path: Optional[str] = None,
                              detector=detect_flag_events) -> Dict[str, Any]:
    """Mede o detector nas lutas com gabarito manual completo."""
    from src.engine import weak_labels as wl
    labels = wl.load_match_labels(labels_path or wl.DEFAULT_LABELS_PATH)["videos"]
    rows, tot = [], {"tp": 0, "fp": 0, "fn": 0}
    for name, entry in sorted(labels.items()):
        if not entry.get("pontos_completos") or entry.get("origem") == AUTO_ORIGIN:
            continue
        video = os.path.join(videos_dir, name)
        if not os.path.isfile(video):
            continue
        res = detector(video)
        m = match_events_to_points(res["eventos"], entry)
        rows.append(dict(m, video_name=name, eventos=res["eventos"]))
        for k in tot:
            tot[k] += m[k]
    prf = wl.precision_recall_f1(tot["tp"], tot["fp"], tot["fn"])
    return {"lutas": rows, "total": dict(tot, **prf, videos=len(rows))}


def apply_to_gabarito(videos_dir: str, evaluation: Dict[str, Any], labels_path: Optional[str] = None,
                      detector=detect_flag_events, min_f1: float = MIN_F1_TO_APPLY,
                      min_videos: int = MIN_VIDEOS_TO_APPLY) -> Dict[str, Any]:
    """Grava os pontos das bandeiras nas lutas sem pontos, só se o detector provou F1 alto."""
    from src.engine import weak_labels as wl
    total = evaluation["total"]
    if total["videos"] < min_videos or (total.get("f1") or 0.0) < min_f1:
        return {"aplicado": False, "motivo": f"Detector ainda não confiável: F1 {total.get('f1')} em {total['videos']} luta(s) "
                                            f"(mínimo {min_f1} em {min_videos})."}
    path = labels_path or wl.DEFAULT_LABELS_PATH
    data = wl.load_match_labels(path)
    written = []
    for name, entry in sorted(data["videos"].items()):
        if wl.is_entry_filled(entry) or wl.is_entry_ignored(entry):
            continue  # nunca sobrescreve gabarito manual nem escreve em vídeo que não é luta
        video = os.path.join(videos_dir, name)
        if not os.path.isfile(video):
            continue
        res = detector(video)
        events = res["eventos"]
        entry.update({
            "pontos": [{"kenshi": e["kenshi"], "tecnica": "MEN", "tempo": f"{e['golpe_estimado_s']:.1f}",
                        "bandeira_s": e["bandeira_s"]} for e in events],
            "pontos_completos": True,
            "paradas_bandeiras": [p["bandeira_s"] for p in res.get("paradas", [])],
            "origem": AUTO_ORIGIN,
            "observacoes": f"Pontos lidos das bandeiras dos shinpan (F1 do detector {total.get('f1')}). "
                           "Técnica não identificada (registrada como MEN): revise se for usar a técnica.",
        })
        written.append({"video_name": name, "pontos": len(events)})
    wl.save_match_labels(data, path)
    log_event("INFO", f"Gabarito automático pelas bandeiras gravado em {len(written)} luta(s).", "flag_detector")
    return {"aplicado": True, "lutas": written}


def _fmt(v: Optional[float]) -> str:
    return "—" if v is None else f"{v * 100:.0f}%"


def _main(argv: Optional[List[str]] = None) -> int:
    import argparse
    from src.utils.feature_cache import list_videos
    parser = argparse.ArgumentParser(description="Detector de bandeiras dos shinpan (rótulos automáticos).")
    sub = parser.add_subparsers(dest="command", required=True)
    p_m = sub.add_parser("mine", help="Extrai recortes dos vídeos para rotulagem.")
    p_m.add_argument("target")
    p_m.add_argument("--fps", type=float, default=1.0)
    p_t = sub.add_parser("train", help="Treina com os recortes rotulados.")
    p_t.add_argument("--epochs", type=int, default=40)
    p_t.add_argument("--activate", action="store_true", help="Ativa o modelo treinado.")
    p_e = sub.add_parser("evaluate", help="Mede o detector ativo contra os gabaritos manuais.")
    p_e.add_argument("target")
    p_a = sub.add_parser("apply", help="Grava pontos automáticos nas lutas sem gabarito (se o detector for confiável).")
    p_a.add_argument("target")
    sub.add_parser("status", help="Rótulos e modelo ativo.")
    args = parser.parse_args(argv)

    if args.command == "status":
        st = dataset_stats()
        print(f"Recortes: {st['total']} · vermelha {st['vermelha']} · branca {st['branca']} · nenhuma {st['nenhuma']} · "
              f"pulados {st['pular']} · sem rótulo {st['sem_rotulo']}")
        print(f"Parada (ambas erguidas): {st[STOP_CLASS]}/{MIN_STOP_LABELS} — opcional; "
              + ("entra no próximo treino." if st[STOP_CLASS] >= MIN_STOP_LABELS else "entra no treino ao chegar no mínimo."))
        print(check_enough_labels(st) or "Rótulos suficientes para treinar.")
        print(f"Modelo ativo: {model_registry.get_active_model_path(MODEL_FAMILY) or 'nenhum'}")
        return 0
    if args.command == "mine":
        todo, skipped = mining_candidates(list_videos(args.target))
        print(f"{len(todo)} vídeo(s) novo(s); {skipped['ja_minerados']} já minerado(s), {skipped['nao_luta']} fora (não é luta).")
        added = mine_crops(todo, sample_fps=args.fps, progress=lambda i, n, name: print(f"[{i + 1}/{n}] {name}", flush=True))
        for name, n in added.items():
            print(f"  {n:4d} recortes · {name}")
        return 0
    if args.command == "train":
        try:
            meta = train_flag_classifier(epochs=args.epochs)
        except ValueError as e:
            print(e)
            return 1
        v = meta["validacao"]
        print(f"Modelo: {meta['modelo']}\n  validação ({', '.join(meta['videos_validacao'])}): "
              f"vermelha F1 {_fmt(v['vermelha']['f1'])} · branca F1 {_fmt(v['branca']['f1'])}"
              + (f" · parada F1 {_fmt(v[STOP_CLASS]['f1'])}" if STOP_CLASS in v else "")
              + f" · acurácia {_fmt(v['acuracia'])}")
        if args.activate:
            model_registry.set_active_model(MODEL_FAMILY, meta["modelo"], "Ativado pela linha de comando após o treino.")
            print("  Ativado.")
        return 0
    ev = evaluate_against_gabarito(args.target)
    t = ev["total"]
    for r in ev["lutas"]:
        print(f"  {r['video_name']}: acertos {r['tp']} · falsos {r['fp']} · perdidos {r['fn']}")
    print(f"TOTAL {t['videos']} lutas · precisão {_fmt(t['precisao'])} · recall {_fmt(t['recall'])} · F1 {_fmt(t['f1'])}")
    if args.command == "apply":
        res = apply_to_gabarito(args.target, ev)
        print(res.get("motivo") or f"Gravado em {len(res['lutas'])} luta(s).")
    return 0


if __name__ == "__main__":
    raise SystemExit(_main())
