"""
Encoder de Movimento Auto-supervisionado (Pendência I10 — pré-treino com vídeo sem rótulo).

Aprende a resumir ~1 s de movimento de um kenshi (com o oponente e a ponta do shinai) num vetor de
32 números, usando só as poses já guardadas no cache (I1) — nenhum rótulo é necessário.

Aprendizado por contraste: duas versões distorcidas da mesma janela (ruído, escala, atraso, quadros
faltando, espelhamento) devem ter vetores próximos; janelas diferentes, distantes (perda NT-Xent).

- Validação separada por vídeo, medida sem rótulo: "reconhecimento do par" (em lotes de 256 janelas
  de vídeos não vistos, quantas vezes o par certo é o mais parecido). Mede se o pré-treino aprendeu
  algo estável — NÃO se ajuda a julgar golpes.
- Quem decide se ajuda são o filtro de golpes (I3) e o classificador de técnica (I9): eles treinam com
  e sem estes 32 números e ficam com a versão que validar melhor; a promoção ainda exige o golden.
- Registro único de modelos, família "pose_encoder".

Uso via linha de comando (a partir da pasta Dev/):
    python -m src.analytics.pose_encoder train [--epochs 15] [--activate]
    python -m src.analytics.pose_encoder status | deactivate
"""

import os
import json
import math
import random
import datetime
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

from src.utils import model_registry
from src.utils.logger_manager import log_event

MODEL_FAMILY = "pose_encoder"
SCHEMA_VERSION = 1
EMBED_DIM = 32
WINDOW = 32                 # ~1 s a 30 fps
HALF_BEFORE = 20            # o impacto fica perto do fim da janela (preparo + golpe)
STRIDE = 8
MIN_PRESENT = 0.7
JOINTS = ["LEFT_SHOULDER", "RIGHT_SHOULDER", "LEFT_ELBOW", "RIGHT_ELBOW", "LEFT_WRIST", "RIGHT_WRIST",
          "LEFT_HIP", "RIGHT_HIP", "LEFT_KNEE", "RIGHT_KNEE", "LEFT_ANKLE", "RIGHT_ANKLE"]
FRAME_DIM = (len(JOINTS) + 2) * 2 + 4 + 1   # juntas + cabeça + ponta do shinai, oponente (quadril e cabeça), presença


def embedding_feature_names() -> List[str]:
    return [f"emb_{i}" for i in range(EMBED_DIM)]


# ------------------------------------------------------------------------------
# Janelas de pose normalizadas
# ------------------------------------------------------------------------------

def _xy(pose, name):
    lm = pose.get(name) if pose else None
    if isinstance(lm, dict) and "px" in lm:
        return float(lm["px"]), float(lm["py"])
    return None


def _mid(a, b):
    if a is None or b is None:
        return a or b
    return (a[0] + b[0]) / 2.0, (a[1] + b[1]) / 2.0


def _tip(pose):
    sh = pose.get("SHINAI") if pose else None
    if isinstance(sh, dict) and sh.get("detected") and sh.get("tip_px"):
        return float(sh["tip_px"][0]), float(sh["tip_px"][1])
    return None


def window_tensor(attacker: Sequence[Optional[dict]], defender: Sequence[Optional[dict]], center: int) -> Optional[np.ndarray]:
    """Janela [WINDOW, FRAME_DIM] centrada no quadril do kenshi no quadro de referência, em unidades de tronco."""
    n = len(attacker)
    ref = attacker[center] if 0 <= center < n else None
    hip = _mid(_xy(ref, "LEFT_HIP"), _xy(ref, "RIGHT_HIP"))
    sh = _mid(_xy(ref, "LEFT_SHOULDER"), _xy(ref, "RIGHT_SHOULDER"))
    if hip is None or sh is None:
        return None
    torso = math.hypot(sh[0] - hip[0], sh[1] - hip[1])
    if torso < 5:
        return None
    out = np.zeros((WINDOW, FRAME_DIM), dtype=np.float32)
    present = 0
    for k in range(WINDOW):
        i = center - HALF_BEFORE + k
        pose = attacker[i] if 0 <= i < n else None
        if not pose:
            continue
        present += 1
        pts = [_xy(pose, j) for j in JOINTS]
        pts.append(_mid(_xy(pose, "LEFT_EAR"), _xy(pose, "RIGHT_EAR")))
        pts.append(_tip(pose))
        opp = defender[i] if 0 <= i < len(defender) else None
        pts.append(_mid(_xy(opp, "LEFT_HIP"), _xy(opp, "RIGHT_HIP")))
        pts.append(_mid(_xy(opp, "LEFT_EAR"), _xy(opp, "RIGHT_EAR")))
        row = []
        for p in pts:
            row.extend([0.0, 0.0] if p is None else [(p[0] - hip[0]) / torso, (p[1] - hip[1]) / torso])
        row.append(1.0)
        out[k] = np.clip(np.array(row, dtype=np.float32), -8.0, 8.0)
    return out if present / WINDOW >= MIN_PRESENT else None


def collect_windows(max_windows: int = 60000, seed: int = 7) -> Tuple[np.ndarray, List[str]]:
    """Janelas de todas as lutas do cache (exceto vídeos marcados como "não é luta"), com o vídeo de origem."""
    import gzip
    import pickle
    from src.utils import feature_cache
    from src.engine import weak_labels as wl
    from src.vision.combatant_tracker import CombatantTracker
    labels = wl.load_match_labels()["videos"]
    tracker = CombatantTracker()
    cache_dir = feature_cache.get_feature_cache_dir()
    windows, groups = [], []
    for name in sorted(os.listdir(cache_dir)):
        if not name.endswith(".pkl.gz"):
            continue
        try:
            with gzip.open(os.path.join(cache_dir, name), "rb") as f:
                data = pickle.load(f)
        except Exception:
            continue
        video = str(data.get("video_name") or name)
        if wl.is_entry_ignored(labels.get(video)):
            continue
        aka = tracker.interpolate_missing_poses(data["aka_history"], max_gap=10)
        shiro = tracker.interpolate_missing_poses(data["shiro_history"], max_gap=10)
        for atk, dfd in ((aka, shiro), (shiro, aka)):
            for c in range(HALF_BEFORE, len(atk) - (WINDOW - HALF_BEFORE), STRIDE):
                w = window_tensor(atk, dfd, c)
                if w is not None:
                    windows.append(w)
                    groups.append(video)
    if len(windows) > max_windows:
        idx = random.Random(seed).sample(range(len(windows)), max_windows)
        windows = [windows[i] for i in idx]
        groups = [groups[i] for i in idx]
    return (np.stack(windows) if windows else np.zeros((0, WINDOW, FRAME_DIM), np.float32)), groups


# ------------------------------------------------------------------------------
# Rede e treino por contraste
# ------------------------------------------------------------------------------

def _build_net():
    import torch.nn as nn

    class Encoder(nn.Module):
        def __init__(self):
            super().__init__()
            self.body = nn.Sequential(
                nn.Conv1d(FRAME_DIM, 64, 5, padding=2), nn.ReLU(),
                nn.Conv1d(64, 64, 5, stride=2, padding=2), nn.ReLU(),
                nn.Conv1d(64, 128, 3, stride=2, padding=1), nn.ReLU(),
                nn.AdaptiveAvgPool1d(1),
            )
            self.embed = nn.Linear(128, EMBED_DIM)
            self.project = nn.Sequential(nn.ReLU(), nn.Linear(EMBED_DIM, EMBED_DIM))

        def forward(self, x):  # x: [B, WINDOW, FRAME_DIM]
            h = self.body(x.transpose(1, 2)).squeeze(-1)
            return self.embed(h)

    return Encoder()


def augment(x, rng=None):
    """Distorções que não mudam "qual movimento é": ruído, escala, atraso, quadros faltando, espelhamento."""
    import torch
    b = x.shape[0]
    out = x.clone()
    coords = out[..., :-1]
    coords *= torch.empty(b, 1, 1, device=x.device).uniform_(0.85, 1.15)
    coords += torch.randn_like(coords) * 0.03
    flip = (torch.rand(b, device=x.device) < 0.5)
    coords[flip, :, 0::2] *= -1
    shift = int(torch.randint(-3, 4, (1,)).item())
    out = torch.roll(out, shifts=shift, dims=1)
    drop = (torch.rand(b, out.shape[1], 1, device=x.device) < 0.1)
    out = out.masked_fill(drop, 0.0)
    return out


def nt_xent(z1, z2, temperature: float = 0.2):
    import torch
    import torch.nn.functional as F
    z = F.normalize(torch.cat([z1, z2]), dim=1)
    sim = z @ z.t() / temperature
    n = z1.shape[0]
    sim.fill_diagonal_(-1e9)
    targets = torch.cat([torch.arange(n, 2 * n), torch.arange(0, n)]).to(z.device)
    return F.cross_entropy(sim, targets)


def pair_recognition(net, X: np.ndarray, device: str, batch: int = 256, seed: int = 0) -> Optional[float]:
    """Sem rótulo: em lotes de janelas não vistas, quantas vezes a versão distorcida certa é a mais parecida."""
    import torch
    import torch.nn.functional as F
    if len(X) < 8:
        return None
    torch.manual_seed(seed)
    net.eval()
    hits = total = 0
    with torch.no_grad():
        for s in range(0, len(X) - 1, batch):
            xb = torch.from_numpy(X[s:s + batch]).to(device)
            if len(xb) < 8:
                break
            a = F.normalize(net(augment(xb)), dim=1)
            b = F.normalize(net(augment(xb)), dim=1)
            hits += int((torch.argmax(a @ b.t(), dim=1) == torch.arange(len(xb), device=device)).sum())
            total += len(xb)
    return round(hits / total, 4) if total else None


def split_by_video(groups: List[str], val_fraction: float = 0.25, seed: int = 7) -> Tuple[np.ndarray, np.ndarray]:
    videos = sorted(set(groups))
    random.Random(seed).shuffle(videos)
    n_val = max(1, int(round(len(videos) * val_fraction))) if len(videos) > 1 else 0
    val = set(videos[:n_val])
    g = np.array(groups)
    return np.where(~np.isin(g, list(val)))[0], np.where(np.isin(g, list(val)))[0]


def train_encoder(X: Optional[np.ndarray] = None, groups: Optional[List[str]] = None, epochs: int = 15,
                  batch: int = 512, device: Optional[str] = None, models_dir: Optional[str] = None,
                  seed: int = 7) -> Dict[str, Any]:
    """Pré-treina o encoder nas janelas do cache e salva a versão (ainda não ativa)."""
    import torch
    if X is None:
        X, groups = collect_windows()
    if len(X) < 512 or len(set(groups or [])) < 2:
        raise ValueError(f"Poucas janelas de pose para pré-treinar ({len(X)} de {len(set(groups or []))} vídeo(s)); "
                         "são precisas ≥ 512 janelas de ≥ 2 lutas no cache.")
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(seed)
    train_idx, val_idx = split_by_video(groups, seed=seed)
    net = _build_net().to(device)
    opt = torch.optim.Adam(net.parameters(), lr=1e-3)
    Xt = torch.from_numpy(X[train_idx])
    before = pair_recognition(net, X[val_idx], device)
    losses = []
    for _ in range(epochs):
        net.train()
        perm = torch.randperm(len(Xt))
        total = 0.0
        steps = 0
        for s in range(0, len(Xt) - batch + 1, batch):
            xb = Xt[perm[s:s + batch]].to(device)
            loss = nt_xent(net.project(net(augment(xb))), net.project(net(augment(xb))))
            opt.zero_grad()
            loss.backward()
            opt.step()
            total += loss.item()
            steps += 1
        losses.append(round(total / max(1, steps), 4))
    after = pair_recognition(net, X[val_idx], device)

    stamp = datetime.datetime.now().strftime("%Y%m%dT%H%M%S")
    d = models_dir or model_registry.get_models_dir(MODEL_FAMILY)
    weights = os.path.join(d, f"pose_encoder_{stamp}.pt")
    torch.save(net.state_dict(), weights)
    meta = {
        "schema": SCHEMA_VERSION, "modelo": weights, "treinado_em": stamp, "embed_dim": EMBED_DIM,
        "janela": WINDOW, "frame_dim": FRAME_DIM, "epocas": epochs, "perda_por_epoca": losses,
        "janelas": {"treino": int(len(train_idx)), "validacao": int(len(val_idx))},
        "videos_validacao": sorted({groups[i] for i in val_idx}),
        "reconhecimento_do_par": {"antes_do_treino": before, "depois_do_treino": after},
    }
    with open(weights.replace(".pt", ".json"), "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2, ensure_ascii=False)
    log_event("INFO", f"Encoder de pose pré-treinado: reconhecimento do par {before} → {after}.", "pose_encoder")
    return meta


class PoseEncoder:
    """Aplica o encoder: janela de pose ao redor do golpe → 32 números."""

    _cache: Dict[str, "PoseEncoder"] = {}

    def __init__(self, weights_path: str, device: Optional[str] = None):
        import torch
        self.path = weights_path
        self.device = device or "cpu"
        self.net = _build_net()
        self.net.load_state_dict(torch.load(weights_path, map_location=self.device))
        self.net.to(self.device).eval()

    @classmethod
    def from_path(cls, path: Optional[str]) -> Optional["PoseEncoder"]:
        if not path or not os.path.isfile(path):
            return None
        if path not in cls._cache:
            cls._cache[path] = cls(path)
        return cls._cache[path]

    @classmethod
    def load_active(cls) -> Optional["PoseEncoder"]:
        return cls.from_path(model_registry.get_active_model_path(MODEL_FAMILY))

    def embed(self, attacker, defender, impact_frame: int) -> np.ndarray:
        import torch
        w = window_tensor(attacker, defender, int(impact_frame))
        if w is None:
            return np.zeros(EMBED_DIM, dtype=np.float32)
        with torch.no_grad():
            return self.net(torch.from_numpy(w[None]).to(self.device)).cpu().numpy()[0]


# ------------------------------------------------------------------------------
# Apoio aos classificadores (I3, I9): mesma lista de features com ou sem o encoder
# ------------------------------------------------------------------------------

def with_embedding(base: Dict[str, float], encoder: Optional[PoseEncoder], attacker, defender, impact_frame: int) -> Dict[str, float]:
    """Acrescenta emb_0..emb_31 às features quando há encoder (senão devolve as features base)."""
    if encoder is None:
        return base
    out = dict(base)
    for k, v in zip(embedding_feature_names(), encoder.embed(attacker, defender, impact_frame)):
        out[k] = float(v)
    return out


def feature_names_for(base_names: List[str], uses_encoder: bool) -> List[str]:
    return base_names + (embedding_feature_names() if uses_encoder else [])


def _main(argv: Optional[List[str]] = None) -> int:
    import argparse
    parser = argparse.ArgumentParser(description="Encoder de movimento auto-supervisionado (pré-treino sem rótulo).")
    sub = parser.add_subparsers(dest="command", required=True)
    p_t = sub.add_parser("train")
    p_t.add_argument("--epochs", type=int, default=15)
    p_t.add_argument("--activate", action="store_true")
    sub.add_parser("status")
    sub.add_parser("deactivate")
    args = parser.parse_args(argv)
    if args.command == "status":
        print(f"Ativo: {model_registry.get_active_model_path(MODEL_FAMILY) or 'nenhum'}")
        return 0
    if args.command == "deactivate":
        model_registry.set_active_model(MODEL_FAMILY, None, "Desativado pela linha de comando.")
        return 0
    X, groups = collect_windows()
    print(f"{len(X)} janelas de pose de {len(set(groups))} luta(s) do cache.")
    try:
        meta = train_encoder(X, groups, epochs=args.epochs)
    except ValueError as e:
        print(e)
        return 1
    r = meta["reconhecimento_do_par"]
    print(f"Encoder: {meta['modelo']}\n  perda por época: {meta['perda_por_epoca']}\n"
          f"  reconhecimento do par em lutas não vistas: {r['antes_do_treino']} → {r['depois_do_treino']}")
    if args.activate:
        model_registry.set_active_model(MODEL_FAMILY, meta["modelo"], "Ativado após o pré-treino.")
        print("  Ativado.")
    return 0


if __name__ == "__main__":
    raise SystemExit(_main())
