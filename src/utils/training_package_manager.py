"""
Módulo de Gerenciamento de Pacotes de Análise de Treinamento e Aprendizado de Kendo (SenpAI).
Permite salvar e exportar a análise completa da sessão de treinamento (vídeo analisado/anotado +
dados do relatório dos 3 Pilares e praticantes) em um pacote portátil (.zip), e importar o pacote
para dar continuidade imediata à análise sem necessidade de reprocessamento do vídeo.
"""

import io
import os
import json
import time
import math
import zipfile
import shutil
from typing import Dict, Any, Optional, Union, BinaryIO, Tuple
import numpy as np

from src.utils.logger_manager import log_event
from src.utils.strike_clip_manager import get_senpai_uploads_dir
from src.analytics.training_analyzer import sync_kendoka_names_in_training_data


PACKAGE_IDENTIFIER = "senpai_training_analysis_package"
PACKAGE_VERSION = "1.0"


def sanitize_for_json(obj: Any) -> Any:
    """
    Converte recursivamente tipos numéricos do NumPy (float64, int64), arrays,
    tuplas e objetos serializáveis para estruturas nativas de Python (JSON-safe).
    """
    if isinstance(obj, dict):
        return {str(k): sanitize_for_json(v) for k, v in obj.items()}
    elif isinstance(obj, (list, tuple, set)):
        return [sanitize_for_json(v) for v in obj]
    elif isinstance(obj, (np.floating, float)):
        if math.isnan(obj) or math.isinf(obj):
            return 0.0
        return float(obj)
    elif isinstance(obj, (np.integer, int)):
        return int(obj)
    elif isinstance(obj, (np.bool_, bool)):
        return bool(obj)
    elif isinstance(obj, np.ndarray):
        return sanitize_for_json(obj.tolist())
    elif hasattr(obj, "to_dict") and callable(getattr(obj, "to_dict")):
        return sanitize_for_json(obj.to_dict())
    return obj


def export_training_session_package(
    analysis_result: Dict[str, Any],
    video_path: Optional[str] = None,
    annotated_video_path: Optional[str] = None,
    kendoka_names: Optional[Dict[str, str]] = None,
    invert_aka_shiro: bool = False,
    session_reviews: Optional[Dict[str, Any]] = None,
    destination: Optional[str] = None
) -> Union[bytes, str]:
    """
    Exporta uma sessão de análise de Treinamento & Aprendizado em um arquivo ZIP consolidado.
    Contém:
    - training_session.json: metadados, relatório analítico dos 3 Pilares, kendokas, timeline e diagnósticos.
    - analyzed_video.mp4: vídeo com as anotações visuais (pose, tracking, HUD), se disponível.
    - original_video.mp4: vídeo original da sessão de treino (se diferente do anotado e existente).

    Retorna:
    - Se destination for fornecido: o caminho absoluto do arquivo .zip gerado.
    - Se destination for None: os bytes do arquivo ZIP.
    """
    train_data = analysis_result.get("training_analysis", {})
    mod_key = train_data.get("modality_key", "treinamento")
    mod_name = train_data.get("modality_name", "Treinamento de Kendo")

    has_annotated = bool(annotated_video_path and os.path.isfile(annotated_video_path))
    has_original = bool(video_path and os.path.isfile(video_path))

    # Identificar o nome original do arquivo
    orig_name = "video_treinamento.mp4"
    if has_original and video_path:
        orig_name = os.path.basename(video_path)
    elif has_annotated and annotated_video_path:
        orig_name = os.path.basename(annotated_video_path)

    metadata: Dict[str, Any] = {
        "package_type": PACKAGE_IDENTIFIER,
        "format_version": PACKAGE_VERSION,
        "exported_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "modality_key": mod_key,
        "modality_name": mod_name,
        "duration_seconds": float(analysis_result.get("duration_seconds", 0.0)),
        "total_frames": int(analysis_result.get("total_frames", 0)),
        "fps": float(analysis_result.get("fps", 30.0)),
        "original_filename": orig_name,
        "has_analyzed_video": has_annotated,
        "has_original_video": has_original,
        "sub_sessions_count": len(train_data.get("sub_sessions", [])),
        "modality_blocks_count": len(train_data.get("modality_blocks", [])),
    }

    clean_result = sanitize_for_json(analysis_result)

    # Sincroniza e consolida os nomes customizados dos Kendokas dentro da estrutura analítica
    # (incluindo kendokas globais, sub_sessions e modality_blocks)
    effective_kendoka_names = dict(kendoka_names or {})
    train_analysis = clean_result.get("training_analysis", {})
    sync_kendoka_names_in_training_data(train_analysis, effective_kendoka_names)

    # Atualiza também no analysis_result de origem para consistência de sessão imediata
    orig_train_analysis = analysis_result.get("training_analysis", {})
    if isinstance(orig_train_analysis, dict):
        sync_kendoka_names_in_training_data(orig_train_analysis, effective_kendoka_names)

    package_content = {
        "metadata": metadata,
        "analysis_result": clean_result,
        "training_kendoka_names": effective_kendoka_names,
        "invert_aka_shiro": invert_aka_shiro,
        "session_reviews": session_reviews or {}
    }

    pkg_json_str = json.dumps(package_content, indent=2, ensure_ascii=False)

    buffer: Optional[io.BytesIO] = None
    # Se um caminho de destino foi passado, escreve diretamente no disco
    if destination:
        os.makedirs(os.path.dirname(os.path.abspath(destination)), exist_ok=True)
        zip_target = destination
        zf = zipfile.ZipFile(zip_target, "w")
    else:
        buffer = io.BytesIO()
        zf = zipfile.ZipFile(buffer, "w")

    try:
        # 1. Escreve o JSON de dados e relatório (comprimido)
        zf.writestr("training_session.json", pkg_json_str.encode("utf-8"), compress_type=zipfile.ZIP_DEFLATED)

        # 2. Adiciona o vídeo anotado/analisado (ZIP_STORED para máxima velocidade sem overhead)
        if has_annotated and annotated_video_path:
            zf.write(annotated_video_path, arcname="analyzed_video.mp4", compress_type=zipfile.ZIP_STORED)

        # 3. Adiciona o vídeo original se for diferente do anotado
        if has_original and video_path:
            is_same_file = (has_annotated and annotated_video_path and os.path.abspath(video_path) == os.path.abspath(annotated_video_path))
            if not is_same_file:
                zf.write(video_path, arcname="original_video.mp4", compress_type=zipfile.ZIP_STORED)
            elif not has_annotated:
                zf.write(video_path, arcname="analyzed_video.mp4", compress_type=zipfile.ZIP_STORED)

    finally:
        zf.close()

    log_event("INFO", f"Pacote de análise de treinamento exportado com sucesso (Modalidade: {mod_name}).", "training_package")

    if destination:
        return os.path.abspath(destination)
    elif buffer is not None:
        buffer.seek(0)
        return buffer.getvalue()
    return b""


def is_valid_training_package(package_source: Union[bytes, BinaryIO, str]) -> Tuple[bool, str]:
    """
    Verifica rapidamente se um arquivo ou buffer representa um pacote válido de análise de treinamento.
    Retorna: (is_valid, reason_or_message)
    """
    try:
        if isinstance(package_source, (bytes, bytearray)):
            file_obj: Union[io.BytesIO, str] = io.BytesIO(package_source)
        elif isinstance(package_source, str):
            if not os.path.isfile(package_source):
                return False, f"Arquivo não encontrado: {package_source}"
            file_obj = package_source
        elif hasattr(package_source, "read"):
            package_source.seek(0)
            file_obj = io.BytesIO(package_source.read())
            package_source.seek(0)
        else:
            return False, "Tipo de entrada não suportado para validação de pacote."

        with zipfile.ZipFile(file_obj, "r") as zf:
            file_list = zf.namelist()
            if "training_session.json" not in file_list:
                return False, "Arquivo 'training_session.json' ausente no pacote ZIP."

            with zf.open("training_session.json") as f:
                data = json.load(f)

            if not isinstance(data, dict):
                return False, "Estrutura JSON inválida no pacote de treinamento."

            meta = data.get("metadata", {})
            pkg_type = meta.get("package_type")
            if pkg_type != PACKAGE_IDENTIFIER and "training_analysis" not in data.get("analysis_result", {}):
                return False, "O pacote não é reconhecido como uma análise de treinamento SenpAI válida."

            return True, "Pacote de análise de treinamento válido."
    except zipfile.BadZipFile:
        return False, "O arquivo fornecido não é um arquivo ZIP válido."
    except Exception as e:
        return False, f"Erro ao validar pacote de treinamento: {str(e)}"


def import_training_session_package(
    package_source: Union[bytes, BinaryIO, str],
    extract_dir: Optional[str] = None
) -> Dict[str, Any]:
    """
    Importa um pacote de análise de treinamento (.zip) exportado anteriormente.
    Extrai o vídeo analisado/original para o disco, reconecta os caminhos dos arquivos
    e reconstitui o resultado analítico completo para dar continuidade à análise no Dashboard.

    Retorna:
    Dicionário com:
    - success: bool
    - analysis_result: Dict[str, Any]
    - video_file_path: str (caminho local do vídeo original ou analisado)
    - annotated_output: Optional[str] (caminho do vídeo anotado, se presente)
    - training_kendoka_names: Dict[str, str]
    - invert_aka_shiro: bool
    - session_reviews: Dict[str, Any]
    - metadata: Dict[str, Any]
    - message: str
    """
    if isinstance(package_source, (bytes, bytearray)):
        file_obj: Union[io.BytesIO, str] = io.BytesIO(package_source)
    elif isinstance(package_source, str):
        if not os.path.isfile(package_source):
            raise FileNotFoundError(f"Arquivo de pacote não encontrado: {package_source}")
        file_obj = package_source
    elif hasattr(package_source, "read"):
        package_source.seek(0)
        file_obj = io.BytesIO(package_source.read())
        package_source.seek(0)
    else:
        raise ValueError("Origem do pacote inválida.")

    # Diretório de destino para extração dos arquivos
    if not extract_dir:
        base_uploads = get_senpai_uploads_dir()
        ts_id = int(time.time() * 1000)
        extract_dir = os.path.join(base_uploads, f"training_session_{ts_id}")

    os.makedirs(extract_dir, exist_ok=True)

    with zipfile.ZipFile(file_obj, "r") as zf:
        namelist = zf.namelist()
        if "training_session.json" not in namelist:
            raise ValueError("O arquivo ZIP selecionado não contém 'training_session.json'.")

        # 1. Carregar dados do relatório JSON
        with zf.open("training_session.json") as f_json:
            data = json.load(f_json)

        meta = data.get("metadata", {})
        analysis_result = data.get("analysis_result", {})
        kendoka_names = dict(data.get("training_kendoka_names", {}))
        invert_aka_shiro = data.get("invert_aka_shiro", False)
        session_reviews = data.get("session_reviews", {})

        # Sincroniza os nomes customizados dos Kendokas bidirecionalmente (incluindo sub_sessions e modality_blocks)
        train_analysis = analysis_result.get("training_analysis", {})
        if isinstance(train_analysis, dict):
            sync_kendoka_names_in_training_data(train_analysis, kendoka_names)

        # 2. Extrair vídeos de forma segura contra Zip Slip
        extracted_annotated: Optional[str] = None
        extracted_original: Optional[str] = None

        for item_name in namelist:
            # Proteção contra Directory Traversal (Zip Slip)
            norm_name = os.path.normpath(item_name)
            if norm_name.startswith("..") or os.path.isabs(norm_name):
                continue

            target_path = os.path.join(extract_dir, norm_name)
            target_dir_check = os.path.dirname(target_path)
            os.makedirs(target_dir_check, exist_ok=True)

            if item_name == "analyzed_video.mp4":
                with zf.open(item_name) as src_f, open(target_path, "wb") as dst_f:
                    shutil.copyfileobj(src_f, dst_f)
                extracted_annotated = os.path.abspath(target_path)
            elif item_name == "original_video.mp4":
                with zf.open(item_name) as src_f, open(target_path, "wb") as dst_f:
                    shutil.copyfileobj(src_f, dst_f)
                extracted_original = os.path.abspath(target_path)
            elif item_name.endswith((".mp4", ".mov", ".avi")):
                # Fallback para outros vídeos contidos no zip
                with zf.open(item_name) as src_f, open(target_path, "wb") as dst_f:
                    shutil.copyfileobj(src_f, dst_f)
                if not extracted_original:
                    extracted_original = os.path.abspath(target_path)

    # 3. Resolução dos caminhos de vídeo para o dashboard
    # Se ambos existirem, o original é o video_file_path e o anotado é annotated_output
    # Se apenas o anotado existir, ambos usam o anotado para garantir reprodutibilidade e visualização
    active_video = extracted_original or extracted_annotated
    if not active_video:
        raise ValueError("Nenhum arquivo de vídeo correspondente foi encontrado dentro do pacote.")

    if not extracted_annotated and extracted_original:
        extracted_annotated = extracted_original

    # Atualiza o caminho do vídeo no resultado da análise para bater com o arquivo local recém-extraído
    analysis_result["video_path"] = os.path.abspath(active_video)

    mod_name = meta.get("modality_name", "Treinamento de Kendo")
    kendokas_count = len(analysis_result.get("training_analysis", {}).get("kendokas", []))

    log_event(
        "INFO",
        f"Pacote de treinamento importado com sucesso: {mod_name} ({kendokas_count} praticantes). Vídeo em {active_video}",
        "training_package"
    )

    return {
        "success": True,
        "analysis_result": analysis_result,
        "video_file_path": os.path.abspath(active_video),
        "annotated_output": os.path.abspath(extracted_annotated) if extracted_annotated else None,
        "training_kendoka_names": kendoka_names,
        "invert_aka_shiro": bool(invert_aka_shiro),
        "session_reviews": session_reviews,
        "metadata": meta,
        "message": f"Sessão de treinamento de {mod_name} importada com sucesso ({kendokas_count} praticante{'s' if kendokas_count != 1 else ''})."
    }
