"""
Módulo de Analytics do ShinpanAI.
Contém motores de detecção de eventos, biomecânica, sonkyo e fusão multi-câmeras.
"""

from src.analytics.event_spotter import EventSpotter, StrikeEvent
from src.analytics.sonkyo_detector import SonkyoDetector
from src.analytics.biomechanics import BiomechanicsAnalyzer
from src.analytics.multi_camera_fusion import (
    MultiCameraFusionEngine,
    CameraFrameEvidence,
    MultiCameraStrikeEvaluation
)
from src.analytics.training_analyzer import (
    TrainingAnalyzer,
    TRAINING_MODALITIES_METADATA,
    TrainingPillarMetrics,
    KendokaTrainingProfile,
    TrainingSessionResult
)

__all__ = [
    "EventSpotter",
    "StrikeEvent",
    "SonkyoDetector",
    "BiomechanicsAnalyzer",
    "MultiCameraFusionEngine",
    "CameraFrameEvidence",
    "MultiCameraStrikeEvaluation",
    "TrainingAnalyzer",
    "TRAINING_MODALITIES_METADATA",
    "TrainingPillarMetrics",
    "KendokaTrainingProfile",
    "TrainingSessionResult"
]
