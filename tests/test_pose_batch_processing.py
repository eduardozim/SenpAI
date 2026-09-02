"""
Testes Automatizados de Processamento em Lote (Batch Inference) e Otimizações de GPU.
Valida o suporte a lotes de frames no PoseDetector, determinação de batch size ideal e integração no SenpAIPipeline.
"""

import unittest
import numpy as np
from unittest.mock import patch, MagicMock

from src.vision.pose_detector import PoseDetector
from src.utils.hardware import get_optimal_batch_size
from src.pipeline import SenpAIPipeline, AsyncVideoBatchReader

class TestPoseBatchProcessing(unittest.TestCase):
    def setUp(self):
        self.detector_cpu = PoseDetector(device="cpu")
        self.dummy_frame1 = np.zeros((480, 640, 3), dtype=np.uint8)
        self.dummy_frame2 = np.ones((480, 640, 3), dtype=np.uint8) * 128

    def test_get_optimal_batch_size_defaults(self):
        """Valida que get_optimal_batch_size retorna 64 para GPU e 1 para CPU."""
        self.assertEqual(get_optimal_batch_size("gpu"), 64)
        self.assertEqual(get_optimal_batch_size("GPU"), 64)
        self.assertEqual(get_optimal_batch_size("cpu"), 1)
        self.assertEqual(get_optimal_batch_size(""), 1)

    def test_get_optimal_batch_size_custom(self):
        """Valida que valores customizados válidos são respeitados com clamping."""
        self.assertEqual(get_optimal_batch_size("gpu", custom_batch_size=16), 16)
        self.assertEqual(get_optimal_batch_size("gpu", custom_batch_size=64), 64)
        self.assertEqual(get_optimal_batch_size("cpu", custom_batch_size=8), 8)
        # Limite superior de segurança (128) e inferior (1)
        self.assertEqual(get_optimal_batch_size("gpu", custom_batch_size=256), 128)
        self.assertEqual(get_optimal_batch_size("gpu", custom_batch_size=0), 64)

    def test_process_frame_candidates_batch_empty(self):
        """Valida que process_frame_candidates_batch retorna lista vazia quando recebe lista vazia de frames."""
        res = self.detector_cpu.process_frame_candidates_batch([])
        self.assertEqual(res, [])

    def test_process_frame_candidates_batch_cpu_parity(self):
        """Valida que no modo CPU o retorno do lote tem o mesmo tamanho e formato que chamadas individuais."""
        frames = [self.dummy_frame1, self.dummy_frame2]
        batch_results = self.detector_cpu.process_frame_candidates_batch(frames)
        
        self.assertIsInstance(batch_results, list)
        self.assertEqual(len(batch_results), 2)
        for cand_list in batch_results:
            self.assertIsInstance(cand_list, list)

    def test_yolo_batch_inference_mock(self):
        """Simula a inferência do YOLO em lote com half=True na GPU para validar a decomposição de resultados."""
        detector = PoseDetector(device="cpu") # Inicializa base
        
        # Simular modelo YOLO com mock
        mock_yolo = MagicMock()
        mock_res1 = MagicMock()
        mock_res2 = MagicMock()
        
        # Simular keypoints do YOLO [1 pessoa, 17 keypoints, 3 coords (px, py, conf)]
        kpts_array = np.zeros((1, 17, 3), dtype=np.float32)
        # Ombro direito (idx 6) e esquerdo (idx 5), Quadril direito (idx 12) e esquerdo (idx 11)
        kpts_array[0, 5] = [200, 150, 0.9] # LEFT_SHOULDER
        kpts_array[0, 6] = [300, 150, 0.9] # RIGHT_SHOULDER
        kpts_array[0, 11] = [220, 300, 0.9] # LEFT_HIP
        kpts_array[0, 12] = [280, 300, 0.9] # RIGHT_HIP
        kpts_array[0, 0] = [250, 100, 0.9] # NOSE

        mock_keypoints = MagicMock()
        mock_tensor = MagicMock()
        mock_tensor.cpu().numpy.return_value = kpts_array
        mock_keypoints.data = mock_tensor
        mock_res1.keypoints = mock_keypoints
        mock_res2.keypoints = mock_keypoints

        mock_yolo.return_value = [mock_res1, mock_res2]
        detector.yolo_model = mock_yolo
        detector.use_gpu = True

        frames = [self.dummy_frame1, self.dummy_frame2]
        batch_results = detector.process_frame_candidates_batch(frames)

        # Validar que o mock_yolo foi chamado com batch=2 na GPU
        mock_yolo.assert_called_once()
        _, kwargs = mock_yolo.call_args
        self.assertEqual(kwargs.get("batch"), 2)
        self.assertEqual(kwargs.get("device"), "cuda:0")

        self.assertEqual(len(batch_results), 2)
        self.assertEqual(len(batch_results[0]), 1)
        self.assertEqual(len(batch_results[1]), 1)
        self.assertIn("NOSE", batch_results[0][0])
        self.assertIn("RIGHT_SHOULDER", batch_results[0][0])
        self.assertIn("LEFT_HIP", batch_results[0][0])

    def test_pipeline_initialization_with_custom_batch_size(self):
        """Valida que o SenpAIPipeline aceita custom_batch_size na inicialização."""
        pipeline = SenpAIPipeline(calibration_profile="normal", device_preference="cpu", custom_batch_size=16)
        self.assertEqual(pipeline.batch_size, 16)

    def test_pipeline_execution_with_batching_on_demo_video(self):
        """Valida a execução completa do pipeline com batch_size configurado sobre vídeo sintético demo."""
        pipeline = SenpAIPipeline(calibration_profile="normal", device_preference="cpu", custom_batch_size=8)
        
        # Usar o vídeo demo existente
        video_path = "demo_kendo_match.mp4"
        result = pipeline.process_video(video_path=video_path, output_video_path=None)

        self.assertIsNotNone(result)
        self.assertIn("scoreboard", result)
        self.assertIn("sonkyo_analysis", result)
        self.assertIn("events", result)

    def test_async_video_batch_reader_lifecycle(self):
        """Valida a leitura assíncrona com prefetching do AsyncVideoBatchReader."""
        video_path = "demo_kendo_match.mp4"
        with AsyncVideoBatchReader(video_path, batch_size=10, max_queue=2) as reader:
            self.assertGreater(reader.total_frames, 0)
            self.assertGreater(reader.fps, 0)
            
            first_batch = reader.read_batch()
            self.assertIsNotNone(first_batch)
            self.assertIsInstance(first_batch, list)
            self.assertLessEqual(len(first_batch), 10)
            
            # Testar parada graciosa
            reader.stop()
            self.assertTrue(reader.stopped)

if __name__ == "__main__":
    unittest.main()
