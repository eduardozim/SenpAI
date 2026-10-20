"""
Testes Unitários para a Fila de Curadoria Ativa:
- Exibição do clipe de vídeo do golpe para Benchmark visual;
- Retreinamento adaptativo disparado pelo retorno do árbitro;
- Exclusão segura do arquivo do clipe de demonstração em disco após a homologação.
"""

import os
import json
import unittest
import shutil
import tempfile

from src.utils.strike_clip_manager import (
    get_or_create_strike_clip,
    delete_strike_clip
)
from src.engine.active_learning import UncertaintySampler
from src.engine.feedback_manager import FeedbackManager
from src.engine.auto_trainer import AutoTrainingEngine


class TestActiveLearningCurationClip(unittest.TestCase):

    def setUp(self):
        self.test_dir = tempfile.mkdtemp()
        self.queue_file = os.path.join(self.test_dir, "test_queue.json")
        self.dataset_file = os.path.join(self.test_dir, "test_feedback.json")
        self.profiles_file = os.path.join(self.test_dir, "test_profiles.json")
        self.clips_dir = os.path.join(self.test_dir, "clips")
        os.makedirs(self.clips_dir, exist_ok=True)

        # Criar vídeo sintético autônomo para testes de recorte de luta
        self.dummy_video_path = os.path.join(self.test_dir, "demo_kendo_match.mp4")
        import cv2
        import numpy as np
        fourcc = cv2.VideoWriter.fourcc(*'mp4v')
        vw = cv2.VideoWriter(self.dummy_video_path, fourcc, 30.0, (320, 240))
        for _ in range(90):
            vw.write(np.zeros((240, 320, 3), dtype=np.uint8))
        vw.release()

        self.sampler = UncertaintySampler(queue_file=self.queue_file, min_threshold=0.45, max_threshold=0.65)
        self.feedback_mgr = FeedbackManager(
            dataset_path=self.dataset_file,
            profiles_path=self.profiles_file
        )
        self.feedback_mgr.uncertainty_sampler = self.sampler
        self.trainer = AutoTrainingEngine(
            profiles_path=self.profiles_file,
            feedback_path=self.dataset_file
        )
        self.trainer.feedback_mgr = self.feedback_mgr

    def tearDown(self):
        shutil.rmtree(self.test_dir, ignore_errors=True)

    def test_get_or_create_strike_clip_and_deletion(self):
        item = {
            "id": "test_unc_men_01",
            "strike_type": "MEN",
            "confidence": 0.52,
            "uncertainty_score": 0.96,
            "video_source": self.dummy_video_path,
            "strike_data": {
                "strike_type": "MEN",
                "scores": {"target_impact": 0.55, "fumikomi_sync": 0.50, "posture": 0.50, "zanshin": 0.50},
                "timestamp_seconds": 1.5
            }
        }

        # 1. Obter e extrair o corte real da luta
        clip_path = get_or_create_strike_clip(item, output_dir=self.clips_dir)
        self.assertIsNotNone(clip_path)
        self.assertTrue(os.path.isfile(clip_path))
        self.assertGreater(os.path.getsize(clip_path), 0)

        # 2. Exclusão segura do clipe
        deleted = delete_strike_clip(clip_path)
        self.assertTrue(deleted)
        self.assertFalse(os.path.isfile(clip_path))

    def test_active_learning_resolution_triggers_retrain_and_clip_cleanup(self):
        # 1. Enfileirar lance incerto com vídeo de luta real
        strike_data = {
            "strike_type": "KOTE",
            "sub_scores": {"target_impact": 0.54, "fumikomi_sync": 0.50, "posture": 0.50, "zanshin": 0.48},
            "total_score": 52.0,
            "timestamp_seconds": 1.8
        }
        enqueued = self.sampler.evaluate_and_enqueue(
            strike_data=strike_data,
            confidence=0.52,
            video_source=self.dummy_video_path,
            profile_name="normal"
        )
        self.assertIsNotNone(enqueued)
        item_id = enqueued["id"]

        # 2. Assegurar que o clipe exista para visualização do benchmark
        clip_path = self.feedback_mgr.ensure_active_learning_clip(enqueued)
        self.assertIsNotNone(clip_path)
        self.assertTrue(os.path.isfile(clip_path))

        # 3. Dar o retorno arbitral: Ippon com retreinamento e exclusão do clipe
        res = self.trainer.resolve_active_learning_item(
            item_id=item_id,
            label_approved=True,
            reviewer_dan=5,
            notes="Homologado via Benchmark Arbitral de Teste",
            trigger_retrain=True,
            cleanup_clip=True
        )

        self.assertTrue(res.get("success"))
        self.assertEqual(res.get("curated_label"), "IPPON")
        self.assertTrue(res.get("retrained"))
        self.assertTrue(res.get("clip_deleted"))

        # 4. Verificar se o clipe de demonstração do golpe foi apagado do disco
        self.assertFalse(os.path.isfile(clip_path), "O arquivo de vídeo do golpe deve ser excluído após a resolução.")

        # 5. Verificar se o item e o clipe foram removidos da fila ativa
        queue = self.sampler.get_queue()
        matched = [it for it in queue if it.get("id") == item_id]
        self.assertEqual(len(matched), 0, "O item curado deve ser removido da lista ativa.")

        # 6. Verificar se o registro foi arquivado no histórico de curadorias
        history = self.sampler.get_history()
        hist_matched = [it for it in history if it.get("id") == item_id]
        self.assertEqual(len(hist_matched), 1, "O item deve estar registrado no histórico de curadorias.")
        self.assertEqual(hist_matched[0]["status"], "curated")
        self.assertTrue(hist_matched[0].get("clip_deleted"))
        self.assertIsNone(hist_matched[0].get("clip_path"))

    def test_three_curation_decision_options(self):
        """
        Valida as 3 opções de retorno arbitral:
        1. IPPON: movimento válido para ponto;
        2. INVALID_STRIKE: houve tentativa de golpe, mas não atendeu aos critérios;
        3. NO_STRIKE: nenhuma tentativa de movimento para pontuação (detecção espúria).
        """
        decisions_to_test = [
            ("IPPON", True, "TP", "IPPON"),
            ("INVALID_STRIKE", False, "FP", "INVALID_STRIKE"),
            ("NO_STRIKE", False, "FP", "NO_STRIKE")
        ]

        for dec, exp_approved, exp_fb_label, exp_cur_label in decisions_to_test:
            with self.subTest(decision=dec):
                strike_data = {
                    "strike_type": "MEN",
                    "scores": {"target_impact": 0.52, "fumikomi_sync": 0.50, "posture": 0.50, "zanshin": 0.50},
                    "total_score": 52.0,
                    "timestamp_seconds": 2.0
                }
                item = self.sampler.evaluate_and_enqueue(
                    strike_data=strike_data,
                    confidence=0.52,
                    video_source=self.dummy_video_path,
                    profile_name="normal"
                )
                self.assertIsNotNone(item)
                it_id = item["id"]

                clip_path = self.feedback_mgr.ensure_active_learning_clip(item)
                self.assertIsNotNone(clip_path)
                self.assertTrue(os.path.isfile(clip_path))

                # Resolver com a decisão específica
                res = self.trainer.resolve_active_learning_item(
                    item_id=it_id,
                    label_approved=exp_approved,
                    reviewer_dan=6,
                    notes=f"Teste para {dec}",
                    trigger_retrain=True,
                    cleanup_clip=True,
                    decision=dec
                )

                self.assertTrue(res.get("success"))
                self.assertEqual(res.get("decision"), dec)
                self.assertEqual(res.get("curated_label"), exp_cur_label)
                self.assertEqual(res.get("approved"), exp_approved)
                self.assertFalse(os.path.isfile(clip_path))

                # Verificar se foi removido da fila ativa
                rem_queue = self.sampler.get_queue()
                self.assertEqual(len([x for x in rem_queue if x.get("id") == it_id]), 0)

                # Verificar histórico arquivado
                hist = self.sampler.get_history()
                h_item = next((x for x in hist if x.get("id") == it_id), None)
                self.assertIsNotNone(h_item)
                self.assertEqual(h_item.get("decision"), dec)
                self.assertEqual(h_item.get("curated_label"), exp_cur_label)


    def test_get_or_create_strike_clip_when_source_video_absent_but_clip_exists(self):
        """
        Garante que em cenários de consolidação servidor -> local (onde o vídeo bruto
        de origem não existe na máquina local, mas o clipe recortado foi sincronizado),
        o clipe de demonstração seja recuperado com sucesso sem exigir o vídeo bruto.
        """
        from src.utils.strike_clip_manager import get_or_create_strike_clip

        dummy_clip_path = os.path.join(self.clips_dir, "test_server_clip_corte.mp4")
        with open(dummy_clip_path, "wb") as f:
            f.write(b"fake_video_clip_content_for_test")

        remote_item = {
            "id": "unc_remote_test_1",
            "video_source": "/var/remote/server/tmp/non_existent_original_fight.mp4",
            "clip_path": dummy_clip_path,
            "strike_type": "MEN",
            "timestamp_seconds": 15.0
        }

        # Deve retornar o clipe existente mesmo sem o arquivo /var/remote/server/...
        resolved = get_or_create_strike_clip(remote_item, output_dir=self.clips_dir)
        self.assertIsNotNone(resolved)
        self.assertTrue(os.path.isfile(resolved))
        self.assertEqual(os.path.abspath(resolved), os.path.abspath(dummy_clip_path))


if __name__ == "__main__":
    unittest.main()
