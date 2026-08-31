"""
Testes Unitários e de Integração para o Módulo de Download e Streaming de Vídeo (video_downloader).
"""

import unittest
from unittest.mock import patch, MagicMock
import os
import tempfile
import cv2

from src.utils.video_downloader import (
    validate_video_url,
    format_video_duration,
    sanitize_filename,
    extract_video_info,
    download_video_stream,
    get_format_selector,
    QUALITY_LABELS,
    VideoDownloadError
)
from src.pipeline import SenpAIPipeline
from src.utils.demo_generator import generate_demo_kendo_video


class TestVideoDownloader(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()

    def tearDown(self):
        self.temp_dir.cleanup()

    def test_validate_video_url_youtube_formats(self):
        """Valida múltiplos formatos oficiais de URLs do YouTube."""
        valid_urls = [
            "https://www.youtube.com/watch?v=dQw4w9WgXcQ",
            "http://www.youtube.com/watch?v=dQw4w9WgXcQ",
            "https://youtube.com/watch?v=dQw4w9WgXcQ&feature=shared",
            "https://youtu.be/dQw4w9WgXcQ",
            "http://youtu.be/dQw4w9WgXcQ?t=42",
            "https://www.youtube.com/shorts/abcdef12345",
            "https://youtube.com/shorts/abcdef12345",
            "https://www.youtube.com/live/liveStream123",
            "https://www.youtube.com/embed/embedVideo123",
            "https://example.com/kendo_match.mp4",
            "http://kendo-stream.org/video.mov",
            "https://cdn.example.org/videos/ippon.mkv",
        ]
        for url in valid_urls:
            self.assertTrue(validate_video_url(url), f"A URL deveria ser válida: {url}")

    def test_validate_video_url_invalid_inputs(self):
        """Valida a rejeição de URLs inválidas, vazias ou formatos incorretos."""
        invalid_urls = [
            "",
            None,
            "   ",
            "not_a_url",
            "ftp://invalid-protocol.com/video.mp4",
            "file:///C:/local/video.mp4",
            "http://",
            "https://",
            12345,
        ]
        for url in invalid_urls:
            self.assertFalse(validate_video_url(url), f"A URL deveria ser inválida: {url}")

    def test_format_video_duration(self):
        """Valida a formatação de duração em segundos para strings amigáveis."""
        self.assertEqual(format_video_duration(0), "00:00")
        self.assertEqual(format_video_duration(None), "00:00")
        self.assertEqual(format_video_duration(-10), "00:00")
        self.assertEqual(format_video_duration(45), "00:45")
        self.assertEqual(format_video_duration(65), "01:05")
        self.assertEqual(format_video_duration(720), "12:00")
        self.assertEqual(format_video_duration(3665), "01:01:05")

    def test_sanitize_filename(self):
        """Valida a remoção de caracteres proibidos em nomes de arquivos."""
        self.assertEqual(sanitize_filename("Kendo: Final Match 2026 / All Japan?"), "Kendo_Final_Match_2026_All_Japan")
        self.assertEqual(sanitize_filename("Video<1>|test*"), "Video1test")
        self.assertEqual(sanitize_filename(""), "video")

    def test_extract_video_info_invalid_url_raises_error(self):
        """Verifica se extract_video_info levanta VideoDownloadError para URLs inválidas."""
        with self.assertRaises(VideoDownloadError):
            extract_video_info("invalid_url_string")

    @patch("yt_dlp.YoutubeDL")
    def test_extract_video_info_success_mock(self, mock_ydl_class):
        """Verifica a extração e estruturação de metadados simulando retorno do yt-dlp."""
        mock_instance = MagicMock()
        mock_instance.extract_info.return_value = {
            "id": "kendo123",
            "title": "71st All Japan Kendo Championship Final",
            "duration": 245.0,
            "uploader": "All Japan Kendo Federation",
            "thumbnail": "https://i.ytimg.com/vi/kendo123/hqdefault.jpg",
            "width": 1920,
            "height": 1080,
            "fps": 60.0,
            "webpage_url": "https://www.youtube.com/watch?v=kendo123",
            "is_live": False,
        }
        mock_ydl_class.return_value.__enter__.return_value = mock_instance

        info = extract_video_info("https://www.youtube.com/watch?v=kendo123")
        self.assertEqual(info["id"], "kendo123")
        self.assertEqual(info["title"], "71st All Japan Kendo Championship Final")
        self.assertEqual(info["duration_formatted"], "04:05")
        self.assertEqual(info["uploader"], "All Japan Kendo Federation")
        self.assertEqual(info["resolution"], "1920x1080")
        self.assertEqual(info["fps"], 60.0)
        self.assertFalse(info["is_live"])

    def test_download_video_stream_cache_reuse(self):
        """Verifica se um vídeo já existente no diretório de cache é reutilizado sem novo download."""
        output_dir = self.temp_dir.name
        cached_file = os.path.join(output_dir, "yt_test123_media_Kendo_Final.mp4")
        
        # Gera um vídeo válido de demonstração com tamanho > 10KB
        generate_demo_kendo_video(cached_file, duration_sec=4, fps=30)
        self.assertTrue(os.path.exists(cached_file))
        self.assertGreater(os.path.getsize(cached_file), 10 * 1024)

        with patch("src.utils.video_downloader.extract_video_info") as mock_info:
            mock_info.return_value = {
                "id": "test123",
                "title": "Kendo Final",
                "duration_seconds": 4.0,
                "duration_formatted": "00:04",
                "uploader": "Kendo Channel",
                "thumbnail": "",
                "resolution": "640x480",
                "fps": 30.0,
                "webpage_url": "https://www.youtube.com/watch?v=test123",
                "is_live": False,
            }

            path, info = download_video_stream("https://www.youtube.com/watch?v=test123", output_dir=output_dir)
            self.assertEqual(path, cached_file)
            self.assertEqual(info["id"], "test123")

    @patch("src.utils.video_downloader.extract_video_info")
    def test_download_video_stream_rejects_live(self, mock_info):
        """Verifica se streams ao vivo não finalizados são rejeitados com erro explicativo."""
        mock_info.return_value = {
            "id": "live123",
            "title": "Live Kendo Tournament",
            "duration_seconds": 0.0,
            "is_live": True,
        }
        with self.assertRaises(VideoDownloadError) as ctx:
            download_video_stream("https://www.youtube.com/watch?v=live123", output_dir=self.temp_dir.name)
        self.assertIn("ao vivo", str(ctx.exception).lower())

    @patch("src.utils.video_downloader.extract_video_info")
    def test_download_video_stream_rejects_excessive_duration(self, mock_info):
        """Verifica se vídeos com duração excessiva são rejeitados."""
        mock_info.return_value = {
            "id": "long123",
            "title": "10 Hours of Kendo",
            "duration_seconds": 36000.0, # 10h
            "is_live": False,
        }
        with self.assertRaises(VideoDownloadError) as ctx:
            download_video_stream("https://www.youtube.com/watch?v=long123", output_dir=self.temp_dir.name, max_duration_seconds=1800)
        self.assertIn("limite máximo", str(ctx.exception).lower())

    def test_pipeline_integration_with_downloaded_video(self):
        """Verifica se o vídeo preparado pelo downloader é compatível com o pipeline completo do SenpAI."""
        output_dir = self.temp_dir.name
        video_path = os.path.join(output_dir, "yt_demo_match.mp4")
        generate_demo_kendo_video(video_path, duration_sec=3, fps=30)

        # Valida que o OpenCV abre normalmente
        cap = cv2.VideoCapture(video_path)
        self.assertTrue(cap.isOpened())
        fps = cap.get(cv2.CAP_PROP_FPS)
        cap.release()
        self.assertGreater(fps, 0)

        # Executa no pipeline
        pipeline = SenpAIPipeline(calibration_profile="normal", device_preference="cpu")
        annotated_out = os.path.join(output_dir, "annotated_yt.mp4")
        result = pipeline.process_video(video_path=video_path, output_video_path=annotated_out)

        self.assertIsNotNone(result)
        self.assertIn("scoreboard", result)
        self.assertIn("sonkyo_analysis", result)
        self.assertTrue(os.path.exists(annotated_out))

    def test_get_format_selector_qualities(self):
        """Valida que get_format_selector retorna as strings de formato corretas para cada qualidade."""
        high_fmt = get_format_selector("alta")
        self.assertIn("bestvideo", high_fmt)
        self.assertIn("bestaudio", high_fmt)

        med_fmt = get_format_selector("media")
        self.assertIn("height<=720", med_fmt)
        self.assertIn("fps<=30", med_fmt)

        low_fmt = get_format_selector("baixa")
        self.assertIn("worst", low_fmt)

        # Default fallback para média
        default_fmt = get_format_selector("qualquer_coisa")
        self.assertEqual(default_fmt, med_fmt)

    def test_download_video_stream_with_quality_cache_tag(self):
        """Verifica que o cache separa downloads por nível de qualidade ('media', 'alta', 'baixa')."""
        output_dir = self.temp_dir.name
        
        with patch("src.utils.video_downloader.extract_video_info") as mock_info:
            mock_info.return_value = {
                "id": "qtest",
                "title": "Match Quality",
                "duration_seconds": 5.0,
                "duration_formatted": "00:05",
                "uploader": "Test Channel",
                "thumbnail": "",
                "resolution": "1280x720",
                "fps": 30.0,
                "webpage_url": "https://www.youtube.com/watch?v=qtest",
                "is_live": False,
            }

            for q in ["media", "alta", "baixa"]:
                cached_file = os.path.join(output_dir, f"yt_qtest_{q}_Match_Quality.mp4")
                generate_demo_kendo_video(cached_file, duration_sec=2, fps=30)
                
                path, info = download_video_stream("https://www.youtube.com/watch?v=qtest", output_dir=output_dir, quality=q)
                self.assertEqual(path, cached_file)
                self.assertEqual(info["quality_selected"], q)
                self.assertEqual(info["quality_label"], QUALITY_LABELS[q])
                self.assertIn("downloaded_resolution", info)
                self.assertIn("downloaded_fps", info)
                self.assertGreater(info["downloaded_file_size_mb"], 0.0)
                self.assertEqual(info["downloaded_resolution"], "640x480")
                self.assertEqual(info["downloaded_fps"], 30.0)


