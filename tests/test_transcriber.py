"""
转录器模块的单元测试
"""

import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch, MagicMock

import numpy as np

from audio_srt.core import Transcriber, TranscriptionOptions, TranscriptionResult


class MockWhisperModel:
    """模拟Whisper模型，用于测试"""
    
    def __init__(self, *args, **kwargs):
        pass
    
    def transcribe(self, audio, **kwargs):
        """模拟转录方法"""
        # 创建一个假的转录结果
        mock_segments = [
            MagicMock(
                id=0,
                start=0.0,
                end=2.0,
                text="这是测试文本一",
                words=[
                    MagicMock(start=0.0, end=0.5, word="这是", probability=0.95),
                    MagicMock(start=0.5, end=2.0, word="测试文本一", probability=0.92)
                ]
            ),
            MagicMock(
                id=1,
                start=2.5,
                end=5.0,
                text="这是测试文本二",
                words=[
                    MagicMock(start=2.5, end=3.0, word="这是", probability=0.93),
                    MagicMock(start=3.0, end=5.0, word="测试文本二", probability=0.94)
                ]
            )
        ]
        
        mock_info = MagicMock(
            language="zh",
            language_probability=0.98
        )
        
        return mock_segments, mock_info


class TestTranscriber(unittest.TestCase):
    """转录器单元测试类"""
    
    def setUp(self):
        """测试前设置"""
        # 创建一个临时的测试音频文件
        self.temp_dir = tempfile.mkdtemp()
        self.test_audio_path = os.path.join(self.temp_dir, "test_audio.wav")
        
        # 创建一个简单的WAV文件用于测试
        self._create_test_audio_file(self.test_audio_path)
    
    def tearDown(self):
        """测试后清理"""
        # 删除临时文件
        if os.path.exists(self.test_audio_path):
            os.unlink(self.test_audio_path)
        if os.path.exists(self.temp_dir):
            os.rmdir(self.temp_dir)
    
    def _create_test_audio_file(self, file_path: str, duration: float = 5.0, sample_rate: int = 16000):
        """创建测试音频文件"""
        import wave
        import struct
        
        # 生成一个简单的正弦波
        t = np.linspace(0, duration, int(duration * sample_rate), False)
        audio_data = 0.5 * np.sin(2 * np.pi * 440 * t)  # 440 Hz正弦波
        audio_data = (audio_data * 32767).astype(np.int16)
        
        # 写入WAV文件
        with wave.open(file_path, "wb") as wav_file:
            wav_file.setnchannels(1)  # 单声道
            wav_file.setsampwidth(2)  # 每个样本16位
            wav_file.setframerate(sample_rate)  # 采样率
            for sample in audio_data:
                wav_file.writeframes(struct.pack("<h", sample))
    
    @patch("audio_srt.core.transcriber.ModelManager")
    @patch("audio_srt.models.model_manager.WhisperModel", MockWhisperModel)
    def test_transcribe(self, mock_model_manager):
        """测试转录功能"""
        # 设置模拟模型管理器
        mock_model_manager_instance = mock_model_manager.return_value
        mock_model_manager_instance.load_model.return_value = MockWhisperModel()
        
        # 创建转录选项
        options = TranscriptionOptions(
            model_size="tiny",
            language=None,
            word_timestamps=True
        )
        
        # 创建转录器
        transcriber = Transcriber(options)
        
        # 模拟验证音频文件和获取音频时长的函数
        with patch("audio_srt.core.transcriber.is_valid_audio_file", return_value=True), \
             patch("audio_srt.core.transcriber.get_audio_duration", return_value=5.0), \
             patch("audio_srt.core.transcriber.convert_to_wav", return_value=Path(self.test_audio_path)):
            
            # 执行转录
            result = transcriber.transcribe(self.test_audio_path)
            
            # 验证结果
            self.assertIsInstance(result, TranscriptionResult)
            self.assertEqual(result.language, "zh")
            self.assertEqual(len(result.segments), 2)
            self.assertEqual(result.segments[0]["text"], "这是测试文本一")
            self.assertEqual(result.segments[1]["text"], "这是测试文本二")
            self.assertEqual(result.text, "这是测试文本一 这是测试文本二")
    
    @patch("audio_srt.core.transcriber.ModelManager")
    @patch("audio_srt.models.model_manager.WhisperModel", MockWhisperModel)
    def test_transcribe_with_segments(self, mock_model_manager):
        """测试分段转录功能"""
        # 设置模拟模型管理器
        mock_model_manager_instance = mock_model_manager.return_value
        mock_model_manager_instance.load_model.return_value = MockWhisperModel()
        
        # 创建转录选项
        options = TranscriptionOptions(
            model_size="tiny",
            language=None,
            word_timestamps=True
        )
        
        # 创建转录器
        transcriber = Transcriber(options)
        
        # 模拟验证音频文件、获取音频时长和分割音频的函数
        with patch("audio_srt.core.transcriber.is_valid_audio_file", return_value=True), \
             patch("audio_srt.core.transcriber.get_audio_duration", return_value=15.0), \
             patch("audio_srt.core.transcriber.split_audio", return_value=[Path(self.test_audio_path)]), \
             patch("audio_srt.core.transcriber.convert_to_wav", return_value=Path(self.test_audio_path)):
            
            # 执行分段转录
            result = transcriber.transcribe_with_segments(
                self.test_audio_path,
                segment_duration=10
            )
            
            # 验证结果
            self.assertIsInstance(result, TranscriptionResult)
            self.assertEqual(result.language, "zh")
            self.assertEqual(len(result.segments), 2)
            self.assertEqual(result.text, "这是测试文本一 这是测试文本二")


if __name__ == "__main__":
    unittest.main() 