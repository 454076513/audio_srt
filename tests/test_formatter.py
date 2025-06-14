"""
字幕格式化器模块的单元测试
"""

import json
import os
import tempfile
import unittest
from pathlib import Path

from audio_srt.core import (
    TranscriptionResult,
    FormatterFactory,
    SRTFormatter,
    WebVTTFormatter,
    JSONFormatter,
    TextFormatter
)


class TestFormatter(unittest.TestCase):
    """字幕格式化器单元测试类"""
    
    def setUp(self):
        """测试前设置"""
        # 创建测试用的转录结果
        segments = [
            {
                "id": 0,
                "start": 0.0,
                "end": 2.5,
                "text": "这是第一行字幕"
            },
            {
                "id": 1,
                "start": 3.2,
                "end": 5.8,
                "text": "这是第二行字幕"
            }
        ]
        
        word_timestamps = [
            {
                "start": 0.0,
                "end": 0.8,
                "word": "这是",
                "probability": 0.95
            },
            {
                "start": 0.9,
                "end": 2.5,
                "word": "第一行字幕",
                "probability": 0.92
            },
            {
                "start": 3.2,
                "end": 4.0,
                "word": "这是",
                "probability": 0.93
            },
            {
                "start": 4.1,
                "end": 5.8,
                "word": "第二行字幕",
                "probability": 0.94
            }
        ]
        
        self.transcription = TranscriptionResult(
            segments=segments,
            language="zh",
            duration=6.0,
            word_timestamps=word_timestamps
        )
        
        # 创建临时目录用于保存测试文件
        self.temp_dir = tempfile.mkdtemp()
    
    def tearDown(self):
        """测试后清理"""
        # 删除临时目录中的所有文件
        for file_name in os.listdir(self.temp_dir):
            file_path = os.path.join(self.temp_dir, file_name)
            if os.path.isfile(file_path):
                os.unlink(file_path)
        
        # 删除临时目录
        os.rmdir(self.temp_dir)
    
    def test_srt_formatter(self):
        """测试SRT格式化器"""
        formatter = SRTFormatter()
        srt_content = formatter.format(self.transcription)
        
        # 验证SRT格式
        self.assertIn("1", srt_content)
        self.assertIn("00:00:00,000 --> 00:00:02,500", srt_content)
        self.assertIn("这是第一行字幕", srt_content)
        self.assertIn("2", srt_content)
        self.assertIn("00:00:03,200 --> 00:00:05,800", srt_content)
        self.assertIn("这是第二行字幕", srt_content)
        
        # 测试保存功能
        output_path = Path(self.temp_dir) / "test.srt"
        formatter.save(self.transcription, output_path)
        
        self.assertTrue(output_path.exists())
        with open(output_path, "r", encoding="utf-8") as f:
            saved_content = f.read()
        
        self.assertEqual(saved_content, srt_content)
    
    def test_webvtt_formatter(self):
        """测试WebVTT格式化器"""
        formatter = WebVTTFormatter()
        vtt_content = formatter.format(self.transcription)
        
        # 验证WebVTT格式
        self.assertIn("WEBVTT", vtt_content)
        self.assertIn("1", vtt_content)
        self.assertIn("00:00:00.000 --> 00:00:02.500", vtt_content)
        self.assertIn("这是第一行字幕", vtt_content)
        self.assertIn("2", vtt_content)
        self.assertIn("00:00:03.200 --> 00:00:05.800", vtt_content)
        self.assertIn("这是第二行字幕", vtt_content)
    
    def test_json_formatter(self):
        """测试JSON格式化器"""
        # 测试不包含词级时间戳
        formatter_no_words = JSONFormatter(include_words=False)
        json_content_no_words = formatter_no_words.format(self.transcription)
        
        # 解析JSON
        json_data_no_words = json.loads(json_content_no_words)
        
        # 验证JSON格式
        self.assertEqual(json_data_no_words["text"], "这是第一行字幕 这是第二行字幕")
        self.assertEqual(len(json_data_no_words["segments"]), 2)
        self.assertEqual(json_data_no_words["segments"][0]["text"], "这是第一行字幕")
        self.assertEqual(json_data_no_words["language"], "zh")
        self.assertEqual(json_data_no_words["duration"], 6.0)
        self.assertNotIn("words", json_data_no_words)
        
        # 测试包含词级时间戳
        formatter_with_words = JSONFormatter(include_words=True)
        json_content_with_words = formatter_with_words.format(self.transcription)
        
        # 解析JSON
        json_data_with_words = json.loads(json_content_with_words)
        
        # 验证JSON格式
        self.assertIn("words", json_data_with_words)
        self.assertEqual(len(json_data_with_words["words"]), 4)
        self.assertEqual(json_data_with_words["words"][0]["word"], "这是")
    
    def test_text_formatter(self):
        """测试纯文本格式化器"""
        # 测试不包含时间戳
        formatter_no_timestamps = TextFormatter(include_timestamps=False)
        text_content_no_timestamps = formatter_no_timestamps.format(self.transcription)
        
        # 验证纯文本格式
        self.assertEqual(text_content_no_timestamps, "这是第一行字幕\n这是第二行字幕")
        
        # 测试包含时间戳
        formatter_with_timestamps = TextFormatter(include_timestamps=True)
        text_content_with_timestamps = formatter_with_timestamps.format(self.transcription)
        
        # 验证带时间戳的文本格式
        self.assertIn("[00:00:00 --> 00:00:02] 这是第一行字幕", text_content_with_timestamps)
        self.assertIn("[00:00:03 --> 00:00:05] 这是第二行字幕", text_content_with_timestamps)
    
    def test_formatter_factory(self):
        """测试格式化器工厂"""
        # 测试获取不同类型的格式化器
        srt_formatter = FormatterFactory.get_formatter("srt")
        vtt_formatter = FormatterFactory.get_formatter("vtt")
        json_formatter = FormatterFactory.get_formatter("json")
        text_formatter = FormatterFactory.get_formatter("txt")
        
        self.assertIsInstance(srt_formatter, SRTFormatter)
        self.assertIsInstance(vtt_formatter, WebVTTFormatter)
        self.assertIsInstance(json_formatter, JSONFormatter)
        self.assertIsInstance(text_formatter, TextFormatter)
        
        # 测试获取可用格式
        available_formats = FormatterFactory.get_available_formats()
        self.assertIn("srt", available_formats)
        self.assertIn("vtt", available_formats)
        self.assertIn("json", available_formats)
        self.assertIn("txt", available_formats)
        
        # 测试不支持的格式
        with self.assertRaises(ValueError):
            FormatterFactory.get_formatter("unknown")


if __name__ == "__main__":
    unittest.main() 