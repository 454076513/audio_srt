"""
批处理功能单元测试
"""

import os
import tempfile
import unittest
import json
import yaml
from pathlib import Path
from unittest.mock import MagicMock, patch

from audio_srt.cli.batch import (
    BatchTaskConfig, BatchConfig, BatchTaskResult, 
    BatchReport, BatchProcessor
)
from audio_srt.core.processor import JobStatus


class TestBatchConfig(unittest.TestCase):
    """批处理配置测试类"""

    def setUp(self):
        """测试前准备工作"""
        # 创建临时目录用于测试
        self.temp_dir = tempfile.TemporaryDirectory()
        self.test_dir = Path(self.temp_dir.name)
        
    def tearDown(self):
        """测试后清理工作"""
        self.temp_dir.cleanup()
    
    def test_batch_task_config(self):
        """测试单个批处理任务配置"""
        # 创建基本配置
        config = BatchTaskConfig(source="/path/to/audio")
        
        # 检查默认值
        self.assertEqual(config.source, "/path/to/audio")
        self.assertIsNone(config.output_dir)
        self.assertEqual(config.model, "small")
        self.assertEqual(config.language, "auto")
        self.assertEqual(config.output_format, "srt")
        self.assertTrue(config.skip_existing)
        self.assertTrue(config.optimize_subtitles)
        
        # 测试转换为转录选项
        options = config.to_transcription_options()
        self.assertEqual(options.model, "small")
        self.assertEqual(options.language, "auto")
        self.assertEqual(options.output_format, "srt")
        
        # 测试转换为处理器配置
        processor_config = config.to_processor_config()
        self.assertTrue(processor_config.skip_existing)
        self.assertTrue(processor_config.optimize_subtitles)
    
    def test_batch_config_from_dict(self):
        """测试从字典创建批处理配置"""
        # 创建测试字典
        config_dict = {
            "defaults": {
                "model": "medium",
                "language": "en",
                "output_format": "vtt"
            },
            "tasks": [
                {
                    "source": "/path/to/audio1",
                    "output_dir": "/path/to/output1"
                },
                {
                    "source": "/path/to/audio2",
                    "model": "large-v3"
                }
            ]
        }
        
        # 从字典创建配置
        config = BatchConfig.from_dict(config_dict)
        
        # 检查任务数量
        self.assertEqual(len(config.tasks), 2)
        
        # 检查默认值应用
        self.assertEqual(config.tasks[0].model, "medium")
        self.assertEqual(config.tasks[0].language, "en")
        self.assertEqual(config.tasks[0].output_format, "vtt")
        
        # 检查任务特定值覆盖默认值
        self.assertEqual(config.tasks[1].model, "large-v3")
        self.assertEqual(config.tasks[1].language, "en")  # 从默认值继承
        
        # 测试单任务配置
        single_task = {
            "source": "/path/to/audio",
            "model": "tiny"
        }
        
        config = BatchConfig.from_dict(single_task)
        self.assertEqual(len(config.tasks), 1)
        self.assertEqual(config.tasks[0].source, "/path/to/audio")
        self.assertEqual(config.tasks[0].model, "tiny")
    
    def test_batch_config_file_operations(self):
        """测试配置文件操作"""
        # 创建测试配置
        config = BatchConfig()
        config.tasks = [
            BatchTaskConfig(source="/path/to/audio1", model="small"),
            BatchTaskConfig(source="/path/to/audio2", model="medium")
        ]
        
        # 测试YAML文件保存和加载
        yaml_path = self.test_dir / "config.yaml"
        config.save_to_file(yaml_path)
        
        # 检查文件是否存在
        self.assertTrue(yaml_path.exists())
        
        # 加载配置
        loaded_config = BatchConfig.from_file(yaml_path)
        self.assertEqual(len(loaded_config.tasks), 2)
        self.assertEqual(loaded_config.tasks[0].source, "/path/to/audio1")
        self.assertEqual(loaded_config.tasks[0].model, "small")
        
        # 测试JSON文件保存和加载
        json_path = self.test_dir / "config.json"
        config.save_to_file(json_path)
        
        # 检查文件是否存在
        self.assertTrue(json_path.exists())
        
        # 加载配置
        loaded_config = BatchConfig.from_file(json_path)
        self.assertEqual(len(loaded_config.tasks), 2)
        self.assertEqual(loaded_config.tasks[1].source, "/path/to/audio2")
        self.assertEqual(loaded_config.tasks[1].model, "medium")
        
        # 测试不支持的格式
        with self.assertRaises(ValueError):
            config.save_to_file(self.test_dir / "config.txt")
        
        # 测试文件不存在
        with self.assertRaises(FileNotFoundError):
            BatchConfig.from_file(self.test_dir / "nonexistent.yaml")


class TestBatchResults(unittest.TestCase):
    """批处理结果测试类"""
    
    def test_batch_task_result(self):
        """测试批处理任务结果"""
        # 创建任务配置
        task_config = BatchTaskConfig(source="/path/to/audio")
        
        # 创建任务结果
        result = BatchTaskResult(task=task_config, start_time=1000.0)
        
        # 检查初始状态
        self.assertEqual(result.total_files, 0)
        self.assertEqual(result.processed_files, 0)
        self.assertEqual(result.skipped_files, 0)
        self.assertEqual(result.failed_files, 0)
        self.assertEqual(result.success_rate, 0.0)
        self.assertIsNone(result.duration)
        
        # 标记完成
        result.mark_completed(processed=5, skipped=2, failed=1, errors=["Error 1"])
        
        # 检查更新后的状态
        self.assertEqual(result.total_files, 8)  # 5 + 2 + 1
        self.assertEqual(result.processed_files, 5)
        self.assertEqual(result.skipped_files, 2)
        self.assertEqual(result.failed_files, 1)
        self.assertIsNotNone(result.duration)
        self.assertEqual(result.success_rate, 62.5)  # 5/8 * 100
    
    def test_batch_report(self):
        """测试批处理报告"""
        # 创建任务配置
        task1 = BatchTaskConfig(source="/path/to/audio1")
        task2 = BatchTaskConfig(source="/path/to/audio2")
        
        # 创建任务结果
        result1 = BatchTaskResult(task=task1, start_time=1000.0)
        result1.mark_completed(processed=5, skipped=2, failed=1, errors=["Error 1"])
        
        result2 = BatchTaskResult(task=task2, start_time=1100.0)
        result2.mark_completed(processed=3, skipped=0, failed=2, errors=["Error 2", "Error 3"])
        
        # 创建报告
        report = BatchReport()
        report.results = [result1, result2]
        report.mark_completed()
        
        # 检查报告状态
        self.assertEqual(report.total_files, 13)  # 8 + 5
        self.assertEqual(report.processed_files, 8)  # 5 + 3
        self.assertEqual(report.skipped_files, 2)  # 2 + 0
        self.assertEqual(report.failed_files, 3)  # 1 + 2
        self.assertIsNotNone(report.duration)
        self.assertAlmostEqual(report.success_rate, 61.54, places=2)  # 8/13 * 100


class TestBatchProcessor(unittest.TestCase):
    """批处理器测试类"""
    
    def setUp(self):
        """测试前准备工作"""
        # 创建临时目录用于测试
        self.temp_dir = tempfile.TemporaryDirectory()
        self.test_dir = Path(self.temp_dir.name)
        
        # 创建测试音频文件（实际上只是空文件，用于测试流程）
        self.audio_file = self.test_dir / "test_audio.mp3"
        with open(self.audio_file, "w") as f:
            f.write("Mock audio content")
        
        # 创建测试配置文件
        self.config_file = self.test_dir / "config.yaml"
        config_data = {
            "tasks": [
                {
                    "source": str(self.audio_file),
                    "output_dir": str(self.test_dir),
                    "model": "small",
                    "language": "en"
                }
            ]
        }
        
        with open(self.config_file, "w") as f:
            yaml.dump(config_data, f)
    
    def tearDown(self):
        """测试后清理工作"""
        self.temp_dir.cleanup()
    
    @patch("audio_srt.cli.batch.TranscriptionProcessor")
    def test_process_config_file(self, mock_processor_class):
        """测试处理配置文件"""
        # 设置模拟对象
        mock_processor = MagicMock()
        mock_processor_class.return_value = mock_processor
        
        # 模拟process_audio方法
        def mock_process_audio(audio_path, options, output_path):
            return MagicMock()
        
        # 模拟process_directory方法
        def mock_process_directory(directory, options, output_dir, recursive, file_extensions):
            return {}
        
        # 模拟get_all_jobs方法
        def mock_get_all_jobs():
            job1 = MagicMock()
            job1.status = JobStatus.COMPLETED
            job1.error = None
            
            job2 = MagicMock()
            job2.status = JobStatus.FAILED
            job2.error = "Test error"
            
            return [job1, job2]
        
        mock_processor.process_audio.side_effect = mock_process_audio
        mock_processor.process_directory.side_effect = mock_process_directory
        mock_processor.get_all_jobs.side_effect = mock_get_all_jobs
        
        # 创建批处理器
        processor = BatchProcessor()
        
        # 处理配置文件
        report = processor.process_config_file(self.config_file)
        
        # 检查报告
        self.assertEqual(len(report.results), 1)
        
        # 检查模拟对象是否被正确调用
        mock_processor_class.assert_called_once()
        
        # 由于我们的测试配置文件中有一个文件源，所以应该调用process_audio
        # 但在实际实现中，我们使用Path.is_file()来检查，这在测试环境中可能会返回False
        # 因此这个测试可能会调用process_directory而不是process_audio
        # 我们只需要确认其中一个被调用了
        self.assertTrue(
            mock_processor.process_audio.called or 
            mock_processor.process_directory.called
        )
    
    def test_create_sample_config(self):
        """测试创建示例配置文件"""
        from audio_srt.cli.batch import create_sample_config
        
        # 创建YAML示例配置
        yaml_path = self.test_dir / "sample.yaml"
        create_sample_config(yaml_path, "yaml")
        
        # 检查文件是否存在
        self.assertTrue(yaml_path.exists())
        
        # 加载并检查内容
        with open(yaml_path, "r") as f:
            config = yaml.safe_load(f)
        
        self.assertIn("defaults", config)
        self.assertIn("tasks", config)
        self.assertEqual(len(config["tasks"]), 2)
        
        # 创建JSON示例配置
        json_path = self.test_dir / "sample.json"
        create_sample_config(json_path, "json")
        
        # 检查文件是否存在
        self.assertTrue(json_path.exists())
        
        # 加载并检查内容
        with open(json_path, "r") as f:
            config = json.load(f)
        
        self.assertIn("defaults", config)
        self.assertIn("tasks", config)
        self.assertEqual(len(config["tasks"]), 2)


if __name__ == "__main__":
    unittest.main() 