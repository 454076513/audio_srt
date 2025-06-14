"""
并行处理器单元测试
"""

import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

from audio_srt.core.processor import (
    TranscriptionProcessor, 
    ProcessorConfig, 
    TranscriptionJob, 
    JobStatus
)
from audio_srt.core.transcriber import TranscriptionOptions, TranscriptionResult


class TestTranscriptionProcessor(unittest.TestCase):
    """转录处理器测试类"""

    def setUp(self):
        """测试前准备工作"""
        # 创建临时目录用于测试
        self.temp_dir = tempfile.TemporaryDirectory()
        self.test_dir = Path(self.temp_dir.name)
        
        # 创建测试音频文件（实际上只是空文件，用于测试流程）
        self.audio_files = []
        for i in range(3):
            audio_path = self.test_dir / f"test_audio_{i}.mp3"
            with open(audio_path, "w") as f:
                f.write(f"Mock audio content {i}")
            self.audio_files.append(audio_path)
        
        # 创建模拟的转录器
        self.mock_transcriber = MagicMock()
        
        # 设置转录器的返回值
        def mock_transcribe(audio_path, options=None):
            result = TranscriptionResult(
                text=f"Transcription for {Path(audio_path).name}",
                segments=[{"start": 0.0, "end": 1.0, "text": f"Segment for {Path(audio_path).name}"}],
                language="en"
            )
            return result
        
        self.mock_transcriber.transcribe.side_effect = mock_transcribe
        
        # 默认转录选项
        self.options = TranscriptionOptions(
            output_format="txt"
        )

    def tearDown(self):
        """测试后清理工作"""
        self.temp_dir.cleanup()

    def test_job_lifecycle(self):
        """测试转录任务的生命周期"""
        audio_path = self.audio_files[0]
        job = TranscriptionJob(
            audio_path=audio_path,
            output_path=None,  # 将自动生成
            options=self.options
        )
        
        # 检查初始状态
        self.assertEqual(job.status, JobStatus.PENDING)
        self.assertIsNone(job.error)
        self.assertIsNone(job.result)
        self.assertEqual(job.progress, 0.0)
        self.assertIsNone(job.start_time)
        self.assertIsNone(job.end_time)
        
        # 检查自动生成的输出路径
        self.assertEqual(job.output_path, audio_path.with_suffix(".txt"))
        
        # 测试状态转换
        job.mark_running()
        self.assertEqual(job.status, JobStatus.RUNNING)
        self.assertIsNotNone(job.start_time)
        
        # 创建测试结果
        result = TranscriptionResult(
            text="Test transcription",
            segments=[],
            language="en"
        )
        
        job.mark_completed(result)
        self.assertEqual(job.status, JobStatus.COMPLETED)
        self.assertEqual(job.result, result)
        self.assertEqual(job.progress, 1.0)
        self.assertIsNotNone(job.end_time)
        self.assertIsNotNone(job.duration)
        
        # 测试失败状态
        job = TranscriptionJob(
            audio_path=audio_path,
            output_path=None,
            options=self.options
        )
        
        job.mark_running()
        job.mark_failed("Test error")
        self.assertEqual(job.status, JobStatus.FAILED)
        self.assertEqual(job.error, "Test error")
        self.assertIsNotNone(job.end_time)
        
        # 测试跳过状态
        job = TranscriptionJob(
            audio_path=audio_path,
            output_path=None,
            options=self.options
        )
        
        job.mark_skipped("Already exists")
        self.assertEqual(job.status, JobStatus.SKIPPED)
        self.assertEqual(job.error, "Already exists")
        self.assertEqual(job.progress, 1.0)

    def test_processor_config(self):
        """测试处理器配置"""
        # 测试默认配置
        config = ProcessorConfig()
        self.assertEqual(config.max_workers, 0)  # 自动
        self.assertTrue(config.skip_existing)
        self.assertFalse(config.force_overwrite)
        
        # 测试自定义配置
        config = ProcessorConfig(
            max_workers=4,
            batch_size=2,
            skip_existing=False,
            force_overwrite=True
        )
        self.assertEqual(config.max_workers, 4)
        self.assertEqual(config.batch_size, 2)
        self.assertFalse(config.skip_existing)
        self.assertTrue(config.force_overwrite)
        
        # 测试有效工作线程计算
        self.assertEqual(config.get_effective_workers(), 4)
        
        # 测试无效配置
        with self.assertRaises(ValueError):
            config = ProcessorConfig(max_workers=-1)
            
        with self.assertRaises(ValueError):
            config = ProcessorConfig(batch_size=0)

    def test_process_batch_empty(self):
        """测试处理空批次"""
        processor = TranscriptionProcessor(
            transcriber=self.mock_transcriber,
            config=ProcessorConfig()
        )
        
        results = processor.process_batch([])
        self.assertEqual(len(results), 0)

    def test_process_single_job(self):
        """测试处理单个任务"""
        processor = TranscriptionProcessor(
            transcriber=self.mock_transcriber,
            config=ProcessorConfig(
                optimize_subtitles=False  # 关闭字幕优化以简化测试
            )
        )
        
        audio_path = self.audio_files[0]
        output_path = self.test_dir / "output.txt"
        
        job = TranscriptionJob(
            audio_path=audio_path,
            output_path=output_path,
            options=self.options
        )
        
        # 处理任务
        results = processor.process_batch([job])
        
        # 验证结果
        self.assertEqual(len(results), 1)
        self.assertEqual(job.status, JobStatus.COMPLETED)
        self.assertIsNotNone(job.result)
        self.assertTrue(output_path.exists())
        
        # 检查文件内容
        with open(output_path, "r") as f:
            content = f.read()
            self.assertIn(f"Transcription for {audio_path.name}", content)

    def test_skip_existing(self):
        """测试跳过已存在的文件"""
        # 先创建输出文件
        output_path = self.test_dir / "existing.txt"
        with open(output_path, "w") as f:
            f.write("Existing content")
        
        # 创建处理器
        processor = TranscriptionProcessor(
            transcriber=self.mock_transcriber,
            config=ProcessorConfig(
                skip_existing=True
            )
        )
        
        # 创建任务
        job = TranscriptionJob(
            audio_path=self.audio_files[0],
            output_path=output_path,
            options=self.options
        )
        
        # 处理任务
        results = processor.process_batch([job])
        
        # 验证结果
        self.assertEqual(len(results), 0)  # 没有结果
        self.assertEqual(job.status, JobStatus.SKIPPED)
        
        # 检查文件内容未改变
        with open(output_path, "r") as f:
            content = f.read()
            self.assertEqual(content, "Existing content")
        
        # 使用强制覆盖模式
        processor = TranscriptionProcessor(
            transcriber=self.mock_transcriber,
            config=ProcessorConfig(
                skip_existing=True,
                force_overwrite=True
            )
        )
        
        # 重置任务状态
        job.status = JobStatus.PENDING
        
        # 处理任务
        results = processor.process_batch([job])
        
        # 验证结果
        self.assertEqual(len(results), 1)
        self.assertEqual(job.status, JobStatus.COMPLETED)
        
        # 检查文件内容已更改
        with open(output_path, "r") as f:
            content = f.read()
            self.assertIn(f"Transcription for {self.audio_files[0].name}", content)

    def test_process_multiple_jobs(self):
        """测试处理多个任务"""
        processor = TranscriptionProcessor(
            transcriber=self.mock_transcriber,
            config=ProcessorConfig(
                max_workers=2,  # 使用2个工作线程
                optimize_subtitles=False
            )
        )
        
        # 创建多个任务
        jobs = []
        for i, audio_path in enumerate(self.audio_files):
            output_path = self.test_dir / f"output_{i}.txt"
            job = TranscriptionJob(
                audio_path=audio_path,
                output_path=output_path,
                options=self.options
            )
            jobs.append(job)
        
        # 处理任务
        results = processor.process_batch(jobs)
        
        # 验证结果
        self.assertEqual(len(results), len(jobs))
        for job in jobs:
            self.assertEqual(job.status, JobStatus.COMPLETED)
            self.assertTrue(job.output_path.exists())
            
            # 检查文件内容
            with open(job.output_path, "r") as f:
                content = f.read()
                self.assertIn(f"Transcription for {job.audio_path.name}", content)

    def test_process_directory(self):
        """测试处理目录"""
        processor = TranscriptionProcessor(
            transcriber=self.mock_transcriber,
            config=ProcessorConfig(
                optimize_subtitles=False
            )
        )
        
        # 创建输出目录
        output_dir = self.test_dir / "output"
        output_dir.mkdir(exist_ok=True)
        
        # 处理目录
        with patch('pathlib.Path.iterdir') as mock_iterdir:
            # 模拟目录内容
            mock_iterdir.return_value = self.audio_files
            
            results = processor.process_directory(
                directory=self.test_dir,
                options=self.options,
                output_dir=output_dir,
                recursive=False
            )
        
        # 验证结果
        self.assertEqual(len(results), len(self.audio_files))
        
        # 检查输出文件
        for audio_file in self.audio_files:
            expected_output = output_dir / audio_file.with_suffix(".txt").name
            self.assertTrue(expected_output.exists())
            
            # 检查文件内容
            with open(expected_output, "r") as f:
                content = f.read()
                self.assertIn(f"Transcription for {audio_file.name}", content)

    def test_error_handling(self):
        """测试错误处理"""
        # 创建会抛出异常的模拟转录器
        error_transcriber = MagicMock()
        error_transcriber.transcribe.side_effect = RuntimeError("Transcription failed")
        
        processor = TranscriptionProcessor(
            transcriber=error_transcriber,
            config=ProcessorConfig()
        )
        
        # 创建任务
        job = TranscriptionJob(
            audio_path=self.audio_files[0],
            output_path=self.test_dir / "error_output.txt",
            options=self.options
        )
        
        # 处理任务
        results = processor.process_batch([job])
        
        # 验证结果
        self.assertEqual(len(results), 0)
        self.assertEqual(job.status, JobStatus.FAILED)
        self.assertIn("Transcription failed", job.error)
        self.assertFalse(job.output_path.exists())

    def test_process_audio_convenience(self):
        """测试便捷处理单个音频文件的方法"""
        processor = TranscriptionProcessor(
            transcriber=self.mock_transcriber,
            config=ProcessorConfig(
                optimize_subtitles=False
            )
        )
        
        audio_path = self.audio_files[0]
        output_path = self.test_dir / "single_output.txt"
        
        # 处理单个音频
        result = processor.process_audio(
            audio_path=audio_path,
            options=self.options,
            output_path=output_path
        )
        
        # 验证结果
        self.assertIsNotNone(result)
        self.assertEqual(result.text, f"Transcription for {audio_path.name}")
        self.assertTrue(output_path.exists())
        
        # 测试错误处理
        error_transcriber = MagicMock()
        error_transcriber.transcribe.side_effect = RuntimeError("Single file error")
        
        processor = TranscriptionProcessor(
            transcriber=error_transcriber,
            config=ProcessorConfig()
        )
        
        with self.assertRaises(RuntimeError) as context:
            processor.process_audio(audio_path)
        
        self.assertIn("Single file error", str(context.exception))


if __name__ == "__main__":
    unittest.main() 