"""
批处理功能模块，提供配置文件驱动的批量音频转录功能
"""

import os
import sys
import time
import logging
import json
import yaml
from pathlib import Path
from typing import List, Dict, Any, Optional, Union, Set
from dataclasses import dataclass, field, asdict

from ..core.transcriber import TranscriptionOptions
from ..core.processor import TranscriptionProcessor, ProcessorConfig, JobStatus
from ..models.model_manager import ModelManager


@dataclass
class BatchTaskConfig:
    """单个批处理任务配置"""
    source: str  # 源文件或目录
    output_dir: Optional[str] = None  # 输出目录，None表示使用源文件所在目录
    recursive: bool = False  # 是否递归处理子目录
    file_extensions: List[str] = field(default_factory=lambda: [".mp3", ".wav", ".flac", ".ogg", ".m4a", ".mp4"])  # 要处理的文件扩展名
    
    # 转录选项
    model: str = "small"  # 模型大小
    language: str = "auto"  # 语言
    task: str = "transcribe"  # 任务类型（transcribe 或 translate）
    output_format: str = "srt"  # 输出格式
    device: Optional[str] = None  # 设备类型
    compute_type: Optional[str] = None  # 计算类型
    threads: int = 0  # 处理线程数
    word_timestamps: bool = False  # 是否生成词级时间戳
    vad_filter: bool = False  # 是否使用VAD过滤
    
    # 处理选项
    skip_existing: bool = True  # 跳过已存在的文件
    force_overwrite: bool = False  # 强制覆盖已存在的文件
    optimize_subtitles: bool = True  # 是否优化字幕
    
    # 字幕优化选项
    max_gap_seconds: float = 1.5  # 合并片段间的最大间隔（秒）
    min_segment_duration: float = 1.0  # 最小片段时长（秒）
    max_chars_per_line: int = 50  # 每行字幕最多字符数
    max_lines_per_segment: int = 2  # 每个片段最大行数
    split_on_sentence_end: bool = True  # 是否在句末分割字幕
    
    def to_transcription_options(self) -> TranscriptionOptions:
        """转换为转录选项"""
        return TranscriptionOptions(
            model=self.model,
            language=self.language,
            task=self.task,
            output_format=self.output_format,
            device=self.device,
            compute_type=self.compute_type,
            threads=self.threads,
            word_timestamps=self.word_timestamps,
            vad_filter=self.vad_filter
        )
    
    def to_processor_config(self) -> ProcessorConfig:
        """转换为处理器配置"""
        from ..core.subtitle_optimizer import SubtitleOptimizerConfig
        
        optimizer_config = SubtitleOptimizerConfig(
            max_gap_seconds=self.max_gap_seconds,
            min_segment_duration=self.min_segment_duration,
            max_chars_per_line=self.max_chars_per_line,
            max_lines_per_segment=self.max_lines_per_segment,
            split_on_sentence_end=self.split_on_sentence_end,
            language=self.language
        )
        
        return ProcessorConfig(
            max_workers=0,  # 使用自动线程数
            skip_existing=self.skip_existing,
            force_overwrite=self.force_overwrite,
            optimize_subtitles=self.optimize_subtitles,
            optimizer_config=optimizer_config,
            show_progress=True
        )


@dataclass
class BatchConfig:
    """批处理配置"""
    tasks: List[BatchTaskConfig] = field(default_factory=list)  # 任务列表
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'BatchConfig':
        """从字典创建配置"""
        tasks = []
        
        # 获取全局默认值（如果有）
        defaults = data.get("defaults", {})
        
        # 处理任务列表
        task_list = data.get("tasks", [])
        if not task_list and "source" in data:
            # 如果没有tasks字段但有source字段，则将整个配置视为单个任务
            task_list = [data]
        
        for task_data in task_list:
            # 合并默认值和任务特定值
            task_config = {**defaults, **task_data}
            
            # 创建任务配置
            task = BatchTaskConfig(**task_config)
            tasks.append(task)
        
        return cls(tasks=tasks)
    
    @classmethod
    def from_file(cls, file_path: Union[str, Path]) -> 'BatchConfig':
        """从文件加载配置"""
        file_path = Path(file_path)
        
        if not file_path.exists():
            raise FileNotFoundError(f"配置文件不存在: {file_path}")
        
        with open(file_path, 'r', encoding='utf-8') as f:
            if file_path.suffix.lower() in ['.yml', '.yaml']:
                try:
                    data = yaml.safe_load(f)
                except yaml.YAMLError as e:
                    raise ValueError(f"YAML解析错误: {str(e)}")
            elif file_path.suffix.lower() == '.json':
                try:
                    data = json.load(f)
                except json.JSONDecodeError as e:
                    raise ValueError(f"JSON解析错误: {str(e)}")
            else:
                raise ValueError(f"不支持的配置文件格式: {file_path.suffix}")
        
        return cls.from_dict(data)
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        return {
            "tasks": [asdict(task) for task in self.tasks]
        }
    
    def save_to_file(self, file_path: Union[str, Path]) -> None:
        """保存配置到文件"""
        file_path = Path(file_path)
        
        data = self.to_dict()
        
        with open(file_path, 'w', encoding='utf-8') as f:
            if file_path.suffix.lower() in ['.yml', '.yaml']:
                yaml.dump(data, f, default_flow_style=False, allow_unicode=True)
            elif file_path.suffix.lower() == '.json':
                json.dump(data, f, ensure_ascii=False, indent=2)
            else:
                raise ValueError(f"不支持的配置文件格式: {file_path.suffix}")


@dataclass
class BatchTaskResult:
    """批处理任务结果"""
    task: BatchTaskConfig  # 任务配置
    start_time: float  # 开始时间
    end_time: Optional[float] = None  # 结束时间
    total_files: int = 0  # 总文件数
    processed_files: int = 0  # 处理的文件数
    skipped_files: int = 0  # 跳过的文件数
    failed_files: int = 0  # 失败的文件数
    errors: List[str] = field(default_factory=list)  # 错误信息
    
    @property
    def duration(self) -> Optional[float]:
        """获取任务执行时间（秒）"""
        if self.end_time:
            return self.end_time - self.start_time
        return None
    
    @property
    def success_rate(self) -> float:
        """获取成功率"""
        if self.total_files == 0:
            return 0.0
        return (self.processed_files / self.total_files) * 100
    
    def mark_completed(self, processed: int, skipped: int, failed: int, errors: List[str]) -> None:
        """标记任务为已完成"""
        self.end_time = time.time()
        self.processed_files = processed
        self.skipped_files = skipped
        self.failed_files = failed
        self.errors = errors


@dataclass
class BatchReport:
    """批处理报告"""
    results: List[BatchTaskResult] = field(default_factory=list)
    start_time: float = field(default_factory=time.time)
    end_time: Optional[float] = None
    
    @property
    def duration(self) -> Optional[float]:
        """获取总执行时间（秒）"""
        if self.end_time:
            return self.end_time - self.start_time
        return None
    
    @property
    def total_files(self) -> int:
        """获取总文件数"""
        return sum(result.total_files for result in self.results)
    
    @property
    def processed_files(self) -> int:
        """获取处理的文件数"""
        return sum(result.processed_files for result in self.results)
    
    @property
    def skipped_files(self) -> int:
        """获取跳过的文件数"""
        return sum(result.skipped_files for result in self.results)
    
    @property
    def failed_files(self) -> int:
        """获取失败的文件数"""
        return sum(result.failed_files for result in self.results)
    
    @property
    def success_rate(self) -> float:
        """获取总成功率"""
        if self.total_files == 0:
            return 0.0
        return (self.processed_files / self.total_files) * 100
    
    def mark_completed(self) -> None:
        """标记报告为已完成"""
        self.end_time = time.time()
    
    def print_summary(self) -> None:
        """打印报告摘要"""
        print("\n批处理摘要:")
        print(f"总任务数: {len(self.results)}")
        print(f"总文件数: {self.total_files}")
        print(f"处理成功: {self.processed_files}")
        print(f"已跳过: {self.skipped_files}")
        print(f"处理失败: {self.failed_files}")
        print(f"成功率: {self.success_rate:.1f}%")
        
        if self.duration:
            minutes, seconds = divmod(self.duration, 60)
            hours, minutes = divmod(minutes, 60)
            print(f"总执行时间: {int(hours)}小时 {int(minutes)}分钟 {seconds:.1f}秒")
        
        if self.failed_files > 0:
            print("\n错误摘要:")
            for i, result in enumerate(self.results):
                if result.errors:
                    print(f"\n任务 {i+1} ({Path(result.task.source).name}) 错误:")
                    for error in result.errors[:5]:  # 只显示前5个错误
                        print(f"  - {error}")
                    
                    if len(result.errors) > 5:
                        print(f"  ... 还有 {len(result.errors) - 5} 个错误未显示")


class BatchProcessor:
    """批处理器，负责执行批处理任务"""
    
    def __init__(self, model_manager: Optional[ModelManager] = None):
        """
        初始化批处理器
        
        Args:
            model_manager: 模型管理器，如果为None则创建新实例
        """
        self.model_manager = model_manager or ModelManager()
        self.logger = logging.getLogger("audio_srt.cli.batch")
    
    def process_config(self, config: BatchConfig) -> BatchReport:
        """
        处理批处理配置
        
        Args:
            config: 批处理配置
            
        Returns:
            BatchReport: 批处理报告
        """
        report = BatchReport()
        
        for task_config in config.tasks:
            task_result = self._process_task(task_config)
            report.results.append(task_result)
        
        report.mark_completed()
        return report
    
    def process_config_file(self, config_file: Union[str, Path]) -> BatchReport:
        """
        处理配置文件
        
        Args:
            config_file: 配置文件路径
            
        Returns:
            BatchReport: 批处理报告
        """
        config = BatchConfig.from_file(config_file)
        return self.process_config(config)
    
    def _process_task(self, task_config: BatchTaskConfig) -> BatchTaskResult:
        """
        处理单个任务
        
        Args:
            task_config: 任务配置
            
        Returns:
            BatchTaskResult: 任务结果
        """
        source_path = Path(task_config.source)
        result = BatchTaskResult(task=task_config, start_time=time.time())
        
        # 创建处理器
        processor_config = task_config.to_processor_config()
        transcription_options = task_config.to_transcription_options()
        processor = TranscriptionProcessor(config=processor_config)
        
        try:
            if source_path.is_dir():
                # 处理目录
                self.logger.info(f"处理目录: {source_path}")
                output_dir = Path(task_config.output_dir) if task_config.output_dir else None
                
                # 创建输出目录（如果指定）
                if output_dir:
                    output_dir.mkdir(parents=True, exist_ok=True)
                
                # 处理目录
                job_results = processor.process_directory(
                    directory=source_path,
                    options=transcription_options,
                    output_dir=output_dir,
                    recursive=task_config.recursive,
                    file_extensions=task_config.file_extensions
                )
                
                # 统计结果
                all_jobs = processor.get_all_jobs()
                result.total_files = len(all_jobs)
                result.processed_files = sum(1 for job in all_jobs if job.status == JobStatus.COMPLETED)
                result.skipped_files = sum(1 for job in all_jobs if job.status == JobStatus.SKIPPED)
                result.failed_files = sum(1 for job in all_jobs if job.status == JobStatus.FAILED)
                result.errors = [job.error for job in all_jobs if job.status == JobStatus.FAILED and job.error]
                
            elif source_path.is_file():
                # 处理单个文件
                self.logger.info(f"处理文件: {source_path}")
                
                # 确定输出路径
                if task_config.output_dir:
                    output_dir = Path(task_config.output_dir)
                    output_dir.mkdir(parents=True, exist_ok=True)
                    
                    # 获取文件扩展名
                    ext = self._get_extension_for_format(transcription_options.output_format)
                    output_path = output_dir / source_path.with_suffix(ext).name
                else:
                    output_path = None  # 使用默认输出路径
                
                # 处理文件
                try:
                    processor.process_audio(
                        audio_path=source_path,
                        options=transcription_options,
                        output_path=output_path
                    )
                    result.total_files = 1
                    result.processed_files = 1
                except Exception as e:
                    result.total_files = 1
                    result.failed_files = 1
                    result.errors.append(str(e))
            else:
                # 源不存在
                error_msg = f"源路径不存在或无法访问: {source_path}"
                self.logger.error(error_msg)
                result.errors.append(error_msg)
        
        except Exception as e:
            # 处理任务级别的错误
            error_msg = f"处理任务失败: {str(e)}"
            self.logger.error(error_msg)
            result.errors.append(error_msg)
        
        # 标记任务完成
        result.mark_completed(
            processed=result.processed_files,
            skipped=result.skipped_files,
            failed=result.failed_files,
            errors=result.errors
        )
        
        return result
    
    def _get_extension_for_format(self, output_format: str) -> str:
        """
        根据输出格式获取文件扩展名
        
        Args:
            output_format: 输出格式
            
        Returns:
            str: 文件扩展名
        """
        format_to_ext = {
            "srt": ".srt",
            "vtt": ".vtt",
            "json": ".json",
            "txt": ".txt"
        }
        return format_to_ext.get(output_format.lower(), ".txt")


def create_sample_config(output_path: Union[str, Path], format: str = 'yaml') -> None:
    """
    创建示例配置文件
    
    Args:
        output_path: 输出路径
        format: 输出格式，'yaml' 或 'json'
    """
    sample_config = {
        "defaults": {
            "model": "small",
            "language": "auto",
            "output_format": "srt",
            "skip_existing": True,
            "optimize_subtitles": True
        },
        "tasks": [
            {
                "source": "/path/to/audio/directory",
                "output_dir": "/path/to/output/directory",
                "recursive": True,
                "file_extensions": [".mp3", ".wav", ".flac"],
                "model": "medium",
                "language": "en"
            },
            {
                "source": "/path/to/single/file.mp3",
                "output_dir": "/path/to/output/directory",
                "word_timestamps": True,
                "output_format": "json"
            }
        ]
    }
    
    output_path = Path(output_path)
    
    # 确保输出目录存在
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, 'w', encoding='utf-8') as f:
        if format.lower() == 'yaml':
            yaml.dump(sample_config, f, default_flow_style=False, allow_unicode=True)
        else:
            json.dump(sample_config, f, ensure_ascii=False, indent=2)
    
    print(f"已创建示例配置文件: {output_path}") 