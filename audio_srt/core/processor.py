"""
并行处理模块，负责音频转录任务的高效并行执行
"""

import os
import time
import logging
import traceback
from pathlib import Path
from typing import List, Dict, Any, Optional, Union, Callable, Tuple
from dataclasses import dataclass, field
from concurrent.futures import ThreadPoolExecutor, as_completed
from enum import Enum, auto

import joblib
from tqdm import tqdm

from .transcriber import Transcriber, TranscriptionOptions, TranscriptionResult
from .subtitle_formatter import FormatterFactory
from .subtitle_optimizer import SubtitleOptimizer, SubtitleOptimizerConfig


class JobStatus(Enum):
    """转录任务状态枚举"""
    PENDING = auto()     # 等待处理
    RUNNING = auto()     # 正在处理
    COMPLETED = auto()   # 已完成
    FAILED = auto()      # 失败
    SKIPPED = auto()     # 跳过（例如已存在文件）


@dataclass
class TranscriptionJob:
    """
    单个转录任务描述
    """
    audio_path: Path              # 音频文件路径
    output_path: Optional[Path]   # 输出文件路径，如果为None则自动生成
    options: TranscriptionOptions  # 转录选项
    status: JobStatus = JobStatus.PENDING  # 任务状态
    error: Optional[str] = None   # 错误信息（如果有）
    result: Optional[TranscriptionResult] = None  # 转录结果
    progress: float = 0.0         # 进度（0-1）
    start_time: Optional[float] = None  # 任务开始时间
    end_time: Optional[float] = None    # 任务结束时间

    def __post_init__(self):
        """初始化后处理"""
        # 确保路径是Path对象
        if isinstance(self.audio_path, str):
            self.audio_path = Path(self.audio_path)
        
        if self.output_path and isinstance(self.output_path, str):
            self.output_path = Path(self.output_path)
        
        # 如果未指定输出路径，则自动生成
        if self.output_path is None:
            self.output_path = self._generate_output_path()
    
    def _generate_output_path(self) -> Path:
        """根据音频文件自动生成输出路径"""
        # 获取音频文件所在目录和文件名（不含扩展名）
        parent_dir = self.audio_path.parent
        stem = self.audio_path.stem
        
        # 根据输出格式确定文件扩展名
        output_format = self.options.output_format.lower()
        if output_format == "srt":
            ext = ".srt"
        elif output_format == "vtt":
            ext = ".vtt"
        elif output_format == "json":
            ext = ".json"
        else:  # 默认为 txt
            ext = ".txt"
        
        # 组合成最终路径
        return parent_dir / f"{stem}{ext}"
    
    @property
    def duration(self) -> Optional[float]:
        """获取任务执行时间（秒）"""
        if self.start_time and self.end_time:
            return self.end_time - self.start_time
        return None
    
    def mark_running(self):
        """标记任务为运行中"""
        self.status = JobStatus.RUNNING
        self.start_time = time.time()
    
    def mark_completed(self, result: TranscriptionResult):
        """标记任务为已完成"""
        self.status = JobStatus.COMPLETED
        self.end_time = time.time()
        self.result = result
        self.progress = 1.0
    
    def mark_failed(self, error: str):
        """标记任务为失败"""
        self.status = JobStatus.FAILED
        self.end_time = time.time()
        self.error = error
    
    def mark_skipped(self, reason: str = "文件已存在"):
        """标记任务为跳过"""
        self.status = JobStatus.SKIPPED
        self.start_time = time.time()
        self.end_time = time.time()
        self.error = reason
        self.progress = 1.0
    
    def update_progress(self, progress: float):
        """更新任务进度"""
        self.progress = max(0.0, min(1.0, progress))


@dataclass
class ProcessorConfig:
    """处理器配置"""
    max_workers: int = 0  # 最大工作线程数，0表示自动
    batch_size: int = 1   # 批处理大小
    skip_existing: bool = True  # 跳过已存在的文件
    force_overwrite: bool = False  # 强制覆盖已存在的文件
    show_progress: bool = True  # 显示进度条
    optimize_subtitles: bool = True  # 优化字幕
    optimizer_config: Optional[SubtitleOptimizerConfig] = None  # 字幕优化器配置
    progress_callback: Optional[Callable[[TranscriptionJob], None]] = None  # 进度回调函数
    
    def __post_init__(self):
        """验证配置的有效性"""
        if self.max_workers < 0:
            raise ValueError("max_workers 必须大于等于 0")
        
        if self.batch_size < 1:
            raise ValueError("batch_size 必须大于等于 1")
        
        if self.force_overwrite:
            self.skip_existing = False
    
    def get_effective_workers(self) -> int:
        """获取有效的工作线程数"""
        if self.max_workers > 0:
            return self.max_workers
        
        # 自动确定线程数
        try:
            # 获取CPU核心数
            cpu_count = joblib.cpu_count()
            # 使用CPU核心数-1（至少为1）作为默认值
            return max(1, cpu_count - 1)
        except:
            # 如果无法确定，默认使用2个线程
            return 2


class TranscriptionProcessor:
    """
    转录处理器，负责管理和执行并行转录任务
    """
    
    def __init__(self, transcriber: Optional[Transcriber] = None, config: Optional[ProcessorConfig] = None):
        """
        初始化转录处理器
        
        Args:
            transcriber: 转录器实例，如果为None则根据需要创建
            config: 处理器配置，如果为None则使用默认配置
        """
        self.transcriber = transcriber
        self.config = config or ProcessorConfig()
        self.logger = logging.getLogger("audio_srt.core.processor")
        self.subtitle_optimizer = None
        self._jobs = []  # 存储所有处理过的任务
        
        if self.config.optimize_subtitles:
            optimizer_config = self.config.optimizer_config or SubtitleOptimizerConfig()
            self.subtitle_optimizer = SubtitleOptimizer(optimizer_config)
    
    def process_batch(self, jobs: List[TranscriptionJob]) -> Dict[Path, TranscriptionResult]:
        """
        批量处理转录任务
        
        Args:
            jobs: 要处理的转录任务列表
            
        Returns:
            Dict[Path, TranscriptionResult]: 音频路径到转录结果的映射
        """
        if not jobs:
            self.logger.warning("没有要处理的任务")
            return {}
        
        self.logger.info(f"开始处理 {len(jobs)} 个转录任务")
        
        # 添加任务到内部列表
        self._jobs.extend(jobs)
        
        # 预处理任务：检查跳过等情况
        self._preprocess_jobs(jobs)
        
        # 过滤出需要实际处理的任务
        pending_jobs = [job for job in jobs if job.status == JobStatus.PENDING]
        
        if not pending_jobs:
            self.logger.info("没有需要处理的任务（所有任务已跳过）")
            return {job.audio_path: job.result for job in jobs if job.result}
        
        self.logger.info(f"实际需要处理的任务数: {len(pending_jobs)}")
        
        # 创建转录器（如果需要）
        if self.transcriber is None:
            from ..models.model_manager import ModelManager
            
            # 使用第一个任务的选项创建转录器
            first_job = pending_jobs[0]
            model_manager = ModelManager()
            
            self.logger.info("创建转录器...")
            self.transcriber = Transcriber(
                model=first_job.options.model,
                model_manager=model_manager,
                device=first_job.options.device,
                compute_type=first_job.options.compute_type,
                threads=first_job.options.threads
            )
        
        # 基于配置确定并行度
        workers = self.config.get_effective_workers()
        self.logger.info(f"使用 {workers} 个工作线程进行处理")
        
        # 使用线程池处理任务
        results = {}
        
        # 设置进度条
        pbar = None
        if self.config.show_progress:
            pbar = tqdm(total=len(pending_jobs), desc="处理转录任务", unit="文件")
        
        with ThreadPoolExecutor(max_workers=workers) as executor:
            # 提交所有任务
            future_to_job = {
                executor.submit(self._process_single_job, job): job 
                for job in pending_jobs
            }
            
            # 处理完成的任务
            for future in as_completed(future_to_job):
                job = future_to_job[future]
                
                try:
                    result = future.result()
                    if result:
                        results[job.audio_path] = result
                        job.mark_completed(result)
                except Exception as e:
                    error_msg = f"处理任务 {job.audio_path} 失败: {str(e)}"
                    self.logger.error(error_msg)
                    self.logger.debug(traceback.format_exc())
                    job.mark_failed(error_msg)
                
                # 更新进度条
                if pbar:
                    pbar.update(1)
                    
                # 调用进度回调（如果有）
                if self.config.progress_callback:
                    try:
                        self.config.progress_callback(job)
                    except Exception as e:
                        self.logger.error(f"进度回调执行失败: {str(e)}")
        
        # 关闭进度条
        if pbar:
            pbar.close()
        
        # 日志记录处理结果
        completed = sum(1 for job in jobs if job.status == JobStatus.COMPLETED)
        failed = sum(1 for job in jobs if job.status == JobStatus.FAILED)
        skipped = sum(1 for job in jobs if job.status == JobStatus.SKIPPED)
        
        self.logger.info(f"转录任务处理完成。总计: {len(jobs)}, "
                         f"完成: {completed}, 失败: {failed}, 跳过: {skipped}")
        
        return results
    
    def get_all_jobs(self) -> List[TranscriptionJob]:
        """
        获取所有处理过的任务
        
        Returns:
            List[TranscriptionJob]: 所有任务列表
        """
        return self._jobs.copy()
    
    def _preprocess_jobs(self, jobs: List[TranscriptionJob]):
        """
        预处理任务列表，标记需要跳过的任务
        
        Args:
            jobs: 要预处理的任务列表
        """
        for job in jobs:
            # 检查输出文件是否已存在
            if job.output_path.exists():
                if self.config.force_overwrite:
                    # 强制覆盖模式，继续处理
                    continue
                
                if self.config.skip_existing:
                    # 跳过已存在的文件
                    job.mark_skipped(f"输出文件已存在: {job.output_path}")
    
    def _process_single_job(self, job: TranscriptionJob) -> Optional[TranscriptionResult]:
        """
        处理单个转录任务
        
        Args:
            job: 要处理的任务
            
        Returns:
            Optional[TranscriptionResult]: 转录结果，如果任务跳过或失败则为None
        """
        if job.status != JobStatus.PENDING:
            # 任务已经被标记为跳过或其他状态，不需要处理
            return None
        
        try:
            # 标记任务为运行中
            job.mark_running()
            self.logger.debug(f"开始处理任务: {job.audio_path}")
            
            # 执行转录
            result = self.transcriber.transcribe(
                str(job.audio_path),
                options=job.options
            )
            
            # 应用字幕优化（如果启用）
            if self.subtitle_optimizer and result and result.segments:
                self.logger.debug(f"正在优化字幕: {job.audio_path}")
                optimized_segments = self.subtitle_optimizer.optimize_segments(result.segments)
                result.segments = optimized_segments
            
            # 保存转录结果
            self._save_result(job, result)
            
            self.logger.debug(f"任务处理完成: {job.audio_path}")
            return result
            
        except Exception as e:
            error_msg = f"处理任务 {job.audio_path} 失败: {str(e)}"
            self.logger.error(error_msg)
            self.logger.debug(traceback.format_exc())
            job.mark_failed(error_msg)
            return None
    
    def _save_result(self, job: TranscriptionJob, result: TranscriptionResult):
        """
        保存转录结果到文件
        
        Args:
            job: 转录任务
            result: 转录结果
        """
        try:
            # 获取格式化器
            formatter = FormatterFactory.create(job.options.output_format)
            
            # 创建输出目录（如果不存在）
            output_dir = job.output_path.parent
            output_dir.mkdir(parents=True, exist_ok=True)
            
            # 格式化并保存结果
            output_content = formatter.format(result, word_timestamps=job.options.word_timestamps)
            
            with open(job.output_path, "w", encoding="utf-8") as f:
                f.write(output_content)
                
            self.logger.info(f"已保存结果到: {job.output_path}")
                
        except Exception as e:
            error_msg = f"保存结果失败: {str(e)}"
            self.logger.error(error_msg)
            self.logger.debug(traceback.format_exc())
            job.mark_failed(error_msg)
            raise
    
    def process_audio(self, audio_path: Union[str, Path], 
                      options: Optional[TranscriptionOptions] = None,
                      output_path: Optional[Union[str, Path]] = None) -> TranscriptionResult:
        """
        处理单个音频文件（便捷方法）
        
        Args:
            audio_path: 音频文件路径
            options: 转录选项，如果为None则使用默认选项
            output_path: 输出文件路径，如果为None则自动生成
            
        Returns:
            TranscriptionResult: 转录结果
        """
        from .transcriber import TranscriptionOptions
        
        # 创建默认选项（如果需要）
        if options is None:
            options = TranscriptionOptions()
            
        # 创建任务
        job = TranscriptionJob(
            audio_path=audio_path,
            output_path=output_path,
            options=options
        )
        
        # 使用批处理方法处理
        results = self.process_batch([job])
        
        if job.status == JobStatus.FAILED:
            raise RuntimeError(f"转录失败: {job.error}")
        
        return job.result
    
    def process_directory(self, 
                          directory: Union[str, Path],
                          options: Optional[TranscriptionOptions] = None,
                          output_dir: Optional[Union[str, Path]] = None,
                          recursive: bool = False,
                          file_extensions: List[str] = None) -> Dict[Path, TranscriptionResult]:
        """
        处理目录中的所有音频文件
        
        Args:
            directory: 音频目录
            options: 转录选项，如果为None则使用默认选项
            output_dir: 输出目录，如果为None则使用原始目录
            recursive: 是否递归处理子目录
            file_extensions: 要处理的文件扩展名列表，如果为None则使用默认扩展名
            
        Returns:
            Dict[Path, TranscriptionResult]: 音频路径到转录结果的映射
        """
        from .transcriber import TranscriptionOptions
        
        # 默认音频扩展名
        if file_extensions is None:
            file_extensions = [".mp3", ".wav", ".flac", ".ogg", ".m4a", ".mp4"]
        
        # 确保目录是Path对象
        directory = Path(directory)
        
        if not directory.exists() or not directory.is_dir():
            raise ValueError(f"指定的目录不存在或不是一个有效目录: {directory}")
        
        # 扫描音频文件
        audio_files = []
        
        # 处理函数
        def process_dir(dir_path: Path):
            nonlocal audio_files
            for item in dir_path.iterdir():
                if item.is_file() and item.suffix.lower() in file_extensions:
                    audio_files.append(item)
                elif recursive and item.is_dir():
                    process_dir(item)
        
        # 开始扫描
        process_dir(directory)
        
        if not audio_files:
            self.logger.warning(f"目录 {directory} 中没有找到音频文件")
            return {}
        
        self.logger.info(f"在目录 {directory} 中找到 {len(audio_files)} 个音频文件")
        
        # 创建默认选项（如果需要）
        if options is None:
            options = TranscriptionOptions()
        
        # 创建转录任务
        jobs = []
        
        for audio_file in audio_files:
            # 确定输出路径
            if output_dir:
                rel_path = audio_file.relative_to(directory)
                out_path = Path(output_dir) / rel_path.with_suffix(self._get_extension_for_format(options.output_format))
            else:
                out_path = audio_file.with_suffix(self._get_extension_for_format(options.output_format))
            
            # 创建任务
            job = TranscriptionJob(
                audio_path=audio_file,
                output_path=out_path,
                options=options
            )
            jobs.append(job)
        
        # 批量处理任务
        return self.process_batch(jobs)
    
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