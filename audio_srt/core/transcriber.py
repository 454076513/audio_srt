"""
核心转录模块，实现基于Faster-Whisper的音频转文字功能
"""

import os
import logging
import tempfile
from enum import Enum
from pathlib import Path
from typing import Dict, List, Optional, Union, Tuple, Any, Callable

from audio_srt.models import ModelManager
from audio_srt.utils.audio_utils import (
    is_valid_audio_file, convert_to_wav, get_audio_duration, split_audio
)


class TranscriptionStatus(Enum):
    """转录状态枚举"""
    PENDING = "pending"       # 等待处理
    PROCESSING = "processing" # 处理中
    COMPLETED = "completed"   # 完成
    FAILED = "failed"         # 失败


class TranscriptionOptions:
    """转录选项类"""
    
    def __init__(
        self,
        model_size: str = "small",
        language: Optional[str] = None,
        task: str = "transcribe",
        beam_size: int = 5,
        best_of: int = 5,
        temperature: float = 0,
        no_speech_threshold: float = 0.6,
        compression_ratio_threshold: float = 2.4,
        condition_on_previous_text: bool = True,
        initial_prompt: Optional[str] = None,
        word_timestamps: bool = False,
        vad_filter: bool = True,
        vad_parameters: Optional[Dict[str, Any]] = None
    ):
        """
        初始化转录选项
        
        Args:
            model_size: 模型大小，可选值为"tiny", "base", "small", "medium", "large-v1", "large-v2", "large-v3", "large-v3-turbo"
            language: 音频语言代码(如"zh", "en")，None表示自动检测
            task: 任务类型，"transcribe"（转录）或"translate"（翻译成英文）
            beam_size: 波束搜索大小
            best_of: 最佳序列数量
            temperature: 采样温度
            no_speech_threshold: 无语音阈值
            compression_ratio_threshold: 压缩比阈值
            condition_on_previous_text: 是否基于先前文本进行条件处理
            initial_prompt: 初始提示文本
            word_timestamps: 是否生成词级时间戳
            vad_filter: 是否使用语音活动检测过滤
            vad_parameters: VAD参数字典
        """
        self.model_size = model_size
        self.language = language
        self.task = task
        self.beam_size = beam_size
        self.best_of = best_of
        self.temperature = temperature
        self.no_speech_threshold = no_speech_threshold
        self.compression_ratio_threshold = compression_ratio_threshold
        self.condition_on_previous_text = condition_on_previous_text
        self.initial_prompt = initial_prompt
        self.word_timestamps = word_timestamps
        self.vad_filter = vad_filter
        self.vad_parameters = vad_parameters or {}


class TranscriptionResult:
    """转录结果类"""
    
    def __init__(
        self,
        segments: List[Dict[str, Any]],
        language: str,
        duration: float,
        word_timestamps: Optional[List[Dict[str, Any]]] = None
    ):
        """
        初始化转录结果
        
        Args:
            segments: 包含时间戳和文本的片段列表
            language: 检测到的语言
            duration: 音频持续时间
            word_timestamps: 词级时间戳（如果启用）
        """
        self.segments = segments
        self.language = language
        self.duration = duration
        self.word_timestamps = word_timestamps
        
        # 如果有词级时间戳，将它们组织到对应的段落中
        if word_timestamps:
            self._organize_words_by_segments()
    
    def _organize_words_by_segments(self):
        """
        将词级时间戳组织到对应的段落中
        
        将全局的词级时间戳列表按时间范围分配到各个段落中
        """
        if not self.word_timestamps or not self.segments:
            return
            
        # 清理词级时间戳中的空格问题
        self.clean_word_timestamps()
        
        # 为每个段落创建一个空的words列表
        for segment in self.segments:
            segment["words"] = []
        
        # 将词级时间戳分配到对应的段落中
        for word in self.word_timestamps:
            word_start = word["start"]
            word_end = word["end"]
            
            # 找到这个词所属的段落
            for segment in self.segments:
                segment_start = segment["start"]
                segment_end = segment["end"]
                
                # 如果词的时间范围在段落的时间范围内或有重叠，则将其添加到该段落
                if (word_start >= segment_start and word_start < segment_end) or \
                   (word_end > segment_start and word_end <= segment_end) or \
                   (word_start <= segment_start and word_end >= segment_end):
                    # 创建一个新的词对象，不包含probability
                    segment_word = {
                        "word": word["word"],
                        "start": word["start"],
                        "end": word["end"]
                    }
                    segment["words"].append(segment_word)
                    break
        
        # 移除全局的词级时间戳列表，因为现在它们已经被组织到段落中
        self.word_timestamps = None
    
    @property
    def text(self) -> str:
        """
        获取完整的转录文本
        
        Returns:
            str: 转录的文本内容
        """
        return " ".join(segment["text"] for segment in self.segments).strip()
    
    def get_segments_with_timestamps(self) -> List[Dict[str, Any]]:
        """
        获取包含时间戳的段落
        
        Returns:
            List[Dict[str, Any]]: 包含时间戳和文本的段落列表
        """
        return self.segments
    
    def is_empty(self) -> bool:
        """
        检查转录结果是否为空
        
        Returns:
            bool: 如果没有转录内容则返回True
        """
        return len(self.segments) == 0 or not any(segment.get("text") for segment in self.segments)
    
    def clean_word_timestamps(self) -> None:
        """
        清理词级时间戳中的空格问题
        
        处理单词文本中的前导空格，调整时间戳
        """
        if not self.word_timestamps:
            return
            
        for word_dict in self.word_timestamps:
            # 如果单词包含前导空格
            if word_dict["word"].startswith(" "):
                # 估算空格的时长
                word_duration = word_dict["end"] - word_dict["start"]
                space_duration = word_duration * 0.1  # 假设空格占10%
                
                # 调整开始时间
                word_dict["start"] = word_dict["start"] + space_duration
                
                # 清理单词文本
                word_dict["word"] = word_dict["word"].strip()
                
        # 也处理段落中的词级时间戳
        for segment in self.segments:
            if "words" in segment:
                for word_dict in segment["words"]:
                    if word_dict["word"].startswith(" "):
                        word_duration = word_dict["end"] - word_dict["start"]
                        space_duration = word_duration * 0.1
                        word_dict["start"] = word_dict["start"] + space_duration
                        word_dict["word"] = word_dict["word"].strip()


class Transcriber:
    """音频转录器类，基于Faster-Whisper实现音频到文字的转换"""
    
    def __init__(self, options: Optional[TranscriptionOptions] = None):
        """
        初始化转录器
        
        Args:
            options: 转录选项，如果为None则使用默认值
        """
        self.options = options or TranscriptionOptions()
        self.logger = logging.getLogger("audio_srt.transcriber")
        self.model_manager = ModelManager()
        self._model = None  # 延迟加载模型
        
    def _load_model_if_needed(self) -> None:
        """
        按需加载模型
        """
        if self._model is None:
            self.logger.info(f"正在加载模型: {self.options.model_size}")
            self._model = self.model_manager.load_model(
                model_size=self.options.model_size,
                device=None,  # 自动选择
                compute_type=None  # 自动选择
            )
            self.logger.info("模型加载完成")
    
    def transcribe(
        self,
        audio_path: Union[str, Path],
        options: Optional[TranscriptionOptions] = None,
        progress_callback: Optional[Callable[[float], None]] = None
    ) -> TranscriptionResult:
        """
        转录音频文件
        
        Args:
            audio_path: 音频文件路径
            options: 转录选项，如果为None则使用实例选项
            progress_callback: 进度回调函数，接受一个0-1之间的浮点数参数
            
        Returns:
            TranscriptionResult: 转录结果对象
            
        Raises:
            ValueError: 如果音频文件无效
            RuntimeError: 如果转录过程失败
        """
        if not is_valid_audio_file(audio_path):
            raise ValueError(f"无效的音频文件: {audio_path}")
        
        # 使用提供的选项或默认选项
        opts = options or self.options
        
        # 加载模型（如果尚未加载）
        self._load_model_if_needed()
        
        try:
            # 获取音频时长
            duration = get_audio_duration(audio_path)
            self.logger.info(f"音频持续时间: {duration:.2f}秒")
            
            # 转换为WAV格式（Whisper需要的格式）
            wav_path = convert_to_wav(audio_path)
            self.logger.info(f"转换为WAV格式: {wav_path}")
            
            # 执行转录
            self.logger.info("开始转录...")
            
            # 准备转录参数
            # 注意：temperature_increment_on_fallback 参数已被移除，因为最新版本的 Faster-Whisper 不再支持它
            kwargs = {
                "beam_size": opts.beam_size,
                "best_of": opts.best_of,
                "temperature": opts.temperature,
                "language": opts.language,
                "task": opts.task,
                "initial_prompt": opts.initial_prompt,
                "word_timestamps": opts.word_timestamps,
                "condition_on_previous_text": opts.condition_on_previous_text,
                "no_speech_threshold": opts.no_speech_threshold,
                "compression_ratio_threshold": opts.compression_ratio_threshold,
            }
            
            # VAD过滤配置
            if opts.vad_filter:
                kwargs["vad_filter"] = True
                kwargs["vad_parameters"] = opts.vad_parameters
            
            # 执行转录
            segments, info = self._model.transcribe(str(wav_path), **kwargs)
            
            # 收集结果
            detected_language = info.language
            language_probability = info.language_probability
            self.logger.info(f"检测到的语言: {detected_language} (置信度: {language_probability:.2f})")
            
            # 构建转录结果
            segments_list = []
            word_timestamps_list = []
            
            # 处理片段
            for segment in segments:
                segment_dict = {
                    "id": segment.id,
                    "start": segment.start,
                    "end": segment.end,
                    "text": segment.text,
                }
                
                # 如果启用了词级时间戳，为每个段落添加words列表
                if opts.word_timestamps and segment.words:
                    segment_dict["words"] = []
                    for word in segment.words:
                        # 处理单词文本，去除前后空格
                        word_text = word.word.strip()
                        # 调整时间戳：如果单词前有空格，需要相应调整开始时间
                        start_time = word.start
                        if word.word.startswith(" "):
                            # 估算空格的时长，假设空格占单词总时长的一小部分
                            space_duration = (word.end - word.start) * 0.1  # 假设空格占10%
                            start_time = word.start + space_duration
                            
                        word_dict = {
                            "start": start_time,
                            "end": word.end,
                            "word": word_text
                        }
                        segment_dict["words"].append(word_dict)
                        
                        # 同时也添加到全局列表中，用于兼容旧的处理方式
                        word_timestamps_list.append({
                            "start": start_time,
                            "end": word.end,
                            "word": word_text,
                            "probability": word.probability
                        })
                
                segments_list.append(segment_dict)
            
            # 清理临时文件
            if os.path.exists(wav_path) and wav_path != audio_path:
                os.unlink(wav_path)
                
            # 返回转录结果
            result = TranscriptionResult(
                segments=segments_list,
                language=detected_language,
                duration=duration,
                word_timestamps=word_timestamps_list if opts.word_timestamps else None
            )
            
            return result
            
        except Exception as e:
            self.logger.error(f"转录失败: {e}")
            raise RuntimeError(f"转录失败: {e}")
    
    def transcribe_with_segments(
        self,
        audio_path: Union[str, Path],
        segment_duration: int = 600,
        options: Optional[TranscriptionOptions] = None,
        progress_callback: Optional[Callable[[float], None]] = None
    ) -> TranscriptionResult:
        """
        分段转录长音频文件
        
        对于超过指定时长的音频文件，将其分割成多个片段进行处理，
        然后合并结果。这对于处理较长的音频文件很有用。
        
        Args:
            audio_path: 音频文件路径
            segment_duration: 每个片段的最大持续时间（秒）
            options: 转录选项
            progress_callback: 进度回调函数
            
        Returns:
            TranscriptionResult: 合并后的转录结果
            
        Raises:
            ValueError: 如果音频文件无效
            RuntimeError: 如果转录过程失败
        """
        if not is_valid_audio_file(audio_path):
            raise ValueError(f"无效的音频文件: {audio_path}")
        
        # 获取音频时长
        duration = get_audio_duration(audio_path)
        self.logger.info(f"音频持续时间: {duration:.2f}秒")
        
        # 如果音频小于片段时长，直接转录
        if duration <= segment_duration:
            return self.transcribe(audio_path, options, progress_callback)
        
        # 分割音频文件
        self.logger.info(f"音频时长超过{segment_duration}秒，进行分段处理")
        segment_files = split_audio(audio_path, segment_duration)
        self.logger.info(f"分割为{len(segment_files)}个片段")
        
        # 逐个处理片段
        all_segments = []
        detected_language = None
        language_probability = 0.0
        
        for i, segment_file in enumerate(segment_files):
            self.logger.info(f"处理片段 {i+1}/{len(segment_files)}")
            
            # 更新进度
            if progress_callback:
                progress_callback(i / len(segment_files))
            
            # 转录片段
            result = self.transcribe(segment_file, options)
            
            # 记录语言检测结果（使用第一个片段的结果或置信度最高的结果）
            if detected_language is None or result.language_probability > language_probability:
                detected_language = result.language
                language_probability = getattr(result, "language_probability", 1.0)
            
            # 调整时间戳（根据片段在原始音频中的位置）
            segment_start_time = i * segment_duration
            for segment in result.segments:
                segment["start"] += segment_start_time
                segment["end"] += segment_start_time
                
                # 调整词级时间戳
                if "words" in segment:
                    for word in segment["words"]:
                        word["start"] += segment_start_time
                        word["end"] += segment_start_time
                
                all_segments.append(segment)
            
            # 清理临时文件
            if segment_file != audio_path and os.path.exists(segment_file):
                os.unlink(segment_file)
        
        # 完成进度
        if progress_callback:
            progress_callback(1.0)
        
        # 按开始时间排序片段
        all_segments.sort(key=lambda x: x["start"])
        
        # 重新编号片段ID
        for i, segment in enumerate(all_segments):
            segment["id"] = i
        
        result = TranscriptionResult(
            segments=all_segments,
            language=detected_language or "unknown",
            duration=duration,
            word_timestamps=None  # 词级时间戳现在已经在每个段落中
        )
        
        return result 