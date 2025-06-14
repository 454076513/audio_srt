"""
字幕格式化模块，将转录结果转换为不同的字幕格式
"""

import json
import os
from abc import ABC, abstractmethod
from datetime import timedelta
from pathlib import Path
from typing import Dict, List, Optional, Union, Any, TextIO

from audio_srt.core.transcriber import TranscriptionResult


def format_timestamp(seconds: float, include_ms: bool = True) -> str:
    """
    将秒数格式化为字幕时间戳格式 (HH:MM:SS,mmm)
    
    Args:
        seconds: 秒数
        include_ms: 是否包含毫秒
        
    Returns:
        str: 格式化的时间戳
    """
    td = timedelta(seconds=seconds)
    hours = int(td.total_seconds() // 3600)
    minutes = int((td.total_seconds() % 3600) // 60)
    seconds = td.total_seconds() % 60
    
    if include_ms:
        return f"{hours:02d}:{minutes:02d}:{int(seconds):02d},{int(seconds * 1000) % 1000:03d}"
    else:
        return f"{hours:02d}:{minutes:02d}:{int(seconds):02d}"


def format_webvtt_timestamp(seconds: float) -> str:
    """
    将秒数格式化为WebVTT时间戳格式 (HH:MM:SS.mmm)
    
    Args:
        seconds: 秒数
        
    Returns:
        str: 格式化的WebVTT时间戳
    """
    td = timedelta(seconds=seconds)
    hours = int(td.total_seconds() // 3600)
    minutes = int((td.total_seconds() % 3600) // 60)
    seconds = td.total_seconds() % 60
    
    return f"{hours:02d}:{minutes:02d}:{int(seconds):02d}.{int(seconds * 1000) % 1000:03d}"


class SubtitleFormatter(ABC):
    """字幕格式化抽象基类"""
    
    @abstractmethod
    def format(self, transcription: TranscriptionResult) -> str:
        """
        将转录结果格式化为字幕
        
        Args:
            transcription: 转录结果
            
        Returns:
            str: 格式化后的字幕内容
        """
        pass
    
    def save(
        self,
        transcription: TranscriptionResult,
        output_path: Union[str, Path],
        encoding: str = "utf-8"
    ) -> Path:
        """
        将格式化的字幕保存到文件
        
        Args:
            transcription: 转录结果
            output_path: 输出文件路径
            encoding: 文件编码
            
        Returns:
            Path: 保存的文件路径
        """
        output_path = Path(output_path)
        
        # 确保输出目录存在
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        # 格式化并写入文件
        with open(output_path, "w", encoding=encoding) as f:
            f.write(self.format(transcription))
        
        return output_path


class SRTFormatter(SubtitleFormatter):
    """SRT字幕格式化器"""
    
    def __init__(self, max_line_length: int = 42):
        """
        初始化SRT格式化器
        
        Args:
            max_line_length: 每行字幕的最大长度，默认为42个字符
        """
        self.max_line_length = max_line_length
    
    def format(self, transcription: TranscriptionResult) -> str:
        """
        将转录结果格式化为SRT字幕
        
        Args:
            transcription: 转录结果
            
        Returns:
            str: SRT格式的字幕内容
        """
        if not transcription.segments:
            return ""
        
        srt_content = []
        
        for i, segment in enumerate(transcription.segments, start=1):
            # 格式化时间戳
            start_time = format_timestamp(segment["start"])
            end_time = format_timestamp(segment["end"])
            
            # 处理文本，可能会根据最大长度进行分行
            text = segment["text"].strip()
            
            # 添加SRT条目
            srt_content.append(f"{i}\n{start_time} --> {end_time}\n{text}\n")
        
        return "\n".join(srt_content)


class WebVTTFormatter(SubtitleFormatter):
    """WebVTT字幕格式化器"""
    
    def __init__(self, max_line_length: int = 42):
        """
        初始化WebVTT格式化器
        
        Args:
            max_line_length: 每行字幕的最大长度，默认为42个字符
        """
        self.max_line_length = max_line_length
    
    def format(self, transcription: TranscriptionResult) -> str:
        """
        将转录结果格式化为WebVTT字幕
        
        Args:
            transcription: 转录结果
            
        Returns:
            str: WebVTT格式的字幕内容
        """
        if not transcription.segments:
            return "WEBVTT\n\n"
        
        vtt_content = ["WEBVTT\n"]
        
        for i, segment in enumerate(transcription.segments):
            # 格式化时间戳
            start_time = format_webvtt_timestamp(segment["start"])
            end_time = format_webvtt_timestamp(segment["end"])
            
            # 处理文本
            text = segment["text"].strip()
            
            # 添加WebVTT条目
            vtt_content.append(f"{i + 1}\n{start_time} --> {end_time}\n{text}\n")
        
        return "\n".join(vtt_content)


class JSONFormatter(SubtitleFormatter):
    """JSON字幕格式化器"""
    
    def __init__(self, indent: int = 2, include_words: bool = True):
        """
        初始化JSON格式化器
        
        Args:
            indent: JSON缩进空格数，默认为2
            include_words: 是否包含词级时间戳，默认为True
        """
        self.indent = indent
        self.include_words = include_words
    
    def format(self, transcription: TranscriptionResult) -> str:
        """
        将转录结果格式化为JSON
        
        Args:
            transcription: 转录结果
            
        Returns:
            str: JSON格式的字幕内容
        """
        result = {
            "text": transcription.text,
            "segments": transcription.segments,
            "language": transcription.language,
            "duration": transcription.duration,
        }
        
        # 如果不需要包含词级时间戳，则从段落中移除
        if not self.include_words:
            for segment in result["segments"]:
                if "words" in segment:
                    del segment["words"]
        
        return json.dumps(result, ensure_ascii=False, indent=self.indent)


class TextFormatter(SubtitleFormatter):
    """纯文本格式化器"""
    
    def __init__(self, include_timestamps: bool = False):
        """
        初始化纯文本格式化器
        
        Args:
            include_timestamps: 是否在文本中包含时间戳，默认为False
        """
        self.include_timestamps = include_timestamps
    
    def format(self, transcription: TranscriptionResult) -> str:
        """
        将转录结果格式化为纯文本
        
        Args:
            transcription: 转录结果
            
        Returns:
            str: 纯文本格式的转录内容
        """
        if not transcription.segments:
            return ""
        
        lines = []
        
        for segment in transcription.segments:
            text = segment["text"].strip()
            if self.include_timestamps:
                start_time = format_timestamp(segment["start"], include_ms=False)
                end_time = format_timestamp(segment["end"], include_ms=False)
                lines.append(f"[{start_time} --> {end_time}] {text}")
            else:
                lines.append(text)
        
        return "\n".join(lines)


class FormatterFactory:
    """字幕格式化器工厂，用于创建不同类型的字幕格式化器"""
    
    @staticmethod
    def get_formatter(
        format_type: str,
        max_line_length: int = 42,
        json_indent: int = 2,
        include_word_timestamps: bool = False,
        include_text_timestamps: bool = False
    ) -> SubtitleFormatter:
        """
        根据指定的格式类型创建对应的字幕格式化器
        
        Args:
            format_type: 字幕格式类型，可选值为"srt", "vtt", "json", "txt"
            max_line_length: 字幕每行的最大长度
            json_indent: JSON格式的缩进空格数
            include_word_timestamps: 是否在JSON中包含词级时间戳
            include_text_timestamps: 是否在纯文本中包含时间戳
            
        Returns:
            SubtitleFormatter: 对应类型的格式化器实例
            
        Raises:
            ValueError: 如果格式类型不受支持
        """
        format_type = format_type.lower()
        
        if format_type == "srt":
            return SRTFormatter(max_line_length=max_line_length)
        elif format_type in ["vtt", "webvtt"]:
            return WebVTTFormatter(max_line_length=max_line_length)
        elif format_type == "json":
            return JSONFormatter(indent=json_indent, include_words=include_word_timestamps)
        elif format_type in ["txt", "text"]:
            return TextFormatter(include_timestamps=include_text_timestamps)
        else:
            raise ValueError(f"不支持的字幕格式: {format_type}")
    
    @staticmethod
    def get_available_formats() -> List[str]:
        """
        获取所有支持的字幕格式
        
        Returns:
            List[str]: 支持的字幕格式列表
        """
        return ["srt", "vtt", "json", "txt"] 