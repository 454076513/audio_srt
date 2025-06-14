"""
音频处理工具模块，提供音频文件处理、格式转换和验证功能
"""

import os
import tempfile
from pathlib import Path
from typing import Optional, Tuple, List, Union

import ffmpeg
from pydub import AudioSegment


class AudioError(Exception):
    """音频处理相关错误"""
    pass


def is_valid_audio_file(file_path: Union[str, Path]) -> bool:
    """
    检查文件是否为有效的音频文件

    Args:
        file_path: 音频文件路径

    Returns:
        bool: 如果是有效的音频文件则返回True，否则返回False
    """
    if not os.path.exists(file_path):
        return False
    
    try:
        probe = ffmpeg.probe(str(file_path))
        return any(stream.get('codec_type') == 'audio' for stream in probe.get('streams', []))
    except ffmpeg.Error:
        return False


def get_audio_duration(file_path: Union[str, Path]) -> float:
    """
    获取音频文件的持续时间（秒）

    Args:
        file_path: 音频文件路径

    Returns:
        float: 音频持续时间（秒）

    Raises:
        AudioError: 如果文件不是有效的音频文件或无法读取
    """
    if not is_valid_audio_file(file_path):
        raise AudioError(f"无效的音频文件: {file_path}")
    
    try:
        probe = ffmpeg.probe(str(file_path))
        # 查找音频流
        for stream in probe['streams']:
            if stream['codec_type'] == 'audio':
                # 优先使用duration，如果没有则尝试计算
                if 'duration' in stream:
                    return float(stream['duration'])
                elif 'duration_ts' in stream and 'time_base' in stream:
                    # 解析time_base（例如"1/44100"）
                    num, den = map(int, stream['time_base'].split('/'))
                    return stream['duration_ts'] * num / den
        
        # 如果在流中找不到，尝试在格式信息中查找
        if 'format' in probe and 'duration' in probe['format']:
            return float(probe['format']['duration'])
        
        raise AudioError(f"无法确定音频持续时间: {file_path}")
    except ffmpeg.Error as e:
        raise AudioError(f"处理音频文件时出错: {e}")


def convert_to_wav(
    input_file: Union[str, Path],
    output_file: Optional[Union[str, Path]] = None,
    sample_rate: int = 16000,
    channels: int = 1
) -> Path:
    """
    将音频文件转换为WAV格式（用于Whisper模型处理）

    Args:
        input_file: 输入音频文件路径
        output_file: 输出WAV文件路径，如果为None则创建临时文件
        sample_rate: 采样率，默认16kHz（Whisper模型推荐）
        channels: 声道数，默认1（单声道）

    Returns:
        Path: 转换后的WAV文件路径

    Raises:
        AudioError: 转换过程中出错
    """
    if not is_valid_audio_file(input_file):
        raise AudioError(f"无效的音频文件: {input_file}")
    
    try:
        if output_file is None:
            # 创建临时文件
            fd, temp_path = tempfile.mkstemp(suffix=".wav")
            os.close(fd)
            output_file = Path(temp_path)
        else:
            output_file = Path(output_file)
        
        # 使用ffmpeg转换音频
        (
            ffmpeg
            .input(str(input_file))
            .output(
                str(output_file),
                acodec='pcm_s16le',  # 16位PCM
                ac=channels,         # 声道数
                ar=sample_rate       # 采样率
            )
            .overwrite_output()
            .run(quiet=True, capture_stdout=True, capture_stderr=True)
        )
        
        return output_file
    
    except ffmpeg.Error as e:
        if output_file is not None and os.path.exists(output_file):
            os.unlink(output_file)
        stderr = e.stderr.decode() if hasattr(e, 'stderr') and e.stderr else "未知错误"
        raise AudioError(f"音频转换失败: {stderr}")


def get_supported_formats() -> List[str]:
    """
    获取支持的音频格式列表
    
    Returns:
        List[str]: 支持的音频格式扩展名列表
    """
    return [
        ".mp3", ".wav", ".flac", ".ogg", ".m4a", ".wma",
        ".aac", ".mp4", ".mpeg", ".mpga", ".webm"
    ]


def split_audio(
    file_path: Union[str, Path],
    segment_duration: int = 600,
    output_dir: Optional[Union[str, Path]] = None
) -> List[Path]:
    """
    将长音频文件分割为多个较小的片段
    
    Args:
        file_path: 音频文件路径
        segment_duration: 每个片段的最大持续时间（秒），默认10分钟
        output_dir: 输出目录，如果为None则使用临时目录
        
    Returns:
        List[Path]: 分割后的音频文件路径列表
        
    Raises:
        AudioError: 分割过程中出错
    """
    if not is_valid_audio_file(file_path):
        raise AudioError(f"无效的音频文件: {file_path}")
    
    try:
        # 读取音频文件
        audio = AudioSegment.from_file(str(file_path))
        total_duration = len(audio) / 1000  # 转换为秒
        
        # 如果音频小于最大片段时长，则直接返回
        if total_duration <= segment_duration:
            # 创建临时WAV文件供处理
            wav_path = convert_to_wav(file_path)
            return [wav_path]
        
        # 创建输出目录
        if output_dir is None:
            output_dir = Path(tempfile.mkdtemp())
        else:
            output_dir = Path(output_dir)
            output_dir.mkdir(exist_ok=True, parents=True)
        
        # 计算分段数量
        segment_count = int(total_duration / segment_duration) + (1 if total_duration % segment_duration > 0 else 0)
        segment_files = []
        
        # 分割并保存
        file_stem = Path(file_path).stem
        for i in range(segment_count):
            start_ms = i * segment_duration * 1000
            end_ms = min((i + 1) * segment_duration * 1000, len(audio))
            segment = audio[start_ms:end_ms]
            
            # 保存片段
            segment_path = output_dir / f"{file_stem}_part{i+1:03d}.wav"
            segment.export(
                segment_path,
                format="wav",
                parameters=["-ar", "16000", "-ac", "1", "-acodec", "pcm_s16le"]
            )
            segment_files.append(segment_path)
        
        return segment_files
    
    except Exception as e:
        raise AudioError(f"分割音频文件时出错: {e}") 