"""
命令行命令定义模块
"""

import logging
import os
import sys
import time
from pathlib import Path
from typing import List, Optional, Tuple

import click
from tqdm import tqdm

from audio_srt.core import (
    Transcriber, TranscriptionOptions, FormatterFactory
)
from audio_srt.models import ModelManager
from audio_srt.utils.audio_utils import (
    is_valid_audio_file, get_supported_formats
)
from audio_srt.cli.batch import BatchProcessor, create_sample_config


# 创建日志记录器
logger = logging.getLogger("audio_srt")

# 支持的字幕格式
SUPPORTED_FORMATS = FormatterFactory.get_available_formats()

# 支持的配置文件格式
CONFIG_FORMATS = ["yaml", "json"]


def setup_logging(verbose: bool) -> None:
    """
    设置日志级别
    
    Args:
        verbose: 是否启用详细日志
    """
    log_level = logging.DEBUG if verbose else logging.INFO
    
    # 配置日志记录器
    logging.basicConfig(
        level=log_level,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S"
    )


def show_progress(progress: float) -> None:
    """
    显示进度回调函数
    
    Args:
        progress: 0-1之间的进度值
    """
    # 进度回调的实现将在CLI工具中使用tqdm提供
    pass


def find_audio_files(directory: Path) -> List[Path]:
    """
    在目录中查找所有音频文件
    
    Args:
        directory: 要搜索的目录
        
    Returns:
        List[Path]: 音频文件路径列表
    """
    audio_files = []
    supported_exts = get_supported_formats()
    
    for file_path in directory.glob("**/*"):
        if file_path.is_file() and file_path.suffix.lower() in supported_exts:
            audio_files.append(file_path)
    
    return sorted(audio_files)


def get_output_path(
    input_path: Path,
    output_path: Optional[Path],
    output_format: str
) -> Path:
    """
    获取输出文件路径
    
    Args:
        input_path: 输入文件路径
        output_path: 指定的输出路径（可选）
        output_format: 输出格式
        
    Returns:
        Path: 最终的输出文件路径
    """
    # 如果提供了输出路径
    if output_path:
        # 如果输出路径是目录，则在该目录中生成输出文件
        if output_path.is_dir():
            return output_path / f"{input_path.stem}.{output_format}"
        # 否则直接使用指定的输出路径
        return output_path
    
    # 默认情况下，在输入文件所在目录生成同名但扩展名不同的文件
    return input_path.with_suffix(f".{output_format}")


@click.group()
@click.version_option()
@click.option(
    "--verbose", "-v",
    is_flag=True,
    help="启用详细日志输出"
)
@click.pass_context
def cli(ctx: click.Context, verbose: bool) -> None:
    """
    Audio SRT: 基于Faster-Whisper的高效音频转字幕工具
    """
    # 设置日志级别
    setup_logging(verbose)
    
    # 存储在上下文中，供子命令使用
    ctx.ensure_object(dict)
    ctx.obj["verbose"] = verbose


@cli.command()
@click.argument(
    "audio_file",
    type=click.Path(exists=True, file_okay=True, dir_okay=False, path_type=Path)
)
@click.option(
    "--output", "-o",
    type=click.Path(path_type=Path),
    help="输出文件路径"
)
@click.option(
    "--format", "-f",
    type=click.Choice(SUPPORTED_FORMATS),
    default="srt",
    help="输出字幕格式 [default: srt]"
)
@click.option(
    "--model", "-m",
    type=str,
    default=None,
    help="模型大小 (tiny, base, small, medium, large-v1, large-v2, large-v3, large-v3-turbo)"
)
@click.option(
    "--language", "-l",
    type=str,
    default=None,
    help="音频语言 (如: zh, en, auto)"
)
@click.option(
    "--task",
    type=click.Choice(["transcribe", "translate"]),
    default="transcribe",
    help="任务类型: transcribe(转录)或translate(翻译成英文) [default: transcribe]"
)
@click.option(
    "--device",
    type=str,
    default=None,
    help="运行设备 (cuda, mps, cpu, auto)"
)
@click.option(
    "--compute-type",
    type=click.Choice(["float32", "float16", "int8"]),
    default=None,
    help="计算精度类型 [default: 自动选择]"
)
@click.option(
    "--threads",
    type=int,
    default=0,
    help="处理线程数 (0表示自动) [default: 0]"
)
@click.option(
    "--word-timestamps/--no-word-timestamps",
    default=False,
    help="启用词级时间戳 (仅在JSON格式输出中可用)"
)
@click.option(
    "--vad-filter/--no-vad-filter",
    default=True,
    help="使用语音活动检测过滤静音 [default: enabled]"
)
@click.option(
    "--max-line-length",
    type=int,
    default=42,
    help="字幕每行的最大字符数 [default: 42]"
)
@click.pass_context
def convert(
    ctx: click.Context,
    audio_file: Path,
    output: Optional[Path],
    format: str,
    model: Optional[str],
    language: Optional[str],
    task: str,
    device: Optional[str],
    compute_type: Optional[str],
    threads: int,
    word_timestamps: bool,
    vad_filter: bool,
    max_line_length: int
) -> None:
    """
    将单个音频文件转换为字幕
    """
    if not is_valid_audio_file(audio_file):
        click.echo(f"错误: {audio_file} 不是有效的音频文件", err=True)
        sys.exit(1)
    
    # 获取推荐的模型大小（如果未指定）
    if model is None:
        model_manager = ModelManager()
        model = model_manager.get_recommended_model()
        click.echo(f"自动选择模型大小: {model}")
    
    # 创建转录选项
    options = TranscriptionOptions(
        model_size=model,
        language=None if language == "auto" else language,
        task=task,
        word_timestamps=word_timestamps,
        vad_filter=vad_filter
    )
    
    # 创建转录器
    transcriber = Transcriber(options)
    
    # 如果指定了设备或计算精度，需要手动加载模型
    if device is not None or compute_type is not None or threads > 0:
        click.echo("使用自定义性能设置...")
        model_manager = ModelManager()
        
        # 加载模型时指定设备和计算精度
        transcriber._model = model_manager.load_model(
            model_size=model,
            device=device,  # 如果为None，会自动选择
            compute_type=compute_type,  # 如果为None，会自动选择
            num_workers=threads if threads > 0 else None  # 处理线程数
        )
        click.echo(f"模型已加载，设备: {device or '自动'}, 计算精度: {compute_type or '自动'}, 线程数: {threads if threads > 0 else '自动'}")
    
    # 创建格式化器
    formatter = FormatterFactory.get_formatter(
        format_type=format,
        max_line_length=max_line_length,
        include_word_timestamps=word_timestamps
    )
    
    click.echo(f"正在处理: {audio_file}")
    start_time = time.time()
    
    # 执行转录
    try:
        with tqdm(total=100, desc="转录进度") as pbar:
            def update_progress(progress: float) -> None:
                pbar.update(int(progress * 100) - pbar.n)
            
            result = transcriber.transcribe_with_segments(
                audio_file,
                segment_duration=600,  # 10分钟
                progress_callback=update_progress
            )
        
        duration = time.time() - start_time
        click.echo(f"转录完成! 用时: {duration:.2f}秒")
        click.echo(f"检测到的语言: {result.language}")
        
        # 获取输出路径
        output_path = get_output_path(audio_file, output, format)
        
        # 保存结果
        formatter.save(result, output_path)
        click.echo(f"已保存到: {output_path}")
        
    except Exception as e:
        click.echo(f"转录失败: {e}", err=True)
        sys.exit(1)


@cli.command()
@click.argument(
    "directory",
    type=click.Path(exists=True, file_okay=False, dir_okay=True, path_type=Path)
)
@click.option(
    "--output-dir", "-o",
    type=click.Path(file_okay=False, dir_okay=True, path_type=Path),
    help="输出目录"
)
@click.option(
    "--format", "-f",
    type=click.Choice(SUPPORTED_FORMATS),
    default="srt",
    help="输出字幕格式 [default: srt]"
)
@click.option(
    "--model", "-m",
    type=str,
    default=None,
    help="模型大小 (tiny, base, small, medium, large-v1, large-v2, large-v3, large-v3-turbo)"
)
@click.option(
    "--language", "-l",
    type=str,
    default=None,
    help="音频语言 (如: zh, en, auto)"
)
@click.option(
    "--recursive/--no-recursive", "-r/",
    default=False,
    help="递归处理子目录中的文件"
)
@click.option(
    "--workers", "-w",
    type=int,
    default=1,
    help="并行处理的工作线程数 [default: 1]"
)
@click.option(
    "--device",
    type=str,
    default=None,
    help="运行设备 (cuda, mps, cpu, auto)"
)
@click.option(
    "--compute-type",
    type=click.Choice(["float32", "float16", "int8"]),
    default=None,
    help="计算精度类型 [default: 自动选择]"
)
@click.option(
    "--threads",
    type=int,
    default=0,
    help="处理线程数 (0表示自动) [default: 0]"
)
@click.option(
    "--word-timestamps/--no-word-timestamps",
    default=False,
    help="启用词级时间戳 (仅在JSON格式输出中可用)"
)
@click.option(
    "--vad-filter/--no-vad-filter",
    default=True,
    help="使用语音活动检测过滤静音 [default: enabled]"
)
@click.option(
    "--force/--no-force", "-F/",
    default=False,
    help="强制处理已存在的字幕文件 [default: 不强制]"
)
@click.pass_context
def batch(
    ctx: click.Context,
    directory: Path,
    output_dir: Optional[Path],
    format: str,
    model: Optional[str],
    language: Optional[str],
    recursive: bool,
    workers: int,
    device: Optional[str],
    compute_type: Optional[str],
    threads: int,
    word_timestamps: bool,
    vad_filter: bool,
    force: bool
) -> None:
    """
    批量处理目录中的音频文件
    """
    # 创建输出目录（如果指定了）
    if output_dir:
        output_dir.mkdir(parents=True, exist_ok=True)
    
    # 查找所有音频文件
    pattern = "**/*" if recursive else "*"
    audio_files = []
    supported_exts = get_supported_formats()
    
    for file_path in directory.glob(pattern):
        if file_path.is_file() and file_path.suffix.lower() in supported_exts:
            audio_files.append(file_path)
    
    if not audio_files:
        click.echo("未找到任何音频文件")
        sys.exit(0)
    
    click.echo(f"找到 {len(audio_files)} 个音频文件")
    
    # 获取推荐的模型大小（如果未指定）
    if model is None:
        model_manager = ModelManager()
        model = model_manager.get_recommended_model()
        click.echo(f"自动选择模型大小: {model}")
    
    # 如果worker数大于1，需要使用joblib进行并行处理
    # 此部分实现需要依赖processor模块，目前尚未实现
    # 这里先采用串行处理
    click.echo(f"使用串行处理模式（并行处理功能待实现）")
    
    # 创建转录选项
    options = TranscriptionOptions(
        model_size=model,
        language=None if language == "auto" else language,
        task="transcribe",
        word_timestamps=word_timestamps and format == "json",
        vad_filter=vad_filter
    )
    
    # 创建转录器
    transcriber = Transcriber(options)
    
    # 如果指定了设备或计算精度，需要手动加载模型
    if device is not None or compute_type is not None or threads > 0:
        click.echo("使用自定义性能设置...")
        model_manager = ModelManager()
        
        # 加载模型时指定设备和计算精度
        transcriber._model = model_manager.load_model(
            model_size=model,
            device=device,  # 如果为None，会自动选择
            compute_type=compute_type,  # 如果为None，会自动选择
            num_workers=threads if threads > 0 else None  # 处理线程数
        )
        click.echo(f"模型已加载，设备: {device or '自动'}, 计算精度: {compute_type or '自动'}, 线程数: {threads if threads > 0 else '自动'}")
    
    # 统计信息
    processed_count = 0
    skipped_count = 0
    failed_count = 0
    
    # 处理每个文件
    for i, audio_file in enumerate(audio_files, start=1):
        # 获取输出路径
        if output_dir:
            rel_path = audio_file.relative_to(directory) if audio_file.is_relative_to(directory) else audio_file.name
            output_path = output_dir / rel_path.with_suffix(f".{format}")
            output_path.parent.mkdir(parents=True, exist_ok=True)
        else:
            output_path = audio_file.with_suffix(f".{format}")
        
        # 检查输出文件是否已存在
        if output_path.exists() and not force:
            click.echo(f"跳过文件 [{i}/{len(audio_files)}]: {audio_file} (输出文件已存在)")
            skipped_count += 1
            continue
        
        click.echo(f"处理文件 [{i}/{len(audio_files)}]: {audio_file}")
        
        try:
            # 执行转录
            with tqdm(total=100, desc="转录进度") as pbar:
                def update_progress(progress: float) -> None:
                    pbar.update(int(progress * 100) - pbar.n)
                
                result = transcriber.transcribe_with_segments(
                    audio_file,
                    segment_duration=600,  # 10分钟
                    progress_callback=update_progress
                )
            
            # 创建格式化器
            formatter = FormatterFactory.get_formatter(
                format_type=format,
                include_word_timestamps=word_timestamps and format == "json"
            )
            
            # 保存结果
            formatter.save(result, output_path)
            click.echo(f"已保存到: {output_path}")
            processed_count += 1
            
        except Exception as e:
            click.echo(f"处理文件 {audio_file} 失败: {e}", err=True)
            failed_count += 1
    
    # 输出统计信息
    click.echo("\n批处理完成!")
    click.echo(f"总文件数: {len(audio_files)}")
    click.echo(f"成功处理: {processed_count}")
    click.echo(f"已跳过: {skipped_count}")
    click.echo(f"失败: {failed_count}")


@cli.command()
@click.pass_context
def models(ctx: click.Context) -> None:
    """
    列出可用的模型
    """
    model_manager = ModelManager()
    available_models = model_manager.get_available_models()
    recommended = model_manager.get_recommended_model()
    device = model_manager.get_device()
    
    click.echo("可用的模型大小:")
    for model in available_models:
        info = model_manager.get_model_info(model)
        is_recommended = model == recommended
        marker = "* " if is_recommended else "  "
        click.echo(f"{marker}{model:<10} - 参数量: {info.get('parameters', '未知')}")
    
    click.echo(f"\n推荐模型: {recommended} (根据系统资源自动选择)")
    click.echo(f"检测到的设备: {device}")


@cli.command()
@click.argument(
    "config_file",
    type=click.Path(exists=True, file_okay=True, dir_okay=False, path_type=Path),
    required=False
)
@click.option(
    "--create-sample", "-c",
    type=click.Path(path_type=Path),
    help="创建示例配置文件"
)
@click.option(
    "--format", "-f",
    type=click.Choice(CONFIG_FORMATS),
    default="yaml",
    help="配置文件格式 [default: yaml]"
)
@click.pass_context
def config(
    ctx: click.Context,
    config_file: Optional[Path],
    create_sample: Optional[Path],
    format: str
) -> None:
    """
    使用配置文件批量处理音频文件
    
    如果提供了CONFIG_FILE，将使用该配置文件进行批处理。
    如果使用--create-sample选项，将创建一个示例配置文件。
    """
    # 创建示例配置文件
    if create_sample:
        create_sample_config(create_sample, format)
        return
    
    # 如果没有提供配置文件路径，显示帮助信息
    if not config_file:
        click.echo(ctx.get_help())
        return
    
    # 处理配置文件
    try:
        click.echo(f"使用配置文件: {config_file}")
        
        # 创建批处理器
        processor = BatchProcessor()
        
        # 处理配置文件
        report = processor.process_config_file(config_file)
        
        # 打印报告摘要
        report.print_summary()
        
    except Exception as e:
        click.echo(f"错误: {str(e)}", err=True)
        if ctx.obj["verbose"]:
            import traceback
            click.echo(traceback.format_exc(), err=True)
        sys.exit(1)


def main() -> None:
    """主入口函数"""
    cli(obj={}) 