"""
模型管理模块，负责Faster-Whisper模型的下载、缓存和加载
"""

import os
import logging
from pathlib import Path
from typing import Optional, Union, List, Dict, Any

import torch
from tqdm import tqdm


class ModelManager:
    """Faster-Whisper模型管理器"""
    
    # 可用的模型大小
    AVAILABLE_MODELS = ["tiny", "base", "small", "medium", "large-v1", "large-v2", "large-v3", "large-v3-turbo"]
    
    # 模型默认缓存目录
    DEFAULT_CACHE_DIR = Path.home() / ".cache" / "audio-srt" / "models"
    
    def __init__(self, cache_dir: Optional[Union[str, Path]] = None):
        """
        初始化模型管理器
        
        Args:
            cache_dir: 模型缓存目录，默认为~/.cache/audio-srt/models
        """
        self.cache_dir = Path(cache_dir) if cache_dir else self.DEFAULT_CACHE_DIR
        self.logger = logging.getLogger("audio_srt.models")
        
        # 确保缓存目录存在
        self.cache_dir.mkdir(parents=True, exist_ok=True)
    
    def get_available_models(self) -> List[str]:
        """
        获取可用的模型列表
        
        Returns:
            List[str]: 支持的模型大小列表
        """
        return self.AVAILABLE_MODELS.copy()
    
    def get_recommended_model(self) -> str:
        """
        获取推荐的默认模型大小
        
        根据系统资源情况，推荐合适的模型大小
        
        Returns:
            str: 推荐的模型大小名称
        """
        if torch.cuda.is_available():
            cuda_gb = torch.cuda.get_device_properties(0).total_memory / (1024**3)
            if cuda_gb >= 16:
                return "large-v3-turbo"
            elif cuda_gb >= 12:
                return "large-v3"
            elif cuda_gb >= 8:
                return "medium"
            elif cuda_gb >= 4:
                return "small"
            else:
                return "base"
        else:
            # CPU模式下更保守的推荐
            try:
                import psutil
                mem_gb = psutil.virtual_memory().total / (1024**3)
                if mem_gb >= 16:
                    return "medium"
                elif mem_gb >= 8:
                    return "small"
                elif mem_gb >= 4:
                    return "base"
                else:
                    return "tiny"
            except ImportError:
                # 如果无法检测内存，返回安全的默认值
                return "base"
    
    def get_device(self) -> str:
        """
        获取建议的设备类型
        
        Returns:
            str: 'cuda', 'mps', 或 'cpu'
        """
        if torch.cuda.is_available():
            return "cuda"
        elif hasattr(torch, 'backends') and hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
            # 检测 Apple Silicon (M1/M2/M3) 的 MPS 加速
            self.logger.info("检测到 Apple Silicon MPS 加速")
            return "mps"
        else:
            return "cpu"
    
    def load_model(
        self,
        model_size: str = "small",
        device: Optional[str] = None,
        compute_type: Optional[str] = None,
        download_root: Optional[Union[str, Path]] = None,
        num_workers: Optional[int] = None,
        **kwargs
    ) -> Any:
        """
        加载Faster-Whisper模型
        
        Args:
            model_size: 模型大小，可选值为"tiny", "base", "small", "medium", "large-v1", "large-v2", "large-v3", "large-v3-turbo"
            device: 运行设备，可选值为"cpu", "cuda", "mps", None（自动选择）
            compute_type: 计算类型，例如"float16", "int8"
            download_root: 模型下载目录，默认为缓存目录
            num_workers: 处理线程数，None表示使用默认值
            **kwargs: 传递给Faster-Whisper模型构造函数的额外参数
            
        Returns:
            WhisperModel: Faster-Whisper模型实例
            
        Raises:
            ValueError: 如果模型大小不受支持
            ImportError: 如果未安装faster-whisper
        """
        if model_size not in self.AVAILABLE_MODELS:
            raise ValueError(
                f"不支持的模型大小: {model_size}。可用选项: {', '.join(self.AVAILABLE_MODELS)}"
            )
        
        try:
            from faster_whisper import WhisperModel
        except ImportError:
            raise ImportError("请安装faster-whisper: pip install faster-whisper")
        
        # 设置默认值
        if device is None:
            device = self.get_device()
        
        # MPS 设备兼容性检查 - Faster-Whisper 目前不支持 MPS
        if device == "mps":
            self.logger.warning("Faster-Whisper 不支持 MPS 设备，回退到 CPU 模式")
            device = "cpu"
            
        if compute_type is None:
            # 根据设备类型选择计算精度
            if device == "cuda":
                compute_type = "float16"
            elif device == "mps":  # 虽然已经回退到 CPU，但保留这个逻辑以备将来支持
                compute_type = "float16"
            else:
                compute_type = "int8"
        
        # 使用缓存目录作为下载根目录
        if download_root is None:
            download_root = self.cache_dir
        
        self.logger.info(f"正在加载模型 {model_size}，设备: {device}，计算类型: {compute_type}")
        
        # 优化 CPU 线程使用
        if device == "cpu" and num_workers is not None:
            # 设置 ctranslate2 的线程数
            import os
            
            # 如果用户指定了线程数
            if num_workers > 0:
                # 设置总线程数
                os.environ["CT2_VERBOSE"] = "1"  # 启用详细日志
                os.environ["CT2_NUM_THREADS"] = str(num_workers)
                
                # 计算 intra_threads 和 inter_threads
                # 对于小模型，更多的 intra_threads 更好
                # 对于大模型，更多的 inter_threads 更好
                if model_size in ["tiny", "base", "small"]:
                    intra_threads = max(2, num_workers // 2)
                    inter_threads = max(1, num_workers // 4)
                else:
                    intra_threads = max(1, num_workers // 4)
                    inter_threads = max(2, num_workers // 2)
                
                os.environ["CT2_INTRA_THREADS"] = str(intra_threads)
                os.environ["CT2_INTER_THREADS"] = str(inter_threads)
                
                self.logger.info(f"设置 CPU 线程优化: 总线程={num_workers}, intra={intra_threads}, inter={inter_threads}")
        
        # 如果指定了线程数，添加到参数中
        if num_workers is not None and num_workers > 0:
            kwargs["num_workers"] = num_workers
            self.logger.info(f"设置处理线程数: {num_workers}")
        
        # 加载模型
        model = WhisperModel(
            model_size,
            device=device,
            compute_type=compute_type,
            download_root=str(download_root),
            **kwargs
        )
        
        return model
    
    def get_model_info(self, model_size: str) -> Dict[str, Any]:
        """
        获取模型的信息
        
        Args:
            model_size: 模型大小
            
        Returns:
            Dict[str, Any]: 包含模型信息的字典
        """
        if model_size not in self.AVAILABLE_MODELS:
            raise ValueError(
                f"不支持的模型大小: {model_size}。可用选项: {', '.join(self.AVAILABLE_MODELS)}"
            )
        
        # 模型大小（参数量）信息
        model_params = {
            "tiny": {"parameters": "39M", "english_only": False, "multilingual": True},
            "base": {"parameters": "74M", "english_only": False, "multilingual": True},
            "small": {"parameters": "244M", "english_only": False, "multilingual": True},
            "medium": {"parameters": "769M", "english_only": False, "multilingual": True},
            "large-v1": {"parameters": "1550M", "english_only": False, "multilingual": True},
            "large-v2": {"parameters": "1550M", "english_only": False, "multilingual": True},
            "large-v3": {"parameters": "1550M", "english_only": False, "multilingual": True},
            "large-v3-turbo": {"parameters": "1550M", "english_only": False, "multilingual": True},
        }
        
        return model_params[model_size]
    
    def clear_cache(self, model_size: Optional[str] = None) -> None:
        """
        清除模型缓存
        
        Args:
            model_size: 要清除的特定模型大小，如果为None则清除所有模型
            
        Returns:
            None
        """
        if model_size:
            if model_size not in self.AVAILABLE_MODELS:
                raise ValueError(
                    f"不支持的模型大小: {model_size}。可用选项: {', '.join(self.AVAILABLE_MODELS)}"
                )
            model_path = self.cache_dir / model_size
            if model_path.exists():
                import shutil
                shutil.rmtree(model_path)
                self.logger.info(f"已清除模型缓存: {model_size}")
        else:
            # 清除所有模型
            for model in self.AVAILABLE_MODELS:
                model_path = self.cache_dir / model
                if model_path.exists():
                    import shutil
                    shutil.rmtree(model_path)
            self.logger.info("已清除所有模型缓存") 