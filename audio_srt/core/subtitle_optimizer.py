"""
字幕优化模块，用于优化字幕分段、合并和格式化
"""

import re
import logging
from dataclasses import dataclass
from typing import List, Dict, Any, Tuple, Optional

@dataclass
class SubtitleOptimizerConfig:
    """字幕优化器的配置参数"""
    
    # 字幕合并相关参数
    max_gap_seconds: float = 1.5  # 合并片段之间的最大间隔（秒）
    min_segment_duration: float = 1.0  # 最小片段时长（秒）
    
    # 行长度控制相关参数
    max_chars_per_line: int = 50  # 每行字幕最多字符数
    max_lines_per_segment: int = 2  # 每个片段最大行数
    
    # 分段优化相关参数
    split_on_sentence_end: bool = True  # 是否在句末分割字幕
    sentence_end_pattern: str = r'[.!?。！？]'  # 句末匹配模式
    min_segment_gap: float = 0.2  # 分割后片段之间的最小间隔（秒）
    
    # 语言相关设置
    language: str = "auto"  # 字幕语言，用于某些特定语言的优化规则
    
    def __post_init__(self):
        """验证配置参数的有效性"""
        if self.max_gap_seconds < 0:
            raise ValueError("max_gap_seconds 必须大于等于 0")
        
        if self.min_segment_duration < 0:
            raise ValueError("min_segment_duration 必须大于等于 0")
        
        if self.max_chars_per_line < 1:
            raise ValueError("max_chars_per_line 必须大于 0")
        
        if self.max_lines_per_segment < 1:
            raise ValueError("max_lines_per_segment 必须大于 0")


class SubtitleOptimizer:
    """字幕优化器，用于优化转录结果的字幕效果"""
    
    def __init__(self, config: Optional[SubtitleOptimizerConfig] = None):
        """
        初始化字幕优化器
        
        Args:
            config: 优化器配置，如果为None则使用默认配置
        """
        self.config = config or SubtitleOptimizerConfig()
        self.logger = logging.getLogger("audio_srt.core.optimizer")
    
    def optimize_segments(self, segments: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        优化字幕片段
        
        Args:
            segments: 原始字幕片段列表，每个片段是一个字典，包含 start, end, text 等字段
            
        Returns:
            List[Dict[str, Any]]: 优化后的字幕片段列表
        """
        self.logger.info(f"开始优化字幕片段，共 {len(segments)} 个片段")
        
        # 先合并过短的片段
        merged_segments = self._merge_short_segments(segments)
        self.logger.debug(f"合并后的片段数: {len(merged_segments)}")
        
        # 然后控制行长度
        formatted_segments = self._control_line_length(merged_segments)
        self.logger.debug(f"格式化后的片段数: {len(formatted_segments)}")
        
        # 最后进行分段优化
        optimized_segments = self._optimize_segmentation(formatted_segments)
        self.logger.debug(f"优化后的片段数: {len(optimized_segments)}")
        
        return optimized_segments
    
    def _merge_short_segments(self, segments: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        合并过短的片段
        
        Args:
            segments: 原始字幕片段列表
            
        Returns:
            List[Dict[str, Any]]: 合并后的字幕片段列表
        """
        if not segments:
            return []
        
        result = [segments[0].copy()]
        
        for segment in segments[1:]:
            prev = result[-1]
            curr = segment.copy()
            
            # 计算与前一个片段的时间间隔
            gap = curr["start"] - prev["end"]
            
            # 如果片段间隔小于阈值且上一个片段时长较短，则合并
            if (gap <= self.config.max_gap_seconds and 
                (prev["end"] - prev["start"]) < self.config.min_segment_duration):
                
                # 更新文本和结束时间
                prev["text"] += " " + curr["text"]
                prev["end"] = curr["end"]
                
                # 如果有词级时间戳，也需要合并
                if "words" in prev and "words" in curr:
                    prev["words"].extend(curr["words"])
            else:
                result.append(curr)
        
        return result
    
    def _control_line_length(self, segments: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        控制每行字幕的长度
        
        Args:
            segments: 字幕片段列表
            
        Returns:
            List[Dict[str, Any]]: 格式化后的字幕片段列表
        """
        result = []
        
        for segment in segments:
            new_segment = segment.copy()
            text = segment["text"]
            
            # 如果文本长度超过限制，进行分行处理
            if len(text) > self.config.max_chars_per_line:
                new_text = self._split_text_into_lines(text)
                new_segment["text"] = new_text
            
            result.append(new_segment)
        
        return result
    
    def _split_text_into_lines(self, text: str) -> str:
        """
        将长文本分割为多行
        
        Args:
            text: 原始文本
            
        Returns:
            str: 分行后的文本
        """
        max_chars = self.config.max_chars_per_line
        max_lines = self.config.max_lines_per_segment
        
        # 如果文本长度本身小于最大字符数，直接返回
        if len(text) <= max_chars:
            return text
        
        # 分割策略: 尝试在标点或空格处分行
        lines = []
        remaining_text = text
        
        # 生成不超过最大行数的行
        while remaining_text and len(lines) < max_lines:
            if len(remaining_text) <= max_chars:
                lines.append(remaining_text)
                remaining_text = ""
            else:
                # 尝试在标点或空格处分行
                split_pos = self._find_split_position(remaining_text, max_chars)
                
                lines.append(remaining_text[:split_pos].strip())
                remaining_text = remaining_text[split_pos:].strip()
        
        # 如果还有剩余文本，将其添加到最后一行（可能超出最大字符数）
        if remaining_text and lines:
            if len(lines) == max_lines:
                # 达到最大行数，将剩余文本添加到最后一行
                lines[-1] = lines[-1] + " " + remaining_text
            else:
                # 未达到最大行数，添加一行
                lines.append(remaining_text)
        
        # 使用换行符连接行
        return "\n".join(lines)
    
    def _find_split_position(self, text: str, max_length: int) -> int:
        """
        在文本中找到合适的分割位置
        
        Args:
            text: 要分割的文本
            max_length: 最大长度
            
        Returns:
            int: 分割位置索引
        """
        # 如果文本长度小于最大长度，直接返回文本长度
        if len(text) <= max_length:
            return len(text)
        
        # 逆序查找标点符号和空格作为分割点
        for i in range(max_length, 0, -1):
            # 先检查是否是中文或英文的标点符号
            if i < len(text) and re.match(r'[,.;!?，。；！？]', text[i]):
                return i + 1  # 包括标点符号
            
            # 再检查是否是空格
            if i < len(text) and text[i] == ' ':
                return i + 1  # 包括空格
        
        # 如果没有找到合适的分割点，就在最大长度处分割
        return max_length
    
    def _optimize_segmentation(self, segments: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        优化字幕分段，尝试在句子边界断句
        
        Args:
            segments: 字幕片段列表
            
        Returns:
            List[Dict[str, Any]]: 优化分段后的字幕片段列表
        """
        # 如果不需要在句子边界分段，直接返回原始片段
        if not self.config.split_on_sentence_end:
            return segments
        
        result = []
        
        for segment in segments:
            # 检查是否需要在句子边界分段
            split_segments = self._split_on_sentence_boundary(segment)
            result.extend(split_segments)
        
        return result
    
    def _split_on_sentence_boundary(self, segment: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        在句子边界处分割字幕片段
        
        Args:
            segment: 要分割的字幕片段
            
        Returns:
            List[Dict[str, Any]]: 分割后的字幕片段列表
        """
        text = segment["text"]
        
        # 查找句子结束标记的位置
        pattern = self.config.sentence_end_pattern
        matches = list(re.finditer(pattern, text))
        
        # 如果没有找到句子结束标记或者只找到一个句子，直接返回原始片段
        if len(matches) <= 1:
            return [segment]
        
        # 原始片段的时长和开始时间
        duration = segment["end"] - segment["start"]
        start_time = segment["start"]
        
        # 分割后的片段列表
        split_segments = []
        
        # 上一个分割点的位置
        last_pos = 0
        
        # 遍历每个句子分割点
        for i, match in enumerate(matches[:-1]):  # 不处理最后一个句子分割点
            # 当前句子结束的位置
            end_pos = match.end()
            
            # 分割的比例
            ratio = end_pos / len(text)
            
            # 计算新片段的结束时间
            end_time = start_time + duration * ratio
            
            # 创建新的片段
            new_segment = segment.copy()
            new_segment["text"] = text[last_pos:end_pos].strip()
            new_segment["start"] = start_time
            new_segment["end"] = end_time
            
            # 处理词级时间戳（如果有）
            if "words" in segment:
                new_segment["words"] = self._filter_words_by_time(
                    segment["words"], start_time, end_time
                )
            
            split_segments.append(new_segment)
            
            # 更新下一个片段的开始时间和上一个分割点位置
            start_time = end_time + self.config.min_segment_gap
            last_pos = end_pos
        
        # 添加最后一个片段
        if last_pos < len(text):
            last_segment = segment.copy()
            last_segment["text"] = text[last_pos:].strip()
            last_segment["start"] = start_time
            
            # 处理词级时间戳（如果有）
            if "words" in segment:
                last_segment["words"] = self._filter_words_by_time(
                    segment["words"], start_time, segment["end"]
                )
            
            split_segments.append(last_segment)
        
        return split_segments
    
    def _filter_words_by_time(
        self, words: List[Dict[str, Any]], start_time: float, end_time: float
    ) -> List[Dict[str, Any]]:
        """
        根据时间范围过滤词级时间戳
        
        Args:
            words: 词级时间戳列表
            start_time: 开始时间
            end_time: 结束时间
            
        Returns:
            List[Dict[str, Any]]: 过滤后的词级时间戳列表
        """
        return [
            word for word in words 
            if word["start"] >= start_time and word["end"] <= end_time
        ] 