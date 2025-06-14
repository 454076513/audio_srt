"""
字幕优化器单元测试
"""

import unittest
from audio_srt.core.subtitle_optimizer import SubtitleOptimizer, SubtitleOptimizerConfig


class TestSubtitleOptimizer(unittest.TestCase):
    """字幕优化器测试类"""

    def test_merge_short_segments(self):
        """测试合并短片段功能"""
        # 配置优化器
        config = SubtitleOptimizerConfig(max_gap_seconds=1.0, min_segment_duration=2.0)
        optimizer = SubtitleOptimizer(config)

        # 测试数据
        segments = [
            {"start": 0.0, "end": 1.5, "text": "这是第一段"},
            {"start": 2.0, "end": 3.0, "text": "这是第二段"},  # 间隔 0.5 秒，第一段小于 min_duration，应当合并
            {"start": 5.0, "end": 7.0, "text": "这是第三段"},  # 间隔 2.0 秒，大于 max_gap，不应当合并
            {"start": 7.5, "end": 8.0, "text": "这是第四段"},  # 间隔 0.5 秒，第三段大于 min_duration，不应当合并
        ]

        merged = optimizer._merge_short_segments(segments)

        # 断言
        self.assertEqual(len(merged), 3)
        self.assertEqual(merged[0]["start"], 0.0)
        self.assertEqual(merged[0]["end"], 3.0)
        self.assertEqual(merged[0]["text"], "这是第一段 这是第二段")
        self.assertEqual(merged[1]["start"], 5.0)
        self.assertEqual(merged[2]["start"], 7.5)

    def test_control_line_length(self):
        """测试行长控制功能"""
        # 配置优化器
        config = SubtitleOptimizerConfig(max_chars_per_line=10, max_lines_per_segment=2)
        optimizer = SubtitleOptimizer(config)

        # 测试数据
        segments = [
            {"start": 0.0, "end": 2.0, "text": "这是一个短句"},  # 不需要分行
            {"start": 3.0, "end": 5.0, "text": "这是一个超过十个字符的长句子"},  # 需要分行
            {"start": 6.0, "end": 8.0, "text": "这是一个非常非常非常非常长的句子需要被截断的"},  # 超过两行需要截断
        ]

        formatted = optimizer._control_line_length(segments)

        # 断言
        self.assertEqual(len(formatted), 3)
        self.assertEqual(formatted[0]["text"], "这是一个短句")  # 不变
        
        # 检查第二个分段是否正确分行
        self.assertIn("\n", formatted[1]["text"])  # 应包含换行符
        lines = formatted[1]["text"].split("\n")
        self.assertEqual(len(lines), 2)  # 应该有两行
        self.assertLessEqual(len(lines[0]), 10)  # 每行不超过10字符
        
        # 检查第三个分段是否正确分行和截断
        self.assertIn("\n", formatted[2]["text"])  # 应包含换行符
        lines = formatted[2]["text"].split("\n")
        self.assertEqual(len(lines), 2)  # 应该有两行，不能超过max_lines
        self.assertLessEqual(len(lines[0]), 10)  # 第一行不超过10字符

    def test_split_on_sentence_boundary(self):
        """测试在句子边界分段功能"""
        # 配置优化器
        config = SubtitleOptimizerConfig(split_on_sentence_end=True)
        optimizer = SubtitleOptimizer(config)

        # 测试数据：包含多个句子的长片段
        segment = {
            "start": 0.0,
            "end": 10.0,
            "text": "这是第一个句子。这是第二个句子！这是第三个句子？",
            "words": [
                {"word": "这", "start": 0.5, "end": 1.0},
                {"word": "是", "start": 1.0, "end": 1.5},
                {"word": "第一个", "start": 1.5, "end": 2.5},
                {"word": "句子", "start": 2.5, "end": 3.0},
                {"word": "。", "start": 3.0, "end": 3.5},
                {"word": "这", "start": 4.0, "end": 4.5},
                {"word": "是", "start": 4.5, "end": 5.0},
                {"word": "第二个", "start": 5.0, "end": 6.0},
                {"word": "句子", "start": 6.0, "end": 7.0},
                {"word": "！", "start": 7.0, "end": 7.5},
                {"word": "这", "start": 8.0, "end": 8.5},
                {"word": "是", "start": 8.5, "end": 9.0},
                {"word": "第三个", "start": 9.0, "end": 9.5},
                {"word": "句子", "start": 9.5, "end": 10.0},
                {"word": "？", "start": 10.0, "end": 10.0}
            ]
        }

        split_segments = optimizer._split_on_sentence_boundary(segment)

        # 断言
        self.assertEqual(len(split_segments), 3)  # 应分为3个片段
        self.assertEqual(split_segments[0]["text"], "这是第一个句子。")
        self.assertEqual(split_segments[1]["text"], "这是第二个句子！")
        self.assertEqual(split_segments[2]["text"], "这是第三个句子？")
        
        # 检查时间戳是否正确分配
        self.assertAlmostEqual(split_segments[0]["start"], 0.0)
        self.assertLess(split_segments[0]["end"], 5.0)  # 应小于第二个片段的开始
        
        self.assertGreater(split_segments[1]["start"], split_segments[0]["end"])
        self.assertLess(split_segments[1]["end"], 10.0)
        
        self.assertGreater(split_segments[2]["start"], split_segments[1]["end"])
        self.assertAlmostEqual(split_segments[2]["end"], 10.0)
        
        # 检查词级时间戳是否正确过滤
        self.assertTrue(all(word["start"] >= split_segments[0]["start"] and word["end"] <= split_segments[0]["end"] for word in split_segments[0]["words"]))
        self.assertTrue(all(word["start"] >= split_segments[1]["start"] and word["end"] <= split_segments[1]["end"] for word in split_segments[1]["words"]))
        self.assertTrue(all(word["start"] >= split_segments[2]["start"] and word["end"] <= split_segments[2]["end"] for word in split_segments[2]["words"]))

    def test_end_to_end_optimization(self):
        """测试端到端的字幕优化完整流程"""
        # 配置优化器
        config = SubtitleOptimizerConfig(
            max_gap_seconds=1.0,
            min_segment_duration=2.0,
            max_chars_per_line=20,
            max_lines_per_segment=2,
            split_on_sentence_end=True
        )
        optimizer = SubtitleOptimizer(config)

        # 测试数据
        segments = [
            {"start": 0.0, "end": 1.0, "text": "这是第一段"},
            {"start": 1.5, "end": 2.0, "text": "短片段应该被合并"},  # 应该与第一段合并
            {"start": 4.0, "end": 10.0, "text": "这是一个很长的句子。应该在句号处分段。这样可以提高可读性。"},  # 应在句号处分段
        ]

        optimized = optimizer.optimize_segments(segments)

        # 基本断言
        self.assertGreater(len(optimized), 1)  # 至少应该有多个片段
        self.assertLessEqual(len(optimized), 5)  # 不应该产生太多片段

        # 检查文本和格式
        first_segment_text = optimized[0]["text"]
        self.assertIn("这是第一段", first_segment_text)
        self.assertIn("短片段应该被合并", first_segment_text)
        
        # 检查后续片段是否在句号处分段
        if len(optimized) > 1:
            for segment in optimized[1:]:
                # 每个分段应该是完整的句子（以句号、问号或感叹号结尾）
                self.assertTrue(
                    segment["text"].strip().endswith(".") or 
                    segment["text"].strip().endswith("。") or
                    segment["text"].strip().endswith("!") or 
                    segment["text"].strip().endswith("！") or
                    segment["text"].strip().endswith("?") or 
                    segment["text"].strip().endswith("？")
                )


if __name__ == "__main__":
    unittest.main() 