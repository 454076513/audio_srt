# Audio SRT

基于 Faster-Whisper 的高效音频转字幕工具，支持多种字幕格式输出。

## 特点

- **高效转录**：使用 Faster-Whisper 实现快速准确的语音识别
- **轻量设计**：优化资源占用，适用于普通 PC 配置
- **多格式支持**：输出 SRT、WebVTT、JSON 和纯文本格式
- **词级时间戳**：支持在 JSON 格式中输出词级别的时间戳
- **批量处理**：支持批量处理多个音频文件，自动跳过已处理文件
- **字幕优化**：智能合并短句，限制字幕长度，提升阅读体验
- **多语言支持**：继承 Whisper 的多语言识别能力
- **多种模型**：支持从 tiny 到 large-v3-turbo 多种模型大小
- **硬件加速**：支持 CUDA、Apple Silicon MPS 等硬件加速

## 安装

```bash
# 从PyPI安装
pip install audio_srt

# 或从源码安装
git clone https://github.com/yourusername/audio_srt.git
cd audio_srt
pip install -e .
```

## 使用方法

### 基本用法

```bash
# 转换单个音频文件为SRT字幕
audio-srt convert audio.mp3 --format srt

# 指定输出文件
audio-srt convert audio.mp3 --output subtitles.srt

# 选择不同的模型大小
audio-srt convert audio.mp3 --model medium
```

### 支持的模型

Audio SRT 支持以下模型大小，根据您的硬件资源选择合适的模型：

- `tiny`: 39M 参数，适用于资源受限的环境
- `base`: 74M 参数，基础性能
- `small`: 244M 参数，平衡性能和资源占用
- `medium`: 769M 参数，更好的准确性
- `large-v1`: 1550M 参数，高准确性
- `large-v2`: 1550M 参数，改进版本
- `large-v3`: 1550M 参数，最新版本
- `large-v3-turbo`: 1550M 参数，优化的最新版本

### 词级时间戳

```bash
# 生成带有词级时间戳的JSON格式字幕
audio-srt convert audio.mp3 --format json --word-timestamps
```

JSON 输出示例：

```json
{
  "text": "Hello, my name is John.",
  "segments": [
    {
      "id": "seg1",
      "start": 1.0,
      "end": 4.0,
      "text": "Hello, my name is John.",
      "words": [
        { "word": "Hello", "start": 1.0, "end": 1.5 },
        { "word": "my", "start": 1.7, "end": 1.9 },
        { "word": "name", "start": 2.0, "end": 2.3 },
        { "word": "is", "start": 2.4, "end": 2.6 },
        { "word": "John", "start": 2.8, "end": 3.5 }
      ]
    }
  ],
  "language": "en",
  "duration": 4.0
}
```

### 批量转换

```bash
# 转换目录中的所有音频文件
audio-srt batch /path/to/audio/files --format srt

# 指定输出目录
audio-srt batch /path/to/audio/files --output-dir /path/to/output

# 递归处理子目录
audio-srt batch /path/to/audio/files --recursive

# 强制重新处理已存在的文件
audio-srt batch /path/to/audio/files --force
```

### 批量处理带词级时间戳的 JSON

```bash
# 批量处理并生成带词级时间戳的JSON文件
audio-srt batch /path/to/audio/files --format json --word-timestamps

# 使用大型模型并递归处理
audio-srt batch /path/to/audio/files --model large-v3-turbo --format json --word-timestamps --recursive
```

### 性能优化

在 Mac M3 等 Apple Silicon 设备上，可以使用以下参数优化性能：

```bash
# 控制处理线程数（推荐）
audio-srt convert audio.mp3 --threads 4

# 使用较小的模型减少资源占用
audio-srt convert audio.mp3 --model small
```

> **注意**：虽然 Audio SRT 检测到 Apple Silicon MPS 加速，但 Faster-Whisper 目前不直接支持 MPS 设备，将自动回退到 CPU 模式。未来版本可能会添加完整的 MPS 支持。

对于长音频文件，可以尝试：

```bash
# 批处理时使用较小的模型和优化的线程数
audio-srt batch /path/to/files --model small --threads 4
```

**提高 Mac 性能的建议**：

- 使用 `tiny` 或 `base` 模型以减少内存和 CPU 使用
- 设置合适的线程数（通常为核心数的一半）
  ```bash
  # 例如，在 M3 上使用 4-6 个线程通常效果最佳
  audio-srt convert audio.mp3 --threads 4
  ```
- 对于批处理，使用较小的音频文件或先将大文件分割
- 关闭其他占用 CPU 的应用程序
- 线程优化会自动应用于 CTranslate2 引擎，根据模型大小调整内部线程分配

**线程优化说明**：

- 小型模型 (tiny, base, small): 优先分配更多的 intra_threads 以提高单操作性能
- 大型模型 (medium, large): 优先分配更多的 inter_threads 以提高批处理性能
- 这些优化是自动的，只需使用 `--threads` 参数指定总线程数即可

### 高级选项

```bash
# 设置语言（自动检测语言时可选）
audio-srt convert audio.mp3 --language zh

# 翻译成英文
audio-srt convert audio.mp3 --task translate

# 使用VAD过滤静音
audio-srt convert audio.mp3 --vad-filter
```

## 命令行参数

### convert 命令

```
audio-srt convert [OPTIONS] AUDIO_FILE
```

| 选项                | 描述                                                                               |
| ------------------- | ---------------------------------------------------------------------------------- |
| `--output, -o`      | 输出文件路径                                                                       |
| `--format, -f`      | 输出格式 (srt, vtt, json, txt)                                                     |
| `--model, -m`       | 模型大小 (tiny, base, small, medium, large-v1, large-v2, large-v3, large-v3-turbo) |
| `--language, -l`    | 音频语言 (zh, en, auto)                                                            |
| `--task`            | 任务类型 (transcribe, translate)                                                   |
| `--device`          | 运行设备 (cuda, mps, cpu, auto)                                                    |
| `--compute-type`    | 计算精度类型 (float32, float16, int8)                                              |
| `--threads`         | 处理线程数 (0 表示自动)                                                            |
| `--word-timestamps` | 启用词级时间戳 (仅 JSON 格式)                                                      |
| `--vad-filter`      | 使用语音活动检测过滤静音                                                           |

### batch 命令

```
audio-srt batch [OPTIONS] DIRECTORY
```

| 选项                | 描述                                                                               |
| ------------------- | ---------------------------------------------------------------------------------- |
| `--output-dir, -o`  | 输出目录                                                                           |
| `--format, -f`      | 输出格式 (srt, vtt, json, txt)                                                     |
| `--model, -m`       | 模型大小 (tiny, base, small, medium, large-v1, large-v2, large-v3, large-v3-turbo) |
| `--language, -l`    | 音频语言 (zh, en, auto)                                                            |
| `--recursive, -r`   | 递归处理子目录                                                                     |
| `--device`          | 运行设备 (cuda, mps, cpu, auto)                                                    |
| `--compute-type`    | 计算精度类型 (float32, float16, int8)                                              |
| `--threads`         | 处理线程数 (0 表示自动)                                                            |
| `--word-timestamps` | 启用词级时间戳 (仅 JSON 格式)                                                      |
| `--force, -F`       | 强制处理已存在的字幕文件                                                           |

## 许可证

MIT

```bash
audio-srt batch ../bilibili_downloader/englishpod -o englishpod_json --model large-v3-turbo --format json --word-timestamps --language en
```
