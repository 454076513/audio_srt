from setuptools import setup, find_packages

setup(
    name="audio_srt",
    version="0.1.0",
    packages=find_packages(),
    install_requires=[
        "faster-whisper>=0.9.0",
        "torch>=2.0.0",
        "numpy>=1.22.0",
        "pydub>=0.25.1",
        "tqdm>=4.65.0",
        "click>=8.1.3",
        "joblib>=1.2.0",
        "ffmpeg-python>=0.2.0",
    ],
    entry_points={
        "console_scripts": [
            "audio_srt=audio_srt.cli.main:main",
        ],
    },
    author="Frank",
    author_email="example@email.com",
    description="高效的音频转字幕工具，基于Faster-Whisper",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    url="https://github.com/yourusername/audio_srt",
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: End Users/Desktop",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
    ],
    python_requires=">=3.8",
) 