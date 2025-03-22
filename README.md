# ExtraInput
A Python package for generating Mandarin Chinese language learning audio content using AI text generation and speech synthesis.

## Installation

```bash
pip install git+https://github.com/tebby24/extrainput.git
```
## Usage

### Text Generation

```python
from extrainput import TextGenerator

text_gen = TextGenerator("your_deepseek_api_key")
example = text_gen.generate_text("请用'好'字造一个简单的句子")
print(example)
```

### Speech Synthesis

```python
from extrainput import SimpleTTSGenerator

tts = SimpleTTSGenerator("your_azure_speech_key", "your_azure_speech_region")
tts.synthesize_speech("你好世界", "zh-CN-XiaoxiaoNeural", "hello_world.wav")
```

### Speech with Subtitles

```python
from extrainput import TTSWithSubsGenerator

tts_subs = TTSWithSubsGenerator("your_azure_speech_key", "your_azure_speech_region")
tts_subs.synthesize_speech_with_srt(
    "这是一个测试。我们正在生成语音和字幕。", 
    "zh-CN-XiaoxiaoNeural", 
    "output.mp3", 
    "output.srt"
)
```

### Example Sentence Listener Generation

```python
from extrainput import ExampleSentenceListenerGenerator

esl_gen = ExampleSentenceListenerGenerator("your_deepseek_api_key", "your_azure_speech_key", "your_azure_speech_region")
words = ["你好", "谢谢", "再见"]
output_file = esl_gen.generate_example_sentence_listener(words, "output.wav")
print(f"Generated audio saved to {output_file}")
```

### Video Creation

```python
from extrainput import VideoGenerator

VideoGenerator.generate_video("image.jpg", "audio.mp3", "output.mp4")
```

## Available Voices

These are the voices accepted by the speech generation classes:

| Voice Name | Description |
|------------|-------------|
| zh-CN-YunxiaNeural | Male - Child |
| zh-CN-XiaoshuangNeural | Female - Child |
| zh-CN-YunxiNeural | Male - Young Adult |
| zh-CN-XiaoxiaoNeural | Female - Young Adult |
| zh-CN-YunjianNeural | Male - Adult |
| zh-CN-XiaorouNeural | Female - Adult |
| zh-CN-YunyeNeural | Male - Senior |
| zh-CN-XiaoqiuNeural | Female - Senior |

## Requirements

- Python 3.6+
- DeepSeek API key for text generation
- Azure Speech Service key and region for speech synthesis