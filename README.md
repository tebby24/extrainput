# ExtraInput

A Python package for generating language learning audio content using AI text generation and speech synthesis.

## Installation

```bash
pip install git+https://github.com/tebby24/extrainput.git
```

## Usage

```python
from extrainput import TextGenerator, TTSEngine, ExtraInputGenerator

# Initialize components
text_generator = TextGenerator("your_deepseek_api_key")
tts_engine = TTSEngine("your_azure_speech_key", "your_azure_speech_region")
generator = ExtraInputGenerator(text_generator, tts_engine)

# Generate example sentences
words = ["你好", "谢谢", "再见"]
output_file = generator.generate_example_sentence_listener(words, "output.wav")
print(f"Generated audio saved to {output_file}")
```
