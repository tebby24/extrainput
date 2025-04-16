# ExtraInput

A Python library for generating (Chinese) language learning content through AI-powered text, speech, and image generation.

## About

I only really wrote this up for use in my personal Chinese learning workflow. The code is adequate for my current usecase but is extremely messy. Honestly I don't think it made that much sense to combine these 3 technologies into one library as they're basically just seperate pieces in a pipline. Maybe in the future I'll just do seperate repos for the TTS functionality and the video editing functionality. Realistically the LLM stuff doesn't need its own wrapper.  

## Functionality

ExtraInput provides some functions for creating multimedia language learning materials by frankensteining together several AI services:
- Text generation using various language models via OpenRouter
- Text-to-speech synthesis (with matching subtitles) with Azure Speech Services
- Image generation with StabilityAI
- Video generation combining images and audio
- Smart voice pairing for text content

## Installation

```bash
pip install git+https://github.com/tebby24/extrainput.git
```

## API Requirements

You'll need API keys for the following services:
- OpenRouter for text generation
- Azure Speech Services for text-to-speech
- StabilityAI for image generation


## Example Usage

```python
from extrainput import ExtraInputGenerator
import os
from dotenv import load_dotenv

# Load API keys from environment variables
load_dotenv()

# Initialize the generator with API keys
generator = ExtraInputGenerator(
    openrouter_api_key=os.getenv("OPENROUTER_API_KEY"),
    azure_speech_key=os.getenv("AZURE_SPEECH_KEY"),
    azure_speech_region=os.getenv("AZURE_SPEECH_REGION"),
    stabilityai_api_key=os.getenv("STABILITYAI_API_KEY")
)

# Generate text
chinese_text = generator.generate_text("请写一个关于春天的短故事", model="deepseek/deepseek-r1:free")
print(f"Generated text: {chinese_text}")

# Find appropriate voice for the content
voice = generator.pair_voice_to_article(chinese_text)
print(f"Selected voice: {voice}")

# Generate speech with synchronized subtitles
generator.synthesize_speech_with_srt(
    chinese_text,
    voice, 
    "output/spring_audio.mp3", 
    "output/spring_subtitles.srt"
)

# Generate an image based on the text
generator.generate_image(
    "Spring landscape with cherry blossoms in China",
    "output/spring_image.png"
)

# Create a video combining the image and audio
generator.generate_video(
    "output/spring_image.png",
    "output/spring_audio.mp3",
    "output/spring_video.mp4"
)
```

