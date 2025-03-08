import random
from pydub import AudioSegment
from openai import OpenAI
from azure.cognitiveservices.speech import SpeechConfig, SpeechSynthesizer, AudioConfig, ResultReason, CancellationReason
import concurrent.futures
from typing import List, Tuple
import srt
from datetime import timedelta
import os
import requests
import json
import time
import uuid
import zipfile
import tempfile
from moviepy import AudioFileClip, ImageClip

class TextGenerator:
    def __init__(self, deepseek_api_key):
        self.client = OpenAI(api_key=deepseek_api_key, base_url="https://api.deepseek.com")

    def generate_text(self, prompt):
        response = self.client.chat.completions.create(
            model="deepseek-chat",
            messages=[
                {"role": "system", "content": "You are a helpful assistant"},
                {"role": "user", "content": prompt},
            ],
            stream=False
        )
        return response.choices[0].message.content

class ImageGenerator:
    def __init__(self, openai_api_key):
        """Initialize the DALL-E image generator with OpenAI API key.
        
        Args:
            openai_api_key (str): OpenAI API key for accessing DALL-E
        """
        self.client = OpenAI(api_key=openai_api_key)
    
    def generate_image(self, prompt, output_filepath, size="1792x1024", quality="standard", model="dall-e-3"):
        """Generate a landscape-oriented image using DALL-E 3.
        
        Args:
            prompt (str): Description of the image to generate
            output_filepath (str, optional): Path to save the generated image
            size (str, optional): Size of the image - using 1792x1024 for landscape orientation
            quality (str, optional): Image quality - "standard" or "hd"
            model (str, optional): DALL-E model version
            
        Returns:
            str: URL of the generated image or path to saved image if output_filepath is provided
        """
        # Enhance prompt to ensure landscape orientation
        enhanced_prompt = f"{prompt} (landscape orientation, wide format, horizontal composition)"
        
        try:
            response = self.client.images.generate(
                model=model,
                prompt=enhanced_prompt,
                size=size,
                quality=quality,
                n=1
            )
            
            image_url = response.data[0].url
            
            # Save the image if output filepath is provided
            image_data = requests.get(image_url).content
            with open(output_filepath, 'wb') as image_file:
                image_file.write(image_data)
            return output_filepath
            
        except Exception as e:
            print(f"Error generating image: {str(e)}")
            return None

class SimpleTTSGenerator:
    voices = [
        {"name": "zh-CN-YunxiaNeural", "description": "Male - Child"},
        {"name": "zh-CN-XiaoshuangNeural", "description": "Female - Child"},
        {"name": "zh-CN-YunxiNeural", "description": "Male - Young Adult"},
        {"name": "zh-CN-XiaoxiaoNeural", "description": "Female - Young Adult"},
        {"name": "zh-CN-YunjianNeural", "description": "Male - Adult"},
        {"name": "zh-CN-XiaorouNeural", "description": "Female - Adult"},
        {"name": "zh-CN-YunyeNeural", "description": "Male - Senior"},
        {"name": "zh-CN-XiaoqiuNeural", "description": "Female - Senior"},
    ]

    def __init__(self, azure_speech_key, azure_speech_region):
        self.speech_config = SpeechConfig(subscription=azure_speech_key, region=azure_speech_region)

    def synthesize_speech(self, text, voice, output_filepath):
        self.speech_config.speech_synthesis_voice_name = voice
        audio_config = AudioConfig(filename=output_filepath)
        synthesizer = SpeechSynthesizer(speech_config=self.speech_config, audio_config=audio_config)
        result = synthesizer.speak_text_async(text).get()

        if result.reason == ResultReason.Canceled:
            cancellation_details = result.cancellation_details
            print(f"Speech synthesis canceled: {cancellation_details.reason}")
            if cancellation_details.reason == CancellationReason.Error:
                print(f"Error details: {cancellation_details.error_details}")
        elif result.reason == ResultReason.SynthesizingAudioCompleted:
            with open(output_filepath, "wb") as audio_file:
                audio_file.write(result.audio_data)
                return output_filepath
            print(f"Speech synthesized for text [{text}] and saved to [{output_filepath}]")


class ExampleSentenceListenerGenerator:
    def __init__(self, deepseek_api_key, azure_speech_key, azure_speech_region):
        self.text_generator = TextGenerator(deepseek_api_key)
        self.tts_engine = SimpleTTSGenerator(azure_speech_key, azure_speech_region)
    
    def get_example_sentences(self, word, num_sentences=3):
        prompt = f"""请用中文词语'{word}'造{num_sentences}个例句。
    要求：
    1. 每个例句单独成行
    2. 不要包含编号
    3. 不要有任何解释，只需要例句
    4. 例句应该简单且实用"""
        
        generated_text = self.text_generator.generate_text(prompt)
        return [line.strip() for line in generated_text.split('\n') if line.strip()]

    def get_translation(self, word):
        prompt = f"请你把'{word}'这个词翻译成英文，只需要提供英文翻译，不要包含其他内容。"
        return self.text_generator.generate_text(prompt)

    def get_random_voice(self, exclude="none"):
        voice = ""
        while (voice == "" or voice == exclude):
            voice = random.choice(SimpleTTSGenerator.voices)["name"]
        return voice

    def generate_example_sentence_listener(self, words, output_filepath):
        # Functions to get data for a single word
        def get_word_data(word) -> Tuple[str, List[str]]:
            translation = self.get_translation(word)
            example_sentences = self.get_example_sentences(word)
            return translation, example_sentences

        # Use ThreadPoolExecutor to fetch data in parallel
        word_data_map = {}
        with concurrent.futures.ThreadPoolExecutor() as executor:
            # Submit all tasks and store future objects
            future_to_word = {executor.submit(get_word_data, word): word for word in words}
            
            # Process completed tasks as they finish
            for future in concurrent.futures.as_completed(future_to_word):
                word = future_to_word[future]
                try:
                    translation, example_sentences = future.result()
                    word_data_map[word] = (translation, example_sentences)
                except Exception as e:
                    print(f"Error processing word '{word}': {e}")
        
        # Generate audio files after all data is fetched
        audios = []
        for word in words:
            if word not in word_data_map:
                continue
                
            translation, example_sentences = word_data_map[word]
            
            # Ensure the directory exists
            os.makedirs("bin/tmp", exist_ok=True)
            audios.append(self.tts_engine.synthesize_speech(word, self.get_random_voice(), f"bin/tmp/{word}_word.wav"))
            audios.append(self.tts_engine.synthesize_speech(translation, self.get_random_voice(), f"bin/tmp/{word}_translation.wav"))

            for index, sentence in enumerate(example_sentences):
                voice_1 = self.get_random_voice()
                voice_2 = self.get_random_voice(exclude=voice_1)
                audios.append(self.tts_engine.synthesize_speech(
                    sentence, 
                    voice_1, 
                    f"bin/tmp/{word}_sentence_{index+1}_a.wav"
                ))
                audios.append(self.tts_engine.synthesize_speech(
                    sentence, 
                    voice_2, 
                    f"bin/tmp/{word}_sentence_{index+1}_b.wav"

                ))
        combined_audio = AudioSegment.silent(duration=200)
        for audio_path in audios:
            if ('_word' in audio_path):
                combined_audio += AudioSegment.silent(duration=1000) # 1 second of silence between each word subgroup
            audio_segment = AudioSegment.from_wav(audio_path)
            combined_audio += audio_segment + AudioSegment.silent(duration=500)  # 0.5 second of silence between segments

        combined_audio.export(output_filepath, format="wav")
        return output_filepath



class TTSWithSubsGenerator:
    voices = [
        {"name": "zh-CN-YunxiaNeural", "description": "Male - Child"},
        {"name": "zh-CN-XiaoshuangNeural", "description": "Female - Child"},
        {"name": "zh-CN-YunxiNeural", "description": "Male - Young Adult"},
        {"name": "zh-CN-XiaoxiaoNeural", "description": "Female - Young Adult"},
        {"name": "zh-CN-YunjianNeural", "description": "Male - Adult"},
        {"name": "zh-CN-XiaorouNeural", "description": "Female - Adult"},
        {"name": "zh-CN-YunyeNeural", "description": "Male - Senior"},
        {"name": "zh-CN-XiaoqiuNeural", "description": "Female - Senior"},
    ]

    def __init__(self, azure_speech_key, azure_speech_region):
        """Initialize the TTS generator with Azure credentials from environment variables."""
        self.speech_key = azure_speech_key
        self.speech_region = azure_speech_region
        self.speech_endpoint = f"https://{self.speech_region}.api.cognitive.microsoft.com"
        self.api_version = "2024-04-01"
        
        if not self.speech_key or not self.speech_region:
            raise ValueError("Azure Speech credentials not found in environment variables")

    def synthesize_speech_with_srt(self, text, voice, mp3_output_filepath, srt_output_filepath):
        """
        Generate speech from text and create a synchronized subtitle file.
        
        Args:
            text (str): The text to synthesize
            voice (str): The voice name to use
            mp3_output_filepath (str): Path to save the MP3 audio output
            srt_output_filepath (str): Path to save the SRT subtitle output
        """
        voice_names = [v["name"] for v in self.voices]
        if voice not in voice_names:
            raise ValueError(f"Voice '{voice}' not found. Available voices: {voice_names}")

        # Create a unique job ID
        job_id = str(uuid.uuid4())
        
        # Create the batch synthesis request
        headers = {
            "Ocp-Apim-Subscription-Key": self.speech_key,
            "Content-Type": "application/json"
        }

        body = {
            "inputKind": "PlainText",
            "synthesisConfig": {
                "voice": voice,
            },
            "inputs": [
                {
                    "content": text
                },
            ],
            "properties": {
                "outputFormat": "audio-16khz-32kbitrate-mono-mp3",
                "wordBoundaryEnabled": True
            }
        }

        # Submit the synthesis job
        url = f"{self.speech_endpoint}/texttospeech/batchsyntheses/{job_id}?api-version={self.api_version}"
        print(f"Starting batch synthesis job...")
        
        response = requests.put(url, data=json.dumps(body), headers=headers)
        if not response.ok:
            print(f"Error: {response.status_code} - {response.text}")
            response.raise_for_status()
        
        print(f"Job created successfully")

        # Poll for the batch synthesis result
        status_url = f"{self.speech_endpoint}/texttospeech/batchsyntheses/{job_id}?api-version={self.api_version}"
        print(f"Waiting for synthesis to complete...")
        
        while True:
            response = requests.get(status_url, headers=headers)
            if not response.ok:
                response.raise_for_status()
            
            result = response.json()
            status = result.get("status")
            
            if status == "Succeeded":
                break
            elif status == "Failed":
                error_message = result.get("errors", ["Unknown error"])[0]
                raise Exception(f"Batch synthesis failed: {error_message}")
            
            time.sleep(10)

        # Download the synthesized audio
        outputs = result.get("outputs", {})
        if not outputs:
            raise Exception("No outputs found in the response")
        
        # Handle the ZIP file
        zip_url = outputs.get("result")
        if not zip_url:
            raise Exception("Result ZIP URL not found in the response")
        
        print(f"Downloading results...")
        zip_response = requests.get(zip_url)
        if not zip_response.ok:
            raise Exception(f"Failed to download ZIP: {zip_response.status_code}")
        
        self._process_zip_response(zip_response, text, mp3_output_filepath, srt_output_filepath)

        print("Speech synthesis complete!")
        print(f"Generated: {mp3_output_filepath} and {srt_output_filepath}")

    def _process_zip_response(self, zip_response, text, mp3_output_filepath, srt_output_filepath):
        """Process the ZIP file response from the TTS service."""
        # Create a temporary directory to extract the ZIP
        with tempfile.TemporaryDirectory() as temp_dir:
            zip_path = os.path.join(temp_dir, "results.zip")
            
            # Save the ZIP file
            with open(zip_path, "wb") as zip_file:
                zip_file.write(zip_response.content)
            
            # Extract the ZIP file
            try:
                with zipfile.ZipFile(zip_path, "r") as z:
                    # Find the audio file and word boundary file
                    audio_file_name = None
                    word_boundary_file_name = None
                    
                    for file_name in z.namelist():
                        if file_name.endswith(".mp3"):
                            audio_file_name = file_name
                        elif file_name.endswith(".word.json"):
                            word_boundary_file_name = file_name
                    
                    if not audio_file_name:
                        raise Exception("Audio file not found in the ZIP")
                    
                    # Extract the audio file and save it to the output path
                    with open(mp3_output_filepath, "wb") as f:
                        f.write(z.read(audio_file_name))
                    
                    # Process word boundaries if available
                    if word_boundary_file_name:
                        word_boundaries = json.loads(z.read(word_boundary_file_name).decode("utf-8"))
                        
                        # Process word boundaries to generate SRT file
                        timestamps = []
                        for item in word_boundaries:
                            if "Text" in item and "AudioOffset" in item:
                                word_info = {
                                    "text": item.get("Text", ""),
                                    "start_time": item.get("AudioOffset")  # Already in milliseconds
                                }
                                timestamps.append(word_info)
                        
                        self._save_srt_file(timestamps, srt_output_filepath)
                    else:
                        print("Warning: Word boundaries not available. Creating default subtitle.")
                        # Create a simple SRT with just the entire text
                        with open(srt_output_filepath, "w", encoding="utf-8") as srt_file:
                            srt_file.write("1\n00:00:00,000 --> 00:05:00,000\n" + text)
            except zipfile.BadZipFile as e:
                print(f"Error: The downloaded file is not a valid ZIP file: {e}")
                # Try to save the raw content as MP3 directly
                with open(mp3_output_filepath, "wb") as f:
                    f.write(zip_response.content)
                
                # Create a simple SRT with just the entire text
                with open(srt_output_filepath, "w", encoding="utf-8") as srt_file:
                    srt_file.write("1\n00:00:00,000 --> 00:05:00,000\n" + text)

    def _build_subtitle(self, timestamps, index):
        """
        Takes a list of timestamps and returns a single srt.Subtitle object 
        that groups all the timestamps in the list.
        """
        subtitle_text = "".join([ts["text"] for ts in timestamps])
        start_time = timedelta(milliseconds=timestamps[0]["start_time"]) if timestamps else timedelta(0)
        end_time = timedelta(milliseconds=timestamps[-1]["start_time"] + 500) if timestamps else timedelta(500)
        return srt.Subtitle(
            index=index,
            start=start_time,
            end=end_time,
            content=subtitle_text.strip()
        )

    def _save_srt_file(self, timestamps, filename):
        """
        Generate an SRT subtitle file from word timestamps.
        
        This groups words into subtitle entries based on punctuation.
        """
        chinese_punctuation = "，。！？、"""

        subtitles = []
        curr_group = []

        subtitle_index = 1

        i = 0
        while i < len(timestamps):
            if timestamps[i]["text"] in chinese_punctuation:
                while (i+1 < len(timestamps)) and (timestamps[i+1]["text"] in chinese_punctuation):
                    curr_group.append(timestamps[i])
                    i += 1
                    if i == len(timestamps):
                        break
                curr_group.append(timestamps[i])
                subtitles.append(self._build_subtitle(curr_group, subtitle_index))
                subtitle_index += 1
                curr_group = []
            else:
                curr_group.append(timestamps[i])
            i += 1

        # adjust improperly placed closing dialogue characters
        i = 1
        while i < len(subtitles):
            if subtitles[i].content[0] == '"':
                subtitles[i-1].content += '"'
                subtitles[i].content = subtitles[i].content[1:]
            i += 1

        # strip subtitles
        for subtitle in subtitles:
            subtitle.content = subtitle.content.lstrip(" \t\n")

        # fix overlapping subtitles
        i = 0
        while i < len(subtitles) - 1:
            if subtitles[i].end > subtitles[i+1].start:
                subtitles[i].end = subtitles[i+1].start
            i += 1

        # write the subtitle content
        srt_content = srt.compose(subtitles)
        with open(filename, "w", encoding="utf-8") as srt_file:
            srt_file.write(srt_content)

class VideoGenerator:
    def generate_video(image_filepath, mp3_filepath, mp4_output_filepath):
        audio = AudioFileClip(mp3_filepath)
        clip = ImageClip(image_filepath).with_duration(audio.duration)
        clip = clip.with_audio(audio)
        clip.write_videofile(mp4_output_filepath, fps=24)