from openai import OpenAI
from azure.cognitiveservices.speech import SpeechConfig, SpeechSynthesizer, AudioConfig, ResultReason, CancellationReason
import json
import os
import requests
import uuid
import zipfile
import tempfile
import time
from datetime import timedelta
import srt
from moviepy import AudioFileClip, ImageClip
from typing import List

class ExtraInputGenerator:
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

    def __init__(self, openrouter_api_key, azure_speech_key, azure_speech_region, stabilityai_api_key):
        """Initialize the ExtraInputGenerator with required API keys.
        
        Args:
            openrouter_api_key (str): API key for OpenRouter services
            azure_speech_key (str): API key for Azure Speech services
            azure_speech_region (str): Region for Azure Speech services
            stabilityai_api_key (str): API key for StabilityAI image generation
        """
        if not openrouter_api_key:
            raise ValueError("OpenRouter API key is required")
        
        self.openai_client = OpenAI(
            base_url="https://openrouter.ai/api/v1",
            api_key=openrouter_api_key
        )
        self.speech_key = azure_speech_key
        self.speech_region = azure_speech_region
        self.speech_config = SpeechConfig(subscription=azure_speech_key, region=azure_speech_region)
        self.speech_endpoint = f"https://{self.speech_region}.api.cognitive.microsoft.com"
        self.api_version = "2024-04-01"
        self.stabilityai_api_key = stabilityai_api_key

    def generate_text(self, prompt, model="openai/gpt-4.1-nano", temperature=1):
        """Generate text based on a prompt using the specified model.
        
        Args:
            prompt (str): The prompt to generate text from
            model (str, optional): Any OpenRouter model name. Defaults to "openai/gpt-4.1-nano"
            temperature (float, optional): Controls randomness. Higher values (e.g., 1.0) make output more random, 
                                         lower values (e.g., 0.1) make it more deterministic. Defaults to 1.
                
        Returns:
            str: Generated text response
        """
        response = self.openai_client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": "You are a helpful assistant"},
                {"role": "user", "content": prompt},
            ],
            temperature=temperature,
            stream=False
        )
        return response.choices[0].message.content

    def generate_image(self, prompt, output_filepath, aspect_ratio="16:9", output_format="png"):
        """Generate an image based on a prompt using stability AI.

        Args:
            content (str): The content or description for the image generation
            output_filepath (str): The file path where the generated image will be saved
            aspect_ratio (str, optional): The aspect ratio of the image. Defaults to "1:1"
            output_format (str, optional): The format of the output image. Defaults to "png"

        Returns:
            str: The file path to the saved image or None if generation failed
        """
        try:
            response = requests.post(
                "https://api.stability.ai/v2beta/stable-image/generate/ultra",
                headers={
                    "authorization": f"Bearer {self.stabilityai_api_key}",
                    "accept": "image/*"
                },
                files={"none": ''},
                data={
                    "prompt": prompt,
                    "aspect_ratio": aspect_ratio,
                    "output_format": output_format,
                },
            )

            if response.status_code == 200:
                with open(output_filepath, 'wb') as file:
                    file.write(response.content)
                return output_filepath
            else:
                error_info = str(response.json() if response.headers.get('content-type') == 'application/json' else response.text)
                print(f"Error generating image: Status code {response.status_code} - {error_info}")
                return None
                
        except Exception as e:
            print(f"Error generating image: {str(e)}")
            return None

    def synthesize_speech(self, text, voice, output_filepath):
        """Generate speech from text using Azure Speech Services.
        
        Args:
            text (str): The text to synthesize to speech
            voice (str): The voice name to use
            output_filepath (str): Path to save the audio output
            
        Returns:
            str: The path to the saved audio file if successful, None otherwise
        """
        self.speech_config.speech_synthesis_voice_name = voice
        audio_config = AudioConfig(filename=output_filepath)
        synthesizer = SpeechSynthesizer(speech_config=self.speech_config, audio_config=audio_config)
        result = synthesizer.speak_text_async(text).get()

        if result.reason == ResultReason.Canceled:
            cancellation_details = result.cancellation_details
            print(f"Speech synthesis canceled: {cancellation_details.reason}")
            if cancellation_details.reason == CancellationReason.Error:
                print(f"Error details: {cancellation_details.error_details}")
            return None
        elif result.reason == ResultReason.SynthesizingAudioCompleted:
            with open(output_filepath, "wb") as audio_file:
                audio_file.write(result.audio_data)
            print(f"Speech synthesized for text and saved to [{output_filepath}]")
            return output_filepath
        return None

    def synthesize_speech_with_srt(self, text, voice, mp3_output_filepath, srt_output_filepath):
        """Generate speech from text and create a synchronized subtitle file.
        
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
        
        This groups words into subtitle entries based on punctuation,
        keeping punctuation at the end of subtitles.
        """
        chinese_punctuation = "，。！？、：—；"  # Added the em dash "—"
    
        subtitles = []
        curr_group = []
    
        subtitle_index = 1
    
        i = 0
        while i < len(timestamps):
            # Add the current word to the group
            curr_group.append(timestamps[i])
            
            # Check if current word is a punctuation that should trigger a split
            if any(char in chinese_punctuation for char in timestamps[i]["text"]):
                # Skip any consecutive punctuation marks (they should all stay in the same subtitle)
                while (i+1 < len(timestamps)) and (timestamps[i+1]["text"] in chinese_punctuation):
                    i += 1
                    curr_group.append(timestamps[i])
                    
                # Create subtitle from current group and reset
                subtitles.append(self._build_subtitle(curr_group, subtitle_index))
                subtitle_index += 1
                curr_group = []
            
            i += 1
        
        # Add any remaining words as the final subtitle
        if curr_group:
            subtitles.append(self._build_subtitle(curr_group, subtitle_index))
        
        # Handle the special case with closing dialogue characters (Chinese style subtitle)
        i = 1
        while i < len(subtitles):
            if subtitles[i].content and subtitles[i].content[0] == '”':
                # Move the closing quote to the end of the previous subtitle
                subtitles[i-1].content += subtitles[i].content[0]
                subtitles[i].content = subtitles[i].content[1:]
            i += 1
        


        # Handle the special case with closing dialogue characters (English style subtitle)
        i = 1
        needs_cosed = False
        while i < len(subtitles):
            if subtitles[i].content and subtitles[i].content[0] == '"':
                if not needs_cosed:
                    needs_cosed = True
                else:
                    # Move the closing quote to the end of the previous subtitle
                    subtitles[i-1].content += subtitles[i].content[0]
                    subtitles[i].content = subtitles[i].content[1:]
                    needs_cosed = False
            i += 1
        
        # Strip whitespace from subtitles
        for subtitle in subtitles:
            subtitle.content = subtitle.content.lstrip(" \t\n")
    
        # Fix overlapping subtitles
        i = 0
        while i < len(subtitles) - 1:
            if subtitles[i].end > subtitles[i+1].start:
                subtitles[i].end = subtitles[i+1].start
            i += 1
    
        # Write the subtitle content
        srt_content = srt.compose(subtitles)
        with open(filename, "w", encoding="utf-8") as srt_file:
            srt_file.write(srt_content)
    
    def pair_voice_to_article(self, content, voices=None):
        """
        Determines the most suitable voice from the available voices to narrate given content.
        
        Args:
            content (str): The Chinese text content that needs narration
            voices (list, optional): A list of dictionaries containing voice information.
                                     If None, uses the default voices list.
        
        Returns:
            str: The name of the selected voice that best matches the content
        """
        if voices is None:
            voices = self.voices

        voices_as_json = json.dumps(voices, ensure_ascii=False)
        prompt = f"""
        I have a piece of Chinese writing and I plan to use text to speech to synthasize it into speech. Given a set of voices, descriptions of the voices, and my content, I want you to determine the most suitable voice to serve as the narrator for my writing. 
        For example, if it sounds like the narrirator of the content is a little girl, respond with the name of the voice who's description describes a little girl. If the content seems like a news broadcast, maybe choose the voice who's description matches an older man. 
        Here are the voice names and their descriptions:
        ```json
        {voices_as_json}
        ```

        Here is my Chinese writing:
        ```text
        {content}
        ```
        Which voice would be best to narrate my content? 
        Response requirements:
        - write only the name of the voice and nothing else
        - do NOT provide any formatting such as quotes or brackets
        - the voice must match one of the names of the voices provided
        """

        response = self.generate_text(prompt, model="openai/gpt-4.1-nano")
        return response.strip()

    def generate_video(self, image_filepath, mp3_filepath, mp4_output_filepath):
        """
        Generates a video by combining an image and an audio file.

        Args:
            image_filepath (str): The file path to the image to be used in the video.
            mp3_filepath (str): The file path to the MP3 audio file to be used in the video.
            mp4_output_filepath (str): The file path where the generated MP4 video will be saved.

        Returns:
            str: The path to the generated video file
        """
        audio = AudioFileClip(mp3_filepath)
        clip = ImageClip(image_filepath).with_duration(audio.duration)
        clip = clip.with_audio(audio)
        clip.write_videofile(mp4_output_filepath, fps=24)
        return mp4_output_filepath

