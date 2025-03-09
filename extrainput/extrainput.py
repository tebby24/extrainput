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
    def __init__(self, deepseek_api_key=None, openai_api_key=None):
        """Initialize text generator with API keys for different providers.
        
        Args:
            deepseek_api_key (str, optional): API key for Deepseek models
            openai_api_key (str, optional): API key for OpenAI models
        """
        self.deepseek_client = None
        self.openai_client = None
        
        if deepseek_api_key:
            self.deepseek_client = OpenAI(api_key=deepseek_api_key, base_url="https://api.deepseek.com")
        
        if openai_api_key:
            self.openai_client = OpenAI(api_key=openai_api_key)

    def get_models(self):
        """
        Retrieves a tuple of available model names.

        Returns:
            tuple: A tuple containing the names of the available models.
        """
        return ("deepseek-chat", "gpt-4o-mini", "gpt-4o")

    def generate_text(self, prompt, model="gpt-4o-mini", json_mode=False):
        """Generate text based on a prompt using the specified model.
        
        Args:
            prompt (str): The prompt to generate text from
            model (str, optional): Model to use. Options: 
                - "gpt-4o-mini" (default)
                - "gpt-4o"
                - "deepseek-chat" 
            json_mode (bool, optional): If True, instructs the model to return JSON. 
                Works with OpenAI models only, ignored for deepseek-chat.
                
        Returns:
            str: Generated text response
            
        Raises:
            ValueError: If the required API client for the selected model is not initialized
            ValueError: If an unsupported model is specified
            ValueError: If json_mode is True but model doesn't support it
        """
        if model == "deepseek-chat":
            if not self.deepseek_client:
                raise ValueError("Deepseek API key is required to use the deepseek-chat model")
                
            if json_mode:
                # Deepseek doesn't support json_mode parameter directly
                # Add JSON instructions to the prompt
                system_content = "You are a helpful assistant. Always respond with valid JSON."
                user_content = f"Return your response as valid JSON. {prompt}"
            else:
                system_content = "You are a helpful assistant"
                user_content = prompt
                
            response = self.deepseek_client.chat.completions.create(
                model="deepseek-chat",
                messages=[
                    {"role": "system", "content": system_content},
                    {"role": "user", "content": user_content},
                ],
                stream=False
            )
            return response.choices[0].message.content
            
        elif model in ["gpt-4o-mini", "gpt-4o"]:
            if not self.openai_client:
                raise ValueError(f"OpenAI API key is required to use the {model} model")
                
            response = self.openai_client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": "You are a helpful assistant"},
                    {"role": "user", "content": prompt},
                ],
                response_format={"type": "json_object"} if json_mode else None,
                stream=False
            )
            return response.choices[0].message.content
            
        else:
            raise ValueError(f"Unsupported model: {model}. Supported models: deepseek-chat, gpt-4o-mini, gpt-4o")

class ImageGenerator:
    def __init__(self, openai_api_key):
        """
        Initialize the ImageGenerator with the OpenAI API key.

        Args:
            openai_api_key (str): The OpenAI API key for accessing DALL-E and GPT models.
        """
        self.client = OpenAI(api_key=openai_api_key)
        self.text_generator = TextGenerator(openai_api_key=openai_api_key)
    
    def generate_image(self, prompt, output_filepath, size="1792x1024", quality="standard", model="dall-e-3"):
        """
        Generate an image using DALL-E 3 based on the provided prompt.

        Args:
            prompt (str): The description of the image to generate.
            output_filepath (str): The file path where the generated image will be saved.
            size (str, optional): The size of the image. Defaults to "1792x1024".
            quality (str, optional): The quality of the image. Defaults to "standard".
            model (str, optional): The DALL-E model version to use. Defaults to "dall-e-3".

        Returns:
            str: The file path to the saved image.
        """
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
            
            image_data = requests.get(image_url).content
            with open(output_filepath, 'wb') as image_file:
                image_file.write(image_data)
            return output_filepath
            
        except Exception as e:
            print(f"Error generating image: {str(e)}")
            return None

    def generate_image_representing_content(self, content, output_filepath, size="1792x1024", quality="standard", model="dall-e-3"):
        """
        Generate an image representing the given content using DALL-E 3.

        Args:
            content (str): The content or story to be represented in the image.
            output_filepath (str): The file path where the generated image will be saved.
            size (str, optional): The size of the image. Defaults to "1792x1024".
            quality (str, optional): The quality of the image. Defaults to "standard".
            model (str, optional): The DALL-E model version to use. Defaults to "dall-e-3".

        Returns:
            str: The file path to the saved image.
        """
        image_meta_prompt = "请编写 Dall-E 3 提示语，用于生成一个图像来表现下面的故事。只提供提示，不提供其他文字。:\n\n" + content
        image_prompt = self.text_generator.generate_text(image_meta_prompt, model="gpt-4o-mini")
        return self.generate_image(image_prompt, output_filepath, size=size, quality=quality, model=model)


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
        
        This groups words into subtitle entries based on punctuation,
        keeping punctuation at the end of subtitles.
        """
        chinese_punctuation = "，。！？、：—"  # Added the em dash "—"
    
        subtitles = []
        curr_group = []
    
        subtitle_index = 1
    
        i = 0
        while i < len(timestamps):
            # Add the current word to the group
            curr_group.append(timestamps[i])
            
            # Check if current word is a punctuation that should trigger a split
            if timestamps[i]["text"] in chinese_punctuation:
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
        
        # Handle the special case with closing dialogue characters (both English and Chinese)
        i = 1
        while i < len(subtitles):
            if subtitles[i].content and (subtitles[i].content[0] == '"' or subtitles[i].content[0] == '”'):
                # Move the closing quote to the end of the previous subtitle
                subtitles[i-1].content += subtitles[i].content[0]
                subtitles[i].content = subtitles[i].content[1:]
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

class VideoGenerator:
    def generate_video(self, image_filepath, mp3_filepath, mp4_output_filepath):
        """
        Generates a video by combining an image and an audio file.

        Args:
            image_filepath (str): The file path to the image to be used in the video.
            mp3_filepath (str): The file path to the MP3 audio file to be used in the video.
            mp4_output_filepath (str): The file path where the generated MP4 video will be saved.

        Returns:
            None
        """
        audio = AudioFileClip(mp3_filepath)
        clip = ImageClip(image_filepath).with_duration(audio.duration)
        clip = clip.with_audio(audio)
        clip.write_videofile(mp4_output_filepath, fps=24)


class ExampleContentGenerator:
    def __init__(self, openai_api_key):
        self.text_generator = TextGenerator(openai_api_key=openai_api_key)
    
    def create_word_groups(self, words, group_min_size, group_max_size):
        """
        Groups a list of words into smaller lists based on specified size constraints.

        Args:
            words (list): The list of words to be grouped.
            group_min_size (int): The minimum number of words in each group.
            group_max_size (int): The maximum number of words in each group.

        Returns:
            list: A list of lists, where each sublist contains grouped words.
        """
        words_formatted_as_list = json.dumps(words, ensure_ascii=False)
        print(f"Input words: {words_formatted_as_list}")

        prompt = f"""
You will be given a list of words. Your task is to group these words into lists, ensuring that:
- Each group contains between {group_min_size} and {group_max_size} words.
- Words in each group should be related and likely to appear together in the same paragraph.
- Return ONLY a valid JSON list of lists, with no explanations or markdown formatting.
- Do not include backticks, json tags, or any other text besides the actual JSON array.

Example input:
["苹果", "香蕉", "樱桃", "狗", "猫", "大象", "吉他", "钢琴", "小提琴", "河流", "海洋", "湖泊"]

Example output format (but adjusted to your group size requirements):
[
    ["苹果", "香蕉", "樱桃", "梨"],
    ["狗", "猫", "大象", "老虎"],
    ["吉他", "钢琴", "小提琴", "萨克斯", "架子鼓"],
    ["河流", "海洋", "湖泊", "池塘"]
]

Now, here is the actual list of words you should process:
{words_formatted_as_list}
"""

        # Get response with json_mode=False
        response = self.text_generator.generate_text(prompt, model="gpt-4o", json_mode=False)
        
        # Clean up the response to extract just the JSON part
        cleaned_response = response.strip()
        
        # Remove code block markers if present
        if cleaned_response.startswith("```") and cleaned_response.endswith("```"):
            cleaned_response = cleaned_response[3:-3].strip()
        
        # Remove json language specifier if present
        if cleaned_response.startswith("```json"):
            cleaned_response = cleaned_response[7:].strip()
            if cleaned_response.endswith("```"):
                cleaned_response = cleaned_response[:-3].strip()
        
        # Try to parse the JSON
        try:
            result = json.loads(cleaned_response)
            
            # Validate the structure - we expect a list of lists
            if not isinstance(result, list):
                print(f"Error: Expected a list but got {type(result).__name__}")
                return []
                
            for group in result:
                if not isinstance(group, list):
                    print(f"Error: Expected each group to be a list but got {type(group).__name__}")
                    return []
                    
            return result
        except json.JSONDecodeError as e:
            print(f"JSON parsing error: {e}")
            print(f"Failed to parse: {cleaned_response}")
            # Return empty list as fallback
            return []

    def generate_example_content(self, words):
        """
        Generates a Chinese article based on a given list of words.
        This method takes a list of words and creates a prompt to generate a Chinese article
        that uses each word at least once. The article can be in various forms such as a story,
        essay, news report, poem, dialogue, or argumentative essay. The generated text should
        be fluent, natural, and conform to Chinese expression habits.
        Args:
            words (list): A list of words to be included in the generated article.
        Returns:
            str: The generated Chinese article as a string.
        """
        words_formatted_as_list = json.dumps(words, ensure_ascii=False)
         
        prompt = f"""
请根据以下给定的词语创作一篇中文文章，要求如下：
- **必须使用每个词语至少一次**，但可以自然地分布在文章中。
- **自由选择文体**（可以是故事、散文、新闻报道、诗歌、对话、议论文等）。请确保文章有清晰的结构。
- **语言流畅自然，符合汉语表达习惯**，避免生硬地逐字使用词语，而是让它们有机地融入文章。
- **适当扩展内容**，使文章完整、富有表现力，不要仅仅是简单的词组拼接。

### **词语列表：**
{words_formatted_as_list}

请用**正式、自然、可读性强的中文**写作，不要额外解释，只输出完整的创作文本。
"""

        response = self.text_generator.generate_text(prompt, model="gpt-4o", json_mode=False)
        return response

    def pair_voice_to_content(self, content, voices):
        """
        Determines the most suitable voice from a list of available voices to narrate given content.
        
        Uses AI to analyze the content's tone, style, and context to select an appropriate voice
        that would be the most natural fit for narrating the text.
        
        Args:
            content (str): The Chinese text content that needs narration
            voices (list): A list of dictionaries containing voice information with format:
                           [{"name": "voicename", "description": "voice_description"}, ...]
        
        Returns:
            str: The name of the selected voice that best matches the content
        """

        voices_as_json = json.dumps(voices, ensure_ascii=False)
        prompt = f"""
I have a piece of Chinese writing and I plan to use text to speech to synthasize it into speech. Given a set of voices, descriptions of the voices, and my content, I want you to determine the most suitable voice to serve as the narrator for my writing. 
For example, if it sounds like the narrirator of the content is a little girl, respond with the name of the voice who's description describes a little girl. If the content seems like a news broadcast, maybe choose the voice who's description matches an older man. 
Here are the voice names and their descriptions:
```json
{voices}
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

        response = self.text_generator.generate_text(prompt, model="gpt-4o-mini", json_mode=False)
        return response