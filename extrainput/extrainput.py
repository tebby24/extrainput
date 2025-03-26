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
import numpy as np
import fasttext
import fasttext.util
from sklearn.cluster import KMeans
from sklearn.metrics.pairwise import cosine_similarity

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
    
    # Define supported OpenRouter models
    openrouter_models = [
        "deepseek/deepseek-r1:free",
        "deepseek/deepseek-chat-v3-0324:free",
        "qwen/qwen2.5-vl-32b-instruct:free",
        "eva-unit-01/eva-qwen-2.5-72b",
        "qwen/qwq-32b:free"
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

    def generate_text(self, prompt, model="deepseek/deepseek-r1:free"):
        """Generate text based on a prompt using the specified model.
        
        Args:
            prompt (str): The prompt to generate text from
            model (str, optional): Model to use. Options: 
                - "deepseek/deepseek-r1:free" (default)
                - "deepseek/deepseek-chat-v3-0324:free"
                - "qwen/qwen2.5-vl-32b-instruct:free"
                - "eva-unit-01/eva-qwen-2.5-72b"
                
        Returns:
            str: Generated text response
            
        Raises:
            ValueError: If an unsupported model is specified
        """
        if model in self.openrouter_models:
            response = self.openai_client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": "You are a helpful assistant"},
                    {"role": "user", "content": prompt},
                ],
                stream=False
            )
            return response.choices[0].message.content
        else:
            supported_models = ", ".join(self.openrouter_models)
            raise ValueError(f"Unsupported model: {model}. Supported models: {supported_models}")

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
        chinese_punctuation = "，。！？、：—"  # Added the em dash "—"
    
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

    def create_word_groups(self, words, group_min_size, group_max_size):
        """
        Groups a list of words into smaller lists based on semantic similarity using fastText embeddings.

        Args:
            words (list): The list of words to be grouped.
            group_min_size (int): The minimum number of words in each group.
            group_max_size (int): The maximum number of words in each group.

        Returns:
            list: A list of lists, where each sublist contains semantically related words.
        """
        # Calculate the target group size (average of min and max)
        target_group_size = min(max(len(words) // 5, group_min_size), group_max_size)
        
        try:
            # Create models directory if it doesn't exist
            models_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "models")
            os.makedirs(models_dir, exist_ok=True)
            
            # Check if fastText model exists in models directory, if not download it
            model_filename = "cc.zh.300.bin"
            model_path = os.path.join(models_dir, model_filename)
            
            if not os.path.exists(model_path):
                print("Downloading Chinese fastText model (this may take some time)...")
                # Download to current directory first
                downloaded_file = fasttext.util.download_model('zh', if_exists='ignore')
                
                if downloaded_file:
                    # Move the file to our models directory
                    if os.path.exists(downloaded_file):
                        os.rename(downloaded_file, model_path)
                        print(f"Moved model to {model_path}")
                    # Check if we need to extract .gz file
                    elif os.path.exists(f"{downloaded_file}.gz"):
                        fasttext.util.gunzip(f"{downloaded_file}.gz")
                        if os.path.exists(downloaded_file):
                            os.rename(downloaded_file, model_path)
                            print(f"Extracted and moved model to {model_path}")
                
                if not os.path.exists(model_path):
                    raise FileNotFoundError(f"Failed to download and setup fastText model at {model_path}")
            
            # Load the model
            print("Loading fastText model...")
            model = fasttext.load_model(model_path)
            
            # Get word vectors
            print(f"Getting word vectors for {len(words)} words...")
            vectors = []
            valid_words = []
            for word in words:
                try:
                    vec = model.get_word_vector(word)
                    vectors.append(vec)
                    valid_words.append(word)
                except Exception as e:
                    print(f"Error getting vector for word '{word}': {str(e)}")
            
            if not valid_words:
                print("No valid word vectors found, falling back to AI-based grouping")
                return self._fallback_create_word_groups(words, group_min_size, group_max_size)
            
            print(f"Generated {len(vectors)} word vectors successfully")
            vectors = np.array(vectors)
            
            # Compute the number of groups needed
            num_groups = max(1, len(valid_words) // target_group_size)
            
            # Cluster words into semantically similar groups
            if len(valid_words) <= target_group_size:
                # If we have fewer words than the target size, return them as a single group
                return [valid_words]
            
            # Calculate pairwise cosine similarities
            similarity_matrix = cosine_similarity(vectors)
            
            # Group words using the greedy approach for equal-sized groups
            unassigned_indices = set(range(len(valid_words)))
            groups = []
            
            while len(unassigned_indices) >= group_min_size:
                # Find the word with the highest average similarity to all unassigned words
                best_start_idx = None
                best_avg_similarity = -float('inf')
                
                for idx in unassigned_indices:
                    unassigned_similarity = [similarity_matrix[idx][j] for j in unassigned_indices if j != idx]
                    avg_similarity = np.mean(unassigned_similarity) if unassigned_similarity else 0
                    if avg_similarity > best_avg_similarity:
                        best_avg_similarity = avg_similarity
                        best_start_idx = idx
                
                # Start a new group with this word
                current_group_indices = [best_start_idx]
                unassigned_indices.remove(best_start_idx)
                
                # Add words one by one until we reach the target group size
                current_size = min(target_group_size, len(unassigned_indices) + 1)
                while len(current_group_indices) < current_size and unassigned_indices:
                    # Calculate average similarity of each unassigned word to current group
                    best_idx = None
                    best_avg_similarity = -float('inf')
                    
                    for idx in unassigned_indices:
                        avg_similarity = np.mean([similarity_matrix[idx][group_idx] for group_idx in current_group_indices])
                        if avg_similarity > best_avg_similarity:
                            best_avg_similarity = avg_similarity
                            best_idx = idx
                    
                    # Add the best word to the group
                    current_group_indices.append(best_idx)
                    unassigned_indices.remove(best_idx)
                
                # Add the current group to our results
                current_group = [valid_words[idx] for idx in current_group_indices]
                groups.append(current_group)
            
            # If there are any remaining words, add them as the final group
            # or distribute them among existing groups
            if unassigned_indices:
                if len(unassigned_indices) >= group_min_size:
                    final_group = [valid_words[idx] for idx in unassigned_indices]
                    groups.append(final_group)
                else:
                    # Distribute remaining words to their most similar groups
                    for idx in unassigned_indices:
                        best_group_idx = -1
                        best_similarity = -float('inf')
                        
                        for group_idx, group in enumerate(groups):
                            group_word_indices = [valid_words.index(word) for word in group]
                            avg_similarity = np.mean([similarity_matrix[idx][group_word_idx] for group_word_idx in group_word_indices])
                            
                            if avg_similarity > best_similarity:
                                best_similarity = avg_similarity
                                best_group_idx = group_idx
                        
                        if best_group_idx >= 0:
                            groups[best_group_idx].append(valid_words[idx])
            
            print(f"Created {len(groups)} word groups using semantic clustering")
            return groups
            
        except Exception as e:
            print(f"Error in fastText grouping: {str(e)}")
            print("Falling back to AI-based grouping method")
            return self._fallback_create_word_groups(words, group_min_size, group_max_size)
    
    def _fallback_create_word_groups(self, words, group_min_size, group_max_size):
        """Fallback method that uses AI-based grouping when fastText clustering fails."""
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
        response = self.generate_text(prompt, model="deepseek/deepseek-r1:free")
        
        # Clean response and parse initial groups
        initial_groups = self._clean_and_parse_json_response(response)
        if not initial_groups:
            # If we failed to get valid groups, return a fallback with all words in one group
            print("Failed to parse initial groups, falling back to simple grouping")
            return [words]
            
        # Remove duplicates across groups (keep first occurrence)
        seen_words = set()
        cleaned_groups = []
        
        for group in initial_groups:
            cleaned_group = []
            for word in group:
                if word not in seen_words:
                    seen_words.add(word)
                    cleaned_group.append(word)
            if cleaned_group:  # Only add non-empty groups
                cleaned_groups.append(cleaned_group)
        
        # Find unused words
        unused_words = [word for word in words if word not in seen_words]
        
        # If there are unused words, ask the model to incorporate them
        if unused_words:
            print(f"Found {len(unused_words)} unused words. Asking model to incorporate them.")
            
            unused_formatted = json.dumps(unused_words, ensure_ascii=False)
            groups_formatted = json.dumps(cleaned_groups, ensure_ascii=False)
            
            second_prompt = f"""
            I have some word groups, but some words from my original list were not included. 
            Please add these unused words to the existing groups where they fit best semantically.
            
            IMPORTANT: Return ONLY a valid JSON array of arrays like this exact format:
            [["word1", "word2"], ["word3", "word4"]]
            
            Current word groups:
            {groups_formatted}
            
            Words to add:
            {unused_formatted}
            
            Rules:
            - Add each unused word to whichever existing group it fits best with semantically
            - Maintain the existing groupings as much as possible
            - Return ONLY the final complete groups as a JSON array of arrays
            - Do not add any explanation, comments, or code formatting - just the raw JSON
            """
            
            response = self.generate_text(second_prompt, model="gpt-4o")
            print(f"Second response: {response.strip()}")
            
            # Clean and parse the second response
            final_groups = self._clean_and_parse_json_response(response)
            
            # If we got a valid response, use it; otherwise fall back to the first result
            if final_groups:
                # Verify that all words are included
                all_grouped_words = set()
                for group in final_groups:
                    for word in group:
                        all_grouped_words.add(word)
                
                missing_words = [word for word in words if word not in all_grouped_words]
                if missing_words:
                    print(f"Warning: {len(missing_words)} words still not included after second attempt")
                
                # Ensure that all groups meet the min/max size requirements
                valid_groups = []
                for group in final_groups:
                    if len(group) >= group_min_size and len(group) <= group_max_size:
                        valid_groups.append(group)
                    else:
                        # Try to split or merge groups that don't meet requirements
                        if len(group) > group_max_size:
                            # Split into multiple groups
                            for i in range(0, len(group), group_max_size):
                                chunk = group[i:i + group_max_size]
                                if len(chunk) >= group_min_size:
                                    valid_groups.append(chunk)
                                elif valid_groups:  # Add to the last group if possible
                                    valid_groups[-1].extend(chunk)
                        elif len(group) < group_min_size and valid_groups:
                            # Try to merge with the last group if it won't exceed max
                            if len(valid_groups[-1]) + len(group) <= group_max_size:
                                valid_groups[-1].extend(group)
                            else:
                                valid_groups.append(group)  # Keep as is if can't merge
                
                return valid_groups
            else:
                print("Failed to parse the second response, falling back to first grouping")
        
        # Ensure that all groups meet the min/max size requirements
        valid_groups = []
        for group in cleaned_groups:
            if len(group) >= group_min_size and len(group) <= group_max_size:
                valid_groups.append(group)
            else:
                # Try to split or merge groups that don't meet requirements
                if len(group) > group_max_size:
                    # Split into multiple groups
                    for i in range(0, len(group), group_max_size):
                        chunk = group[i:i + group_max_size]
                        if len(chunk) >= group_min_size:
                            valid_groups.append(chunk)
                        elif valid_groups:  # Add to the last group if possible
                            valid_groups[-1].extend(chunk)
                elif len(group) < group_min_size and valid_groups:
                    # Try to merge with the last group if it won't exceed max
                    if len(valid_groups[-1]) + len(group) <= group_max_size:
                        valid_groups[-1].extend(group)
                    else:
                        valid_groups.append(group)  # Keep as is if can't merge
        
        return valid_groups if valid_groups else [words]  # Fallback if no valid groups

    def _clean_and_parse_json_response(self, response):
        """Helper method to clean and parse JSON responses from the model."""
        try:
            # Start by trying to parse the raw response
            try:
                return json.loads(response.strip())
            except json.JSONDecodeError:
                pass
                
            # Clean up and try different formats
            cleaned_response = response.strip()
            
            # Try to extract just the JSON array part by finding the first [ and last ]
            start_idx = cleaned_response.find('[')
            end_idx = cleaned_response.rfind(']')
            
            if (start_idx != -1) and (end_idx != -1) and (start_idx < end_idx):
                json_part = cleaned_response[start_idx:end_idx+1]
                try:
                    return json.loads(json_part)
                except json.JSONDecodeError:
                    pass
            
            # Remove code block markers if present
            if cleaned_response.startswith("```") and "```" in cleaned_response[3:]:
                cleaned_response = cleaned_response[3:]
                end_marker = cleaned_response.rfind("```")
                if end_marker != -1:
                    cleaned_response = cleaned_response[:end_marker].strip()
            
            # Remove json language specifier if present
            if cleaned_response.startswith("```json"):
                cleaned_response = cleaned_response[7:].strip()
                if "```" in cleaned_response:
                    end_marker = cleaned_response.rfind("```")
                    if end_marker != -1:
                        cleaned_response = cleaned_response[:end_marker].strip()
                        
            # Try parsing again
            try:
                result = json.loads(cleaned_response)
                
                # Verify it's a list of lists
                if isinstance(result, list) and all(isinstance(item, list) for item in result):
                    return result
                else:
                    print(f"Parsed JSON is not a list of lists: {type(result).__name__}")
                    return []
            except json.JSONDecodeError as e:
                print(f"JSON parsing error: {e}")
                print(f"Failed to parse: {cleaned_response}")
                return []
        except Exception as e:
            print(f"Unexpected error in JSON parsing: {str(e)}")
            return []
    
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

        response = self.generate_text(prompt, model="deepseek/deepseek-r1:free")
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

