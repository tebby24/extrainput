import random
from pydub import AudioSegment
from openai import OpenAI
from azure.cognitiveservices.speech import SpeechConfig, SpeechSynthesizer, AudioConfig, ResultReason, CancellationReason
import concurrent.futures
from typing import Dict, List, Tuple


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


class TTSEngine:
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


class ExtraInputGenerator:
    def __init__(self, text_generator: TextGenerator, tts_engine: TTSEngine):
        self.text_generator = text_generator
        self.tts_engine = tts_engine
    
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
            voice = random.choice(TTSEngine.voices)["name"]
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