import os
import unittest
from dotenv import load_dotenv
from extrainput import (
    TextGenerator, 
    SimpleTTSGenerator, 
    TTSWithSubsGenerator, 
    ExampleSentenceListenerGenerator, 
    VideoGenerator,
    ImageGenerator
)

# Load environment variables from .env file
load_dotenv()

class TestExtraInput(unittest.TestCase):
    def setUp(self):
        # Get API keys from environment variables
        self.deepseek_api_key = os.getenv("DEEPSEEK_API_KEY")
        self.azure_speech_key = os.getenv("AZURE_SPEECH_KEY")
        self.azure_speech_region = os.getenv("AZURE_SPEECH_REGION")
        self.openai_api_key = os.getenv("OPENAI_API_KEY")
        
        # Skip tests if credentials are not available
        if not self.deepseek_api_key:
            print("Warning: DEEPSEEK_API_KEY not found, some tests will be skipped")
        if not all([self.azure_speech_key, self.azure_speech_region]):
            print("Warning: Azure Speech credentials not found, some tests will be skipped")
        if not self.openai_api_key:
            print("Warning: OPENAI_API_KEY not found, some tests will be skipped")
        
        # Ensure output directories exist
        os.makedirs("bin/test/output", exist_ok=True)
        
        # Paths to test input files
        self.test_text_path = "bin/test/input/test_text.txt"
        self.test_audio_path = "bin/test/input/test_audio.wav"
        self.test_image_path = "bin/test/input/test_image.png"
        
        # Check if test files exist
        if not os.path.exists(self.test_text_path):
            print(f"Warning: Test text file not found at {self.test_text_path}")
        if not os.path.exists(self.test_audio_path):
            print(f"Warning: Test audio file not found at {self.test_audio_path}")
        if not os.path.exists(self.test_image_path):
            print(f"Warning: Test image file not found at {self.test_image_path}")
    
    def test_text_generator(self):
        """Test text generation functionality"""
        if not self.deepseek_api_key:
            self.skipTest("Deepseek API key not available")
            
        text_gen = TextGenerator(self.deepseek_api_key)
        result = text_gen.generate_text("请用'好'字造一个简单的句子")
        
        # Check that we got a non-empty result
        self.assertTrue(result and isinstance(result, str))
        print(f"Generated text: {result}")
        
        # Test with longer prompt from file
        with open(self.test_text_path, "r", encoding="utf-8") as f:
            sample_text = f.read()
        
        # Use first paragraph as context for generation
        paragraphs = sample_text.split("\n\n")
        context = paragraphs[1] if len(paragraphs) > 1 else paragraphs[0]
        prompt = f"请基于以下文本写一个短评论:\n{context}\n"
        
        result = text_gen.generate_text(prompt)
        self.assertTrue(result and isinstance(result, str))
        print(f"Generated text based on input file: {result}")
    
    def test_simple_tts(self):
        """Test simple TTS generation"""
        if not all([self.azure_speech_key, self.azure_speech_region]):
            self.skipTest("Azure Speech credentials not available")
            
        tts = SimpleTTSGenerator(self.azure_speech_key, self.azure_speech_region)
        output_path = "bin/test/output/simple_tts_test.wav"
        
        # Test with simple text
        result_path = tts.synthesize_speech("你好世界", "zh-CN-XiaoxiaoNeural", output_path)
        
        # Check that file was created
        self.assertTrue(os.path.exists(result_path))
        self.assertEqual(result_path, output_path)
        
        # Test with text from file
        with open(self.test_text_path, "r", encoding="utf-8") as f:
            sample_text = f.read().split("\n\n")[0]  # Use first paragraph
            
        output_path_2 = "bin/test/output/simple_tts_file_test.wav"
        result_path = tts.synthesize_speech(sample_text, "zh-CN-XiaoxiaoNeural", output_path_2)
        
        # Check that file was created
        self.assertTrue(os.path.exists(result_path))
        self.assertEqual(result_path, output_path_2)
    
    def test_tts_with_subtitles(self):
        """Test TTS with subtitle generation"""
        if not all([self.azure_speech_key, self.azure_speech_region]):
            self.skipTest("Azure Speech credentials not available")
            
        tts_subs = TTSWithSubsGenerator(self.azure_speech_key, self.azure_speech_region)
        mp3_output = "bin/test/output/tts_with_subs_test.mp3"
        srt_output = "bin/test/output/tts_with_subs_test.srt"
        
        # Test with simple text
        test_text = "这是一个测试。我们正在生成语音和字幕。"
        tts_subs.synthesize_speech_with_srt(test_text, "zh-CN-XiaoxiaoNeural", mp3_output, srt_output)
        
        # Check that both files were created
        self.assertTrue(os.path.exists(mp3_output))
        self.assertTrue(os.path.exists(srt_output))
        
        # Test with text from file
        with open(self.test_text_path, "r", encoding="utf-8") as f:
            sample_text = f.read().split("\n\n")[0]  # Use first paragraph
            
        mp3_output_2 = "bin/test/output/tts_with_subs_file_test.mp3"
        srt_output_2 = "bin/test/output/tts_with_subs_file_test.srt"
        
        tts_subs.synthesize_speech_with_srt(sample_text, "zh-CN-XiaoxiaoNeural", mp3_output_2, srt_output_2)
        
        # Check that both files were created
        self.assertTrue(os.path.exists(mp3_output_2))
        self.assertTrue(os.path.exists(srt_output_2))
    
    def test_example_sentence_listener(self):
        """Test example sentence listener generation"""
        if not all([self.deepseek_api_key, self.azure_speech_key, self.azure_speech_region]):
            self.skipTest("API credentials not available")
            
        esl_gen = ExampleSentenceListenerGenerator(
            self.deepseek_api_key, 
            self.azure_speech_key, 
            self.azure_speech_region
        )
        
        output_path = "bin/test/output/example_sentences_test.wav"
        words = ["你好", "谢谢"]  # Use a small set for testing
        
        result_path = esl_gen.generate_example_sentence_listener(words, output_path)
        
        # Check that file was created
        self.assertTrue(os.path.exists(result_path))
        self.assertEqual(result_path, output_path)
        
        # Extract words from the test text file
        with open(self.test_text_path, "r", encoding="utf-8") as f:
            text = f.read()
            # Extract simple vocabulary words (between 1-2 characters)
            import re
            words = list(set(re.findall(r'[\u4e00-\u9fff]{1,2}', text)))[:5]  # Get first 5 unique words
        
        output_path_2 = "bin/test/output/example_sentences_file_test.wav"
        result_path = esl_gen.generate_example_sentence_listener(words, output_path_2)
        
        # Check that file was created
        self.assertTrue(os.path.exists(result_path))
        self.assertEqual(result_path, output_path_2)
    
    def test_image_generator(self):
        """Test image generation"""
        if not self.openai_api_key:
            self.skipTest("OpenAI API key not available")
            
        # Initialize image generator
        img_gen = ImageGenerator(self.openai_api_key)
        
        # Test with simple prompt
        output_path = "bin/test/output/generated_image_test.png"
        result_path = img_gen.generate_image("一只可爱的小猫", output_path)
        
        # Check that file was created
        self.assertTrue(os.path.exists(result_path))
        self.assertEqual(result_path, output_path)
        
        # Test with text from file for prompt
        with open(self.test_text_path, "r", encoding="utf-8") as f:
            sample_text = f.read().split("\n\n")[1]  # Use second paragraph
            
        # Create a concise prompt from the text
        prompt = f"根据这段文字创建一个简单的插图: {sample_text[:100]}..."
        
        output_path_2 = "bin/test/output/generated_image_file_test.png"
        result_path = img_gen.generate_image(prompt, output_path_2)
        
        # Check that file was created
        self.assertTrue(os.path.exists(result_path))
        self.assertEqual(result_path, output_path_2)
    
    def test_video_generator(self):
        """Test video generation from image and audio"""
        # Generate video from test files
        output_video = "bin/test/output/test_video.mp4"
        
        # Skip if test files don't exist
        if not os.path.exists(self.test_image_path) or not os.path.exists(self.test_audio_path):
            self.skipTest("Test image or audio file not available")
        
        VideoGenerator.generate_video(self.test_image_path, self.test_audio_path, output_video)
        
        # Check that video file was created
        self.assertTrue(os.path.exists(output_video))
        
        # Test with a generated image and audio if APIs are available
        if all([self.openai_api_key, self.azure_speech_key, self.azure_speech_region]):
            # Generate an image
            img_gen = ImageGenerator(self.openai_api_key)
            image_path = "bin/test/output/video_test_image.png"
            img_gen.generate_image("一座美丽的山脉", image_path)
            
            # Generate audio
            tts = SimpleTTSGenerator(self.azure_speech_key, self.azure_speech_region)
            audio_path = "bin/test/output/video_test_audio.wav"
            tts.synthesize_speech("这是用于测试的视频，展示了一座美丽的山脉。", "zh-CN-XiaoxiaoNeural", audio_path)
            
            # Generate video
            output_video_2 = "bin/test/output/generated_video_test.mp4"
            VideoGenerator.generate_video(image_path, audio_path, output_video_2)
            
            # Check that video file was created
            self.assertTrue(os.path.exists(output_video_2))

if __name__ == "__main__":
    unittest.main()
    

