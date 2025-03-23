import os
import unittest
from dotenv import load_dotenv
from extrainput.extrainput import ExtraInputGenerator

# Load environment variables from .env file
load_dotenv()

# Dictionary to store test configurations
# Set to True to run the test, False to skip
TEST_CONFIG = {
    'text_generator': False,         # Test text generation
    'image_generator': False,        # Test image generation
    'simple_tts': False,             # Test simple TTS
    'tts_with_subs': False,          # Test TTS with subtitles
    'word_groups': True,            # Test word grouping
    'voice_pairing': False,          # Test voice pairing
    'video_generator': False,        # Test video generation
}

class TestExtraInput(unittest.TestCase):
    def setUp(self):
        # Get API keys from environment variables
        self.azure_speech_key = os.getenv("AZURE_SPEECH_KEY")
        self.azure_speech_region = os.getenv("AZURE_SPEECH_REGION")
        self.openai_api_key = os.getenv("OPENAI_API_KEY")
        
        # Skip tests if credentials are not available
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
            
        # Initialize the ExtraInputGenerator if credentials are available
        if all([self.openai_api_key, self.azure_speech_key, self.azure_speech_region]):
            self.generator = ExtraInputGenerator(
                openai_api_key=self.openai_api_key,
                azure_speech_key=self.azure_speech_key,
                azure_speech_region=self.azure_speech_region
            )
        else:
            self.generator = None
    
    def test_text_generator(self):
        """Test text generation functionality"""
        if not self.generator:
            self.skipTest("API credentials not available")
            
        result = self.generator.generate_text("请用'好'字造一个简单的句子")
        
        # Check that we got a non-empty result
        self.assertTrue(result and isinstance(result, str))
        print(f"Generated text: {result}")
        
        # Test with longer prompt from file
        if os.path.exists(self.test_text_path):
            with open(self.test_text_path, "r", encoding="utf-8") as f:
                sample_text = f.read()
            
            # Use first paragraph as context for generation
            paragraphs = sample_text.split("\n\n")
            context = paragraphs[1] if len(paragraphs) > 1 else paragraphs[0]
            prompt = f"请基于以下文本写一个短评论:\n{context}\n"
            
            result = self.generator.generate_text(prompt)
            self.assertTrue(result and isinstance(result, str))
            print(f"Generated text based on input file: {result}")
    
    def test_image_generator(self):
        """Test image generation"""
        if not self.generator:
            self.skipTest("API credentials not available")
            
        # Test with simple prompt
        output_path = "bin/test/output/generated_image_test.png"
        result_path = self.generator.generate_image("一只可爱的小猫", output_path)
        
        # Check that file was created
        self.assertTrue(os.path.exists(result_path))
        self.assertEqual(result_path, output_path)
        
        # Test with text from file for prompt
        if os.path.exists(self.test_text_path):
            with open(self.test_text_path, "r", encoding="utf-8") as f:
                sample_text = f.read().split("\n\n")[1]  # Use second paragraph
                
            # Create a concise prompt from the text
            prompt = f"根据这段文字创建一个简单的插图: {sample_text[:100]}..."
            
            output_path_2 = "bin/test/output/generated_image_file_test.png"
            result_path = self.generator.generate_image(prompt, output_path_2)
            
            # Check that file was created
            self.assertTrue(os.path.exists(result_path))
            self.assertEqual(result_path, output_path_2)
    
    def test_simple_tts(self):
        """Test simple TTS generation"""
        if not self.generator:
            self.skipTest("API credentials not available")
            
        output_path = "bin/test/output/simple_tts_test.wav"
        
        # Test with simple text
        result_path = self.generator.synthesize_speech("你好世界", "zh-CN-XiaoxiaoNeural", output_path)
        
        # Check that file was created
        self.assertTrue(os.path.exists(result_path))
        self.assertEqual(result_path, output_path)
        
        # Test with text from file
        if os.path.exists(self.test_text_path):
            with open(self.test_text_path, "r", encoding="utf-8") as f:
                sample_text = f.read().split("\n\n")[0]  # Use first paragraph
                
            output_path_2 = "bin/test/output/simple_tts_file_test.wav"
            result_path = self.generator.synthesize_speech(sample_text, "zh-CN-XiaoxiaoNeural", output_path_2)
            
            # Check that file was created
            self.assertTrue(os.path.exists(result_path))
            self.assertEqual(result_path, output_path_2)
    
    def test_tts_with_subtitles(self):
        """Test TTS with subtitle generation"""
        if not self.generator:
            self.skipTest("API credentials not available")
            
        mp3_output = "bin/test/output/tts_with_subs_test.mp3"
        srt_output = "bin/test/output/tts_with_subs_test.srt"
        
        # Test with simple text
        test_text = "这是一个测试。我们正在生成语音和字幕。"
        self.generator.synthesize_speech_with_srt(test_text, "zh-CN-XiaoxiaoNeural", mp3_output, srt_output)
        
        # Check that both files were created
        self.assertTrue(os.path.exists(mp3_output))
        self.assertTrue(os.path.exists(srt_output))
        
        # Test with text from file
        if os.path.exists(self.test_text_path):
            with open(self.test_text_path, "r", encoding="utf-8") as f:
                sample_text = f.read()
                
            mp3_output_2 = "bin/test/output/tts_with_subs_file_test.mp3"
            srt_output_2 = "bin/test/output/tts_with_subs_file_test.srt"
            
            self.generator.synthesize_speech_with_srt(sample_text, "zh-CN-XiaoxiaoNeural", mp3_output_2, srt_output_2)
            
            # Check that both files were created
            self.assertTrue(os.path.exists(mp3_output_2))
            self.assertTrue(os.path.exists(srt_output_2))
    
    def test_word_groups(self):
        """Test word grouping functionality"""
        if not self.generator:
            self.skipTest("API credentials not available")
            
        # Test with sample words
        words = [
            "你好", "再见", "谢谢", "不客气", "是", "不是", "什么", "谁",
            "哪里", "为什么", "多少", "时间", "今天", "明天", "昨天", "早上",
            "晚上", "中午", "吃饭", "喝水", "工作", "学习", "睡觉", "朋友",
            "家人", "爱", "喜欢", "不喜欢", "电脑", "手机", "书", "笔",
            "桌子", "椅子", "房子", "车", "钱", "名字", "国家", "城市"
        ]
        grouped_words = self.generator.create_word_groups(words, group_min_size=3, group_max_size=5)
        
        # Check that we got a list of groups
        self.assertTrue(isinstance(grouped_words, list))
        
        # Check that each group has the required size
        for group in grouped_words:
            self.assertTrue(isinstance(group, list))
            self.assertTrue(3 <= len(group) <= 5)
            
        print(f"Grouped words: {grouped_words}")
    
    def test_voice_pairing(self):
        """Test voice pairing functionality"""
        if not self.generator:
            self.skipTest("API credentials not available")
            
        # Test with a sample text
        content = "我是一个小女孩，今年六岁了。我喜欢唱歌和跳舞。"
        voice = self.generator.pair_voice_to_article(content)
        
        # Check that we got a valid voice
        self.assertTrue(voice in [v["name"] for v in self.generator.voices])
        print(f"Selected voice for child content: {voice}")
        
        # Test with another sample text
        content = "作为经验丰富的教授，我认为教育是培养未来领袖的关键。"
        voice = self.generator.pair_voice_to_article(content)
        
        # Check that we got a valid voice
        self.assertTrue(voice in [v["name"] for v in self.generator.voices])
        print(f"Selected voice for professor content: {voice}")
    
    def test_video_generator(self):
        """Test video generation from image and audio"""
        if not self.generator:
            self.skipTest("API credentials not available")
            
        # Generate video from test files
        if not os.path.exists(self.test_image_path) or not os.path.exists(self.test_audio_path):
            self.skipTest("Test image or audio file not available")
        
        output_video = "bin/test/output/test_video.mp4"
        result_path = self.generator.generate_video(self.test_image_path, self.test_audio_path, output_video)
        
        # Check that video file was created
        self.assertTrue(os.path.exists(result_path))
        self.assertEqual(result_path, output_video)
        
        # Test with generated content if possible
        # Generate an image
        image_path = "bin/test/output/video_test_image.png"
        self.generator.generate_image("一座美丽的山脉", image_path)
        
        # Generate audio
        audio_path = "bin/test/output/video_test_audio.wav"
        self.generator.synthesize_speech("这是用于测试的视频，展示了一座美丽的山脉。", "zh-CN-XiaoxiaoNeural", audio_path)
        
        # Generate video
        output_video_2 = "bin/test/output/generated_video_test.mp4"
        result_path = self.generator.generate_video(image_path, audio_path, output_video_2)
        
        # Check that video file was created
        self.assertTrue(os.path.exists(result_path))
        self.assertEqual(result_path, output_video_2)

def run_tests():
    """Run the selected tests based on the TEST_CONFIG dictionary"""
    suite = unittest.TestSuite()
    
    # Add tests based on configuration
    if TEST_CONFIG.get('text_generator', False):
        suite.addTest(TestExtraInput('test_text_generator'))
        
    if TEST_CONFIG.get('image_generator', False):
        suite.addTest(TestExtraInput('test_image_generator'))
        
    if TEST_CONFIG.get('simple_tts', False):
        suite.addTest(TestExtraInput('test_simple_tts'))
        
    if TEST_CONFIG.get('tts_with_subs', False):
        suite.addTest(TestExtraInput('test_tts_with_subtitles'))
        
    if TEST_CONFIG.get('word_groups', False):
        suite.addTest(TestExtraInput('test_word_groups'))
        
    if TEST_CONFIG.get('voice_pairing', False):
        suite.addTest(TestExtraInput('test_voice_pairing'))
        
    if TEST_CONFIG.get('video_generator', False):
        suite.addTest(TestExtraInput('test_video_generator'))

    runner = unittest.TextTestRunner(verbosity=2)
    runner.run(suite)

if __name__ == "__main__":
    # Option 1: Run specific tests by modifying the TEST_CONFIG dictionary
    run_tests()
    
    # Option 2: Run all tests with unittest.main()
    # unittest.main()