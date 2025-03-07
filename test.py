from extrainput.extrainput import TextGenerator, TTSEngine, ExtraInputGenerator
from dotenv import load_dotenv
import os


def test_text_generator(text_generator):
    sample_prompt = "Translate the following sentence to Chinese: 'Hello, how are you?'"
    try:
        result = text_generator.generate_text(sample_prompt)
        print("TextGenerator output:", result)
    except Exception as e:
        print("Error in TextGenerator:", e)


def test_tts_engine(tts_engine: TTSEngine):
    sample_text = "hello! I can also speak english"
    try:
        tts_engine.synthesize_speech(sample_text, "zh-CN-YunxiaNeural", "bin/test/output/tts_engine_test.wav")
    except Exception as e:
        print("Error in TTSEngine:", e)

def test_extra_input_generator(extra_input_generator: ExtraInputGenerator):
    def test_get_example_sentences():
        sample_word = "特别"
        try:
            result = extra_input_generator.get_example_sentences(sample_word)
            print("ExtraInputGenerator output:", result)
        except Exception as e:
            print("Error in ExtraInputGenerator:", e)

    def test_generate_example_sentence_listener():
        words = ["上下文", "再三", "张口结舌"]   
        output_filepath =  "bin/test/output/examples_sentence_listener_test.wav"
        extra_input_generator.generate_example_sentence_listener(words, output_filepath)
        print("ExtraInputGenerator output:", output_filepath)

    # test_get_example_sentences()
    test_generate_example_sentence_listener() 

if __name__ == "__main__":
    load_dotenv()

    deepseek_api_key = os.getenv("DEEPSEEK_API_KEY")
    azure_speech_key = os.getenv("AZURE_SPEECH_KEY")
    azure_speech_region = os.getenv("AZURE_SPEECH_REGION")

    text_generator = TextGenerator(deepseek_api_key)
    tts_engine = TTSEngine(azure_speech_key, azure_speech_region)
    extra_input_generator = ExtraInputGenerator(text_generator, tts_engine)

    # test_text_generator(text_generator)
    # test_tts_engine(tts_engine)
    test_extra_input_generator(extra_input_generator)
    

