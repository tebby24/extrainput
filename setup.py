from setuptools import setup, find_packages

setup(
    name="extrainput",
    version="0.1.2",
    packages=find_packages(), 
    install_requires=[
        "pydub==0.25.1",
        "openai==1.65.4",
        "azure-cognitiveservices-speech==1.42.0",
        "requests==2.32.3",
        "srt==3.5.3",
        "moviepy==2.1.2",
        "dotenv==0.9.9",
        "typing_extensions==4.12.2",
        "fasttext==0.9.3",
        "scikit-learn==1.6.1",
    ],
    author="Teddy Gonyea",
    description="A set of Python classes for generating language learning content",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    url="https://github.com/tebby24/extrainput",
    python_requires=">=3.6",
)