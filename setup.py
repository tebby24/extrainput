from setuptools import setup, find_packages

setup(
    name="extrainput",
    version="0.1.2",
    packages=find_packages(), 
    install_requires=[
        "pydub",
        "openai",
        "azure-cognitiveservices-speech",
        "requests",
        "srt",
        "moviepy",
        "dotenv",
        "typing_extensions",
        "fasttext",
        "scikit-learn",
    ],
    author="Teddy Gonyea",
    description="A set of Python classes for generating language learning content",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    url="https://github.com/tebby24/extrainput",
    python_requires=">=3.6",
)