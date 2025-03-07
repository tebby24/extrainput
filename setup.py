from setuptools import setup, find_packages

setup(
    name="extrainput",
    version="0.1.0",
    packages=find_packages(), 
    install_requires=[
        "pydub",
        "openai",
        "azure-cognitiveservices-speech",
    ],
    author="Teddy Gonyea",
    description="A tool for generating language learning audio content",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    url="https://github.com/tebby24/extrainput",
    python_requires=">=3.6",
)