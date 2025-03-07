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
    author="Your Name",
    author_email="your.email@example.com",
    description="A tool for generating language learning audio content",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    url="https://github.com/yourusername/extrainput",
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.6",
)