from setuptools import setup, find_packages

setup(
    name="jarvis",
    version="0.1.0",
    packages=find_packages(),
    install_requires=[
        "numpy>=1.23.5,<2.0.0",
        "tensorflow-macos==2.15.0",  # Use tensorflow instead of tensorflow-macos on non-Mac systems
        "keras==2.15.0",
        "python-speech-features",
        "webrtcvad",
        "pyaudio",
        "scikit-learn",
    ],
    python_requires=">=3.9",
) 