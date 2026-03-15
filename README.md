# Multimodal Emotion Detection (Fusion Model)

A deep learning project that detects human emotions using both **facial expressions and voice signals**.  
The model combines features from video and audio to improve emotion classification accuracy.

## Idea
Emotion cannot always be detected reliably from a single signal.  
This system uses **multimodal fusion**, combining information from:

- Facial expressions (video frames)
- Voice characteristics (audio)

The features from both modalities are merged before final prediction.

## Architecture

Face Branch  
Video Frame → CNN → Face Feature Vector

Audio Branch  
Audio → MFCC / Spectrogram → CNN/RNN → Audio Feature Vector

Fusion  
Face Features + Audio Features → Concatenation → Dense Layers → Softmax

## Emotion Classes
Examples of predicted emotions:

- Happy
- Sad
- Angry
- Fear
- Surprise
- Neutral

## Tech Stack

- Python
- TensorFlow / Keras
- OpenCV
- Librosa
- NumPy

## Applications

- Human–computer interaction
- Mental health monitoring
- Smart assistants
- Emotion-aware AI systems
