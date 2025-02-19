# MultiConAD: A Unified Multilingual Conversational Dataset for Early Alzheimer’s Detection
This repository contains the code and data processing pipeline for our paper, MultiConAD: A Unified Multilingual Conversational Dataset for Early Alzheimer’s Detection. The repository includes scripts for data collection, preprocessing, transcription, and experimentation.

## Repository Structure
collection.py, CHA.collection.py, TSV.collection.py, ASR_collection.py – Scripts for collecting data from various datasets.
audio_transcription.py – Transcribes audio data.
SR_collection.notebook – Normalizes transcribed data and stores it in the normalized folder.
experiments/ – Contains code for TF-IDF and E5 Large model experiments.
Data Processing Pipeline
### Data Collection:

We gathered data from multiple sources using collection.py, CHA.collection.py, TSV.collection.py, and ASR_collection.py.
### Audio Transcription:

Transcribed audio files using audio_transcription.py.
Normalized and stored transcripts in the normalized folder using SR_collection.notebook.
Dataset Combination & Preprocessing:

### Combined datasets from different languages.
Processed each language separately.
Created English translations for non-English datasets.
## Experiments:

Conducted experiments using TF-IDF and E5 Large models.
The code for these experiments is in the experiments folder.
