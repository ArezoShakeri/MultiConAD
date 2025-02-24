This repository provides the complete codebase and data processing pipeline for submission:  
**MultiConAD: A Unified Multilingual Conversational Dataset for Early Alzheimer’s Detection.**  

It includes scripts for **data collection, preprocessing, transcription, translation, and classification experiments** using both **TF-IDF and E5 Large models** across different linguistic settings.  

## **Repository Structure**  

### **1. Audio Transcription**  
Scripts related to **Automatic Speech Recognition (ASR)** and transcript organization:  
- `ASR_audio_dataset.py` – Transcribes audio data.  
- `ASR_collection.py` – Organizes transcripts of audio-based data generated by **Whisper ASR** into a normalized format.  

### **2. Data Collection & Extraction**  
Scripts for gathering and structuring data from **16 different datasets** into a standardized format:  
- `collection.py` – Defines the **normalized class format** and handles general data collection.  
- `cha.collection.py` – Extracts text data from **DementiaBank Chat format files**.  
- `TSV.collection.py` – Processes text-based data specifically for the **iFlytek dataset**.  
- `ASR_collection.py` – Organizes transcripts from **Whisper ASR** into structured data.  
- `test_ch_collection.py` – Sample script to test the output of the normalized dataset for **chat format files**.  

### **3. Text Preprocessing**  
Scripts for **language-specific text cleaning** to ensure consistency:  
- `text_cleaning_Chinese.py` – Combining all **Chinese** datasets and preprocessing text and splitting to train and test sets.  (NCMMSC2021_AD_Competition,Chinese-predictive challenge_iFlytek, Taukdial)
- `text_cleaning_English.py` – Combining all **English** datasets and preprocessing text and splitting to train and test sets. (Pitt, Lu, Baycrest, VAS, Kempler, WLS, Delware, taukdial)
- `text_cleaning_Greek.py` – Combining all **Greek** datasets and preprocessing text and splitting to train and test sets. (ADReSS-M, Ds3, Ds5, Ds7)
- `text_cleaning_Spanish.py` – Combining all **Spanish** datasets and preprocessing text and splitting to train and test sets. (Ivanova, PerLA)

### **4. Translation**  
Scripts for translating non-English datasets into **English** for multilingual experiments:  
- `translation_all_language.py` – Translates **Greek, Spanish, and Chinese** datasets into **English** using **GPT-4** and appends a translation column to the dataset.  

### **5. Experiments & Classification**  
Scripts for **binary and multiclass classification tasks** using different text representation techniques:  
- `TF_IDF_classifier.py` – Implements **TF-IDF-based classification**.  
- `e5_large_classifier.py` – Implements **E5 Large model-based classification**.  
- **Supports three experimental settings:**  
  - **Monolingual:** Training on a single language.  
  - **Combined-Multilingual:** Training on a dataset containing multiple original languages.  
  - **Combined-Translated:** Training on a dataset where all text is translated into English.  


## **Implementation Guide**  

### **Step 1: Audio Transcription**  
1. Place raw **audio files** in the designated folder.  
2. Run the **ASR transcription script**:  
   ```bash
   python ASR_audio_dataset.py
   ```  
3. Normalize and organize transcripts:  
   ```bash
   python ASR_collection.py
   ```  

### **Step 2: Data Collection & Extraction**  
1. Run the main data collection script:  
   ```bash
   python collection.py
   ```  
2. For **DementiaBank Chat format** files:  
   ```bash
   python cha.collection.py
   ```  
3. For **iFlytek dataset (TSV format)**:  
   ```bash
   python TSV.collection.py
   ```  
4. Test the processed dataset output:  
   ```bash
   python test_ch_collection.py
   ```  

### **Step 3: Preprocess Text Data**  
1. Run the preprocessing script for each language:  
   ```bash
   python text_cleaning_Chinese.py
   python text_cleaning_English.py
   python text_cleaning_Greek.py
   python text_cleaning_Spanish.py
   ```  

### **Step 4: Translate Non-English Data**  
1. Translate **Greek, Spanish, and Chinese** datasets into English:  
   ```bash
   python translation_all_language.py --directory_to_input_data=${1} --directory_to_output_translated=${2} --source_language=${3}
   ```  

### **Step 5: Train Classification Models**  
#### **TF-IDF Classifier**  
1. Train and evaluate using **TF-IDF**:  
   ```bash
   python TF_IDF_classifier.py --test_language=${1} --task=${2}  --translated=${3}
   ```  

#### **E5 Large Model Classifier**  
1. Train and evaluate using **E5 Large embeddings**:  
   ```bash
   python e5_large_classifier.py --test_language=${1} --task=${2}  --translated=${3}
   ```  



## **Experimental Settings & Goals**  
The experiments are conducted in three different linguistic setups:  
1. **Monolingual:** Training and evaluation on individual language datasets.  
2. **Combined-Multilingual:** Training on a dataset containing **multiple original languages**.  
3. **Combined-Translated:** Training on a dataset where **all text is translated into English**.  

### **Objectives**  
- Compare **TF-IDF** vs. **E5 Large embeddings** for text classification.  
- Analyze the impact of **monolingual, multilingual, and translated** settings on classification accuracy.  
- Determine whether **translation improves classification performance** in multilingual scenarios.  

