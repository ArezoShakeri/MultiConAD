import os
import json
import whisper
from tqdm import tqdm

# Path to the directory containing audio files
directory_path = 'path_to_audio_files'

# Load the Whisper model
model = whisper.load_model("large-v3")

# List to store results
results = []

# Loop over files in the directory
for filename in tqdm(os.listdir(directory_path)):
    if filename.endswith(".wav"):  # Adjust the file extension as needed
        audio_path = os.path.join(directory_path, filename)
        
        try:
            # Transcribe the audio file
            result = model.transcribe(audio_path)
            
            # Get the detected language and transcription
            detected_language = result['language']
            transcription = result['text']
            
            # Only append if the detected language is 'en' or 'zh'
            if detected_language in ['en', 'zh','el']:
                results.append({
                    "file_name": os.path.splitext(filename)[0],
                    "transcription": transcription,
                    "language": detected_language
                })
        except Exception as e:
            print(f"Error processing file {filename}: {e}")

# Save results to JSON file
output_path = '/kaggle/working/taukdial_train_transcrpt.json'
with open(output_path, "w", encoding="utf-8") as f:
    json.dump(results, f, ensure_ascii=False, indent=4)

print(f"Transcriptions saved to {output_path}")