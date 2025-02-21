import re
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from collection import JSONLCombiner


input_files = [
    "path_to_M_ADReSS.jsonl",
    "path_to_Greek_data_Ds3.jsonl",
    "path_to_Greek_data_Ds5.jsonl",
    "path_to_Greek_data_Ds7.jsonl"
]
output_directory = 'path_to_output_directory'
output_filename = 'combined_jsonl_Greek_M_ADReSS_D3_5_7.jsonl'
combiner = JSONLCombiner(input_files, output_directory, output_filename)
combiner.combine()








# Preprocessing text transcribed by Whisper large v3
def preprocess_text_audio_data(text, TF_IDF= False):
    text = re.sub(r'Speaker \d+:', '', text).strip()
    text = re.sub(r'\.{2,}', '', text)
    if TF_IDF:
        text = re.sub(r'[^\w\s]', '', text) 
    return text


def remove_short_transcripts(df, min_length=60):
    return df[df['Text_length'] >= min_length]



Greek_df= pd.read_json(output_directory + output_filename, lines=True)
# Replace the diagnosis labels
Greek_df['Diagnosis'] = Greek_df['Diagnosis'].replace({
    'Control': 'HC',
    'healthy': 'HC',
    'HEALTHY': 'HC',
    'ProbableAD': 'AD',
    'MCI': 'MCI',
    'ΜCI': 'MCI'
})




Greek_df = Greek_df[Greek_df['Diagnosis'] != 'Unknown']
Greek_df['Text_interviewer_participant'] = Greek_df['Text_interviewer_participant'].apply(preprocess_text_audio_data)
Greek_df['length'] = Greek_df['Text_interviewer_participant'].apply(len)



Greek_df = remove_short_transcripts(Greek_df)
train_gr, test_gr = train_test_split(Greek_df, test_size=0.2,stratify=Greek_df['Diagnosis'], random_state=42)

train_gr.to_json(output_directory + "train_greek.jsonl", orient="records", lines=True, force_ascii=False)
test_gr.to_json(output_directory + "test_greek.jsonl", orient="records", lines=True, force_ascii=False)
