import re
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from collection import JSONLCombiner




input_files = [
    "path/to/data_Chinese -NCMMSC2021_AD_Competition.jsonl",
    "path/to/Chinese-predictive challenge_tsv2_output.jsonl",
    "path/to/TAUKADIAL.jsonl"
]
output_directory = 'path_to_output_directory'
output_filename = 'combined_jsonl_Chinses_NCMMSC_iFlyTek_Taukdial.jsonl'
combiner = JSONLCombiner(input_files, output_directory, output_filename)
combiner.combine()



def remove_english_rows(df):
    # Filter out rows where the 'language' field is 'en' (Taukdial dataset)
    df_filtered = df[df['Languages'] != 'en']
    return df_filtered





def preprocess_text(text):
 
    text = text.replace('//', '').replace('/', '')
    #text = text.replace('&', '')
    text = re.sub(r'\s*&\w+', '', text).strip()
    text = re.sub(r'\[.*?\]', '', text)
    # Remove "Doctor: " and "Patient: "
    text = text.replace('Doctor:', '').replace('Patient:', '')
    text = re.sub(r'\s+', ' ', text).strip()
    
    return text

def preprocess_text_audio_data(text):

    text= re.sub(r'Speaker \d+:', '', text).strip()
    
    return text

def preprocess_dataset(df, dataset_name):
    if dataset_name in df['Dataset'].values:
        df.loc[df['Dataset'] == dataset_name, 'Text_interviewer_participant'] = df.loc[df['Dataset'] == dataset_name, 'Text_interviewer_participant'].apply(preprocess_text)
    return df

def preprocess_dataset_audio(df):
    for index, row in df.iterrows():
        if row["Dataset"] != 'Predictive_Chinese_challenge_Chinese_2019':
            df.at[index, 'Text_interviewer_participant'] = preprocess_text_audio_data(row['Text_interviewer_participant'])
    return df

# filter rows based on mean ± std
def filter_by_length(row, stats):
    mean = stats.loc[row['Diagnosis'], 'mean']
    std = stats.loc[row['Diagnosis'], 'std']
    return mean - std <= row['length'] <= mean + std



df_chinese= pd.read_json(output_directory + output_filename, lines=True)
# Filter rows where the Diagnosis column has the value 'Unknown'
unknown_diagnosis_rows = df_chinese[df_chinese["Diagnosis"] == 'Unknown']
# Save the unique values in a set
unique_values_set = set(unknown_diagnosis_rows["Dataset"])
# Print the unique values set
print(unique_values_set)
# removing the test of iFlytek, due to lack of labels
df_chinese = df_chinese[df_chinese["Diagnosis"] != 'Unknown']
# Rename Diagnosis values: NC to HC
df_chinese["Diagnosis"] = df_chinese["Diagnosis"].replace({'NC': 'HC'})
df_chinese["Diagnosis"] = df_chinese["Diagnosis"].replace({'CTRL': 'HC'})
df_chinese = remove_english_rows(df_chinese)
# Filter rows where the Diagnosis column has the value 'Unknown'
unknown_diagnosis_rows = df_chinese[df_chinese["Diagnosis"] == 'Unknown']
df_chinese = preprocess_dataset(df_chinese, "Predictive_Chinese_challenge_Chinese_2019")
df_chinese = preprocess_dataset_audio(df_chinese)



df_chinese['length'] = df_chinese['Text_interviewer_participant'].apply(len)

# Calculate the min, max, mean, and std length in each category for df
length_stats = df_chinese.groupby('Diagnosis')['length'].agg(['min', 'max', 'mean', 'std'])
df_chinese = df_chinese[df_chinese.apply(filter_by_length, axis=1, stats=length_stats)]
train_cha, test_cha = train_test_split(df_chinese, test_size=0.2,stratify=df_chinese['Diagnosis'], random_state=42)

train_cha.to_json(output_directory + "train_chinses.jsonl", orient="records", lines=True, force_ascii=False)
test_cha.to_json(output_directory + "test_chinese.jsonl", orient="records", lines=True, force_ascii=False)

