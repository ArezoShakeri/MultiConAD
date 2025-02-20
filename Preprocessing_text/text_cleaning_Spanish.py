import re
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from collection import JSONLCombiner

input_files = [
    "path_to_spanish_data_Ivanova.jsonl",
     "path_to_spanish_data_Perla.jsonl"
]
output_directory = 'path_to_output_directory'
output_filename = 'combined_jsonl_spanish_perla_Ivanova.jsonl'


combiner = JSONLCombiner(input_files, output_directory, output_filename)
combiner.combine()






def preprocess_text(text):

    text = re.sub(r'\b[A-Z]{3}\b', '', text)
    text = re.sub(r'xxx', '', text)
    text = re.sub(r'<[^>]*>', '', text)
    # Remove qutation and all punctuation marks, in case of TF-IDF, for e5 you should comment out this part.
    text = re.sub(r'[^\w\s]', '', text)
    text = re.sub(r'\d+', '', text)
    text = text.replace('PAR', '')
    text = text.replace('\n', ' ')
    text = re.sub(r'\s+', ' ', text).strip()
    text = re.sub(r'\\x[0-9A-Za-z_]+\\x', '', text)
    text = re.sub(r'\b\w+:\s*', '', text) 
    text = text.replace('\n', ' ')
    text = text.replace('→', '')
    text = text.replace('(', '').replace(')', '')
    text = re.sub(r'[\\+^"/„]', '', text)
    text = re.sub(r"[_']", '', text)
    text = text.replace('\t', ' ')
    text = re.sub(r'\[.*?\]', '', text)
    text = text.replace('&=laughs', '')
    text = text.replace('&=nods', '')
    text = text.replace('&=coughs', '')
    text = text.replace('&=snaps:tongue', '')
    text = text.replace('<', '').replace('>', '')
    text = text.replace('*', '').replace('&', '')
    text = re.sub(r'\s+', ' ', text).strip()
    text = re.sub(r'([.,!?;:])\s+\1', r'\1', text)
    text = re.sub(r'(\.\s*){2,}', '.', text)
    if '.' in text:
        text = text.rsplit('.', 1)[0] + '.'  # Keep the text before the last period and add the period

    return text


def plot_text_lengths(dataset, dataset_name):
    # Filter the dataset for "PerLA"
    df_perla = dataset[dataset["Dataset"] == dataset_name]

    # Calculate the length of each "Text_interviewer_participant"
    df_perla['length'] = df_perla['Text_interviewer_participant'].apply(lambda x: len(str(x).split()))

    # Group by PID to ensure unique PIDs
    df_perla_grouped = df_perla.groupby('PID', as_index=False).agg({'length': 'sum'})

    # Plot the lengths as a point plot
    plt.figure(figsize=(10, 6))
    sns.pointplot(x='PID', y='length', data=df_perla_grouped)
    plt.xlabel('PatientID')
    plt.ylabel('Length (Number of Words)')
    plt.title('Point Plot of Text Lengths in PerLA Dataset')
    plt.xticks(rotation=90)  # Rotate x-axis labels if needed
    plt.show()



def process_transcripts(df, word_limits):
    def process_text(text, min_words, max_words):
        words = text.split()
        if len(words) < min_words:
            return None
        if len(words) > max_words:
            return ' '.join(words[:max_words])
        return text

    processed_data = []
    for index, row in df.iterrows():
        dataset_type = row['Dataset']
        text = row['Text_interviewer_participant']
        if dataset_type in word_limits:
            min_words, max_words = word_limits[dataset_type]
            processed_text = process_text(text, min_words, max_words)
            if processed_text:
                row['Text_interviewer_participant'] = processed_text
                processed_data.append(row)

    return pd.DataFrame(processed_data)



spanish_data_path= "path_to_spanish_data_Ivanova.jsonl"
df_spanish_per_Iva= pd.read_json(output_directory + output_filename, lines=True)
# Replace 'DTA' with 'AD' in the 'Diagnosis' column
df_spanish_per_Iva["Diagnosis"] = df_spanish_per_Iva["Diagnosis"].replace('DTA', 'AD')
# Remove rows with 'unknown' or empty values in the 'Diagnosis' column
df_spanish_per_Iva = df_spanish_per_Iva[df_spanish_per_Iva["Diagnosis"].notnull() & (df_spanish_per_Iva["Diagnosis"].str.strip() != '') & (df_spanish_per_Iva["Diagnosis"] != 'Unknown')]
# Preprocess the 'Text_interviewer_participant' column using the 'preprocess_text' function
df_spanish_per_Iva["Text_interviewer_participant"] = df_spanish_per_Iva["Text_interviewer_participant"].apply(preprocess_text)
# Adding the lengh count column to dataset
df_spanish_per_Iva['length'] = df_spanish_per_Iva['Text_interviewer_participant'].apply(lambda x: len(str(x).split()))



# Removing too short and too long transcripts
word_limits = {
    'Ivanova': (40, 100),
    'PerLA': (250, 1500)
}
processed_df = process_transcripts(df_spanish_per_Iva, word_limits)
train_df_spa, test_df_spa = train_test_split(processed_df, test_size=0.2, stratify=processed_df['Diagnosis'], random_state=42)




# Print the distribution of diagnoses in the processed dataset
diagnosis_counts = processed_df["Diagnosis"].value_counts()
print(diagnosis_counts)

# Save train and test datasets as JSONL
train_df_spa.to_json(output_directory + "train_spanish.jsonl", orient="records", lines=True, force_ascii=False)
test_df_spa.to_json(output_directory + "test_spanish.jsonl", orient="records", lines=True, force_ascii=False)

