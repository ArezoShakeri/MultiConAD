import openai
import json
import argparse


parser = argparse.ArgumentParser()
parser.add_argument('--directory_to_input_data', required=True)
parser.add_argument('--directory_to_output_translated', required=True)
parser.add_argument('--source_language', required=True)
args_slurm = parser.parse_args()

# Initialize OpenAI client with your API key
client = openai.OpenAI(api_key="your-api-key")

def translate_text(text, source_lang, target_lang="English", max_tokens=2048):
    """Translates text to English using OpenAI GPT model, handling long texts by splitting them into chunks."""
    prompt_template = f"Translate the following {source_lang} text to {target_lang}:\n\n{{}}\n\nTranslation:"
    translated_text = ""
    
    # Split text into chunks
    text_chunks = [text[i:i+max_tokens] for i in range(0, len(text), max_tokens)]
    
    for chunk in text_chunks:
        prompt = prompt_template.format(chunk)
        
        response = client.chat.completions.create(
            model="gpt-4",  
            messages=[
                {"role": "system", "content": "You are a helpful translator."},
                {"role": "user", "content": prompt}
            ],
            temperature=0  # Ensures accurate translations
        )
        
        translated_text += response.choices[0].message.content.strip() + " "
    
    return translated_text.strip()

def translate_dataset(input_path, output_path, source_lang):
    """Translates the text in the dataset and saves it to a new file."""
    with open(input_path, 'r', encoding='utf-8') as f:
        lines = f.readlines()

    translated_lines = []
    for i, line in enumerate(lines):
        data = json.loads(line)
        text = data.get("Text_interviewer_participant", "")
        print(f"Translating line {i+1}: {text[:50]}...")  # Print the first 50 characters of the text
        translated_text = translate_text(text, source_lang)
        data["translated"] = translated_text
        translated_lines.append(json.dumps(data, ensure_ascii=False))

    with open(output_path, 'w', encoding='utf-8') as f:
        for line in translated_lines:
            f.write(line + '\n')


translate_dataset(input_path=args_slurm.directory_to_input_data, output_path=args_slurm.directory_to_output_translated, source_lang=args_slurm.source_language)