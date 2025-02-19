# This file Implements language-specific logic and the execution pipeline for TSV file from iFlytek dataset.
import os
from typing import Iterator, Callable
from collection import Collection, RawDataPoint, NormalizedDataPoint
import csv
import json
from dataclasses import asdict


     

class TSVCollection(Collection):
    def __iter__(self) -> Iterator[RawDataPoint]:
        """
        Iterate through all .tsv files in the specified path and yield raw data points.
        """
        for filename in os.listdir(self.path):
            if filename.endswith(".tsv"):
                file_path = os.path.join(self.path, filename)
                yield from self.parse_tsv_file(file_path)

    def parse_tsv_file(self, file_path: str) -> Iterator[RawDataPoint]:
        """
        Parse a TSV file and extract speaker-specific data.
        """
        info = {
            "age": "Unknown",
            "gender": "Unknown",
            "PID": "Unknown",
            "Languages": "Unknown",
            "Participants": [],
            "File_ID": "Unknown",
            "Media": "Unknown",
            "Education": "Unknown",
            "Modality": "Automatic_iFlytek_transcrption_ and_manually_annotated_text",
            "Task": [" Cookie_Theft_picture_description"],
            "Dataset": "Predictive_Chinese_challenge_Chinese_2019",
            "Diagnosis": "Unknown",
            "MMSE": "Unknown",
            "Continents": "Unknown",
            "Countries": "Unknown",
            "Duration": "Unknown",
            "Location": "Unknown",
            "Date": "Unknown",
            "Transcriber": "Unknown",
            "Moca": "Unknown",
            "Setting": "Unknown",
            "Comment": "Unknown",
            "text_participant": [],
            "text_interviewer": [],
            "text_interviewer_participant": [],
        }
        with open(file_path, 'r', encoding='utf-8') as file:
            info["PID"] = file_path.split("/")[-1].split(".")[0]
            reader = csv.DictReader(file, delimiter='\t')
            for line in reader:
                speaker = line.get("speaker", "").strip()  # Get speaker column
                value = line.get("value", "").strip()      # Get value column

                # Append the value if speaker is not 'sil'
                if speaker == "<A>":
                    if "Doctor" not in info["Participants"]:
                        info["Participants"].append("Doctor")
                    info["text_interviewer"].append("Doctor: " + value)
                    info["text_interviewer_participant"].append("Doctor: " + value)
                elif speaker == "<B>":
                    if "Patient" not in info["Participants"]:
                        info["Participants"].append("Patient")
                    info["text_participant"].append("Patient: " + value)
                    info["text_interviewer_participant"].append("Patient: " + value)
        info["text_interviewer_participant"] = " ".join(info["text_interviewer_participant"])
        info["text_participant"] = " ".join(info["text_participant"])
        info["text_interviewer"] = " ".join(info["text_interviewer"])
        yield info

    def normalize_datapoint(self, raw_datapoint: RawDataPoint) -> NormalizedDataPoint:
        """
        Normalize a raw data point into a standardized format.
        """
        return NormalizedDataPoint(
            PID=raw_datapoint["PID"],
            Languages=raw_datapoint["Languages"],
            MMSE=raw_datapoint["MMSE"],
            Diagnosis=raw_datapoint["Diagnosis"],
            Participants=raw_datapoint["Participants"],
            Dataset=raw_datapoint["Dataset"],
            Modality=raw_datapoint["Modality"],
            Task=raw_datapoint["Task"],
            File_ID=raw_datapoint["File_ID"],
            Media=raw_datapoint["Media"],
            Age=raw_datapoint["age"],
            Gender=raw_datapoint["gender"],
            Education=raw_datapoint["Education"],
            Source="TSV Dataset",
            Continents=raw_datapoint["Continents"],
            Countries=raw_datapoint["Countries"],
            Duration=raw_datapoint["Duration"],
            Location=raw_datapoint["Location"],
            Date=raw_datapoint["Date"],
            Transcriber=raw_datapoint["Transcriber"],
            Moca=raw_datapoint["Moca"],
            Setting=raw_datapoint["Setting"],
            Comment=raw_datapoint["Comment"],
            Text_interviewer_participant=raw_datapoint["text_interviewer_participant"],
            Text_participant=raw_datapoint["text_participant"],
            Text_interviewer=raw_datapoint["text_interviewer"]
        )



path_to_cha_files =  "path_to_tsv_files" # Path to the directory containing the .tsv files


if __name__ == '__main__':
    
    collection = TSVCollection(path_to_cha_files,language="")
    
    # Making the file name for the output file
    last_words= path_to_cha_files.split('/')[-3:]
    output_file_name= f"{last_words[0]}_{last_words[1]}_{last_words[2]}_output.jsonl"
    
    # Writing the normalized data to the output file
    output_file_path = os.path.join("jsonl_files", output_file_name)
    with open(output_file_path, "w",encoding="utf-8") as outfile:
        for normalized_datapoint in collection.get_normalized_data():
            normalized_dict = asdict(normalized_datapoint)
            json.dump(normalized_dict, outfile, ensure_ascii=False)
            outfile.write("\n")
    