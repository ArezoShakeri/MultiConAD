import os
from typing import Iterator, Callable
from collection import Collection, RawDataPoint, NormalizedDataPoint
import csv
import json
from dataclasses import asdict


class CHACollection(Collection):
    def __iter__(self) -> Iterator[RawDataPoint]:
        """
        Iterate through all .cha files in the specified path and yield raw data points.
        """
        for filename in os.listdir(self.path):
            if filename.endswith(".cha"):
                file_path = os.path.join(self.path, filename)
                if self.language == "english" or self.language == "chinese":
                    yield from self.parse_cha_file(file_path, self._parse_line_english_chinese)
                elif self.language == "spanish":
                    yield from self.parse_cha_file(file_path, self._parse_line_spanish)
                else:
                    raise ValueError(f"Unsupported language: {self.language}")

    def parse_cha_file(self, file_path: str, parse_line_func: Callable[[dict, str], None]) -> Iterator[RawDataPoint]:
        """
        Parses a .cha file and yields a single raw data point after processing the entire file.
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
            "Modality": "Unknown",
            "Task": [""],
            "Dataset": "Unknown",
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
            "text_interviewer":[],
            "text_interviewer_participant": [],
        }


        with open(file_path, 'r') as file:
            for line in file:
                parse_line_func(info, line,file_path)

        # Finalize the text field by joining collected transcript lines
        info["text_participant"] = " ".join(info["text_participant"])
        info["text_interviewer"] = " ".join(info["text_interviewer"])
        info["text_interviewer_participant"] = " ".join(info["text_interviewer_participant"])

        yield info
    
    def _parse_line_english_chinese(self, info: dict, line: str,file_path: str): # Chinese Lu datset from DementiaBank
        """
        Language-specific line parser for English.
        """
        if line.startswith("@PID:"):
            info["PID"] = line.split(":")[1].strip()
        elif line.startswith("@Date:"):
            info["Date"] = line.split(":")[1].strip()
        elif line.startswith("@Languages:"):
            info["Languages"] = line.split(":")[1].strip()
        elif line.startswith("@Participants:"): # e.g., PAR Participant, INV Investigator
            info["Participants"] = line.split(":")[1].strip()
        elif line.startswith("@Situation:"):
            info["Task"].append("Situation: "+line.split(":")[1].strip())
        elif line.startswith("@Activities:"):
            info["Task"].append("Activities: "+line.split(":")[1].strip())
        elif line.startswith("@Bg:"):
            info["Task"].append(line.split(":")[1].strip())
        elif line.startswith("@G:"):
            info["Task"].append(line.split(":")[1].strip())
        elif line.startswith("@comment:"):
            info["Comment"] = line.split(":")[1].strip()
        elif line.startswith("@ID:") and "Participant" in line:
            parts = line.split("|")
            info["Languages"] = parts[0].split()[-1].strip()
            info["Dataset"] = parts[1].strip()
            info["Diagnosis"] = parts[5].strip() 
            age_info = parts[3].split(';')[0].strip()
            if age_info.isdigit():
                info["age"] = int(age_info)
            info["gender"] = parts[4].strip()
            if parts[9].isdigit():
                info["MMSE"] = int(parts[9])
                #info["Moca"] = int(parts[9]) # Dalware
            elif parts[8].isdigit():
                info["MMSE"] = int(parts[8])
                #info["Moca"] = int(parts[8]) # Baycrest dataset
        elif line.startswith("@Media:"):
            media_parts = line.split(":")[1].strip().split(",")
            if len(media_parts) > 1:
                info["File_ID"] = media_parts[0].strip()
                info["Media"] = media_parts[1].strip()
        elif line.startswith("*PAR:"):
            participant_text = line.replace("*PAR:", "").strip()
            info["text_participant"].append(participant_text)
            info["text_interviewer_participant"].append("PAR: "+participant_text)
        elif line.startswith("*INV:"):
            interviewer_text = line.replace("*INV:", "").strip()
            info["text_interviewer"].append(interviewer_text)
            info["text_interviewer_participant"].append("INT: "+interviewer_text)
            

    def _parse_line_spanish(self, info: dict, line: str,file_path: str):
        """
        Language-specific line parser for Spanish.
        """
        info["File_ID"] = os.path.splitext(os.path.basename(file_path))[0]# PerLA
        if line.startswith("@PID:"):
            info["PID"] = line.split(":")[1].strip()
        elif line.startswith("@Transcriber:"):
            info["Transcriber"] = line.split(":")[1].strip()
        elif line.startswith("@Date:"):
            info["Date"] = line.split(":")[1].strip()
        elif line.startswith("@Location:"): 
            info["Location"] = line.split(":")[1].strip()
        elif line.startswith("@Time Duration:"):
            info["Duration"] = line.split(":", 1)[1].strip()
        elif line.startswith("@Languages:"):
            info["Languages"] = line.split(":")[1].strip()
            #info["Languages"]= 'spanish' # Ivanova dataset
        elif line.startswith("@Participants:"): # e.g., PAR Participant, INV Investigator
            info["Participants"] = line.split(":")[1].strip()
        elif line.startswith("@G:"):
            info["Task"].append(line.split(":")[1].strip())
        elif line.startswith("@Situation:"):
            info["Task"].append("Situation: "+line.split(":")[1].strip())
        elif line.startswith("@Bg:"):
            info["Task"].append(line.split(":")[1].strip())
        elif line.startswith("@G:"):
            info["Task"].append(line.split(":")[1].strip())
        elif line.startswith("@comment:"):
            info["Comment"] = line.split(":")[1].strip()
        elif line.startswith("@Media:"):
            media_parts = line.split(":")[1].strip().split(",")
            if len(media_parts) > 1:
                #info["File_ID"] = media_parts[0].strip()# Ivanova
                info["Media"] = media_parts[1].strip()
        elif line.startswith("@ID:") and "Target_Adult" in line:
            parts = line.split("|")
            info["Languages"] = parts[0].split()[-1].strip()
            info["Dataset"] = parts[1].strip()
            info["Diagnosis"] = parts[5].strip()
            age_info = parts[3].split(';')[0].strip()
            if age_info.isdigit():
                info["age"] = int(age_info)
            info["gender"] = parts[4].strip()
            if parts[9].isdigit():
                info["MMSE"] = int(parts[9])
            elif parts[8].isdigit():
                info["MMSE"] = int(parts[8])
        elif line.startswith("@Media:"):
            media_parts = line.split(":")[1].strip().split(",")
            if len(media_parts) > 1:
                info["File_ID"] = media_parts[0].strip()
                info["Media"] = media_parts[1].strip()
        
        elif line.startswith("*PAR:"): # Ivanova
            participant_text = line.replace("*PAR:", "").strip()
            info["text_participant"].append(participant_text)
            info["text_interviewer_participant"].append("PAR: "+participant_text)
        
        elif line.startswith("*"): # PerLA
                 interviewer_text = line
                 info["text_interviewer"].append(interviewer_text)
                 info["text_interviewer_participant"].append(interviewer_text)

        



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
            Modality=raw_datapoint["Media"],
            Task=raw_datapoint["Task"],
            File_ID=raw_datapoint["File_ID"],
            Media=raw_datapoint["Media"],
            Age=raw_datapoint["age"],
            Gender=raw_datapoint["gender"],
            Education=raw_datapoint["Education"],
            Source="CHA Dataset",
            Continents=raw_datapoint["Continents"],
            Countries=raw_datapoint["Countries"],
            Duration=raw_datapoint["Duration"],
            Location=raw_datapoint["Location"],
            Date=raw_datapoint["Date"],
            Transcriber=raw_datapoint["Transcriber"],
            Moca=raw_datapoint["Moca"],
            Setting=raw_datapoint["Setting"],
            Comment=raw_datapoint["Comment"],
            Text_interviewer_participant = raw_datapoint["text_interviewer_participant"],
            Text_participant = raw_datapoint["text_participant"],
            Text_interviewer=raw_datapoint["text_interviewer"]
        )
        





  




path_to_cha_files =  "path_to_cha_files" # path to the folder containing .cha files




if __name__ == '__main__':
    collection = CHACollection(path_to_cha_files,language="chinese")
    #collection = CHACollection(path_to_cha_files,language="spanish")
    #collection = CHACollection(path_to_cha_files,language="english")
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
            

    

    





