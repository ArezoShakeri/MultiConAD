#  This file defines the data structures and the base functionality.
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Iterator, Any, Dict, List
import json
import os


RawDataPoint = Dict[str, Any]

@dataclass
class NormalizedDataPoint:
    PID: str
    Languages: str
    MMSE: Any
    Diagnosis: str
    Participants: Any
    Dataset: str
    Modality: str
    Task: List[str]
    File_ID: str
    Media: str
    Age: Any
    Gender: str
    Education: Any
    Source: str
    Continents: Any
    Countries: Any
    Duration: Any
    Location: str
    Date: Any
    Transcriber: Any
    Moca: Any
    Moca: Any
    Setting: Any
    Comment: Any
    Text_interviewer_participant: str
    Text_participant: str
    Text_interviewer: str


class Collection(ABC):
    def __init__(self, path: str, language: str):
        self.path = path
        self.language = language.lower()

    @abstractmethod
    def __iter__(self) -> Iterator[RawDataPoint]:
        raise NotImplementedError

    @abstractmethod
    def normalize_datapoint(self, raw_datapoint: RawDataPoint) -> NormalizedDataPoint:
        raise NotImplementedError

    def get_normalized_data(self) -> Iterator[NormalizedDataPoint]:
        for raw_datapoint in self:
            yield self.normalize_datapoint(raw_datapoint)










class JSONLCombiner:
    def __init__(self, input_files: List[str], output_directory: str, output_filename: str):
        self.input_files = input_files
        self.output_directory = output_directory
        self.output_file = os.path.join(output_directory, output_filename)

    def combine(self):
        # Ensure the output directory exists
        os.makedirs(self.output_directory, exist_ok=True)

        combined_data = []

        # Read the contents of each JSONL file
        for file_path in self.input_files:
            with open(file_path, 'r', encoding='utf-8') as file:
                data = [json.loads(line) for line in file]
                combined_data.extend(data)

        # Write the combined contents to the new JSONL file
        with open(self.output_file, 'w', encoding='utf-8') as outfile:
            for entry in combined_data:
                outfile.write(json.dumps(entry, ensure_ascii=False) + '\n')

        print(f"Combined JSONL file saved to: {self.output_file}")

