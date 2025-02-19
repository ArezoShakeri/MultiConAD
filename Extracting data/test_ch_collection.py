import unittest
from cha_collection import CHACollection, NormalizedDataPoint

class TestCHACollection(unittest.TestCase):
    def test_normalize_datapoint(self):
        collection = CHACollection("", language="english")
        raw_datapoint = {
            "PID": "123",
            "Languages": "English",
            "MMSE": 30,
            "Diagnosis": "Healthy",
            "Participants": ["A", "B"],
            "Dataset": "DementiaBank",
            "Media": "audio",
            "Task": "description",
            "File_ID": "file123",
            "age": 75,
            "gender": "M",
            "Education": "High School",
            "Continents": "Europe",
            "Countries": "Norway",
            "Moca": 28,
            "Setting": "Clinic",
            "text_interviewer_participant": "Sample transcript text.",
            "text_participant": "Sample transcript text.",
            "text_interviewer": "Sample transcript text."
        }
        normalized_datapoint = collection.normalize_datapoint(raw_datapoint)
        self.assertEqual(normalized_datapoint.PID, "123")
        self.assertEqual(normalized_datapoint.Languages, "English")
        self.assertEqual(normalized_datapoint.MMSE, 30)
        self.assertEqual(normalized_datapoint.Diagnosis, "Healthy")
        self.assertEqual(normalized_datapoint.Participants, ["A", "B"])
        self.assertEqual(normalized_datapoint.Dataset, "DementiaBank")
        self.assertEqual(normalized_datapoint.Media, "audio")
        self.assertEqual(normalized_datapoint.Task, "description")
        self.assertEqual(normalized_datapoint.File_ID, "file123")
        self.assertEqual(normalized_datapoint.Age, 75)
        self.assertEqual(normalized_datapoint.Gender, "M")
        self.assertEqual(normalized_datapoint.Education, "High School")
        self.assertEqual(normalized_datapoint.Continents, "Europe")
        self.assertEqual(normalized_datapoint.Countries, "Norway")
        self.assertEqual(normalized_datapoint.Moca, 28)
        self.assertEqual(normalized_datapoint.Setting, "Clinic")
        self.assertEqual(normalized_datapoint.Text_interviewer_participant, "Sample transcript text.")
        self.assertEqual(normalized_datapoint.Text_participant, "Sample transcript text.")
        self.assertEqual(normalized_datapoint.Text_interviewer, "Sample transcript text.")


if __name__ == '__main__':
    unittest.main()
