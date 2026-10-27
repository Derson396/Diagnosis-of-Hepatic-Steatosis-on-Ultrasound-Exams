import csv
import tempfile
import unittest
from pathlib import Path

from actions.cross_validation import split_patient_validation
from actions.csv_handler import write_roi_data_to_csv
from actions.patient_labels import class_for_patient


class DataHelperTests(unittest.TestCase):
    def test_patient_class_boundaries_are_zero_based(self):
        self.assertEqual([class_for_patient(i) for i in (0, 16, 17)], [0, 0, 1])

    def test_validation_split_keeps_each_patient_in_one_partition(self):
        patients = list(range(20)) * 2
        train, validation = split_patient_validation(patients)

        self.assertTrue(train.isdisjoint(validation))
        self.assertEqual(train | validation, set(patients))

    def test_saving_same_roi_replaces_its_existing_csv_row(self):
        with tempfile.TemporaryDirectory(dir=Path.cwd()) as directory:
            path = Path(directory) / "roi_data.csv"
            write_roi_data_to_csv(path, ["ROI_0_0.png", 0, "old", "right", 1.0])
            write_roi_data_to_csv(path, ["ROI_0_0.png", 1, "latest", "right", 1.2])

            with path.open(newline="", encoding="utf-8") as file:
                rows = list(csv.reader(file))

        self.assertEqual(len(rows), 2)
        self.assertEqual(rows[1], ["ROI_0_0.png", "1", "latest", "right", "1.2"])


if __name__ == "__main__":
    unittest.main()
