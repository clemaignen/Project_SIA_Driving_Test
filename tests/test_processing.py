import unittest
import pandas as pd
from scripts.data_processing import load_data, preprocess_data

class TestDataProcessing(unittest.TestCase):

    def setUp(self):
        self.file_path = "/Users/camillelemaignen/Desktop/projet_python_SIA/data/DataDSVA_R.xlsm"
        self.df = load_data(self.file_path)

    def test_load_data_not_empty(self):
        self.assertFalse(self.df.empty, "Le DataFrame chargé est vide.")

    def test_columns_exist(self):
        expected_cols = [
            'Location', 'Age', 'Conducted_Male', 'Passes_Male', 'PassRate_Male',
            'Conducted_Female', 'Passes_Female', 'PassRate_Female',
            'Conducted_Total', 'Passes_Total', 'PassRate_Total', 'Year'
        ]
        for col in expected_cols:
            self.assertIn(col, self.df.columns)

    def test_preprocess_outputs_nonempty(self):
        df_filtered = preprocess_data(self.df)
        self.assertFalse(df_filtered.empty, "Le DataFrame pré-traité est vide.")

if __name__ == '__main__':
    unittest.main()