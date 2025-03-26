import unittest
from scripts.data_processing import load_data, preprocess_data
from scripts.modelling import train_models

class TestModelling(unittest.TestCase):

    def setUp(self):
        self.file_path = "/Users/camillelemaignen/Desktop/projet_python_SIA/data/DataDSVA_R.xlsm"
        df = load_data(self.file_path)
        self.df_filtered = preprocess_data(df)

    def test_model_training(self):
        logit_model, probit_model = train_models(self.df_filtered)
        self.assertIsNotNone(logit_model, "Le modèle Logit est None")
        self.assertIsNotNone(probit_model, "Le modèle Probit est None")
        self.assertTrue(hasattr(logit_model, 'params'), "Le modèle Logit n’a pas d’attribut 'params'")

if __name__ == '__main__':
    unittest.main()