import unittest

import pandas as pd

from src.data_preprocessing import REQUIRED_INPUT_COLUMNS, preprocess_input


class PreprocessingTests(unittest.TestCase):
    def sample_customer(self):
        values = {
            "gender": "Male", "SeniorCitizen": 0, "Partner": "Yes", "Dependents": "No",
            "tenure": 12, "PhoneService": "Yes", "MultipleLines": "No",
            "InternetService": "Fiber optic", "OnlineSecurity": "No", "OnlineBackup": "Yes",
            "DeviceProtection": "No", "TechSupport": "No", "StreamingTV": "Yes",
            "StreamingMovies": "No", "Contract": "Month-to-month", "PaperlessBilling": "Yes",
            "PaymentMethod": "Electronic check", "MonthlyCharges": 75.5, "TotalCharges": 906.0,
        }
        return values

    def test_output_matches_training_schema(self):
        columns = ["tenure", "MonthlyCharges", "TotalCharges", "gender", "Contract_One year"]
        result = preprocess_input(pd.DataFrame([self.sample_customer()]), columns)
        self.assertEqual(list(result.columns), columns)
        self.assertEqual(result.shape, (1, len(columns)))

    def test_unknown_category_does_not_break_alignment(self):
        customer = self.sample_customer()
        customer["Contract"] = "Unexpected contract"
        result = preprocess_input(pd.DataFrame([customer]), ["Contract_Month-to-month"])
        self.assertEqual(result.iloc[0, 0], 0)

    def test_missing_required_field_raises_clear_error(self):
        customer = self.sample_customer()
        customer.pop("tenure")
        with self.assertRaisesRegex(ValueError, "tenure"):
            preprocess_input(pd.DataFrame([customer]), ["tenure"])


if __name__ == "__main__":
    unittest.main()
