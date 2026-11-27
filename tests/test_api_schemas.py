import unittest

from pydantic import ValidationError

from backend.schemas import CustomerProfile


class CustomerProfileSchemaTests(unittest.TestCase):
    def sample(self):
        return {
            "gender": "Male", "SeniorCitizen": 0, "Partner": "Yes", "Dependents": "No",
            "tenure": 12, "PhoneService": "Yes", "MultipleLines": "No",
            "InternetService": "Fiber optic", "OnlineSecurity": "No", "OnlineBackup": "Yes",
            "DeviceProtection": "No", "TechSupport": "No", "StreamingTV": "Yes",
            "StreamingMovies": "No", "Contract": "Month-to-month", "PaperlessBilling": "Yes",
            "PaymentMethod": "Electronic check", "MonthlyCharges": 75.5, "TotalCharges": 906.0,
        }

    def test_valid_profile_is_accepted(self):
        profile = CustomerProfile.model_validate(self.sample())
        self.assertEqual(profile.tenure, 12)

    def test_out_of_range_tenure_is_rejected(self):
        data = self.sample()
        data["tenure"] = 100
        with self.assertRaises(ValidationError):
            CustomerProfile.model_validate(data)

    def test_unexpected_fields_are_rejected(self):
        data = self.sample()
        data["customerID"] = "sensitive-id"
        with self.assertRaises(ValidationError):
            CustomerProfile.model_validate(data)


if __name__ == "__main__":
    unittest.main()
