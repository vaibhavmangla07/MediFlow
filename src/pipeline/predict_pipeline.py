import joblib

from src.logger import logging
from src.exception import CustomException
from src.components.data_preprocessing import TextPreprocessor

class PredictPipeline:
    def __init__(self):
        self.model = joblib.load("artifacts/best_model.pkl")
        self.vectorizer = joblib.load("artifacts/tfidf_vectorizer.pkl")
        self.label_encoder = joblib.load("artifacts/label_encoder.pkl")
        self.preprocessor = TextPreprocessor()

    def predict(self, text):
        try:
            logging.info("Prediction Started")

            # Text Preprocessing
            clean_text = self.preprocessor.preprocess(text)

            # TF-IDF Transformation
            vector = self.vectorizer.transform([clean_text])

            # Prediction
            prediction = self.model.predict(vector)

            # Decode Label
            department = self.label_encoder.inverse_transform(prediction)

            logging.info("Prediction Completed")
            return department[0]

        except Exception as e:
            logging.error("Prediction Failed")
            raise CustomException(e)


if __name__ == "__main__":
    predictor = PredictPipeline()
    complaint = input("Enter Patient Complaint: ")
    prediction = predictor.predict(complaint)
    print(f"\nRecommended Department: {prediction}")