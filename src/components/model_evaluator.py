import json
import joblib

from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report, confusion_matrix

from src.logger import logging
from src.exception import CustomException


class ModelEvaluator:
    def evaluate(self, model, X_test, y_test, label_encoder):
        try:
            logging.info("Model Evaluation Started")

            y_pred = model.predict(X_test)
            accuracy = accuracy_score(y_test, y_pred)
            precision = precision_score(y_test, y_pred, average="weighted")
            recall = recall_score(y_test, y_pred, average="weighted")
            f1 = f1_score(y_test, y_pred, average="weighted")

            metrics = {
                "accuracy": round(accuracy, 4),
                "precision": round(precision, 4),
                "recall": round(recall, 4),
                "f1_score": round(f1, 4)
            }

            with open("artifacts/metrics.json", "w") as file:
                json.dump(metrics, file, indent=4)

            report = classification_report(y_test, y_pred, target_names=label_encoder.classes_)
            with open("artifacts/classification_report.txt", "w") as file:
                file.write(report)

            cm = confusion_matrix(y_test, y_pred)
            joblib.dump(cm, "artifacts/confusion_matrix.pkl")

            logging.info("Model Evaluation Completed")
            return metrics

        except Exception as e:
            logging.error("Error During Model Evaluation")
            raise CustomException(e)