import joblib
import pandas as pd

from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.ensemble import RandomForestClassifier

from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

from src.logger import logging
from src.exception import CustomException


class ModelTrainer:

    def __init__(self):
        self.models = {
            "Multinomial Naive Bayes": MultinomialNB(),
            "Logistic Regression": LogisticRegression(random_state=42),
            "Linear SVM": LinearSVC(random_state=42),
            "Random Forest": RandomForestClassifier(random_state=42)
        }

    def train_and_evaluate(self, X_train, X_test, y_train, y_test):
        try:
            logging.info("Model Training Started")

            results = []

            best_model = None
            best_accuracy = 0

            for model_name, model in self.models.items():
                model.fit(X_train, y_train)
                y_pred = model.predict(X_test)
                accuracy = accuracy_score(y_test, y_pred)
                precision = precision_score(y_test, y_pred, average="weighted")
                recall = recall_score(y_test, y_pred, average="weighted")
                f1 = f1_score(y_test, y_pred, average="weighted")

                results.append({
                    "Model": model_name,
                    "Accuracy": round(accuracy, 4),
                    "Precision": round(precision, 4),
                    "Recall": round(recall, 4),
                    "F1 Score": round(f1, 4)
                })

                if accuracy > best_accuracy:
                    best_accuracy = accuracy
                    best_model = model

            results_df = pd.DataFrame(results)
            results_df = results_df.sort_values(by="Accuracy", ascending=False).reset_index(drop=True)

            joblib.dump(best_model, "artifacts/best_model.pkl")

            logging.info("Best Model Saved Successfully")
            return results_df

        except Exception as e:
            logging.error("Error During Model Training")
            raise CustomException(e)