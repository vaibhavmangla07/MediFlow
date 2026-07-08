import joblib

from src.logger import logging
from src.exception import CustomException

from src.components.model_trainer import ModelTrainer
from src.components.model_evaluator import ModelEvaluator


class TrainPipeline:
    def run_pipeline(self):
        try:
            logging.info("Training Pipeline Started")

            # Load Training Data
            X_train = joblib.load("artifacts/X_train.pkl")
            X_test = joblib.load("artifacts/X_test.pkl")

            y_train = joblib.load("artifacts/y_train.pkl")
            y_test = joblib.load("artifacts/y_test.pkl")

            label_encoder = joblib.load("artifacts/label_encoder.pkl")

            # Train Models
            trainer = ModelTrainer()

            results = trainer.train_and_evaluate(X_train, X_test, y_train, y_test)
            print("\nModel Performance\n")
            print(results)

            # Load Best Model
            best_model = joblib.load("artifacts/best_model.pkl")

            # Evaluate Best Model
            evaluator = ModelEvaluator()

            metrics = evaluator.evaluate(best_model, X_test, y_test, label_encoder)

            print("\nEvaluation Metrics\n")
            print(metrics)

            logging.info("Training Pipeline Completed")

        except Exception as e:
            logging.error("Training Pipeline Failed")
            raise CustomException(e)


if __name__ == "__main__":
    pipeline = TrainPipeline()
    pipeline.run_pipeline()