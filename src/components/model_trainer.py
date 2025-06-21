import os
import sys
from dataclasses import dataclass
from typing import Dict, Any

import numpy as np
from catboost import CatBoostRegressor
from sklearn.ensemble import (
    RandomForestRegressor,
    GradientBoostingRegressor,
    AdaBoostRegressor
)
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from xgboost import XGBRegressor

from project_config import PROJECT_ROOT
from src.exception import CustomException
from src.logger import logging
from src.utils import evaluate_models, save_obj


@dataclass
class ModelTrainerConfig:
    """Configuration for model training artifacts"""
    trained_model_file_path: str = os.path.join(PROJECT_ROOT, 'artifacts', 'model.pkl')


class ModelTrainer:
    def __init__(self):
        """Initialize model trainer with configuration"""
        self.model_trainer_config = ModelTrainerConfig()
        self.models = self._initialize_models()
        self.params = self._initialize_hyperparameters()

    def _initialize_models(self) -> Dict[str, Any]:
        """Initialize machine learning models with default parameters"""
        return {
            "Decision Tree": DecisionTreeRegressor(),
            "Random Forest": RandomForestRegressor(),
            "Gradient Boosting": GradientBoostingRegressor(),
            "Linear Regression": LinearRegression(),
            "XGBoost": XGBRegressor(),
            "CatBoost": CatBoostRegressor(verbose=False),
            "AdaBoost": AdaBoostRegressor()
        }

    def _initialize_hyperparameters(self) -> Dict[str, Any]:
        """Define hyperparameter search spaces for each model"""
        return {
            "Decision Tree": {
                'criterion': ['squared_error', 'friedman_mse', 'absolute_error', 'poisson'],
            },
            "Random Forest": {
                'n_estimators': [8, 16, 32, 64, 128, 256]
            },
            "Gradient Boosting": {
                'learning_rate': [.1, .01, .05, .001],
                'subsample': [0.6, 0.7, 0.75, 0.8, 0.85, 0.9],
                'n_estimators': [8, 16, 32, 64, 128, 256]
            },
            "Linear Regression": {},
            "XGBoost": {
                'learning_rate': [.1, .01, .05, .001],
                'n_estimators': [8, 16, 32, 64, 128, 256]
            },
            "CatBoost": {
                'depth': [6, 8, 10],
                'learning_rate': [0.01, 0.05, 0.1],
                'iterations': [30, 50, 100]
            },
            "AdaBoost": {
                'learning_rate': [.1, .01, 0.5, .001],
                'n_estimators': [8, 16, 32, 64, 128, 256]
            }
        }

    def initiate_model_trainer(self, train_array: np.ndarray, test_array: np.ndarray) -> float:
        """
        Train and evaluate multiple models, then save the best performing one

        Args:
            train_array: Numpy array containing training features and target
            test_array: Numpy array containing test features and target

        Returns:
            float: R2 score of the best model on test data
        """
        try:
            logging.info("Starting model training pipeline")

            # Split data into features and target
            logging.info("Splitting data into features and target")
            X_train, y_train = train_array[:, :-1], train_array[:, -1]
            X_test, y_test = test_array[:, :-1], test_array[:, -1]

            # Evaluate all models
            logging.info("Evaluating models with hyperparameter tuning")
            model_report = evaluate_models(
                X_train=X_train,
                y_train=y_train,
                X_test=X_test,
                y_test=y_test,
                models=self.models,
                param=self.params
            )

            # Select best model
            best_model_name = max(model_report, key=model_report.get)
            best_model_score = model_report[best_model_name]
            best_model = self.models[best_model_name]

            logging.info(f"Best model: {best_model_name} with R2 score: {best_model_score:.4f}")

            # Validate model performance
            if best_model_score < 0.6:
                logging.warning("Best model performance below threshold (R2 < 0.6)")
                raise ValueError("No suitable model found (R2 < 0.6)")

            # Save best model
            logging.info("Saving best model to disk")
            save_obj(
                file_path=self.model_trainer_config.trained_model_file_path,
                obj=best_model
            )

            return best_model_score

        except Exception as e:
            logging.error(f"Error during model training: {str(e)}")
            raise CustomException(e, sys)