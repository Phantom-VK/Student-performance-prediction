import os
import sys
from dataclasses import dataclass

from catboost import CatBoostRegressor
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, AdaBoostRegressor
from sklearn.metrics import r2_score
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from xgboost import XGBRegressor

from project_config import PROJECT_ROOT
from src.exception import CustomException
from src.logger import logging
from src.utils import evaluate_models, save_obj


@dataclass
class ModelTrainerConfig:
    trained_model_file_path = os.path.join(PROJECT_ROOT, 'artifacts', 'model.pkl')





class ModelTrainer:
    def __init__(self):
        self.model_trainer_config = ModelTrainerConfig()

    def initiate_model_trainer(self, train_array, test_array):
        try:
            logging.info("Entered model trainer method or component")
            logging.info("Split training and test input data")
            X_train, y_train, X_test, y_test = (
                train_array[:, :-1],
                train_array[:, -1],
                test_array[:, :-1],
                test_array[:, -1]
            )
            models = {
                "LinearRegression": LinearRegression(),
                "DecisionTreeRegressor": DecisionTreeRegressor(),
                "RandomForestRegressor": RandomForestRegressor(),
                "XGBRegressor": XGBRegressor(),
                "Decision Tree":DecisionTreeRegressor(),
                "Gradient Boosting": GradientBoostingRegressor(),
                "AdaBoost": AdaBoostRegressor(),
                "KNN": KNeighborsRegressor(),
                "CatBoost": CatBoostRegressor(verbose=False)
            }

            logging.info("Start model evaluation")
            model_report:dict = evaluate_models(X_train, X_test, y_train, y_test, models)

            ## Get best model
            best_model_score = max(model_report.values())
            best_model_name = max(model_report, key=model_report.get)
            best_model = models[best_model_name]
            if best_model_score < 0.6:
                raise Exception("Model performance is not good enough")
            logging.info(f"Best model is {best_model_name} with score {model_report[best_model_name]}")

            save_obj(
                file_path = self.model_trainer_config.trained_model_file_path,
                obj = best_model
            )
            logging.info("Saved object successfully")
            return best_model_score

        except Exception as e:
            raise CustomException(e, sys)
