import logging
import sys
from pathlib import Path
from typing import Any, Dict

import dill
from sklearn.metrics import r2_score
from sklearn.model_selection import GridSearchCV

from src.exception import CustomException


def save_obj(file_path: str, obj: Any) -> None:
    """
    Save a Python object to disk using dill (with pickle protocol).

    Args:
        file_path: Path where the object will be saved
        obj: Python object to be serialized

    Raises:
        CustomException: If any error occurs during saving
    """
    try:
        # Create directory if it doesn't exist
        Path(file_path).parent.mkdir(parents=True, exist_ok=True)

        logging.info(f"Saving object to {file_path}")
        with open(file_path, 'wb') as file:
            dill.dump(obj, file, protocol=dill.HIGHEST_PROTOCOL)
        logging.info("Object saved successfully")

    except Exception as e:
        logging.error(f"Error saving object to {file_path}: {str(e)}")
        raise CustomException(e, sys)


def evaluate_models(
        X_train,
        X_test,
        y_train,
        y_test,
        models: Dict[str, Any],
        param: Dict[str, Dict[str, Any]]
) -> Dict[str, float]:
    """
    Evaluate multiple models using GridSearchCV and return their test R2 scores.

    Args:
        X_train: Training features
        X_test: Test features
        y_train: Training target
        y_test: Test target
        models: Dictionary of model names and instances
        param: Dictionary of hyperparameter grids for each model

    Returns:
        Dictionary mapping model names to their test R2 scores

    Raises:
        CustomException: If any error occurs during evaluation
    """
    try:
        logging.info("Starting model evaluation process")
        report = {}

        for name, model in models.items():
            logging.info(f"Evaluating model: {name}")

            # Hyperparameter tuning
            current_params = param.get(name, {})
            if current_params:  # Only do GridSearch if parameters are provided
                logging.info(f"Performing GridSearchCV for {name} with params: {current_params}")
                gs = GridSearchCV(model, current_params, cv=5, n_jobs=-1)
                gs.fit(X_train, y_train)
                model.set_params(**gs.best_params_)
                logging.info(f"Best params for {name}: {gs.best_params_}")

            # Train model with best parameters
            model.fit(X_train, y_train)

            # Evaluate on both train and test sets
            for dataset, X, y in [("train", X_train, y_train), ("test", X_test, y_test)]:
                y_pred = model.predict(X)
                score = r2_score(y, y_pred)
                logging.info(f"{name} {dataset} R2 score: {score:.4f}")

                if dataset == "test":
                    report[name] = score

        logging.info("Completed model evaluation")
        return report

    except Exception as e:
        logging.error(f"Error during model evaluation: {str(e)}")
        raise CustomException(e, sys)


def load_object(file_path:str):
    try:
        with open(file_path, 'rb') as file:
            return dill.load(file)
    except Exception as e:
        raise CustomException(e, sys)
