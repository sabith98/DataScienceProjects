import os
import sys
import dill

from sklearn.metrics import r2_score
from sklearn.model_selection import GridSearchCV

import numpy as np
import pandas as pd

from src.exception import CustomException

def save_object(file_path, obj):
    try:
        dir_path = os.path.dirname(file_path)
        os.makedirs(dir_path, exist_ok=True)
        with open(file_path, 'wb') as file_obj:
            dill.dump(obj, file_obj)

    except Exception as e:
        raise CustomException(e, sys)
    
def evaluate_models(X_train, y_train, X_test, y_test, models):
    """Evaluate a list of trained models on the given test data and return their performance metrics"""
    try:
        best_models = {}

        for name, (model, param_grid) in models.items():
            # Use GridSearchCV for hyperparameter tuning
            grid_search = GridSearchCV(model, param_grid, cv=5, scoring='accuracy')
            grid_search.fit(X_train, y_train)

            best_model = grid_search.best_estimator_
            best_params = grid_search.best_params_

            best_models[name] = {
                "model": best_model,
                "best_params": best_params,
                "train_score": grid_search.best_score_,
                "test_score": grid_search.score(X_test, y_test)
            }
            
            # print(f"Best Hyperparameters: {grid_search.best_params_}")

            # Cross-validation
            # cv_scores = cross_val_score(best_model, X, y, cv=5, scoring='accuracy')
            # print(f"Cross-Validation Scores: {cv_scores}")
            # print(f"Mean CV Accuracy: {cv_scores.mean():.2f}\n")

        return best_models
    
    except Exception as e:
        raise CustomException(e, sys)