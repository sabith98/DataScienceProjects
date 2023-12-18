# src > components > model_trainer.py

import os
import sys
from dataclasses import dataclass

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB

from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

from src.exception import CustomException
from src.logger import logging

from src.utils import save_object, evaluate_models

@dataclass
class ModelTrainerConfig:
    """Data class for model training configurations."""
    trained_model_file_path=os.path.join("artifacts","model.pkl")

class ModelTrainer:
    def __init__(self):
        self.model_trainer_config=ModelTrainerConfig()

    def initiate_model_trainer(self,train_array,test_array):
        try:
            logging.info("Split training and test input data")
            x_train,y_train,X_test,y_test=(
                train_array[:,:-1],
                train_array[:,-1],
                test_array[:,:-1],
                test_array[:,-1]
            )
            # Define models with hyperparameter grids
            models = {
                'Logistic Regression': (LogisticRegression(random_state=2), {
                    'C': [0.001, 0.01, 0.1, 1, 10, 100]
                }),
                'Random Forest': (RandomForestClassifier(random_state=2), {
                'n_estimators': [50, 100, 200, 300],
                'max_depth': [None, 10, 20, 30]
                }),
                'SVM': (SVC(random_state=2), {
                'C': [0.1, 1, 10],
                'kernel': ['linear', 'rbf']
                }),
                'KNN': (KNeighborsClassifier(), {
                'n_neighbors': [3, 5, 7, 9],
                'p': [1, 2]
                }),
                'Decision Tree': (DecisionTreeClassifier(random_state=2), {
                'max_depth': [None, 10, 20, 30],
                'min_samples_split': [2, 5, 10]
                }),
                'Gradient Boosting': (GradientBoostingClassifier(random_state=2), {
                'n_estimators': [50, 100, 150],
                'learning_rate': [0.01, 0.1, 0.2]
                }),
                'Naive Bayes': (GaussianNB(), {}),
            }

            best_models:dict=evaluate_models(X_train=x_train,y_train=y_train,
                                             X_test=X_test,y_test=y_test,models=models)
            
            best_model_name = max(best_models, key=lambda k: best_models[k]['test_score'])
            best_model_training_score = best_models[best_model_name]['train_score']
            best_model_score = best_models[best_model_name]['test_score']
            best_model_params = best_models[best_model_name]['best_params']

            overfitting_difference = best_model_training_score-best_model_score

            best_model = best_models[best_model_name]['model']

            if best_model_score < 0.6:
                raise CustomException("No best model found")
            elif overfitting_difference > 0.05:
                raise CustomException(f"{best_model_name} model overfit by {overfitting_difference*100}%")
            
            logging.info("Best found model on both training and testing dataset")
            
            save_object(
                file_path=self.model_trainer_config.trained_model_file_path,
                obj=best_model
            )

            predicted=best_model.predict(X_test)

            accuracy_score_value = accuracy_score(y_test, predicted)

            return (
                best_model_name,
                best_model_params,
                accuracy_score_value
            )

        except Exception as e:
            raise CustomException(e,sys)