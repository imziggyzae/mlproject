import sys
import os

import numpy as np
from dataclasses import dataclass
from pathlib import Path
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import GridSearchCV

project_root = Path(__file__).resolve().parents[2]
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

current_working_dir = Path.cwd()
if str(current_working_dir) not in sys.path:
    sys.path.insert(0, str(current_working_dir))

from src.exception import CustomException
from src.logger import logging
from src.utils import save_object


@dataclass
class ModelTrainerConfig:
    model_path: str = os.path.join('artifacts', 'model.pkl')


class ModelTrainer:
    def __init__(self):
        self.model_trainer_config = ModelTrainerConfig()

    def initiate_model_trainer(self, train_array, test_array):
        try:
            logging.info('Splitting training and testing input data')
            X_train, y_train = train_array[:, :-1], train_array[:, -1]
            X_test, y_test = test_array[:, :-1], test_array[:, -1]

            logging.info('Starting hyperparameter tuning for RandomForestClassifier')
            param_grid = {
                'n_estimators': [100, 200, 300],
                'max_depth': [None, 5, 10],
                'min_samples_split': [2, 5, 10],
            }

            base_model = RandomForestClassifier(random_state=42)
            grid_search = GridSearchCV(
                estimator=base_model,
                param_grid=param_grid,
                cv=5,
                n_jobs=-1,
                scoring='accuracy',
                verbose=1
            )
            grid_search.fit(X_train, y_train)

            best_model = grid_search.best_estimator_
            logging.info(f'Best hyperparameters found: {grid_search.best_params_}')

            y_pred = best_model.predict(X_test)
            model_score = accuracy_score(y_test, y_pred)

            logging.info(f'Model accuracy: {model_score}')

            save_object(
                file_path=self.model_trainer_config.model_path,
                obj=best_model
            )

            logging.info('Model saved successfully')
            return model_score

        except Exception as e:
            raise CustomException(e, sys)


