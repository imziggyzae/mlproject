import os
import sys
from pathlib import Path
from dataclasses import dataclass

import pandas as pd
from sklearn.model_selection import train_test_split

project_root = Path(__file__).resolve().parents[2]
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

current_working_dir = Path.cwd()
if str(current_working_dir) not in sys.path:
    sys.path.insert(0, str(current_working_dir))

from src.exception import CustomException
from src.logger import logging
from src.components.data_transformation import DataTransformation
from src.components.model_trainer import ModelTrainer

@dataclass
class DataIngestionConfig:
    train_data_path: str = "artifacts/train.csv"
    test_data_path: str = "artifacts/test.csv"
    raw_data_path: str = "artifacts/data.csv"

class DataIngestion:
    def __init__(self):
        self.ingestion_config = DataIngestionConfig()

    def initiate_data_ingestion(self):
        logging.info("Entered the data ingestion method or component")
        try:
            project_root = Path(__file__).resolve().parents[2]
            dataset_candidates = list(project_root.rglob("Teen_Mental_Health_Dataset.csv"))

            if not dataset_candidates:
                raise FileNotFoundError(f"Dataset not found under project root: {project_root}")

            data_file = dataset_candidates[0]

            if not data_file.exists():
                raise FileNotFoundError(f"Dataset not found: {data_file}")

            df = pd.read_csv(data_file)
            logging.info("Read the dataset as dataframe")

            artifacts_dir = project_root / "artifacts"
            artifacts_dir.mkdir(parents=True, exist_ok=True)

            df.to_csv(artifacts_dir / "data.csv", index=False, header=True)

            logging.info("Train test split initiated")
            train_set, test_set = train_test_split(df, test_size=0.2, random_state=42)

            train_set.to_csv(artifacts_dir / "train.csv", index=False, header=True)
            test_set.to_csv(artifacts_dir / "test.csv", index=False, header=True)

            logging.info("Ingestion of the data is completed")

            return (
                str(artifacts_dir / "train.csv"),
                str(artifacts_dir / "test.csv"),
            )
        except Exception as e:
            raise CustomException(e, sys)

if __name__ == "__main__":
    obj = DataIngestion()
    train_data, test_data = obj.initiate_data_ingestion()

    data_transformation = DataTransformation()
    train_arr, test_arr, _ = data_transformation.initiate_data_transformation(train_data, test_data)

    modeltrainer = ModelTrainer()
    print(modeltrainer.initiate_model_trainer(train_arr, test_arr))