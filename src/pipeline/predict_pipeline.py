import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from src.utils import load_object

project_root = Path(__file__).resolve().parents[2]
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))


@dataclass
class CustomData:
    age: float
    gender: str
    daily_social_media_hours: float
    platform_usage: str
    sleep_hours: float
    screen_time_before_sleep: float
    academic_performance: float
    physical_activity: float
    social_interaction_level: str
    stress_level: float
    anxiety_level: float
    depression_label: float

    def get_data_as_data_frame(self) -> pd.DataFrame:
        data = {
            "age": [self.age],
            "gender": [self.gender],
            "daily_social_media_hours": [self.daily_social_media_hours],
            "platform_usage": [self.platform_usage],
            "sleep_hours": [self.sleep_hours],
            "screen_time_before_sleep": [self.screen_time_before_sleep],
            "academic_performance": [self.academic_performance],
            "physical_activity": [self.physical_activity],
            "social_interaction_level": [self.social_interaction_level],
            "stress_level": [self.stress_level],
            "anxiety_level": [self.anxiety_level],
            "depression_label": [self.depression_label],
        }
        return pd.DataFrame(data)


class PredictPipeline:
    def __init__(self):
        project_root = Path(__file__).resolve().parents[2]
        self.model_path = project_root / "artifacts" / "model.pkl"
        self.preprocessor_path = project_root / "artifacts" / "preprocessor.pkl"

        self.model = load_object(str(self.model_path))
        self.preprocessor = load_object(str(self.preprocessor_path))

    def predict(self, features: pd.DataFrame):
        transformed_features = self.preprocessor.transform(features)
        predictions = self.model.predict(transformed_features)
        return np.array(predictions)

