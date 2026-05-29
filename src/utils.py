import pickle
from pathlib import Path
from typing import Any

def save_object(file_path: str, obj: Any) -> None:
    file_path = Path(file_path)
    file_path.parent.mkdir(parents=True, exist_ok=True)
    with file_path.open("wb") as file_obj:
        pickle.dump(obj, file_obj)

def load_object(file_path: str) -> Any:
    with open(file_path, "rb") as file_obj:
        return pickle.load(file_obj)
