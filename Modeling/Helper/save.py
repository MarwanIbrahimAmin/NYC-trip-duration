import os 
import joblib


def save_object(obj, file_path: str):
    try:
        dir_name = os.path.dirname(file_path)
        if dir_name:
            os.makedirs(dir_name, exist_ok=True)
            
        joblib.dump(obj, file_path)
            
        print(f"Successfully saved object to: {file_path}")
    except Exception as e:
        print(f"Error while saving object: {e}")

def load_object(file_path: str):
    try:
        return joblib.load(file_path)
    except Exception as e:
        print(f"Error while loading object from {file_path}: {e}")
        return None
