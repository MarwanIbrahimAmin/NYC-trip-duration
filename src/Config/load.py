import yaml
from pathlib import Path

config_path = Path(__file__).parent / "config.yaml"
def load_config() -> dict:
    with open(config_path , "r") as f:
        return yaml.safe_load(f)