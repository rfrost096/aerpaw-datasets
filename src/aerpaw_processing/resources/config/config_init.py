import yaml
from importlib import resources
from aerpaw_processing.resources.config.config_class import Config
from dotenv import load_dotenv, find_dotenv


def load_config() -> Config:
    details = resources.files("aerpaw_processing.resources") / "config.yaml"

    with details.open("r") as f:
        return Config(**yaml.safe_load(f))


def load_env():
    load_dotenv(find_dotenv("config.env"))


CONFIG = load_config()
