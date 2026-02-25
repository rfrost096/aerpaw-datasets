import logging
import os
import yaml
from dotenv import load_dotenv, find_dotenv
from aerpaw_processing.resources.config.config_class import Config


script_path = os.path.realpath(__file__)
script_directory = os.path.dirname(script_path)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(pathname)s:%(lineno)d - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler()],
)


def load_config() -> Config:
    details = os.path.join(script_directory, "config_file.yaml")

    with open(details, "r") as f:
        return Config(**yaml.safe_load(f))


def load_env():
    load_dotenv(find_dotenv("config.env"))


CONFIG = load_config()
