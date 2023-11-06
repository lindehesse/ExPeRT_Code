import json
from pathlib import Path
from typing import Union


def load_json(json_path: Union[Path, str]) -> dict:
    """Loads a json file
    Args:
        json_path (string): path of json
    Returns:
        [dict]: dictionary with content of json
    """
    with open(json_path) as f:
        dict = json.load(f)
    return dict
