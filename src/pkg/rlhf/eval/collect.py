import pickle
import re
from pathlib import Path

import pandas as pd


def extract_reward(sub_dir: Path):
    match = re.match(r"(.+?)_\d", sub_dir.name)
    return match.group(1) if match else None


def extract_reward_seed(sub_dir: Path):
    # extract reward seed
    seed_match = re.search(r"reward_seed_(\d+)", sub_dir.name)
    return seed_match.group(1) if seed_match else None


def extract_scale(model_config: str) -> str:
    return model_config.split("_")[0]


def collect_babylm_metrics(file_path: Path):
    with open(file_path, "rb") as f:
        data = pickle.load(f)
    return data["best_zorro"], data["best_blimp"]


def collect_gen_metrics(file_path: Path):
    if not file_path.exists():
        return {}

    result = pd.read_csv(file_path)

    if result.empty or result.shape[1] <= 1:
        return {}

    # explicitly drop the first column by position
    result = result.iloc[:, 1:]

    # assume single-row result file
    return result.iloc[0].to_dict()
