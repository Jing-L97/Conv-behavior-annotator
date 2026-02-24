import pickle
import re
from pathlib import Path

import pandas as pd

from pkg import settings


def collect_babylm_metrics(file_path: Path):
    with open(file_path, "rb") as f:
        data = pickle.load(f)
    return data["best_zorro"], data["best_blimp"]


def collect_gen_metrics1(file_path: Path):
    if not file_path.exists():
        return {}

    result = pd.read_csv(file_path)

    if result.empty:
        return {}

    # assume single-row result file
    return result.iloc[0].to_dict()


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


model_config = "1e6_reward_seed_3_entropy_001_lm_loss_001_target_6"


def extract_reward(sub_dir: Path):
    match = re.match(r"(.+?)_\d", sub_dir.name)
    return match.group(1) if match else None


results_df = pd.DataFrame()

# grammar metrics path
model_root = settings.PATH.model_dir / "ppo" / model_config

# generation metrics path (different root!)
gen_root = settings.PATH.result_dir / "ppo" / model_config


for sub_dir in model_root.iterdir():
    if not sub_dir.is_dir():
        continue

    reward = extract_reward(sub_dir)

    # -----------------------
    # 1. Collect grammar metrics
    # -----------------------
    best_zorro, best_blimp = collect_babylm_metrics(sub_dir / "best_reward" / "metrics.pkl")

    row_dict = {
        "reward": reward,
        "best_zorro": best_zorro,
        "best_blimp": best_blimp,
    }

    # -----------------------
    # 2. Collect gen metrics
    # -----------------------
    gen_subdir = gen_root / sub_dir.name
    gen_file = gen_subdir / "result_best_reward.csv"

    gen_metrics = collect_gen_metrics(gen_file)

    # merge dictionaries
    row_dict.update(gen_metrics)

    # -----------------------
    # 3. Append row
    # -----------------------
    results_df = pd.concat(
        [results_df, pd.DataFrame(row_dict, index=[0])],
        ignore_index=True,
    )


results_df.to_csv(
    settings.PATH.result_dir / f"{model_config}.csv",
    index=False,
)
