import pickle
import re
from pathlib import Path

import pandas as pd

from pkg import settings


def collect_babylm_metrics(file_path: Path):
    with open(file_path, "rb") as f:
        data = pickle.load(f)
    return data["best_zorro"], data["best_blimp"]


# go over the directory to collect
model_config = "1e6_reward_seed_3_entropy_001_lm_loss_001_target_6"


def extract_reward(sub_dir: Path):
    match = re.match(r"(.+?)_\d", sub_dir.name)
    prefix = match.group(1) if match else None
    return prefix


results_df = pd.DataFrame()
for sub_dir in (settings.PATH.model_dir / "ppo" / model_config).iterdir():
    if sub_dir.is_dir():
        reward = extract_reward(sub_dir)
        best_zorro, best_blimp = collect_babylm_metrics(sub_dir / "best_reward" / "metrics.pkl")
        results_df = pd.concat(
            [
                results_df,
                pd.DataFrame(
                    {
                        "reward": reward,
                        "best_zorro": best_zorro,
                        "best_blimp": best_blimp,
                    },
                    index=[0],
                ),
            ],
            ignore_index=True,
        )

results_df.to_csv(settings.PATH.result_dir / f"{model_config}.csv", index=False)
