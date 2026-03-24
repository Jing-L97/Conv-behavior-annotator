import re
from pathlib import Path

import pandas as pd

from pkg import settings

model_config = "1e6_reward_seed_3_entropy_001_lm_loss_001_target_6"


def extract_reward(sub_dir: Path):
    match = re.match(r"(.+?)_\d", sub_dir.name)
    return match.group(1) if match else None


seed_lst = [3, 123, 999, 1024]
target_row_num = 100

dfs = []

for seed in seed_lst:
    gen_root = settings.PATH.result_dir / "ppo" / model_config / str(seed)

    for sub_dir in Path(gen_root).iterdir():
        if not sub_dir.is_dir():
            continue

        reward = extract_reward(sub_dir)

        df = pd.read_csv(sub_dir / "utt_best_reward.csv")
        df = df.sample(n=target_row_num, replace=True, random_state=1)

        df["seed"] = seed
        df["reward"] = reward

        dfs.append(df)

final_df = pd.concat(dfs, ignore_index=True)

final_df.to_excel(
    settings.PATH.result_dir / "ppo" / f"{model_config}_sampled_100.xlsx",
    index=False,
)
