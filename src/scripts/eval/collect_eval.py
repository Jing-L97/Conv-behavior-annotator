#!/usr/bin/env python3

import argparse
from pathlib import Path

import pandas as pd
from tqdm import tqdm

from pkg import settings
from pkg.rlhf.eval.collect import (
    collect_babylm_metrics,
    collect_gen_metrics,
    extract_reward,
    extract_reward_seed,
    extract_scale,
)

DEFAULT_MODEL_CONFIGS = [
    "1e5_entropy_001_lm_loss_001_target_6",
    "1e6_entropy_001_lm_loss_001_target_6",
    "1e7_entropy_001_lm_loss_001_target_6",
]

DEFAULT_SEEDS = [1, 2, 3, 123, 999, 1024]


def get_args():
    p = argparse.ArgumentParser(description="Collect PPO / baseline / childes evaluation results")

    p.add_argument("--model-configs", nargs="+", default=DEFAULT_MODEL_CONFIGS)
    p.add_argument("--seeds", nargs="+", type=int, default=DEFAULT_SEEDS)
    p.add_argument("--target-row-num", type=int, default=100)
    p.add_argument("--scales", nargs="+", default=["1e5", "1e6", "1e7"])
    p.add_argument("--output-dir", type=str, default=None)

    # ✅ new flag
    p.add_argument(
        "--collect_utterances",
        action="store_true",
        help="Whether to collect and sample utterances",
    )

    return p.parse_args()


# -------------------------
# Baseline
# -------------------------
def collect_baseline(args, rows, dfs):
    baseline_root = settings.PATH.result_dir / "baseline"

    if not baseline_root.exists():
        print(f"Warning: baseline directory not found at {baseline_root}")
        return

    for scale in tqdm(args.scales):
        scale_dir = baseline_root / scale
        if not scale_dir.exists():
            continue

        for lm_seed_dir in scale_dir.iterdir():
            if not lm_seed_dir.is_dir():
                continue

            try:
                lm_seed = int(lm_seed_dir.name)
            except ValueError:
                continue

            if lm_seed not in args.seeds:
                continue

            metrics_file = lm_seed_dir / "benchmark.pkl"
            best_zorro, best_blimp = None, None
            if metrics_file.exists():
                best_zorro, best_blimp = collect_babylm_metrics(metrics_file)

            for gen_seed_dir in lm_seed_dir.iterdir():
                if not gen_seed_dir.is_dir():
                    continue

                try:
                    gen_seed = int(gen_seed_dir.name)
                except ValueError:
                    gen_seed = gen_seed_dir.name

                row_dict = {
                    "model_config": "baseline",
                    "scale": scale,
                    "pretrain_seed": lm_seed,
                    "fine_tune_seed": "none",
                    "generation_seed": gen_seed,
                    "reward": "none",
                    "reward_seed": "none",
                }

                if best_zorro is not None:
                    row_dict["best_zorro"] = best_zorro
                    row_dict["best_blimp"] = best_blimp

                gen_file = gen_seed_dir / "result.csv"
                if gen_file.exists():
                    row_dict.update(collect_gen_metrics(gen_file))

                rows.append(row_dict)

                # ✅ controlled utterance collection
                if args.collect_utterances:
                    utt_file = gen_seed_dir / "utt.csv"
                    if utt_file.exists():
                        df = pd.read_csv(utt_file).sample(n=args.target_row_num, replace=True, random_state=1)
                        df["model_config"] = "baseline"
                        df["scale"] = scale
                        df["pretrain_seed"] = lm_seed
                        df["fine_tune_seed"] = "none"
                        df["generation_seed"] = gen_seed
                        df["reward"] = "none"
                        df["reward_seed"] = "none"
                        dfs.append(df)


# -------------------------
# PPO
# -------------------------


def collect_ppo(args, rows, dfs):
    for model_config in tqdm(args.model_configs):
        scale = extract_scale(model_config)
        if scale not in set(args.scales):
            continue

        config_root_model = settings.PATH.model_dir / "ppo" / model_config
        config_root_gen = settings.PATH.result_dir / "ppo" / model_config

        if not config_root_model.exists():
            continue

        # lm_seed level (e.g., 3)
        for lm_seed_dir in config_root_model.iterdir():
            if not lm_seed_dir.is_dir():
                continue

            try:
                lm_seed = int(lm_seed_dir.name)
            except ValueError:
                continue

            if lm_seed not in args.seeds:
                continue

            # fine-tune seed level (e.g., 123)
            for finetune_seed_dir in lm_seed_dir.iterdir():
                if not finetune_seed_dir.is_dir():
                    continue

                fine_tune_seed = finetune_seed_dir.name

                # reward directory level
                for reward_dir in finetune_seed_dir.iterdir():
                    if not reward_dir.is_dir():
                        continue

                    reward = extract_reward(reward_dir)
                    reward_seed = extract_reward_seed(reward_dir)

                    # metrics (from model_dir)
                    metrics_file = reward_dir / "best_reward" / "metrics.pkl"
                    best_zorro, best_blimp = None, None
                    if metrics_file.exists():
                        best_zorro, best_blimp = collect_babylm_metrics(metrics_file)

                    # corresponding result dir (mirror structure in result_dir)
                    gen_root = config_root_gen / str(lm_seed) / fine_tune_seed / reward_dir.name

                    if not gen_root.exists():
                        continue

                    # generation seed level
                    for gen_seed_dir in gen_root.iterdir():
                        if not gen_seed_dir.is_dir():
                            continue

                        try:
                            gen_seed = int(gen_seed_dir.name)
                        except ValueError:
                            gen_seed = gen_seed_dir.name

                        row_dict = {
                            "model_config": model_config,
                            "scale": scale,
                            "pretrain_seed": lm_seed,
                            "fine_tune_seed": fine_tune_seed,
                            "generation_seed": gen_seed,
                            "reward": reward,
                            "reward_seed": reward_seed,
                        }

                        if best_zorro is not None:
                            row_dict["best_zorro"] = best_zorro
                            row_dict["best_blimp"] = best_blimp

                        gen_file = gen_seed_dir / "result_best_reward.csv"
                        if gen_file.exists():
                            row_dict.update(collect_gen_metrics(gen_file))

                        rows.append(row_dict)

                        # ✅ controlled utterance collection
                        if args.collect_utterances:
                            utt_file = gen_seed_dir / "utt_best_reward.csv"
                            if utt_file.exists():
                                df = pd.read_csv(utt_file).sample(
                                    n=args.target_row_num,
                                    replace=True,
                                    random_state=1,
                                )
                                df["model_config"] = model_config
                                df["scale"] = scale
                                df["pretrain_seed"] = lm_seed
                                df["fine_tune_seed"] = fine_tune_seed
                                df["generation_seed"] = gen_seed
                                df["reward"] = reward
                                df["reward_seed"] = reward_seed
                                dfs.append(df)


# -------------------------
# Childes
# -------------------------
def collect_childes(args, rows, dfs):
    childes_root = settings.PATH.result_dir / "childes"

    if not childes_root.exists():
        print(f"Warning: childes directory not found at {childes_root}")
        return

    subdirs = ["response_transcript_clean", "utt_transcript_clean"]

    for sub in subdirs:
        sub_dir = childes_root / sub
        if not sub_dir.exists():
            continue

        row_dict = {
            "model_config": "childes",
            "scale": "childes",
            "pretrain_seed": "none",
            "fine_tune_seed": "none",
            "generation_seed": sub,
            "reward": "childes",
            "reward_seed": "none",
        }

        result_file = sub_dir / "result.csv"
        if result_file.exists():
            row_dict.update(collect_gen_metrics(result_file))

        rows.append(row_dict)

        # ✅ controlled utterance collection
        if args.collect_utterances:
            utt_file = sub_dir / "utt.csv"
            if utt_file.exists():
                df = pd.read_csv(utt_file).sample(n=args.target_row_num, replace=True, random_state=1)
                df["model_config"] = "childes"
                df["scale"] = "childes"
                df["pretrain_seed"] = "none"
                df["fine_tune_seed"] = "none"
                df["generation_seed"] = sub
                df["reward"] = "none"
                df["reward_seed"] = "none"
                dfs.append(df)


# -------------------------
# Main
# -------------------------
def main():
    args = get_args()

    rows, dfs = [], []

    output_dir = Path(args.output_dir) if args.output_dir else settings.PATH.result_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    collect_baseline(args, rows, dfs)
    collect_ppo(args, rows, dfs)
    collect_childes(args, rows, dfs)

    results_df = pd.DataFrame(rows)

    if args.collect_utterances and dfs:
        final_df = pd.concat(dfs, ignore_index=True)
        final_df.to_excel(output_dir / f"sampled_{args.target_row_num}.xlsx", index=False)
        print(f"Saved utterances to {output_dir}/sampled_{args.target_row_num}.xlsx")
    else:
        print("Skipping utterance export")

    results_df.to_csv(output_dir / "result.csv", index=False)
    print(f"Saved results to {output_dir}/result.csv")


if __name__ == "__main__":
    main()
