#!/usr/bin/env python3

import argparse
from pathlib import Path

import pandas as pd
from tqdm import tqdm

from pkg import settings
from pkg.rlhf.eval.collect import collect_babylm_metrics, collect_gen_metrics, extract_reward, extract_scale

DEFAULT_MODEL_CONFIGS = [
    "1e5_entropy_001_lm_loss_001_target_6",
    "1e6_entropy_001_lm_loss_001_target_6",
    "1e7_entropy_001_lm_loss_001_target_6",
]

DEFAULT_SEEDS = [2, 3, 123, 999, 1024]


def get_args():
    p = argparse.ArgumentParser(description="Collect PPO evaluation results")

    p.add_argument(
        "--model-configs",
        nargs="+",
        default=DEFAULT_MODEL_CONFIGS,
        help="List of model configs",
    )

    p.add_argument(
        "--seeds",
        nargs="+",
        type=int,
        default=DEFAULT_SEEDS,
        help="Random seeds",
    )

    p.add_argument(
        "--target-row-num",
        type=int,
        default=100,
        help="Number of sampled utterances per run",
    )

    p.add_argument(
        "--scales",
        nargs="+",
        default=["1e5", "1e6", "1e7"],
        help="Keep only these scales",
    )

    p.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Override output directory",
    )

    return p.parse_args()


def collect_baseline(args, rows, dfs):
    """Collect results from baseline directory: baseline/{scale}/{seed}/"""
    baseline_root = settings.PATH.result_dir / "baseline"

    if not baseline_root.exists():
        print(f"Warning: baseline directory not found at {baseline_root}")
        return

    for scale in tqdm(args.scales):
        scale_dir = baseline_root / scale
        if not scale_dir.exists():
            continue

        for seed_dir in scale_dir.iterdir():
            if not seed_dir.is_dir():
                continue

            try:
                seed = int(seed_dir.name)
            except ValueError:
                continue

            if seed not in args.seeds:
                continue

            row_dict = {
                "model_config": "baseline",
                "scale": scale,
                "seed": seed,
                "reward": "baseline",
            }

            # -----------------------
            # Grammar metrics (optional)
            # -----------------------
            metrics_file = seed_dir / "benchmark.pkl"
            if metrics_file.exists():
                best_zorro, best_blimp = collect_babylm_metrics(metrics_file)
                row_dict["best_zorro"] = best_zorro
                row_dict["best_blimp"] = best_blimp

            # -----------------------
            # Generation metrics
            # -----------------------
            gen_file = seed_dir / "result.csv"
            if gen_file.exists():
                gen_metrics = collect_gen_metrics(gen_file)
                row_dict.update(gen_metrics)

            rows.append(row_dict)

            # -----------------------
            # Sample utterances
            # -----------------------
            utt_file = seed_dir / "utt.csv"
            if utt_file.exists():
                df = pd.read_csv(utt_file)
                df = df.sample(
                    n=args.target_row_num,
                    replace=True,
                    random_state=1,
                )
                df["model_config"] = "baseline"
                df["scale"] = scale
                df["seed"] = seed
                df["reward"] = "baseline"
                dfs.append(df)


def collect_ppo(args, rows, dfs):
    """Collect results from ppo directory: ppo/{model_config}/{seed}/{reward_subdir}/"""
    for model_config in args.model_configs:
        scale = extract_scale(model_config)

        if scale not in set(args.scales):
            continue

        for seed in tqdm(args.seeds):
            model_root = settings.PATH.model_dir / "ppo" / model_config / str(seed)
            gen_root = settings.PATH.result_dir / "ppo" / model_config / str(seed)

            if not model_root.exists():
                continue

            for sub_dir in Path(model_root).iterdir():
                if not sub_dir.is_dir():
                    continue

                reward = extract_reward(sub_dir)

                row_dict = {
                    "model_config": model_config,
                    "scale": scale,
                    "seed": seed,
                    "reward": reward,
                }

                # -----------------------
                # Grammar metrics (optional)
                # -----------------------
                metrics_file = sub_dir / "best_reward" / "metrics.pkl"
                if metrics_file.exists():
                    best_zorro, best_blimp = collect_babylm_metrics(metrics_file)
                    row_dict["best_zorro"] = best_zorro
                    row_dict["best_blimp"] = best_blimp

                # -----------------------
                # Generation metrics (optional)
                # -----------------------
                gen_subdir = gen_root / sub_dir.name
                gen_file = gen_subdir / "result_best_reward.csv"
                if gen_file.exists():
                    gen_metrics = collect_gen_metrics(gen_file)
                    row_dict.update(gen_metrics)

                rows.append(row_dict)

                # -----------------------
                # Sample utterances (optional)
                # -----------------------
                utt_file = gen_subdir / "utt_best_reward.csv"
                if utt_file.exists():
                    df = pd.read_csv(utt_file)
                    df = df.sample(
                        n=args.target_row_num,
                        replace=True,
                        random_state=1,
                    )
                    df["model_config"] = model_config
                    df["scale"] = scale
                    df["seed"] = seed
                    df["reward"] = reward
                    dfs.append(df)


def main():
    args = get_args()

    rows = []
    dfs = []

    output_dir = Path(args.output_dir) if args.output_dir else settings.PATH.result_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    collect_baseline(args, rows, dfs)
    collect_ppo(args, rows, dfs)

    # -----------------------
    # Final outputs
    # -----------------------
    results_df = pd.DataFrame(rows)
    final_df = pd.concat(dfs, ignore_index=True) if dfs else pd.DataFrame()

    results_df.to_csv(output_dir / "result.csv", index=False)
    final_df.to_excel(output_dir / f"sampled_{args.target_row_num}.xlsx", index=False)
    print(f"Saved results to {output_dir}/result.csv")
    print(f"Saved results to {output_dir}/sampled_{args.target_row_num}.xlsx")


if __name__ == "__main__":
    main()
