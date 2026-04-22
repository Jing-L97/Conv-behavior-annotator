import argparse
import os
import pickle
import warnings


def eval_babylm_metrics(ckpt_dir, eval_data_dir=None, eval_batch_size=1024):
    from lm_eval import evaluator

    from pkg.rlhf.eval.grammar_util import (
        DEFAULT_EVAL_METRICS,
    )
    from pkg.rlhf.utilities import (
        parse_babylm_metrics_results,
    )

    model_args = f"pretrained={ckpt_dir},add_bos_token=True"

    if eval_data_dir is not None:
        os.chdir(eval_data_dir)
        print(f"[INFO] Changed working directory to: {eval_data_dir}")

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        out = evaluator.simple_evaluate(
            model="hf",
            model_args=model_args,
            tasks=DEFAULT_EVAL_METRICS,
            batch_size=eval_batch_size,
            device="cuda",
            cache_requests=True,
        )

    return parse_babylm_metrics_results(out)


def eval_models(args):
    all_results = []
    skipped = []

    if args.eval_data_dir is not None:
        os.environ["EVAL_DATA_DIR"] = args.eval_data_dir
        print(f"[INFO] Using evaluation data directory: {args.eval_data_dir}")

    for model_path in args.model_paths:
        if not os.path.isdir(model_path):
            print(f"[SKIP] Non-existing path: {model_path}")
            skipped.append(model_path)
            continue

        pickle_path = os.path.join(args.output_dir, "benchmark.pkl")

        # ── skip check BEFORE heavy deps ───────────────────────────────────────
        if args.skip_existing and os.path.exists(pickle_path):
            print(f"[SKIP] Already exists: {pickle_path}")
            continue

        print(f"Evaluating {model_path}...")

        # only now we run expensive evaluation
        results = eval_babylm_metrics(model_path, args.eval_data_dir)

        results["model"] = model_path
        all_results.append(results)

        metrics = {
            "best_zorro": results["zorro_filtered_childes"],
            "best_blimp": results["blimp_filtered_childes"],
        }

        with open(pickle_path, "wb") as f:
            pickle.dump(metrics, f)

        print(f"[SAVE] Metrics written to {pickle_path}")

    print(f"[DONE] Skipped {len(skipped)} models")


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_paths", type=str, nargs="+", required=True)

    parser.add_argument(
        "--output_dir",
        type=str,
        default=None,
        help="Optional root directory to store evaluation results",
    )

    parser.add_argument(
        "--eval_data_dir",
        type=str,
        default=None,
    )

    parser.add_argument(
        "--skip_existing",
        action="store_true",
        help="Skip evaluation if output pickle already exists",
    )

    return parser.parse_args()


if __name__ == "__main__":
    args = get_args()
    eval_models(args)
