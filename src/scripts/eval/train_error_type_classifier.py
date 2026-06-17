#!/usr/bin/env python3
"""Train the multi-label grammatical error-type classifier.

Fine-tunes a DeBERTa-style encoder (``AutoModelForSequenceClassification`` with a
multi-label head, ``BCEWithLogitsLoss`` + class weights) on the human-annotated
child utterances. Splitting is grouped by ``transcript_file`` so no transcript
leaks across train/test. Emits predictions in the wide ``clf_<label>`` schema that
mirrors the LLM-judge method.

All taxonomy / data-loading / adapter logic lives in
``pkg.rlhf.eval.error_type_classifier`` -- this script only handles the model, the
training loop, and the CLI. The label set is read from that spec (never hardcoded
in the loop).

Example
-------
    python -m scripts.eval.train_error_type_classifier \
        --data_dir   sample_data \
        --output_dir runs/error_type_deberta \
        --model      microsoft/deberta-v3-base \
        --context_length 0 --num_cv_folds 5

Smoke test (handful of rows, ~2 optimizer steps, asserts clf_* columns appear):
    python -m scripts.eval.train_error_type_classifier --data_dir sample_data --smoke_test
"""

import argparse
import json
import os

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset

from pkg.rlhf.eval.error_type_classifier import (
    ERROR_TYPE_LABELS,
    PREDICTION_COLUMNS,
    build_input_text,
    compute_multilabel_metrics,
    describe_label_distribution,
    encode_targets,
    grouped_split_indices,
    load_annotated_dir,
    predictions_dataframe,
)

DEFAULT_MODEL = "microsoft/deberta-v3-base"


# ----------------------------
# Torch dataset
# ----------------------------
class ErrorTypeDataset(Dataset):
    """Tokenizes on the fly; targets are float multi-hot vectors for BCE."""

    def __init__(self, texts, targets, tokenizer, max_len=128):
        self.texts = list(texts)
        self.targets = np.asarray(targets, dtype=np.float32)
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, i):
        enc = self.tokenizer(
            self.texts[i], truncation=True, max_length=self.max_len, padding="max_length", return_tensors="pt"
        )
        item = {k: v.squeeze(0) for k, v in enc.items()}
        item["labels"] = torch.tensor(self.targets[i], dtype=torch.float32)
        return item


# ----------------------------
# Class weights for imbalanced multi-label BCE (pos_weight per class)
# ----------------------------
def compute_pos_weights(y: np.ndarray) -> torch.Tensor:
    """pos_weight = (#neg / #pos) per class, clipped to keep rare classes sane."""
    y = np.asarray(y)
    pos = y.sum(axis=0)
    neg = y.shape[0] - pos
    with np.errstate(divide="ignore", invalid="ignore"):
        w = np.where(pos > 0, neg / np.maximum(pos, 1), 1.0)
    return torch.tensor(np.clip(w, 1.0, 50.0), dtype=torch.float32)


# ----------------------------
# Train / evaluate one fold
# ----------------------------
def train_one_fold(
    train_texts,
    train_y,
    val_texts,
    val_y,
    model_name,
    output_dir,
    epochs=5,
    batch_size=16,
    lr=2e-5,
    max_steps=None,
    device=None,
):
    from transformers import AutoModelForSequenceClassification, AutoTokenizer

    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)
    model = AutoModelForSequenceClassification.from_pretrained(
        model_name,
        num_labels=len(ERROR_TYPE_LABELS),
        problem_type="multi_label_classification",
    ).to(device)

    train_ds = ErrorTypeDataset(train_texts, train_y, tokenizer)
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)

    pos_weight = compute_pos_weights(train_y).to(device)
    loss_fct = torch.nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    optim = torch.optim.AdamW(model.parameters(), lr=lr)

    model.train()
    step = 0
    for epoch in range(epochs):
        for batch in train_loader:
            labels = batch.pop("labels").to(device)
            batch = {k: v.to(device) for k, v in batch.items()}
            logits = model(**batch).logits
            loss = loss_fct(logits, labels)
            optim.zero_grad()
            loss.backward()
            optim.step()
            step += 1
            if step % 10 == 0 or (max_steps and step >= max_steps):
                print(f"  epoch {epoch} step {step} loss {loss.item():.4f}", flush=True)
            if max_steps and step >= max_steps:
                break
        if max_steps and step >= max_steps:
            break

    # --- Validation probabilities ---
    val_probs = predict_probs(model, tokenizer, val_texts, batch_size, device)
    return model, tokenizer, val_probs


def predict_probs(model, tokenizer, texts, batch_size, device) -> np.ndarray:
    ds = ErrorTypeDataset(texts, np.zeros((len(texts), len(ERROR_TYPE_LABELS))), tokenizer)
    loader = DataLoader(ds, batch_size=batch_size, shuffle=False)
    model.eval()
    out = []
    with torch.no_grad():
        for batch in loader:
            batch.pop("labels")
            batch = {k: v.to(device) for k, v in batch.items()}
            logits = model(**batch).logits
            out.append(torch.sigmoid(logits).cpu().numpy())
    return np.vstack(out) if out else np.zeros((0, len(ERROR_TYPE_LABELS)))


# ----------------------------
# Orchestration
# ----------------------------
def run(args):
    os.makedirs(args.output_dir, exist_ok=True)

    # --- Load + descriptive stats (written BEFORE training) ---
    df = load_annotated_dir(args.data_dir, utt_column=args.utt_column, speaker_code=args.speaker_code)
    describe_label_distribution(df, output_dir=args.output_dir)

    texts = build_input_text(df, utt_column=args.utt_column, context_length=args.context_length).tolist()
    y, _ = encode_targets(df)

    # --- Save the label mapping (single source of truth, persisted) ---
    with open(os.path.join(args.output_dir, "label_mapping.json"), "w") as f:
        json.dump({"labels": ERROR_TYPE_LABELS, "index": {l: i for i, l in enumerate(ERROR_TYPE_LABELS)}}, f, indent=2)

    folds = grouped_split_indices(df, num_cv_folds=args.num_cv_folds)
    if args.smoke_test:
        folds = folds[:1]

    all_metrics = []
    oof_frames = []
    for fold_i, (tr, te) in enumerate(folds):
        print(f"\n=== Fold {fold_i + 1}/{len(folds)} | train={len(tr)} test={len(te)} ===")
        model, tokenizer, test_probs = train_one_fold(
            [texts[i] for i in tr],
            y[tr],
            [texts[i] for i in te],
            y[te],
            model_name=args.model,
            output_dir=args.output_dir,
            epochs=args.epochs,
            batch_size=args.batch_size,
            lr=args.lr,
            max_steps=2 if args.smoke_test else None,
        )

        preds = (test_probs >= args.threshold).astype(int)
        metrics = compute_multilabel_metrics(y[te], preds)
        metrics["fold"] = fold_i
        all_metrics.append(metrics)
        print(
            f"  fold {fold_i}: micro_f1={metrics['micro_f1']:.3f} "
            f"macro_f1={metrics['macro_f1']:.3f} samples_f1={metrics['samples_f1']:.3f}"
        )

        fold_df = df.iloc[te].copy()
        oof_frames.append(predictions_dataframe(fold_df, test_probs, args.model, args.threshold))

        # Save the last fold's model + tokenizer as the deliverable checkpoint.
        if fold_i == len(folds) - 1:
            ckpt_dir = os.path.join(args.output_dir, "model")
            model.save_pretrained(ckpt_dir)
            tokenizer.save_pretrained(ckpt_dir)
            print(f"[saved] model + tokenizer -> {ckpt_dir}")

    # --- Aggregate metrics report ---
    summary = {
        "model": args.model,
        "num_folds": len(all_metrics),
        "micro_f1_mean": float(np.mean([m["micro_f1"] for m in all_metrics])),
        "macro_f1_mean": float(np.mean([m["macro_f1"] for m in all_metrics])),
        "samples_f1_mean": float(np.mean([m["samples_f1"] for m in all_metrics])),
        "macro_f1_nonsparse_mean": float(np.nanmean([m["macro_f1_nonsparse"] for m in all_metrics])),
        "folds": all_metrics,
    }
    with open(os.path.join(args.output_dir, "metrics.json"), "w") as f:
        json.dump(summary, f, indent=2)

    oof = pd.concat(oof_frames, ignore_index=True)
    oof_path = os.path.join(args.output_dir, "predictions_oof.csv")
    oof.to_csv(oof_path, index=False)

    # Smoke-test assertion: the expected wide columns must exist.
    missing = [c for c in PREDICTION_COLUMNS if c not in oof.columns]
    assert not missing, f"Prediction output missing expected columns: {missing}"

    print(f"\n[done] micro_f1={summary['micro_f1_mean']:.3f} macro_f1={summary['macro_f1_mean']:.3f}")
    print(f"[saved] metrics.json, label_mapping.json, predictions_oof.csv -> {args.output_dir}")
    print(f"[ok] prediction schema columns present: {PREDICTION_COLUMNS}")


def get_args():
    p = argparse.ArgumentParser(description="Train multi-label grammatical error-type classifier.")
    p.add_argument("--data_dir", type=str, required=True, help="Directory of annotated CSV(s).")
    p.add_argument("--output_dir", type=str, default="runs/error_type", help="Where to save model/metrics/stats.")
    p.add_argument("--model", type=str, default=DEFAULT_MODEL, help="HF encoder name.")
    p.add_argument("--utt_column", type=str, default="transcript_clean", help="Column with the child utterance.")
    p.add_argument("--speaker_code", type=str, default="[CHI]", help="Speaker code to keep (or '' for all).")
    p.add_argument("--context_length", type=int, default=0, help="# preceding utterances to prepend as context.")
    p.add_argument("--num_cv_folds", type=int, default=5)
    p.add_argument("--epochs", type=int, default=5)
    p.add_argument("--batch_size", type=int, default=16)
    p.add_argument("--lr", type=float, default=2e-5)
    p.add_argument("--threshold", type=float, default=0.5, help="Decision threshold on per-class probability.")
    p.add_argument("--smoke_test", action="store_true", help="Tiny 1-fold, 2-step run to validate the pipeline.")
    args = p.parse_args()
    if args.speaker_code == "":
        args.speaker_code = None
    return args


if __name__ == "__main__":
    run(get_args())
