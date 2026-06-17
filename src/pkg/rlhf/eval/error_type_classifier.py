"""Centralized helper for the grammatical error-type classifier.

This is the *single source of truth* for the multi-label error-type classifier. It
bundles three concerns, kept as separate, independently testable functions, so the
label set can be edited without touching the training loop:

    1. TAXONOMY SPEC      -- the canonical multi-label set, alias map, and the
                             label <-> multi-hot encode/decode helpers. Column
                             names (``clf_<label>``) are derived from here.
    2. DATA LOADING+STATS -- glob/concat the annotated CSVs, keep labeled child
                             rows, build input text (+ optional context), encode
                             targets to a multi-hot matrix, grouped-by-transcript
                             split, and the descriptive-statistics deliverable.
    3. PREDICTION ADAPTER -- map model probabilities into the wide pipeline row
                             format (one binary ``clf_<label>`` column per class,
                             paralleling the LLM-judge's ``judge_<label>``), plus
                             multi-label metric computation.

Design choices (locked in Phase A):
    - Target:   ``labels`` column, comma-separated, MULTI-LABEL, 12 classes.
    - Text:     ``speaker_code + transcript_clean`` for [CHI] rows (+ optional
                preceding-utterance context), mirroring the existing grammaticality
                pipeline.
    - Split:    grouped by ``transcript_file`` (no transcript leaks across splits).
    - Output:   wide multi-hot ``clf_<label>`` columns, mirroring ``judge_<label>``.

This module depends only on pandas / numpy / scikit-learn so the taxonomy, data,
and adapter stay unit-testable without torch/transformers (which live in the
training script).
"""

from __future__ import annotations

import glob
import os
from collections import Counter
from itertools import combinations

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# 1. TAXONOMY SPEC  (single source of truth for the label space)
# ---------------------------------------------------------------------------
#
# Canonical, ordered label list. Order drives the column order of the multi-hot
# target matrix and the ``clf_<label>`` output columns. These are the 12 labels
# observed in the annotated data (``annotated_childes_db.csv``). Edit *here* to
# change the label space -- nothing in the data loader or training loop hardcodes
# labels.

ERROR_TYPE_LABELS: list[str] = [
    "determiner",
    "subject",
    "verb",
    "auxiliary",
    "other",
    "tense_aspect",
    "object",
    "present_progressive",
    "preposition",
    "sv_agreement",
    "possessive",
    "plural",
]

# Aliases -> canonical spelling. Lets us reconcile the LLM-judge taxonomy (which
# uses "progressive", from grammaticality.utils.ERR_PROGRESSIVE) with this data's
# "present_progressive" without editing either call site. Add raw spellings here.
LABEL_ALIASES: dict[str, str] = {
    "progressive": "present_progressive",
    "present progressive": "present_progressive",
    "sv-agreement": "sv_agreement",
    "subject-verb agreement": "sv_agreement",
    "tense": "tense_aspect",
    "tense/aspect": "tense_aspect",
    "aux": "auxiliary",
    "det": "determiner",
}

# Column-name conventions for the wide output (parallels judge_<label>).
COL_PREFIX = "clf_"
LABEL_COLUMNS: list[str] = [f"{COL_PREFIX}{label}" for label in ERROR_TYPE_LABELS]

# Field names reused from the annotated data / existing pipeline conventions.
UTT_COLUMN_DEFAULT = "transcript_clean"
SPEAKER_COLUMN = "speaker_code"
LABELS_COLUMN = "labels"
TRANSCRIPT_COLUMN = "transcript_file"
AGE_COLUMN = "age"
CHILD_SPEAKER_CODE = "[CHI]"


def normalize_label(raw: str) -> str:
    """Lower/strip a raw label token and resolve aliases to canonical spelling."""
    norm = str(raw).strip().lower()
    norm = LABEL_ALIASES.get(norm, norm)
    return norm


def parse_label_cell(cell: object) -> list[str]:
    """Parse one ``labels`` cell -> sorted-by-taxonomy list of canonical labels.

    Empty / NaN cells return ``[]`` (i.e. no error labeled). Comma is the only
    delimiter observed in the data; unknown labels are kept (caller may warn).
    """
    if cell is None or (isinstance(cell, float) and np.isnan(cell)):
        return []
    text = str(cell).strip()
    if not text or text.lower() in {"nan", "none"}:
        return []
    out: list[str] = []
    for tok in text.split(","):
        tok = normalize_label(tok)
        if tok and tok not in out:
            out.append(tok)
    # stable order following the taxonomy
    order = {l: i for i, l in enumerate(ERROR_TYPE_LABELS)}
    return sorted(out, key=lambda l: order.get(l, len(order)))


def labels_to_multihot(label_list: list[str]) -> np.ndarray:
    """Encode a list of canonical labels into a multi-hot vector over the taxonomy."""
    vec = np.zeros(len(ERROR_TYPE_LABELS), dtype=np.int64)
    index = {l: i for i, l in enumerate(ERROR_TYPE_LABELS)}
    for lab in label_list:
        if lab in index:
            vec[index[lab]] = 1
    return vec


def multihot_to_labels(vec) -> list[str]:
    """Decode a multi-hot/boolean vector back into the list of canonical labels."""
    return [ERROR_TYPE_LABELS[i] for i, v in enumerate(vec) if int(v) == 1]


def unknown_labels_in(label_lists: list[list[str]]) -> set[str]:
    """Return any labels present in the data that are not in the taxonomy spec."""
    known = set(ERROR_TYPE_LABELS)
    seen = {lab for lst in label_lists for lab in lst}
    return seen - known


# ---------------------------------------------------------------------------
# 2. DATA LOADING + DESCRIPTIVE STATISTICS
# ---------------------------------------------------------------------------


def load_annotated_dir(
    data_dir: str,
    utt_column: str = UTT_COLUMN_DEFAULT,
    speaker_code: str | None = CHILD_SPEAKER_CODE,
    require_labels: bool = True,
) -> pd.DataFrame:
    """Glob + concat all CSVs in ``data_dir`` and return the labeled child subset.

    Missing/empty utterances are dropped (logged). When ``require_labels`` is True
    only rows with a non-empty ``labels`` cell are kept (the error-type training
    set). A ``__source_file`` column records provenance.
    """
    paths = sorted(glob.glob(os.path.join(data_dir, "*.csv")))
    if not paths:
        raise FileNotFoundError(f"No CSV files found in data directory: {data_dir}")

    frames = []
    for p in paths:
        d = pd.read_csv(p)
        d["__source_file"] = os.path.basename(p)
        frames.append(d)
    df = pd.concat(frames, ignore_index=True)
    n_total = len(df)

    for col in (utt_column, LABELS_COLUMN):
        if col not in df.columns:
            raise ValueError(f"Column '{col}' not found. Available: {list(df.columns)}")

    if speaker_code is not None and SPEAKER_COLUMN in df.columns:
        df = df[df[SPEAKER_COLUMN] == speaker_code].copy()

    # Drop missing/empty utterances.
    utt = df[utt_column].fillna("").astype(str).str.strip()
    empty_utt = (utt == "").sum()
    df = df[utt != ""].copy()

    if require_labels:
        has_labels = df[LABELS_COLUMN].notna() & (
            df[LABELS_COLUMN].astype(str).str.strip().replace({"nan": "", "None": ""}) != ""
        )
        df = df[has_labels].copy()

    print(
        f"[load] {len(paths)} file(s), {n_total} rows -> {len(df)} kept "
        f"(speaker={speaker_code}, require_labels={require_labels}, "
        f"dropped {empty_utt} empty utterances)"
    )
    return df.reset_index(drop=True)


def build_input_text(
    df: pd.DataFrame,
    utt_column: str = UTT_COLUMN_DEFAULT,
    context_length: int = 0,
    sep_token: str = " [SEP] ",
) -> pd.Series:
    """Build model input text: ``speaker_code + utterance`` (+ optional context).

    Context is the ``context_length`` preceding utterances *within the same
    transcript* (ordered by row position), prepended -- mirroring
    ``load_annotated_childes_data_with_context`` in the grammaticality pipeline.
    Requires the full (unfiltered) transcript for real context; with the labeled
    subset only, context is taken from preceding labeled rows of the same file.
    """
    speaker = (
        df[SPEAKER_COLUMN].astype(str) + " " if SPEAKER_COLUMN in df.columns else ""
    )
    base = speaker + df[utt_column].astype(str).str.strip()

    if context_length <= 0 or TRANSCRIPT_COLUMN not in df.columns:
        return base

    texts = []
    by_file = {f: sub.index.tolist() for f, sub in df.groupby(TRANSCRIPT_COLUMN)}
    pos_in_file = {}
    for _f, idxs in by_file.items():
        for k, idx in enumerate(idxs):
            pos_in_file[idx] = (idxs, k)

    for idx in df.index:
        idxs, k = pos_in_file[idx]
        ctx_parts = []
        for j in range(max(0, k - context_length), k):
            ctx_parts.append(base.loc[idxs[j]])
        sentence = base.loc[idx]
        if ctx_parts:
            sentence = sep_token.join(ctx_parts) + sep_token + sentence
        texts.append(sentence)
    return pd.Series(texts, index=df.index)


def encode_targets(df: pd.DataFrame) -> tuple[np.ndarray, list[list[str]]]:
    """Return (multi-hot matrix [n, n_labels], list of per-row canonical labels)."""
    label_lists = [parse_label_cell(c) for c in df[LABELS_COLUMN]]
    unknown = unknown_labels_in(label_lists)
    if unknown:
        print(f"[warn] labels not in taxonomy spec (ignored in targets): {sorted(unknown)}")
    matrix = np.vstack([labels_to_multihot(lst) for lst in label_lists]) if label_lists else np.zeros(
        (0, len(ERROR_TYPE_LABELS)), dtype=np.int64
    )
    return matrix, label_lists


def describe_label_distribution(df: pd.DataFrame, output_dir: str | None = None) -> pd.DataFrame:
    """Compute the label-distribution deliverable; optionally write CSVs.

    Returns the per-label stats DataFrame. When ``output_dir`` is given, writes
    ``label_distribution.csv``, ``label_cooccurrence.csv`` and
    ``labels_per_utterance.csv`` there.
    """
    _, label_lists = encode_targets(df)
    n_utt = len(label_lists)
    counts = Counter(lab for lst in label_lists for lab in lst)
    total_occ = sum(counts.values())

    rows = []
    for label in ERROR_TYPE_LABELS:
        c = counts.get(label, 0)
        rows.append(
            {
                "label": label,
                "count": c,
                "prop_of_utterances": c / n_utt if n_utt else 0.0,
                "prop_of_occurrences": c / total_occ if total_occ else 0.0,
                "sparse_flag": c < 50,
            }
        )
    stats = pd.DataFrame(rows).sort_values("count", ascending=False).reset_index(drop=True)

    n_per_utt = pd.Series([len(lst) for lst in label_lists]).value_counts().sort_index()
    n_per_utt = n_per_utt.rename_axis("n_labels").reset_index(name="n_utterances")

    co = Counter()
    for lst in label_lists:
        for a, b in combinations(sorted(set(lst)), 2):
            co[(a, b)] += 1
    cooc = pd.DataFrame(
        [{"label_a": a, "label_b": b, "count": c} for (a, b), c in co.most_common()]
    )

    print(f"\n=== Label distribution ({n_utt} utterances, {len(ERROR_TYPE_LABELS)} classes) ===")
    print(stats.to_string(index=False))
    sparse = stats.loc[stats["sparse_flag"], "label"].tolist()
    if sparse:
        print(f"[flag] sparse classes (<50 examples): {sparse}")
    print(f"Labels per utterance:\n{n_per_utt.to_string(index=False)}")

    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
        stats.to_csv(os.path.join(output_dir, "label_distribution.csv"), index=False)
        cooc.to_csv(os.path.join(output_dir, "label_cooccurrence.csv"), index=False)
        n_per_utt.to_csv(os.path.join(output_dir, "labels_per_utterance.csv"), index=False)
        print(f"[saved] label-distribution stats -> {output_dir}")
    return stats


def grouped_split_indices(
    df: pd.DataFrame,
    num_cv_folds: int = 5,
    group_column: str = TRANSCRIPT_COLUMN,
    seed: int = 8,
) -> list[tuple[np.ndarray, np.ndarray]]:
    """Yield (train_idx, test_idx) per fold, grouped so a transcript never splits.

    Mirrors the leakage-avoidance intent of ``create_cv_folds`` in the
    grammaticality pipeline, implemented with scikit-learn's ``GroupKFold``.
    """
    from sklearn.model_selection import GroupKFold

    if group_column not in df.columns:
        raise ValueError(f"Group column '{group_column}' not found for grouped split.")
    groups = df[group_column].values
    n_splits = min(num_cv_folds, len(np.unique(groups)))
    gkf = GroupKFold(n_splits=n_splits)
    x_dummy = np.zeros(len(df))
    return list(gkf.split(x_dummy, groups=groups))


def grouped_train_val_split(
    df: pd.DataFrame,
    val_proportion: float = 0.2,
    group_column: str = TRANSCRIPT_COLUMN,
    seed: int = 8,
) -> tuple[np.ndarray, np.ndarray]:
    """Single grouped train/val split (positional indices), no transcript leakage."""
    rng = np.random.default_rng(seed)
    groups = df[group_column].values if group_column in df.columns else np.arange(len(df))
    unique = list(pd.unique(groups))
    rng.shuffle(unique)
    target = int(len(df) * val_proportion)
    val_groups, n = set(), 0
    for g in unique:
        if n >= target:
            break
        val_groups.add(g)
        n += int((groups == g).sum())
    pos = np.arange(len(df))
    val_mask = np.array([g in val_groups for g in groups])
    return pos[~val_mask], pos[val_mask]


# ---------------------------------------------------------------------------
# 3. PREDICTION ADAPTER + METRICS  (-> wide CSV row format, parallels judge)
# ---------------------------------------------------------------------------

PREDICTION_COLUMNS: list[str] = (
    LABEL_COLUMNS  # one binary clf_<label> column per class
    + [
        f"{COL_PREFIX}labels",  # comma-joined predicted labels (matches `labels`)
        f"{COL_PREFIX}is_grammatical",  # 1 if no error predicted else 0
        f"{COL_PREFIX}model",  # provenance
    ]
)


def probs_to_rows(probs: np.ndarray, model_name: str, thresholds: np.ndarray | float = 0.5) -> list[dict]:
    """Map a [n, n_labels] probability matrix to wide ``clf_*`` rows."""
    probs = np.asarray(probs)
    if isinstance(thresholds, (int, float)):
        thr = np.full(probs.shape[1], float(thresholds))
    else:
        thr = np.asarray(thresholds)
    preds = (probs >= thr).astype(int)

    rows = []
    for i in range(probs.shape[0]):
        row = {f"{COL_PREFIX}{label}": int(preds[i, j]) for j, label in enumerate(ERROR_TYPE_LABELS)}
        pred_labels = multihot_to_labels(preds[i])
        row[f"{COL_PREFIX}labels"] = ", ".join(pred_labels)
        row[f"{COL_PREFIX}is_grammatical"] = 0 if pred_labels else 1
        row[f"{COL_PREFIX}model"] = model_name
        rows.append(row)
    return rows


def predictions_dataframe(
    df: pd.DataFrame, probs: np.ndarray, model_name: str, thresholds: np.ndarray | float = 0.5
) -> pd.DataFrame:
    """Append ``clf_*`` columns to ``df`` (row-aligned), returning the wide frame."""
    rows = probs_to_rows(probs, model_name, thresholds)
    pred_df = pd.DataFrame(rows, columns=PREDICTION_COLUMNS)
    out = df.reset_index(drop=True).copy()
    if "row_id" not in out.columns:
        out.insert(0, "row_id", out.index)
    return pd.concat([out, pred_df.reset_index(drop=True)], axis=1)


def compute_multilabel_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    """Per-class + micro/macro/samples P/R/F1 for multi-label predictions."""
    from sklearn.metrics import f1_score, precision_recall_fscore_support

    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    p, r, f, support = precision_recall_fscore_support(
        y_true, y_pred, average=None, zero_division=0, labels=range(len(ERROR_TYPE_LABELS))
    )
    per_class = {
        ERROR_TYPE_LABELS[i]: {
            "precision": float(p[i]),
            "recall": float(r[i]),
            "f1": float(f[i]),
            "support": int(support[i]),
        }
        for i in range(len(ERROR_TYPE_LABELS))
    }
    # macro over non-sparse classes only (secondary, sparsity-robust number).
    nonsparse = [i for i in range(len(ERROR_TYPE_LABELS)) if support[i] >= 50]
    macro_f1_nonsparse = float(np.mean([f[i] for i in nonsparse])) if nonsparse else float("nan")

    return {
        "micro_f1": float(f1_score(y_true, y_pred, average="micro", zero_division=0)),
        "macro_f1": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        "samples_f1": float(f1_score(y_true, y_pred, average="samples", zero_division=0)),
        "macro_f1_nonsparse": macro_f1_nonsparse,
        "per_class": per_class,
    }
