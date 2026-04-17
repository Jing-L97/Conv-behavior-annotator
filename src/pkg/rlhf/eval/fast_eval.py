import hashlib
import math
from collections import Counter, defaultdict

import numpy as np
import pandas as pd
import torch
from sentence_transformers import SentenceTransformer

from pkg.rlhf.eval.gen_util import FeatureExtractor, SemEnt, preprocess  # adjust if needed

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ----------------------------
# Grammaticality scoring
# ----------------------------
def compute_entropy_reg(utterances: list[str]) -> float:
    """entropy_reg = - sum_w p(w) * log(p(w))
    where p(w) = count(w) / total_words over all generated utterances.
    Uses natural log.
    """
    if not utterances:
        return float("nan")

    tokens = []
    for u in utterances:
        if not u:
            continue
        tokens.extend(u.lower().split())

    total = len(tokens)
    if total == 0:
        return float("nan")

    counts = Counter(tokens)
    probs = np.fromiter((c / total for c in counts.values()), dtype=np.float64)

    return float(-(probs * np.log(probs)).sum())


# ============================================================
# FAST + EXACT diversity helpers (closed-form mean cosine)
# ============================================================


def _mean_cosine_closed_form(sum_norm2: float, n: int) -> float:
    # mean over ordered pairs i != j
    return (sum_norm2 - n) / (n * (n - 1))


def mean_pairwise_cosine_dense_closed_form(vectors: list[np.ndarray], eps: float = 1e-12) -> float:
    """Exact mean cosine over i!=j for dense vectors using:
    (||sum x_hat||^2 - n) / (n(n-1))
    """
    valid = [v for v in vectors if isinstance(v, np.ndarray) and v.size > 0 and not np.isnan(v).any()]
    n = len(valid)
    if n < 2:
        return float("nan")

    S = None
    for v in valid:
        v = v.astype(np.float32, copy=False)
        vhat = v / (float(np.linalg.norm(v)) + eps)  # L2 normalize once
        S = vhat.copy() if S is None else (S + vhat)

    sum_norm2 = float(np.dot(S, S))
    return _mean_cosine_closed_form(sum_norm2, n)


def l2_normalize_counter(counter: Counter, eps: float = 1e-12) -> dict[str, float]:
    norm2 = sum(float(c) * float(c) for c in counter.values())
    if norm2 <= eps:
        return {}
    inv = 1.0 / math.sqrt(norm2)
    return {k: float(v) * inv for k, v in counter.items()}


def mean_pairwise_cosine_sparse_closed_form(vhats: list[dict[str, float]]) -> float:
    """Exact mean cosine over i!=j for sparse L2-normalized vectors (dict feature->value)."""
    vhats = [v for v in vhats if v]
    n = len(vhats)
    if n < 2:
        return float("nan")

    sums = defaultdict(float)
    for v in vhats:
        for k, val in v.items():
            sums[k] += float(val)

    sum_norm2 = sum(val * val for val in sums.values())
    return _mean_cosine_closed_form(sum_norm2, n)


# ============================================================
# FAST dep_div: explicit WL feature vector φ(G) from spaCy docs
# (matches your dependency_tree_to_graph node labels & edges)
# ============================================================


def _hash_label(s: str) -> str:
    return hashlib.md5(s.encode("utf-8")).hexdigest()


def spacy_doc_to_adj_and_labels(doc) -> tuple[list[list[int]], list[str]]:
    """Build adjacency + node labels exactly like your dependency_tree_to_graph:
    - undirected edges between token and head
    - node label: POS + DEP + token.text.lower + is_stop + is_punct
    """
    n = len(doc)
    adj = [[] for _ in range(n)]
    labels = [""] * n

    for i, token in enumerate(doc):
        label_parts = [
            token.pos_,
            token.dep_,
            token.text.lower(),
            str(token.is_stop),
            str(token.is_punct),
        ]
        labels[i] = "_".join(label_parts)

        h = token.head.i
        if 0 <= h < n and h != i:
            adj[i].append(h)
            adj[h].append(i)

    return adj, labels


def wl_phi_from_spacy_doc(doc, n_iter: int = 5) -> Counter:
    """Explicit WL feature vector φ(G): histogram of WL node labels across iterations.
    Computed once per doc (no pairwise WL kernel).
    """
    adj, cur = spacy_doc_to_adj_and_labels(doc)
    feats = Counter(cur)

    for _ in range(n_iter):
        new = []
        for i in range(len(cur)):
            neigh = sorted(cur[j] for j in adj[i])
            signature = cur[i] + "|" + "|".join(neigh)
            new.append(_hash_label(signature))
        cur = new
        feats.update(cur)

    return feats


# ----------------------------
# Grammaticality scoring with eval_batch_size
# ----------------------------
def score_in_batches(utterances: list[str], score_fn, model, tokenizer, eval_batch_size: int) -> list:
    """Run a grammaticality scoring function in batches of eval_batch_size."""
    scores = []
    for start in range(0, len(utterances), eval_batch_size):
        batch = utterances[start : start + eval_batch_size]
        batch_scores = score_fn(batch, model, tokenizer)
        scores.extend(list(batch_scores))
    return scores


# ----------------------------
# Feature-based metrics
# ----------------------------
def compute_feature_metrics_for_utts(
    utterances: list[str],
    feature_extractor: FeatureExtractor,
    sent_model: SentenceTransformer,
    feature_list: list[str],
) -> dict:
    """Compute turn-level metrics (word/syn) per utterance and aggregate."""
    print(f"[Features] computing features for {len(utterances)} utterances", flush=True)
    if len(utterances) == 0:
        return pd.DataFrame(), {}

    cleaned = [preprocess(u) for u in utterances]

    df = pd.DataFrame(
        {
            "cleaned": cleaned,
            "speaker": ["CHI"] * len(cleaned),
        }
    )

    print("[Features] spaCy parsing + dependency graphs", flush=True)
    # doc_lst, lemma_lst, dep_graphs = feature_extractor.extract_doc(df["cleaned"], feature_list)
    doc_lst, lemma_lst = feature_extractor.extract_doc(df["cleaned"], feature_list, return_dep_graphs=False)
    dep_graphs = None  # not used anymore
    print("[Features] sentence embeddings", flush=True)
    word_vectors = feature_extractor.extract_vec(df["cleaned"], feature_list)

    # --- Turn-level features (per utterance) ---
    per_utt_features = [
        "freq",
        "conc",
        "word_len",
        "sent_len",
        "ttr",
        "distinct_2",
        "distinct_3",
        "tree_depth",
        "clause",
        "lex_den",
        "func_den",
        "func_den_new",
        "pp_den",
        "non_word_rate",
        "non_word_type_rate",
    ]
    per_utt_features = [f for f in per_utt_features if f in feature_list]

    print(f"[Features] computing turn-level features for {len(cleaned)} utterances", flush=True)
    rows = []
    for i in range(len(cleaned)):
        if i % 100 == 0:
            print(f"[Features] turn-level features {i}/{len(cleaned)}", flush=True)

        rows.append(
            feature_extractor.get_turn_fea(
                df["cleaned"].iloc[i],
                doc_lst[i] if i < len(doc_lst) else "",
                lemma_lst[i] if i < len(lemma_lst) else "",
                per_utt_features,
                ["VERB", "NOUN", "ADV", "ADJ", "PROPN"],  # content_POS
                ["VERB", "NOUN", "PROPN", "ADV", "ADJ", "PRON", "INTJ"],  # target_POS
                "word",  # func_header
            )
        )

    # turn_df: one row per utterance, one column per turn-level feature
    turn_df = pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()

    # Aggregate means for the summary results dict
    agg = {}
    for col in turn_df.columns:
        if pd.api.types.is_numeric_dtype(turn_df[col]):
            agg[f"{col}_mean"] = float(turn_df[col].mean())

    # --- Set-level diversity features (FAST + EXACT; no pairwise matrices) ---
    indices = list(range(len(cleaned)))  # keep for SemEnt below

    # sem_div: closed-form mean cosine on L2-normalized embeddings
    if "sem_div" in feature_list:
        mean_cos = mean_pairwise_cosine_dense_closed_form(word_vectors)
        agg["sem_div_set"] = float("nan") if np.isnan(mean_cos) else float(1.0 - mean_cos)

    # lemma_div: global vocab (implicit) + per-utt count vectors + L2 normalize once + closed-form
    if "lemma_div" in feature_list:
        lemma_vhats = []
        for lem in lemma_lst:
            if isinstance(lem, list) and len(lem) > 0:
                lemma_vhats.append(l2_normalize_counter(Counter(lem)))
        mean_cos = mean_pairwise_cosine_sparse_closed_form(lemma_vhats)
        agg["lemma_div_set"] = float("nan") if np.isnan(mean_cos) else float(1.0 - mean_cos)

    # dep_div: explicit WL feature vector φ(G) computed once per spaCy doc + L2 normalize + closed-form
    if "dep_div" in feature_list:
        wl_vhats = []
        for doc in doc_lst:
            if doc is None or (isinstance(doc, float) and np.isnan(doc)):
                continue
            phi = wl_phi_from_spacy_doc(doc, n_iter=5)
            wl_vhats.append(l2_normalize_counter(phi))
        mean_cos = mean_pairwise_cosine_sparse_closed_form(wl_vhats)
        agg["dep_div_set"] = float("nan") if np.isnan(mean_cos) else float(1.0 - mean_cos)

    # --- Semantic entropy (set-level) ---
    try:
        semEnt = SemEnt(word_vectors, indices=indices)
        sem_ent, k = semEnt.compute_diversity()
        agg["sem_ent_set"] = float(sem_ent)
        agg["sem_ent_k_set"] = float(k)
    except Exception:
        pass

    agg["num_scored_utts"] = int(len(cleaned))
    agg["mean_chars"] = float(np.mean([len(u) for u in utterances]))

    # turn_df may have fewer columns than per_utt_features if some weren't computed;
    # we return it as-is and let the caller align it with utterances/scores.
    return turn_df, agg


# ----------------------------
# Orchestration
# ----------------------------
def build_feature_list(fea_set):
    fea_dict = {
        "word": [
            "freq",
            "conc",
            "word_len",
            "sent_len",
            "ttr",
            "non_word_type_rate",
            "non_word_rate",
            "distinct_2",
            "distinct_3",
        ],
        "syn": ["tree_depth", "clause", "pp_den", "lex_den", "func_den", "func_den_new"],
        "div": ["lemma_div", "dep_div", "sem_div"],
        "semEnt": ["sem_ent"],
    }
    flat = []
    for key in fea_set:
        flat.extend(fea_dict.get(key, []))
    seen = set()
    out = []
    for f in flat:
        if f not in seen:
            out.append(f)
            seen.add(f)
    return out
