import itertools

import pandas as pd

# =========================
# Config
# =========================
result_path = "/store/scratch/jliu/Feedback/results/result.csv"

scales = [1e5, 1e6, 1e7]
pretrain_seeds = [1, 2, 3]
fine_tune_seeds = [3, 999, 1024]
generation_seeds = [1, 2, 3]

rewards = [
    "is_cr",
    "is_acknowledgement",
    "align_lexical_unigram",
    "align_lexical_bigram",
    "align_syntactic",
    "continuous_align_lexical_unigram",
    "continuous_align_lexical_bigram",
    "continuous_align_syntactic",
    "continuous_align_semantic",
    "align_semantic",
    "sent_warmth",
    "sent_engagement",
    "sent_negativity",
    "sent_supportiveness",
    "sent_approval",
    "sent_caring",
    "sent_curiosity",
]

cols = ["scale", "pretrain_seed", "fine_tune_seed", "generation_seed", "reward"]


# =========================
# Normalization
# =========================
def normalize_df(df):
    df = df.copy()

    df["scale"] = pd.to_numeric(df["scale"], errors="coerce")

    for c in ["pretrain_seed", "fine_tune_seed", "generation_seed"]:
        df[c] = pd.to_numeric(df[c], errors="coerce").astype("Int64")

    df["reward"] = df["reward"].astype(str).str.strip().str.lower()

    return df


# =========================
# Load data
# =========================
df = pd.read_csv(result_path)
df = normalize_df(df)
df = df.dropna(subset=cols)

# =========================
# Expected conditions
# =========================
total_expected = len(scales) * len(pretrain_seeds) * len(fine_tune_seeds) * len(generation_seeds) * len(rewards)

print("Total expected conditions:", total_expected)

# =========================
# Full grid
# =========================
full_conditions = pd.DataFrame(
    list(itertools.product(scales, pretrain_seeds, fine_tune_seeds, generation_seeds, rewards)), columns=cols
)

full_conditions = normalize_df(full_conditions)

# =========================
# Build sets
# =========================
full_set = set(map(tuple, full_conditions[cols].values))
df_set = set(map(tuple, df[cols].values))

missing = full_set - df_set

# =========================
# NULL VALUE CHECK
# =========================
# rows that exist but have NaN in grammaticality_childes
null_df = df[df["grammaticality_childes"].isna()]

null_set = set(map(tuple, null_df[cols].values))

# =========================
# Convert to DataFrames
# =========================
missing_df = pd.DataFrame(list(missing), columns=cols)
null_conditions_df = pd.DataFrame(list(null_set), columns=cols)

# =========================
# Print summary
# =========================
print(f"\nMissing rows: {len(missing)} / {total_expected}")
print(f"Rows with NULL grammaticality_childes: {len(null_conditions_df)}")

# =========================
# Print missing combinations
# =========================
if len(missing_df) > 0:
    print("\n=== ❌ Missing combinations (first 20) ===")
    print(missing_df.sort_values(cols).head(20))

    print("\n=== ❌ All missing combinations ===")
    for row in missing_df.sort_values(cols).itertuples(index=False):
        print("[MISSING]", dict(zip(cols, row, strict=False)))
else:
    print("\n✅ No missing combinations")

# =========================
# Print NULL combinations
# =========================
if len(null_conditions_df) > 0:
    print("\n=== ⚠️ NULL grammaticality_childes (first 20) ===")
    print(null_conditions_df.sort_values(cols).head(20))

    print("\n=== ⚠️ All NULL combinations ===")
    for row in null_conditions_df.sort_values(cols).itertuples(index=False):
        print("[NULL]", dict(zip(cols, row, strict=False)))
else:
    print("\n✅ No NULL values in grammaticality_childes")

# =========================
# Optional: save both
# =========================
if len(missing_df) > 0:
    missing_df.to_csv("/store/scratch/jliu/Feedback/results/missing_conditions.csv", index=False)

if len(null_conditions_df) > 0:
    null_conditions_df.to_csv("/store/scratch/jliu/Feedback/results/null_conditions.csv", index=False)

# =========================
# Diagnostics
# =========================
dup_count = df.duplicated(subset=cols).sum()
print("\nDuplicate rows:", dup_count)
