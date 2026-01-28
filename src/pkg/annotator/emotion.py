import numpy as np
import pandas as pd
from tqdm import tqdm
from transformers import pipeline


class SentimentAnnotator:
    """Annotates adult turns with emotion-based sentiment scores."""

    def __init__(
        self, model_name: str = "SamLowe/roberta-base-go_emotions", device: str | None = None, batch_size: int = 32
    ):
        """Initialize sentiment annotator."""
        print(f"Loading emotion model: {model_name}...")
        self.emotion_classifier = pipeline(
            "text-classification", model=model_name, top_k=None, device=0 if device == "cuda" else -1
        )

        # Define emotion categories for child-adult interactions
        self.supportive_emotions = ["admiration", "approval", "caring", "optimism", "pride"]
        self.engagement_emotions = ["curiosity", "excitement", "amusement", "desire"]
        self.warmth_emotions = ["caring", "love", "gratitude", "joy"]
        self.negative_emotions = ["disappointment", "disapproval", "annoyance", "anger"]

    def _compute_composite_scores(self, emotion_dict: dict[str, float]) -> dict[str, float]:
        """Compute composite sentiment scores from emotion distribution."""
        supportiveness = np.mean([emotion_dict.get(e, 0) for e in self.supportive_emotions])

        engagement = np.mean([emotion_dict.get(e, 0) for e in self.engagement_emotions])

        warmth = np.mean([emotion_dict.get(e, 0) for e in self.warmth_emotions])

        negativity = np.mean([emotion_dict.get(e, 0) for e in self.negative_emotions])

        return {
            "supportiveness": float(supportiveness),
            "engagement": float(engagement),
            "warmth": float(warmth),
            "negativity": float(negativity),
            "approval": float(emotion_dict.get("approval", 0)),
            "curiosity": float(emotion_dict.get("curiosity", 0)),
            "caring": float(emotion_dict.get("caring", 0)),
        }

    def annotate_batch1(self, texts: list[str]) -> pd.DataFrame:
        """Annotate a batch of texts with sentiment scores."""
        results = []

        # Process in batches for efficiency
        for text in tqdm(texts, desc="Annotating sentiment"):
            if pd.isna(text) or text.strip() == "":
                # Handle empty/NaN texts
                results.append(
                    {
                        "supportiveness": 0.0,
                        "engagement": 0.0,
                        "warmth": 0.0,
                        "negativity": 0.0,
                        "pedagogical_positivity": 0.0,
                        "approval": 0.0,
                        "curiosity": 0.0,
                        "caring": 0.0,
                    }
                )
            else:
                emotions = self.emotion_classifier(text)[0]
                emotion_dict = {e["label"]: e["score"] for e in emotions}
                scores = self._compute_composite_scores(emotion_dict)
                results.append(scores)

        return pd.DataFrame(results)

    def annotate_batch(self, texts: list[str], batch_size: int = 32) -> pd.DataFrame:
        results = []

        # Split into batches
        for i in tqdm(range(0, len(texts), batch_size), desc="Annotating sentiment"):
            batch = texts[i : i + batch_size]
            # Replace empty strings with placeholder if needed
            non_empty_idx = [j for j, t in enumerate(batch) if pd.notna(t) and t.strip()]
            non_empty_texts = [batch[j] for j in non_empty_idx]

            if non_empty_texts:
                emotions_batch = self.emotion_classifier(non_empty_texts)
                for idx, emo in zip(non_empty_idx, emotions_batch, strict=False):
                    emotion_dict = {e["label"]: e["score"] for e in emo}
                    scores = self._compute_composite_scores(emotion_dict)
                    results.append(scores)
            # Fill empty texts with zeros
            for j in range(len(batch)):
                if j not in non_empty_idx:
                    results.append(  # noqa: PERF401
                        {
                            "supportiveness": 0.0,
                            "engagement": 0.0,
                            "warmth": 0.0,
                            "negativity": 0.0,
                            "pedagogical_positivity": 0.0,
                            "approval": 0.0,
                            "curiosity": 0.0,
                            "caring": 0.0,
                        }
                    )

        return pd.DataFrame(results)
