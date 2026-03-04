import numpy as np
import pandas as pd
from datasets import Dataset
from transformers import pipeline


class SentimentAnnotator:
    """Annotates adult turns with emotion-based sentiment scores using batched GPU inference."""

    def __init__(self, model_name: str = "SamLowe/roberta-base-go_emotions", device: str | None = None):
        """Initialize sentiment annotator."""
        self.device = 0 if device == "cuda" else -1
        print(f"Loading emotion model: {model_name} on {'GPU' if self.device == 0 else 'CPU'}...")
        self.emotion_classifier = pipeline("text-classification", model=model_name, top_k=None, device=self.device)

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

    def annotate_batch(self, texts: list[str], batch_size: int = 32) -> pd.DataFrame:
        """Annotate a list of texts with sentiment scores using batched GPU inference."""
        # Prepare dataset
        ds = Dataset.from_dict({"text": texts})

        def compute_emotions(batch):
            batch_texts = batch["text"]
            # Replace empty/NaN texts with a dummy placeholder to preserve order
            non_empty_idx = [i for i, t in enumerate(batch_texts) if pd.notna(t) and t.strip()]
            non_empty_texts = [batch_texts[i] for i in non_empty_idx]

            scores = [self._compute_composite_scores({}) for _ in batch_texts]  # default zero scores
            if non_empty_texts:
                # Run batch inference on GPU
                emotions_batch = self.emotion_classifier(non_empty_texts)
                for idx, emo in zip(non_empty_idx, emotions_batch, strict=False):
                    emo_dict = {e["label"]: e["score"] for e in emo}
                    scores[idx] = self._compute_composite_scores(emo_dict)
            return {"scores": scores}

        # Map function over dataset with batching
        ds = ds.map(compute_emotions, batched=True, batch_size=batch_size, remove_columns=["text"])

        # Convert back to DataFrame
        df = pd.DataFrame(ds["scores"])
        return df
