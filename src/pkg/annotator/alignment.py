import numpy as np
import spacy
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity


class LexicalAlignmentCalculator:
    """Computes lexical alignment based on lemma overlap rate (TYPE-level)."""

    def __init__(
        self,
        model_name: str = "en_core_web_sm",
        exclude_stopwords: bool = True,
        exclude_punctuation: bool = True,
        exclude_interjections: bool = False,
    ):
        try:
            self.nlp = spacy.load(model_name)
        except OSError:
            print(f"Downloading spaCy model {model_name}...")
            import subprocess

            subprocess.run(["python", "-m", "spacy", "download", model_name], check=False)
            self.nlp = spacy.load(model_name)

        self.exclude_stopwords = exclude_stopwords
        self.exclude_punctuation = exclude_punctuation
        self.exclude_interjections = exclude_interjections  # NEW

    def _get_lemmas(self, text: str) -> list[str]:
        """Extract lemmatized tokens from text (preserving duplicates for counting)."""
        doc = self.nlp(text.lower())
        lemmas = []

        for token in doc:
            # Filter based on settings
            if self.exclude_punctuation and token.is_punct:
                continue
            if self.exclude_stopwords and token.is_stop:
                continue
            if self.exclude_interjections and token.pos_ == "INTJ":  # NEW
                continue
            # Skip whitespace-only tokens
            if token.text.strip() == "":
                continue

            lemmas.append(token.lemma_)

        return lemmas

    def compute_alignment(self, child_turn: str, adult_turn: str) -> float:
        """Compute lexical alignment between child and adult turns (TYPE-level)."""
        child_lemmas = self._get_lemmas(child_turn)
        adult_lemmas = self._get_lemmas(adult_turn)

        # Convert to TYPES (unique lemmas)
        child_types = set(child_lemmas)
        adult_types = set(adult_lemmas)

        # Handle empty cases
        if len(child_types) == 0 and len(adult_types) == 0:
            return 1.0  # Both empty = perfect alignment
        if len(child_types) == 0 or len(adult_types) == 0:
            return 0.0  # One empty = no alignment

        # Overlap rate: intersection / total TYPES
        intersection = len(child_types & adult_types)
        total = len(adult_types)

        alignment = intersection / total if total > 0 else 0.0

        return float(alignment)

    def compute_alignment_detailed(self, child_turn: str, adult_turn: str) -> dict:
        """Compute alignment with detailed breakdown."""
        child_lemmas = self._get_lemmas(child_turn)
        adult_lemmas = self._get_lemmas(adult_turn)

        child_types = set(child_lemmas)
        adult_types = set(adult_lemmas)
        shared_types = child_types & adult_types

        alignment = self.compute_alignment(child_turn, adult_turn)

        return {
            "alignment": alignment,
            "child_lemmas": child_lemmas,  # All tokens
            "adult_lemmas": adult_lemmas,  # All tokens
            "child_types": list(child_types),  # Unique lemmas
            "adult_types": list(adult_types),  # Unique lemmas
            "shared_types": list(shared_types),
            "num_child_tokens": len(child_lemmas),
            "num_adult_tokens": len(adult_lemmas),
            "num_child_types": len(child_types),
            "num_adult_types": len(adult_types),
            "num_shared_types": len(shared_types),
        }


class SyntacticAlignmentCalculator:
    """Computes syntactic alignment based on POS tag bigram overlap rate."""

    def __init__(self, model_name: str = "en_core_web_sm", use_types: bool = True):
        try:
            self.nlp = spacy.load(model_name)
        except OSError:
            print(f"Downloading spaCy model {model_name}...")
            import subprocess

            subprocess.run(["python", "-m", "spacy", "download", model_name], check=False)
            self.nlp = spacy.load(model_name)

        self.use_types = use_types

    def _get_pos_tags(self, text: str) -> list[str]:
        """Extract POS tags from text, excluding punctuation."""
        doc = self.nlp(text)
        # Use universal POS tags (coarse-grained), exclude punctuation
        pos_tags = []
        for token in doc:
            if token.is_punct:
                continue
            # Skip whitespace-only tokens
            if token.text.strip() == "":
                continue
            pos_tags.append(token.pos_)
        return pos_tags

    def _get_pos_bigrams(self, text: str) -> list[str]:
        """Extract POS tag bigrams to capture sequential structure."""
        pos_tags = self._get_pos_tags(text)

        if len(pos_tags) < 2:
            return []

        # Create bigrams: "DET_NOUN", "NOUN_VERB", etc.
        bigrams = [f"{pos_tags[i]}_{pos_tags[i + 1]}" for i in range(len(pos_tags) - 1)]
        return bigrams

    def compute_alignment(self, child_turn: str, adult_turn: str) -> float:
        """Compute syntactic alignment between child and adult turns using POS bigrams."""
        child_bigrams = self._get_pos_bigrams(child_turn)
        adult_bigrams = self._get_pos_bigrams(adult_turn)

        # Handle empty cases
        if len(child_bigrams) == 0 and len(adult_bigrams) == 0:
            return 1.0  # Both empty = perfect alignment
        if len(child_bigrams) == 0 or len(adult_bigrams) == 0:
            return 0.0  # One empty = no alignment

        if self.use_types:
            # TYPE-level: unique bigram patterns
            child_set = set(child_bigrams)
            adult_set = set(adult_bigrams)

            intersection = len(child_set & adult_set)
            total = len(adult_set)
        else:
            # TOKEN-level: count all bigrams
            intersection = len(set(child_bigrams) & set(adult_bigrams))
            total = len(adult_bigrams)

        alignment = intersection / total if total > 0 else 0.0

        return float(alignment)

    def compute_alignment_detailed(self, child_turn: str, adult_turn: str) -> dict:
        """Compute alignment with detailed breakdown."""
        child_tags = self._get_pos_tags(child_turn)
        adult_tags = self._get_pos_tags(adult_turn)

        child_bigrams = self._get_pos_bigrams(child_turn)
        adult_bigrams = self._get_pos_bigrams(adult_turn)

        child_bigram_types = set(child_bigrams)
        adult_bigram_types = set(adult_bigrams)
        shared_bigram_types = child_bigram_types & adult_bigram_types

        alignment = self.compute_alignment(child_turn, adult_turn)

        return {
            "alignment": alignment,
            "child_pos_tags": child_tags,
            "adult_pos_tags": adult_tags,
            "child_pos_bigrams": child_bigrams,  # All tokens
            "adult_pos_bigrams": adult_bigrams,  # All tokens
            "child_bigram_types": list(child_bigram_types),  # Unique
            "adult_bigram_types": list(adult_bigram_types),  # Unique
            "shared_bigram_types": list(shared_bigram_types),
            "num_child_bigrams": len(child_bigrams),
            "num_adult_bigrams": len(adult_bigrams),
            "num_child_bigram_types": len(child_bigram_types),
            "num_adult_bigram_types": len(adult_bigram_types),
            "num_shared_bigram_types": len(shared_bigram_types),
        }


class SemanticAlignmentCalculator:
    """Computes semantic alignment using sentence embeddings."""

    def __init__(self, model_name: str = "all-MiniLM-L6-v2", device: str | None = None):
        print(f"Loading semantic model: {model_name} on {device}...")
        self.model = SentenceTransformer(model_name, device=device)
        self.model_name = model_name

    def _get_embedding(self, text: str) -> np.ndarray:
        """Get sentence embedding for text."""
        embedding = self.model.encode(text, convert_to_numpy=True)
        return embedding

    def compute_alignment(self, child_turn: str, adult_turn: str) -> float:
        """Compute semantic alignment between child and adult turns."""
        # Handle empty inputs
        if not child_turn.strip() or not adult_turn.strip():
            return 0.0

        child_emb = self._get_embedding(child_turn)
        adult_emb = self._get_embedding(adult_turn)

        # Compute cosine similarity
        similarity = cosine_similarity(child_emb.reshape(1, -1), adult_emb.reshape(1, -1))[0, 0]

        # Clip to [0, 1] range
        # (cosine similarity is theoretically [-1, 1], but usually positive)
        alignment = np.clip(similarity, 0, 1)

        return float(alignment)

    def compute_alignment_batch(self, child_turns: list[str], adult_turns: list[str]) -> np.ndarray:
        """Compute semantic alignment for multiple turn pairs efficiently."""
        if len(child_turns) != len(adult_turns):
            raise ValueError("Number of child and adult turns must match")

        # Batch encode for efficiency
        child_embs = self.model.encode(child_turns, convert_to_numpy=True)
        adult_embs = self.model.encode(adult_turns, convert_to_numpy=True)

        # Compute pairwise cosine similarities
        alignments = []
        for child_emb, adult_emb in zip(child_embs, adult_embs, strict=False):
            similarity = cosine_similarity(child_emb.reshape(1, -1), adult_emb.reshape(1, -1))[0, 0]
            alignments.append(np.clip(similarity, 0, 1))

        return np.array(alignments)

    def compute_alignment_detailed(self, child_turn: str, adult_turn: str) -> dict:
        """Compute alignment with embedding details."""
        child_emb = self._get_embedding(child_turn)
        adult_emb = self._get_embedding(adult_turn)

        alignment = self.compute_alignment(child_turn, adult_turn)

        return {
            "alignment": alignment,
            "child_embedding": child_emb,
            "adult_embedding": adult_emb,
            "embedding_dim": len(child_emb),
            "model_name": self.model_name,
        }


class LinguisticAlignmentSuite:
    """Unified interface for computing all alignment metrics."""

    def __init__(
        self,
        spacy_model: str = "en_core_web_sm",
        semantic_model: str = "all-MiniLM-L6-v2",
        exclude_stopwords: bool = True,
        exclude_interjections: bool = False,
        device: str | None = None,
    ):
        """Initialize all alignment calculators."""
        print("Initializing Linguistic Alignment Suite...")

        self.lexical_calc = LexicalAlignmentCalculator(
            model_name=spacy_model, exclude_stopwords=exclude_stopwords, exclude_interjections=exclude_interjections
        )

        self.syntactic_calc = SyntacticAlignmentCalculator(model_name=spacy_model)

        self.semantic_calc = SemanticAlignmentCalculator(model_name=semantic_model, device=device)

        print("✓ Alignment suite ready!")

    def compute_all_alignments(self, child_turn: str, adult_turn: str) -> dict[str, float]:
        """Compute all three alignment types."""
        return {
            "lexical_alignment": self.lexical_calc.compute_alignment(child_turn, adult_turn),
            "syntactic_alignment": self.syntactic_calc.compute_alignment(child_turn, adult_turn),
            "semantic_alignment": self.semantic_calc.compute_alignment(child_turn, adult_turn),
        }

    def compute_all_alignments_detailed(self, child_turn: str, adult_turn: str) -> dict:
        """Compute all alignments with detailed breakdowns."""
        return {
            "lexical": self.lexical_calc.compute_alignment_detailed(child_turn, adult_turn),
            "syntactic": self.syntactic_calc.compute_alignment_detailed(child_turn, adult_turn),
            "semantic": self.semantic_calc.compute_alignment_detailed(child_turn, adult_turn),
        }

    def compute_batch(self, child_turns: list[str], adult_turns: list[str]) -> dict[str, np.ndarray]:
        """Efficiently compute alignments for multiple turn pairs."""
        n = len(child_turns)
        if n != len(adult_turns):
            raise ValueError("Number of child and adult turns must match")

        # Lexical and syntactic need to be computed individually
        lexical_scores = np.array(
            [self.lexical_calc.compute_alignment(c, a) for c, a in zip(child_turns, adult_turns, strict=False)]
        )

        syntactic_scores = np.array(
            [self.syntactic_calc.compute_alignment(c, a) for c, a in zip(child_turns, adult_turns, strict=False)]
        )

        # Semantic can be batched efficiently
        semantic_scores = self.semantic_calc.compute_alignment_batch(child_turns, adult_turns)

        return {
            "lexical_alignment": lexical_scores,
            "syntactic_alignment": syntactic_scores,
            "semantic_alignment": semantic_scores,
        }
