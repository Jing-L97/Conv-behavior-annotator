import numpy as np
import spacy
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity


class LexicalAlignmentCalculator:
    """Computes lexical alignment based on lemma overlap rate.

    Alignment = |lemmas_child ∩ lemmas_adult| / (|lemmas_child| + |lemmas_adult|)
    """

    def __init__(
        self, model_name: str = "en_core_web_sm", exclude_stopwords: bool = True, exclude_punctuation: bool = True
    ):
        """Args:
        model_name: spaCy model to use
        exclude_stopwords: Whether to exclude stopwords from alignment
        exclude_punctuation: Whether to exclude punctuation

        """
        try:
            self.nlp = spacy.load(model_name)
        except OSError:
            print(f"Downloading spaCy model {model_name}...")
            import subprocess

            subprocess.run(["python", "-m", "spacy", "download", model_name], check=False)
            self.nlp = spacy.load(model_name)

        self.exclude_stopwords = exclude_stopwords
        self.exclude_punctuation = exclude_punctuation

    def _get_lemmas(self, text: str) -> list[str]:
        """Extract lemmatized tokens from text."""
        doc = self.nlp(text.lower())
        lemmas = []

        for token in doc:
            # Filter based on settings
            if self.exclude_punctuation and token.is_punct:
                continue
            if self.exclude_stopwords and token.is_stop:
                continue

            lemmas.append(token.lemma_)

        return lemmas

    def compute_alignment(self, child_turn: str, adult_turn: str) -> float:
        """Compute lexical alignment between child and adult turns.

        Args:
            child_turn: Child's utterance
            adult_turn: Adult's utterance

        Returns:
            Alignment score in [0, 1]

        """
        child_lemmas = self._get_lemmas(child_turn)
        adult_lemmas = self._get_lemmas(adult_turn)

        # Handle empty cases
        if len(child_lemmas) == 0 and len(adult_lemmas) == 0:
            return 1.0  # Both empty = perfect alignment
        if len(child_lemmas) == 0 or len(adult_lemmas) == 0:
            return 0.0  # One empty = no alignment

        # Overlap rate: intersection / total
        child_set = set(child_lemmas)
        adult_set = set(adult_lemmas)

        intersection = len(child_set & adult_set)
        total = len(child_lemmas) + len(adult_lemmas)

        alignment = intersection / total if total > 0 else 0.0

        return float(alignment)

    def compute_alignment_detailed(self, child_turn: str, adult_turn: str) -> dict:
        """Compute alignment with detailed breakdown.

        Returns:
            Dictionary with alignment score and shared lemmas

        """
        child_lemmas = self._get_lemmas(child_turn)
        adult_lemmas = self._get_lemmas(adult_turn)

        child_set = set(child_lemmas)
        adult_set = set(adult_lemmas)
        shared_lemmas = child_set & adult_set

        alignment = self.compute_alignment(child_turn, adult_turn)

        return {
            "alignment": alignment,
            "child_lemmas": child_lemmas,
            "adult_lemmas": adult_lemmas,
            "shared_lemmas": list(shared_lemmas),
            "num_child_lemmas": len(child_lemmas),
            "num_adult_lemmas": len(adult_lemmas),
            "num_shared": len(shared_lemmas),
        }


class SyntacticAlignmentCalculator:
    """Computes syntactic alignment based on POS tag bigram overlap rate.

    Uses POS bigrams to capture sequential syntactic structure.
    """

    def __init__(self, model_name: str = "en_core_web_sm"):
        """Args:
        model_name: spaCy model to use

        """
        try:
            self.nlp = spacy.load(model_name)
        except OSError:
            print(f"Downloading spaCy model {model_name}...")
            import subprocess

            subprocess.run(["python", "-m", "spacy", "download", model_name], check=False)
            self.nlp = spacy.load(model_name)

    def _get_pos_tags(self, text: str) -> list[str]:
        """Extract POS tags from text."""
        doc = self.nlp(text)
        # Use universal POS tags (coarse-grained)
        pos_tags = [token.pos_ for token in doc if not token.is_punct]
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
        """Compute syntactic alignment between child and adult turns using POS bigrams.

        Args:
            child_turn: Child's utterance
            adult_turn: Adult's utterance

        Returns:
            Alignment score in [0, 1]

        """
        child_bigrams = self._get_pos_bigrams(child_turn)
        adult_bigrams = self._get_pos_bigrams(adult_turn)

        # Handle empty cases
        if len(child_bigrams) == 0 and len(adult_bigrams) == 0:
            return 1.0  # Both empty = perfect alignment
        if len(child_bigrams) == 0 or len(adult_bigrams) == 0:
            return 0.0  # One empty = no alignment

        # Overlap rate: intersection / total
        child_set = set(child_bigrams)
        adult_set = set(adult_bigrams)

        intersection = len(child_set & adult_set)
        total = len(child_bigrams) + len(adult_bigrams)

        alignment = intersection / total if total > 0 else 0.0

        return float(alignment)

    def compute_alignment_detailed(self, child_turn: str, adult_turn: str) -> dict:
        """Compute alignment with detailed breakdown.

        Returns:
            Dictionary with alignment scores and POS tag information

        """
        child_tags = self._get_pos_tags(child_turn)
        adult_tags = self._get_pos_tags(adult_turn)

        child_bigrams = self._get_pos_bigrams(child_turn)
        adult_bigrams = self._get_pos_bigrams(adult_turn)

        child_bigram_set = set(child_bigrams)
        adult_bigram_set = set(adult_bigrams)
        shared_bigrams = child_bigram_set & adult_bigram_set

        alignment = self.compute_alignment(child_turn, adult_turn)

        return {
            "alignment": alignment,
            "child_pos_tags": child_tags,
            "adult_pos_tags": adult_tags,
            "child_pos_bigrams": child_bigrams,
            "adult_pos_bigrams": adult_bigrams,
            "shared_bigrams": list(shared_bigrams),
            "num_child_bigrams": len(child_bigrams),
            "num_adult_bigrams": len(adult_bigrams),
            "num_shared_bigrams": len(shared_bigrams),
        }


class SemanticAlignmentCalculator:
    """Computes semantic alignment using sentence embeddings.

    Alignment = cosine_similarity(embedding_child, embedding_adult)
    """

    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        """Args:
        model_name: SentenceTransformer model to use
                   Options:
                   - "all-MiniLM-L6-v2" (fast, 384 dim)
                   - "all-mpnet-base-v2" (better quality, 768 dim)
                   - "paraphrase-MiniLM-L6-v2" (paraphrase detection)

        """
        print(f"Loading semantic model: {model_name}...")
        self.model = SentenceTransformer(model_name)
        self.model_name = model_name

    def _get_embedding(self, text: str) -> np.ndarray:
        """Get sentence embedding for text."""
        embedding = self.model.encode(text, convert_to_numpy=True)
        return embedding

    def compute_alignment(self, child_turn: str, adult_turn: str) -> float:
        """Compute semantic alignment between child and adult turns.

        Args:
            child_turn: Child's utterance
            adult_turn: Adult's utterance

        Returns:
            Alignment score in [0, 1]

        Note:
            Cosine similarity ranges from [-1, 1], but in practice
            sentence embeddings rarely give negative values.
            We clip to [0, 1] for consistency.

        """
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
        """Compute semantic alignment for multiple turn pairs efficiently.

        Args:
            child_turns: List of child utterances
            adult_turns: List of adult utterances

        Returns:
            Array of alignment scores

        """
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
        """Compute alignment with embedding details.

        Returns:
            Dictionary with alignment score and embeddings

        """
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
    ):
        """Initialize all alignment calculators.

        Args:
            spacy_model: Model for lexical and syntactic alignment
            semantic_model: Model for semantic alignment
            exclude_stopwords: Whether to exclude stopwords in lexical alignment

        """
        print("Initializing Linguistic Alignment Suite...")

        self.lexical_calc = LexicalAlignmentCalculator(model_name=spacy_model, exclude_stopwords=exclude_stopwords)

        self.syntactic_calc = SyntacticAlignmentCalculator(model_name=spacy_model)

        self.semantic_calc = SemanticAlignmentCalculator(model_name=semantic_model)

        print("✓ Alignment suite ready!")

    def compute_all_alignments(self, child_turn: str, adult_turn: str) -> dict[str, float]:
        """Compute all three alignment types.

        Args:
            child_turn: Child's utterance
            adult_turn: Adult's utterance

        Returns:
            Dictionary with all alignment scores (all in [0, 1])

        """
        return {
            "lexical_alignment": self.lexical_calc.compute_alignment(child_turn, adult_turn),
            "syntactic_alignment": self.syntactic_calc.compute_alignment(child_turn, adult_turn),
            "semantic_alignment": self.semantic_calc.compute_alignment(child_turn, adult_turn),
        }

    def compute_all_alignments_detailed(self, child_turn: str, adult_turn: str) -> dict:
        """Compute all alignments with detailed breakdowns.

        Returns:
            Dictionary with detailed information for each alignment type

        """
        return {
            "lexical": self.lexical_calc.compute_alignment_detailed(child_turn, adult_turn),
            "syntactic": self.syntactic_calc.compute_alignment_detailed(child_turn, adult_turn),
            "semantic": self.semantic_calc.compute_alignment_detailed(child_turn, adult_turn),
        }

    def compute_batch(self, child_turns: list[str], adult_turns: list[str]) -> dict[str, np.ndarray]:
        """Efficiently compute alignments for multiple turn pairs.

        Returns:
            Dictionary of arrays with alignment scores

        """
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
