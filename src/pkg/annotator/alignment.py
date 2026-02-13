import numpy as np
import spacy
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity


class LexicalAlignmentCalculator:
    """Computes lexical alignment based on shared lemmas and bigrams."""

    def __init__(
        self,
        model_name: str = "en_core_web_sm",
        exclude_stopwords: bool = False,
        exclude_punctuation: bool = True,
        exclude_interjections: bool = False,
    ):
        try:
            self.nlp = spacy.load(model_name)
        except OSError:
            import subprocess

            subprocess.run(["python", "-m", "spacy", "download", model_name], check=False)
            self.nlp = spacy.load(model_name)

        self.exclude_stopwords = exclude_stopwords
        self.exclude_punctuation = exclude_punctuation
        self.exclude_interjections = exclude_interjections

    # ---------- helpers ----------

    def _get_lemmas(self, text: str) -> list[str]:
        doc = self.nlp(text.lower())
        lemmas = []

        for token in doc:
            if self.exclude_punctuation and token.is_punct:
                continue
            if self.exclude_stopwords and token.is_stop:
                continue
            if self.exclude_interjections and token.pos_ == "INTJ":
                continue
            if token.text.strip() == "":
                continue

            lemmas.append(token.lemma_)

        return lemmas

    def _get_bigrams(self, seq: list[str]) -> list[tuple[str, str]]:
        return [(seq[i], seq[i + 1]) for i in range(len(seq) - 1)]

    def _overlap_score(self, a: set, b: set) -> float:
        if not a and not b:
            return 1.0
        if not a or not b:
            return 0.0
        return len(a & b) / max(len(a), len(b))

    # ---------- main ----------

    def compute_alignment(self, child_turn: str, adult_turn: str) -> dict[str, float]:
        child = self._get_lemmas(child_turn)
        adult = self._get_lemmas(adult_turn)

        child_uni = set(child)
        adult_uni = set(adult)

        child_bi = set(self._get_bigrams(child))
        adult_bi = set(self._get_bigrams(adult))

        return {
            "lexical_unigram_alignment": self._overlap_score(child_uni, adult_uni),
            "lexical_bigram_alignment": self._overlap_score(child_bi, adult_bi),
        }

    def compute_alignment_detailed(self, child_turn: str, adult_turn: str) -> dict:
        child = self._get_lemmas(child_turn)
        adult = self._get_lemmas(adult_turn)

        child_uni = set(child)
        adult_uni = set(adult)
        child_bi = set(self._get_bigrams(child))
        adult_bi = set(self._get_bigrams(adult))

        return {
            "scores": self.compute_alignment(child_turn, adult_turn),
            "child_unigrams": list(child_uni),
            "adult_unigrams": list(adult_uni),
            "child_bigrams": list(child_bi),
            "adult_bigrams": list(adult_bi),
            "shared_unigrams": list(child_uni & adult_uni),
            "shared_bigrams": list(child_bi & adult_bi),
        }


class SyntacticAlignmentCalculator:
    """Syntactic alignment based on shared POS bigram types."""

    def __init__(self, model_name: str = "en_core_web_sm"):
        try:
            self.nlp = spacy.load(model_name)
        except OSError:
            import subprocess

            subprocess.run(["python", "-m", "spacy", "download", model_name], check=False)
            self.nlp = spacy.load(model_name)

    # ---------- helpers ----------

    def _get_pos_tags(self, text: str) -> list[str]:
        doc = self.nlp(text)
        return [t.pos_ for t in doc if not t.is_punct and t.text.strip()]

    def _get_pos_bigrams(self, pos: list[str]) -> list[tuple[str, str]]:
        return [(pos[i], pos[i + 1]) for i in range(len(pos) - 1)]

    # ---------- main ----------

    def compute_alignment(self, child_turn: str, adult_turn: str) -> float:
        child = set(self._get_pos_bigrams(self._get_pos_tags(child_turn)))
        adult = set(self._get_pos_bigrams(self._get_pos_tags(adult_turn)))

        if not child and not adult:
            return 1.0
        if not child or not adult:
            return 0.0

        shared = len(child & adult)
        denom = max(len(child), len(adult))

        return shared / denom

    def compute_alignment_detailed(self, child_turn: str, adult_turn: str) -> dict:
        child = set(self._get_pos_bigrams(self._get_pos_tags(child_turn)))
        adult = set(self._get_pos_bigrams(self._get_pos_tags(adult_turn)))
        shared = child & adult

        return {
            "alignment": self.compute_alignment(child_turn, adult_turn),
            "child_pos_types": list(child),
            "adult_pos_types": list(adult),
            "shared_pos_types": list(shared),
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
    """Unified interface for computing lexical, syntactic, and semantic alignment."""

    def __init__(
        self,
        spacy_model: str = "en_core_web_sm",
        semantic_model: str = "all-MiniLM-L6-v2",
        exclude_stopwords: bool = False,
        exclude_interjections: bool = False,
        device: str | None = None,
    ):
        print("Initializing Linguistic Alignment Suite...")

        self.lexical_calc = LexicalAlignmentCalculator(
            model_name=spacy_model,
            exclude_stopwords=exclude_stopwords,
            exclude_interjections=exclude_interjections,
        )

        self.syntactic_calc = SyntacticAlignmentCalculator(model_name=spacy_model)
        self.semantic_calc = SemanticAlignmentCalculator(model_name=semantic_model, device=device)

        print("✓ Alignment suite ready!")

    # ---------- single pair ----------

    def compute_all_alignments(self, child_turn: str, adult_turn: str) -> dict[str, float]:
        lexical_scores = self.lexical_calc.compute_alignment(child_turn, adult_turn)

        return {
            "lexical_unigram_alignment": lexical_scores["lexical_unigram_alignment"],
            "lexical_bigram_alignment": lexical_scores["lexical_bigram_alignment"],
            "syntactic_alignment": self.syntactic_calc.compute_alignment(child_turn, adult_turn),
            # "semantic_alignment": self.semantic_calc.compute_alignment(child_turn, adult_turn),
        }

    def compute_all_alignments_detailed(self, child_turn: str, adult_turn: str) -> dict:
        return {
            "lexical": self.lexical_calc.compute_alignment_detailed(child_turn, adult_turn),
            "syntactic": self.syntactic_calc.compute_alignment_detailed(child_turn, adult_turn),
            # "semantic": self.semantic_calc.compute_alignment_detailed(child_turn, adult_turn),
        }

    # ---------- batch ----------

    def compute_batch(self, child_turns: list[str], adult_turns: list[str]) -> dict[str, np.ndarray]:
        if len(child_turns) != len(adult_turns):
            raise ValueError("Number of child and adult turns must match")

        uni_scores = []
        bi_scores = []
        syn_scores = []

        for c, a in zip(child_turns, adult_turns, strict=False):
            lex = self.lexical_calc.compute_alignment(c, a)
            uni_scores.append(lex["lexical_unigram_alignment"])
            bi_scores.append(lex["lexical_bigram_alignment"])
            syn_scores.append(self.syntactic_calc.compute_alignment(c, a))

        # semantic_scores = self.semantic_calc.compute_alignment_batch(child_turns, adult_turns)

        return {
            "lexical_unigram_alignment": np.array(uni_scores),
            "lexical_bigram_alignment": np.array(bi_scores),
            "syntactic_alignment": np.array(syn_scores),
            # "semantic_alignment": semantic_scores,
        }
