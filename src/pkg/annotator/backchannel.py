import re

# Core backchannels with common variations
BACKCHANNEL_PATTERNS = {
    "minimal_affirmatives": [
        r"\byes\b",
        r"\byeah\b",
        r"\byup\b",
        r"\bya\b",
        r"\buh\s*huh\b",
        r"\buh\b",
        r"\bhuh\b",
        r"\bmhm\b",
        r"\bmm\b",
        r"\bm\b",
        r"\bum\b",
        r"\bumm\b",
    ],
    "minimal_interrogatives": [r"\bhuh\b", r"\bwhat\b", r"\bhuh\s*$"],
    "encouragers": [r"\boh\b", r"\bow\b", r"\bcool\b", r"\bnice\b", r"\boh\s+boy\b", r"\boh\s+yeah\b"],
    "continuers": [
        r"\band\s+\w+",
        r"\bthen\b",
        r"\bso\b",  # when standalone
    ],
}


class BackchannelIdentifier:
    """Identifies backchannel utterances."""

    def __init__(self, max_length: int = 15, max_words: int = 3, min_confidence: float = 0.5):
        """Initialize BackchannelIdentifier."""
        self.max_length = max_length
        self.max_words = max_words
        self.min_confidence = min_confidence
        self.BACKCHANNEL_PATTERNS = BACKCHANNEL_PATTERNS
        self._compile_patterns()

    def _compile_patterns(self):
        """Compile regex patterns for efficiency."""
        self.compiled_patterns = {}
        for category, patterns in self.BACKCHANNEL_PATTERNS.items():
            self.compiled_patterns[category] = [re.compile(p, re.IGNORECASE) for p in patterns]

    def is_backchannel(self, text: str) -> bool:
        """Determine if text is a backchannel utterance."""
        if not text or isinstance(text, float):  # Handle NaN/None
            return False

        text = text.strip()

        # Rule 1: Length heuristic
        if len(text) > self.max_length:
            return False

        # Rule 2: Word count heuristic
        word_count = len(text.split())
        if word_count > self.max_words:
            return False

        # Rule 3: Pattern matching
        confidence = self._calculate_pattern_confidence(text)
        return confidence >= self.min_confidence

    def _calculate_pattern_confidence(self, text: str) -> float:
        """Calculate confidence score based on pattern matches."""
        matched_categories = 0
        total_categories = len(self.compiled_patterns)

        for category, patterns in self.compiled_patterns.items():
            for pattern in patterns:
                if pattern.search(text):
                    matched_categories += 1
                    break  # One match per category is enough

        return matched_categories / total_categories if total_categories > 0 else 0.0

    def get_backchannel_type(self, text: str) -> str:
        """Classify the type of backchannel."""
        for category, patterns in self.compiled_patterns.items():
            for pattern in patterns:
                if pattern.search(text):
                    return category
        return "non_backchannel"

    def add_custom_backchannel(self, pattern: str, category: str = "custom"):
        """Allow users to add custom backchannel patterns."""
        if category not in self.BACKCHANNEL_PATTERNS:
            self.BACKCHANNEL_PATTERNS[category] = []
        self.BACKCHANNEL_PATTERNS[category].append(pattern)
        self._compile_patterns()
