import pandas as pd

from .backchannel import BackchannelIdentifier


class ConversationMergeStrategy:
    """Filters and merges child-adult conversation turns into clean CHI→MOT pairs."""

    def __init__(
        self,
        backchannel_identifier: BackchannelIdentifier = None,
        child_col: str = "child_speech",
        adult_col: str = "adult_speech",
    ):
        """Initialize with optional backchannel identifier and column names."""
        self.backchannel_id = backchannel_identifier or BackchannelIdentifier()
        self.child_col = child_col
        self.adult_col = adult_col

    def strategy_1_aggregate_consecutive(self, df: pd.DataFrame) -> pd.DataFrame:
        """Aggregate consecutive child or adult turns while keeping CHI→MOT structure."""
        result = []
        i = 0

        while i < len(df):
            row = df.iloc[i].copy()
            child_speech = self._aggregate_consecutive_same_speaker(df, i, self.child_col)
            j = i + 1
            while j < len(df) and df.iloc[j][self.child_col] in ["<EMPTY>", "<UNINTELLIGIBLE>", None, ""]:
                j += 1

            if j < len(df):
                adult_speech = self._aggregate_consecutive_same_speaker(df, j, self.adult_col)
                row[self.child_col] = child_speech
                row[self.adult_col] = adult_speech
                result.append(row)
                i = j + 1
            else:
                i += 1

        return pd.DataFrame(result).reset_index(drop=True)

    def strategy_2_remove_consecutive_pairs(self, df: pd.DataFrame) -> pd.DataFrame:
        """Keep only alternating CHI→MOT pairs, removing consecutive same-speaker turns."""
        result = []
        for i in range(len(df)):
            row = df.iloc[i].copy()
            child_speech = str(row[self.child_col]).strip()
            adult_speech = str(row[self.adult_col]).strip()
            if self._is_empty_utterance(child_speech) or self._is_empty_utterance(adult_speech):
                continue
            result.append(row)

        filtered_df = pd.DataFrame(result).reset_index(drop=True)
        valid_rows = []
        for i, row in filtered_df.iterrows():
            if i == 0:
                valid_rows.append(i)
                continue
            prev_adult = str(filtered_df.iloc[i - 1][self.adult_col]).strip()
            if not self._is_empty_utterance(prev_adult):
                valid_rows.append(i)

        return filtered_df.iloc[valid_rows].reset_index(drop=True)

    def strategy_3_hybrid(self, df: pd.DataFrame, preserve_backchannel: bool = True) -> pd.DataFrame:
        """Aggregate child turns and remove consecutive adult turns except backchannels."""
        result = []
        i = 0

        while i < len(df):
            row = df.iloc[i].copy()
            child_speech, num_child_attempts = self._aggregate_with_count(df, i, self.child_col)
            child_end_idx = i + num_child_attempts - 1
            j = child_end_idx + 1
            adult_responses = []
            backchannel_count = 0

            while j < len(df):
                adult_text = str(df.iloc[j][self.adult_col]).strip()
                if self._is_empty_utterance(adult_text):
                    j += 1
                    continue
                if self.backchannel_id.is_backchannel(adult_text):
                    backchannel_count += 1
                    if preserve_backchannel:
                        adult_responses.append(adult_text)
                    j += 1
                else:
                    adult_responses.append(adult_text)
                    j += 1
                    break

            if adult_responses:
                row[self.child_col] = child_speech
                row[self.adult_col] = " ".join(adult_responses)
                row["num_child_attempts"] = num_child_attempts
                row["num_backchannels"] = backchannel_count
                result.append(row)

            i = max(child_end_idx + 1, j)

        return pd.DataFrame(result).reset_index(drop=True)

    # ===== Helper Methods =====

    def _is_empty_utterance(self, text: str) -> bool:
        """Return True if utterance is empty or unintelligible."""
        empty_markers = ["<EMPTY>", "<UNINTELLIGIBLE>", "nan", "", None]
        text = str(text).strip().upper()
        return any(marker.upper() in text for marker in empty_markers)

    def _aggregate_consecutive_same_speaker(self, df: pd.DataFrame, start_idx: int, column: str) -> str:
        """Aggregate consecutive non-empty utterances from the same speaker."""
        utterances = []
        i = start_idx
        while i < len(df):
            text = str(df.iloc[i][column]).strip()
            if not self._is_empty_utterance(text):
                utterances.append(text)

            if i + 1 < len(df):
                next_child = str(df.iloc[i + 1][self.child_col]).strip()
                next_adult = str(df.iloc[i + 1][self.adult_col]).strip()
                if (column == self.child_col and self._is_empty_utterance(next_adult)) or (
                    column == self.adult_col and self._is_empty_utterance(next_child)
                ):
                    i += 1
                else:
                    break
            else:
                break

        return " ".join(utterances)

    def _aggregate_with_count(self, df: pd.DataFrame, start_idx: int, column: str) -> tuple[str, int]:
        """Aggregate consecutive utterances and return both text and count."""
        utterances = []
        count = 0
        i = start_idx
        while i < len(df):
            text = str(df.iloc[i][column]).strip()
            if not self._is_empty_utterance(text):
                utterances.append(text)
                count += 1

            if i + 1 < len(df):
                next_child = str(df.iloc[i + 1][self.child_col]).strip()
                next_adult = str(df.iloc[i + 1][self.adult_col]).strip()
                if (column == self.child_col and self._is_empty_utterance(next_adult)) or (
                    column == self.adult_col and self._is_empty_utterance(next_child)
                ):
                    i += 1
                else:
                    break
            else:
                break

        return " ".join(utterances), count

    def get_strategy_stats(self, df: pd.DataFrame, strategy: str) -> dict:
        """Return statistics for the chosen strategy."""
        if strategy == "strategy_1":
            result_df = self.strategy_1_aggregate_consecutive(df)
        elif strategy == "strategy_2":
            result_df = self.strategy_2_remove_consecutive_pairs(df)
        elif strategy == "strategy_3":
            result_df = self.strategy_3_hybrid(df)
        else:
            raise ValueError(f"Unknown strategy: {strategy}")

        loss_pct = ((len(df) - len(result_df)) / len(df) * 100) if len(df) > 0 else 0
        return {
            "original_rows": len(df),
            "result_rows": len(result_df),
            "data_loss_pct": loss_pct,
            "result_df": result_df,
        }
