"""
Evidence retriever — finds the most relevant historical messages for each incoming message.
Uses multi-signal matching: same sender, same group, content similarity, and behavioral patterns.
"""

import re
from collections import Counter

import pandas as pd


class EvidenceRetriever:
    """Retrieves relevant historical message IDs as evidence for routing decisions."""

    def __init__(self, data_loader):
        self.dl = data_loader
        # Pre-compute word sets for all history messages for similarity matching
        self._history_word_sets = {}
        for _, row in data_loader.history_with_events_df.iterrows():
            text = str(row.get("message_text", ""))
            self._history_word_sets[row["message_id"]] = self._tokenize(text)

    @staticmethod
    def _tokenize(text: str) -> set[str]:
        """Simple word tokenization for similarity matching."""
        if not text or pd.isna(text):
            return set()
        # Lowercase, keep alphanumeric words
        words = re.findall(r'[a-z0-9]+', text.lower())
        # Remove very common stop words
        stop_words = {
            'the', 'a', 'an', 'is', 'are', 'was', 'were', 'be', 'been',
            'to', 'of', 'in', 'for', 'on', 'with', 'at', 'by', 'from',
            'and', 'or', 'not', 'this', 'that', 'it', 'you', 'we', 'he',
            'she', 'they', 'i', 'me', 'my', 'your', 'our', 'his', 'her',
            'its', 'can', 'will', 'just', 'so', 'if', 'but', 'all', 'no',
            'do', 'has', 'have', 'had', 'up', 'out', 'as'
        }
        return {w for w in words if w not in stop_words and len(w) > 1}

    @staticmethod
    def _jaccard_similarity(set_a: set, set_b: set) -> float:
        """Jaccard similarity between two word sets."""
        if not set_a or not set_b:
            return 0.0
        intersection = set_a & set_b
        union = set_a | set_b
        return len(intersection) / len(union) if union else 0.0

    def _score_evidence(self, history_msg: dict, incoming_msg: dict, incoming_words: set) -> float:
        """Score a historical message's relevance as evidence.

        Higher score = more relevant. Considers:
        - Same sender (strong signal)
        - Same group (moderate signal)
        - Content similarity (moderate signal)
        - User reaction strength (how strongly user engaged/reacted)
        """
        score = 0.0

        # Same sender bonus
        if (pd.notna(incoming_msg.get("sender_user_id"))
                and incoming_msg.get("sender_user_id") == history_msg.get("sender_user_id")):
            score += 3.0

        # Same business bonus
        if (pd.notna(incoming_msg.get("business_id"))
                and incoming_msg.get("business_id") == history_msg.get("business_id")):
            score += 3.0

        # Same group bonus
        if (pd.notna(incoming_msg.get("group_id"))
                and incoming_msg.get("group_id") == history_msg.get("group_id")):
            score += 1.5

        # Same conversation type bonus
        if incoming_msg.get("conversation_type") == history_msg.get("conversation_type"):
            score += 0.5

        # Content similarity
        hist_words = self._history_word_sets.get(history_msg["message_id"], set())
        similarity = self._jaccard_similarity(incoming_words, hist_words)
        score += similarity * 3.0  # Up to 3.0 for perfect match

        # User reaction signal — messages that user reacted to strongly are more useful
        if history_msg.get("message_reported") == 1:
            score += 2.0  # Strong signal: user reported similar content
        if history_msg.get("muted_after_message") == 1:
            score += 1.5
        if history_msg.get("notification_dismissed") == 1:
            score += 1.0
        if history_msg.get("message_replied") == 1:
            score += 1.0
        if history_msg.get("message_opened") == 1:
            score += 0.5

        # Forwarding pattern similarity
        incoming_fwd = int(incoming_msg.get("forwarded_count", 0))
        hist_fwd = int(history_msg.get("forwarded_count", 0))
        if incoming_fwd > 0 and hist_fwd > 0:
            score += 1.0  # Both are forwards — pattern match

        return score

    def find_evidence(self, message: dict, max_results: int = 3) -> list[dict]:
        """Find the most relevant historical messages as evidence.

        Args:
            message: incoming message dict from messages.csv
            max_results: max number of evidence items to return

        Returns:
            list of dicts with keys: message_id, text_snippet, user_reaction, score
        """
        user_id = message["user_id"]
        user_history = self.dl.get_user_history(user_id)

        if not user_history:
            return []

        incoming_words = self._tokenize(str(message.get("message_text", "")))

        # Score all history messages
        scored = []
        for hist_msg in user_history:
            score = self._score_evidence(hist_msg, message, incoming_words)
            if score > 0.5:  # Minimum relevance threshold
                # Determine user reaction summary
                reaction = self._summarize_reaction(hist_msg)
                text = str(hist_msg.get("message_text", ""))
                snippet = text[:100] + "..." if len(text) > 100 else text

                scored.append({
                    "message_id": hist_msg["message_id"],
                    "text_snippet": snippet,
                    "user_reaction": reaction,
                    "score": score,
                    "sender": hist_msg.get("sender_user_id", hist_msg.get("business_id", "")),
                    "forwarded_count": hist_msg.get("forwarded_count", 0)
                })

        # Sort by score descending, take top N
        scored.sort(key=lambda x: x["score"], reverse=True)
        return scored[:max_results]

    @staticmethod
    def _summarize_reaction(hist_msg: dict) -> str:
        """Summarize how the user reacted to a historical message."""
        reactions = []
        if hist_msg.get("message_reported") == 1:
            reactions.append("reported")
        if hist_msg.get("muted_after_message") == 1:
            reactions.append("muted")
        if hist_msg.get("notification_dismissed") == 1:
            reactions.append("dismissed")
        if hist_msg.get("message_replied") == 1:
            reactions.append("replied")
        if hist_msg.get("message_opened") == 1:
            reactions.append("opened")

        if reactions:
            return ", ".join(reactions)
        return "no interaction"

    def format_evidence_ids(self, evidence_list: list[dict]) -> str:
        """Format evidence list into the output column format.

        Returns semicolon-separated message IDs, or 'none'.
        """
        if not evidence_list:
            return "none"
        return ";".join(e["message_id"] for e in evidence_list)

    def format_evidence_for_prompt(self, evidence_list: list[dict]) -> str:
        """Format evidence into a readable string for the LLM prompt."""
        if not evidence_list:
            return "No relevant historical messages found for this user."

        lines = []
        for i, ev in enumerate(evidence_list, 1):
            lines.append(
                f"  Evidence {i}: [{ev['message_id']}] "
                f"User reaction: {ev['user_reaction']}. "
                f"Content: \"{ev['text_snippet']}\""
            )
        return "\n".join(lines)
