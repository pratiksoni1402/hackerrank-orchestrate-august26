"""
Data loader — loads all CSV files into pandas DataFrames and builds lookup indices.
"""

import pandas as pd
from config import DATASET_DIR


class DataLoader:
    """Loads and indexes all dataset CSV files for fast per-message context retrieval."""

    def __init__(self):
        self._load_all()
        self._build_indices()

    def _load_all(self):
        """Load all CSV files into DataFrames."""
        self.messages_df = pd.read_csv(DATASET_DIR / "messages.csv")
        self.sample_messages_df = pd.read_csv(DATASET_DIR / "sample_messages.csv")
        self.users_df = pd.read_csv(DATASET_DIR / "users.csv")
        self.groups_df = pd.read_csv(DATASET_DIR / "groups.csv")
        self.group_members_df = pd.read_csv(DATASET_DIR / "group_members.csv")
        self.business_accounts_df = pd.read_csv(DATASET_DIR / "business_accounts.csv")
        self.user_business_history_df = pd.read_csv(DATASET_DIR / "user_business_history.csv")
        self.message_history_df = pd.read_csv(DATASET_DIR / "message_history.csv")
        self.message_events_df = pd.read_csv(DATASET_DIR / "message_events.csv")
        self.images_df = pd.read_csv(DATASET_DIR / "images.csv")
        self.voice_notes_df = pd.read_csv(DATASET_DIR / "voice_notes.csv")
        self.daily_summary_df = pd.read_csv(DATASET_DIR / "daily_notification_summary.csv")

        # Merge history with events for easy lookup
        self.history_with_events_df = self.message_history_df.merge(
            self.message_events_df,
            on=["message_id", "user_id"],
            how="left"
        )

    def _build_indices(self):
        """Build dictionaries for fast per-key lookups."""
        # Users indexed by user_id
        self._users_idx = self.users_df.set_index("user_id").to_dict("index")

        # Groups indexed by group_id
        self._groups_idx = self.groups_df.set_index("group_id").to_dict("index")

        # Group members indexed by (group_id, user_id)
        self._group_members_idx = {}
        for _, row in self.group_members_df.iterrows():
            key = (row["group_id"], row["user_id"])
            self._group_members_idx[key] = row.to_dict()

        # Business accounts indexed by business_id
        self._business_idx = self.business_accounts_df.set_index("business_id").to_dict("index")

        # User-business history indexed by (user_id, business_id)
        self._user_biz_idx = {}
        for _, row in self.user_business_history_df.iterrows():
            key = (row["user_id"], row["business_id"])
            self._user_biz_idx[key] = row.to_dict()

        # Images indexed by image_id
        self._images_idx = self.images_df.set_index("image_id").to_dict("index")

        # Voice notes indexed by voice_note_id
        self._voice_notes_idx = self.voice_notes_df.set_index("voice_note_id").to_dict("index")

        # History grouped by user_id for evidence retrieval
        self._history_by_user = {}
        for _, row in self.history_with_events_df.iterrows():
            uid = row["user_id"]
            if uid not in self._history_by_user:
                self._history_by_user[uid] = []
            self._history_by_user[uid].append(row.to_dict())

        # History grouped by (user_id, sender) for sender-specific evidence
        self._history_by_user_sender = {}
        for _, row in self.history_with_events_df.iterrows():
            uid = row["user_id"]
            sender = row.get("sender_user_id", "")
            biz = row.get("business_id", "")
            if pd.notna(sender) and sender:
                key = (uid, "sender", sender)
                self._history_by_user_sender.setdefault(key, []).append(row.to_dict())
            if pd.notna(biz) and biz:
                key = (uid, "business", biz)
                self._history_by_user_sender.setdefault(key, []).append(row.to_dict())
            group = row.get("group_id", "")
            if pd.notna(group) and group:
                key = (uid, "group", group)
                self._history_by_user_sender.setdefault(key, []).append(row.to_dict())

        # Daily summary grouped by user_id (last 7 days)
        self._daily_summary_by_user = {}
        for _, row in self.daily_summary_df.iterrows():
            uid = row["user_id"]
            self._daily_summary_by_user.setdefault(uid, []).append(row.to_dict())

    # --- Public lookup methods ---

    def get_user_profile(self, user_id: str) -> dict | None:
        """Get user notification behavior profile."""
        return self._users_idx.get(user_id)

    def get_group_context(self, group_id: str, user_id: str) -> dict | None:
        """Get group metadata + user's membership context."""
        if not group_id or pd.isna(group_id):
            return None
        group = self._groups_idx.get(group_id)
        membership = self._group_members_idx.get((group_id, user_id))
        if not group:
            return None
        return {
            "group": group,
            "membership": membership
        }

    def get_business_context(self, business_id: str, user_id: str) -> dict | None:
        """Get business account metadata + user-business relationship."""
        if not business_id or pd.isna(business_id):
            return None
        business = self._business_idx.get(business_id)
        relationship = self._user_biz_idx.get((user_id, business_id))
        if not business:
            return None
        return {
            "business": business,
            "relationship": relationship
        }

    def get_user_history(self, user_id: str) -> list[dict]:
        """Get all historical messages for a user (with events)."""
        return self._history_by_user.get(user_id, [])

    def get_sender_history(self, user_id: str, sender_type: str, sender_id: str) -> list[dict]:
        """Get history for a specific sender to a specific user.

        Args:
            sender_type: 'sender', 'business', or 'group'
        """
        return self._history_by_user_sender.get((user_id, sender_type, sender_id), [])

    def get_image_path(self, media_id: str) -> str | None:
        """Get file path for an image."""
        img = self._images_idx.get(media_id)
        return img["file_path"] if img else None

    def get_voice_note_path(self, media_id: str) -> str | None:
        """Get file path for a voice note."""
        vn = self._voice_notes_idx.get(media_id)
        return vn["file_path"] if vn else None

    def get_notification_load(self, user_id: str, last_n_days: int = 7) -> list[dict]:
        """Get recent daily notification summary for a user."""
        entries = self._daily_summary_by_user.get(user_id, [])
        # Sort by date descending, return last N
        sorted_entries = sorted(entries, key=lambda x: x["date"], reverse=True)
        return sorted_entries[:last_n_days]

    def get_all_messages(self) -> list[dict]:
        """Get all 110 messages to route as list of dicts."""
        return self.messages_df.to_dict("records")
