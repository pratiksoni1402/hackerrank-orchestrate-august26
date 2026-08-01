"""
Context assembler — builds a comprehensive context object for each incoming message
by combining user profile, group/business/sender context, media content, and evidence.
"""

import pandas as pd


class ContextAssembler:
    """Assembles per-message context for the LLM routing prompt."""

    def __init__(self, data_loader, media_results: dict):
        self.dl = data_loader
        self.media_results = media_results  # media_id -> processed content

    def assemble(self, message: dict, evidence: list[dict]) -> dict:
        """Build full context for a single message.

        Args:
            message: dict from messages.csv
            evidence: list of evidence dicts from EvidenceRetriever

        Returns:
            dict with all assembled context
        """
        user_id = message["user_id"]
        context = {
            "message": self._build_message_context(message),
            "user": self._build_user_context(user_id),
            "sender_context": self._build_sender_context(message),
            "evidence": evidence,
        }
        return context

    def _build_message_context(self, message: dict) -> dict:
        """Build the message-specific context."""
        msg_ctx = {
            "id": message["message_id"],
            "conversation_type": message["conversation_type"],
            "text": str(message.get("message_text", "")) if pd.notna(message.get("message_text")) else "",
            "forwarded_count": int(message.get("forwarded_count", 0)),
            "timestamp": str(message.get("created_at", "")),
        }

        # Add media content if present
        media_type = message.get("media_type", "")
        media_id = message.get("media_id", "")
        if pd.notna(media_type) and media_type and pd.notna(media_id) and media_id:
            msg_ctx["media_type"] = media_type
            msg_ctx["media_id"] = media_id
            media_content = self.media_results.get(media_id)
            if media_content:
                if media_type == "image":
                    msg_ctx["image_description"] = media_content.get("description", "")
                    msg_ctx["image_text"] = media_content.get("extracted_text", "")
                    msg_ctx["image_category"] = media_content.get("category", "")
                    msg_ctx["image_risk_signals"] = media_content.get("risk_signals", [])
                elif media_type == "voice":
                    msg_ctx["voice_transcription"] = media_content.get("transcription", "")
        else:
            msg_ctx["media_type"] = "none"

        return msg_ctx

    def _build_user_context(self, user_id: str) -> dict:
        """Build the receiving user's profile context."""
        profile = self.dl.get_user_profile(user_id)
        if not profile:
            return {"id": user_id, "profile": "unknown"}

        # Recent notification load
        load = self.dl.get_notification_load(user_id, last_n_days=7)
        avg_notifications = 0
        avg_dismissed = 0
        if load:
            avg_notifications = sum(d.get("notifications_sent", 0) for d in load) / len(load)
            avg_dismissed = sum(d.get("notifications_dismissed", 0) for d in load) / len(load)

        return {
            "id": user_id,
            "dnd_window": profile.get("do_not_disturb_window", ""),
            "messages_opened_30d": profile.get("messages_opened_30d", 0),
            "messages_replied_30d": profile.get("messages_replied_30d", 0),
            "notifications_dismissed_30d": profile.get("notifications_dismissed_30d", 0),
            "messages_reported_30d": profile.get("messages_reported_30d", 0),
            "avg_daily_notifications_7d": round(avg_notifications, 1),
            "avg_daily_dismissed_7d": round(avg_dismissed, 1),
        }

    def _build_sender_context(self, message: dict) -> dict:
        """Build context about the sender (varies by conversation type)."""
        user_id = message["user_id"]
        conv_type = message["conversation_type"]
        ctx = {"conversation_type": conv_type}

        if conv_type == "group":
            group_id = message.get("group_id", "")
            if pd.notna(group_id) and group_id:
                group_ctx = self.dl.get_group_context(group_id, user_id)
                if group_ctx:
                    g = group_ctx["group"]
                    ctx["group_name"] = g.get("group_name", "")
                    ctx["group_type"] = g.get("group_type", "")
                    ctx["group_member_count"] = g.get("member_count", 0)
                    ctx["group_messages_30d"] = g.get("messages_30d", 0)

                    m = group_ctx.get("membership")
                    if m:
                        ctx["user_role_in_group"] = m.get("role", "member")
                        ctx["user_messages_sent_30d"] = m.get("messages_sent_30d", 0)
                        ctx["user_messages_read_30d"] = m.get("messages_read_30d", 0)
                        ctx["user_dismissed_in_group_30d"] = m.get("notifications_dismissed_30d", 0)
                        ctx["user_muted_group"] = bool(m.get("group_muted_by_user", 0))

            # Sender info within group
            sender_id = message.get("sender_user_id", "")
            if pd.notna(sender_id) and sender_id:
                ctx["sender_user_id"] = sender_id
                # Check if sender is admin in this group
                group_id = message.get("group_id", "")
                if pd.notna(group_id) and group_id:
                    sender_membership = self.dl._group_members_idx.get((group_id, sender_id))
                    if sender_membership:
                        ctx["sender_role_in_group"] = sender_membership.get("role", "member")

        elif conv_type == "business":
            business_id = message.get("business_id", "")
            if pd.notna(business_id) and business_id:
                biz_ctx = self.dl.get_business_context(business_id, user_id)
                if biz_ctx:
                    b = biz_ctx["business"]
                    ctx["business_name"] = b.get("display_name", "")
                    ctx["business_brand"] = b.get("brand_name", "")
                    ctx["business_category"] = b.get("category", "")
                    ctx["business_verified"] = bool(b.get("verified", 0))
                    ctx["business_domain"] = b.get("official_domain", "")
                    ctx["domain_used_by_sender"] = b.get("domain_used_by_sender", "")
                    ctx["domain_match"] = (
                        b.get("official_domain", "") == b.get("domain_used_by_sender", "")
                        if b.get("official_domain") and b.get("domain_used_by_sender")
                        else None
                    )
                    ctx["business_account_age_days"] = b.get("account_age_days", 0)
                    ctx["business_reports_30d"] = b.get("user_reports_30d", 0)

                    r = biz_ctx.get("relationship")
                    if r:
                        ctx["user_relationship"] = r.get("why_user_knows_account", "")
                        ctx["user_allows_promotions"] = bool(r.get("allows_promotions", 0))
                        ctx["user_opted_out_promotions"] = bool(
                            pd.notna(r.get("promotions_opted_out_at"))
                            and r.get("promotions_opted_out_at")
                        )
                        ctx["user_activity_count_180d"] = r.get("activity_count_180d", 0)
                        ctx["user_opened_from_biz_30d"] = r.get("messages_opened_30d", 0)
                        ctx["user_dismissed_from_biz_30d"] = r.get("messages_dismissed_30d", 0)
                    else:
                        ctx["user_relationship"] = "none"
                        ctx["user_allows_promotions"] = False

        elif conv_type == "personal":
            sender_id = message.get("sender_user_id", "")
            if pd.notna(sender_id) and sender_id:
                ctx["sender_user_id"] = sender_id
                # Check sender's history with this user
                sender_history = self.dl.get_sender_history(user_id, "sender", sender_id)
                ctx["past_messages_from_sender"] = len(sender_history)
                if sender_history:
                    opened = sum(1 for h in sender_history if h.get("message_opened") == 1)
                    replied = sum(1 for h in sender_history if h.get("message_replied") == 1)
                    reported = sum(1 for h in sender_history if h.get("message_reported") == 1)
                    ctx["sender_opened_rate"] = opened / len(sender_history)
                    ctx["sender_replied_rate"] = replied / len(sender_history)
                    ctx["sender_reported"] = reported > 0
                else:
                    ctx["sender_is_new"] = True

        return ctx

    def format_for_prompt(self, context: dict) -> str:
        """Format the assembled context into a readable string for the LLM prompt."""
        parts = []

        # Message
        msg = context["message"]
        parts.append("=== INCOMING MESSAGE ===")
        parts.append(f"Message ID: {msg['id']}")
        parts.append(f"Conversation Type: {msg['conversation_type']}")
        parts.append(f"Timestamp: {msg['timestamp']}")
        parts.append(f"Forwarded Count: {msg['forwarded_count']}")

        if msg.get("text"):
            parts.append(f"Text Content: \"{msg['text']}\"")
        else:
            parts.append("Text Content: (none)")

        if msg.get("media_type") and msg["media_type"] != "none":
            parts.append(f"Media Type: {msg['media_type']}")
            if msg.get("image_description"):
                parts.append(f"Image Description: {msg['image_description']}")
            if msg.get("image_text"):
                parts.append(f"Image Text (OCR): \"{msg['image_text']}\"")
            if msg.get("image_category"):
                parts.append(f"Image Category: {msg['image_category']}")
            if msg.get("image_risk_signals"):
                parts.append(f"Image Risk Signals: {', '.join(msg['image_risk_signals'])}")
            if msg.get("voice_transcription"):
                parts.append(f"Voice Transcription: \"{msg['voice_transcription']}\"")

        # User
        user = context["user"]
        parts.append("\n=== RECEIVING USER ===")
        parts.append(f"User ID: {user['id']}")
        if user.get("dnd_window"):
            parts.append(f"Do Not Disturb Window: {user['dnd_window']}")
        parts.append(
            f"30-Day Behavior: opened={user.get('messages_opened_30d',0)}, "
            f"replied={user.get('messages_replied_30d',0)}, "
            f"dismissed={user.get('notifications_dismissed_30d',0)}, "
            f"reported={user.get('messages_reported_30d',0)}"
        )
        parts.append(
            f"Recent Load (7d avg): {user.get('avg_daily_notifications_7d',0)} notifications/day, "
            f"{user.get('avg_daily_dismissed_7d',0)} dismissed/day"
        )

        # Sender context
        sender = context["sender_context"]
        parts.append("\n=== SENDER / CONVERSATION CONTEXT ===")
        for k, v in sender.items():
            if k == "conversation_type":
                continue
            parts.append(f"  {k}: {v}")

        # Evidence
        parts.append("\n=== HISTORICAL EVIDENCE ===")
        if context["evidence"]:
            for i, ev in enumerate(context["evidence"], 1):
                parts.append(
                    f"  Evidence {i}: [{ev['message_id']}] "
                    f"User reaction: {ev['user_reaction']}. "
                    f"Content: \"{ev['text_snippet']}\""
                )
        else:
            parts.append("  No relevant historical messages found.")

        return "\n".join(parts)
