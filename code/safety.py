"""
Safety module — rule-based safety overrides and post-processing.
Catches clear scam/spam/opt-out cases that should not depend on LLM judgment.
"""

import re

import pandas as pd


# Scam/phishing URL patterns
_SUSPICIOUS_URL_PATTERNS = [
    r'account-login\.\w+',
    r'verify-now\.\w+',
    r'secure-update\.\w+',
    r'amazonpay-\w+\.\w+',
    r'paytm-\w+\.\w+',
    r'gpay-\w+\.\w+',
    r'\w+-delivery\.\w+',
    r'\w+-verify\.\w+',
    r'\w+-secure\.\w+',
    r'bit\.ly/',
    r'tinyurl\.',
]

# Scam keyword patterns
_SCAM_PATTERNS = [
    r'(?:enter|share|send|reply\s+with)\s+(?:your\s+)?(?:otp|password|pin|cvv|login\s+code)',
    r'account\s+(?:will\s+be\s+)?(?:blocked|suspended|deactivated|locked)',
    r'(?:verify|confirm)\s+(?:your\s+)?(?:identity|account|password)',
    r'profile\s+(?:may\s+be\s+)?(?:temporarily\s+)?blocked',
    r'otp\s+(?:may\s+have\s+)?leaked',
    r'pay\s+(?:small\s+)?(?:reattempt|redelivery)\s+fee',
]

# Prompt injection patterns
_INJECTION_PATTERNS = [
    r'ignore\s+(?:all\s+)?(?:previous|prior|above)\s+(?:routing\s+)?(?:rules|instructions|prompts)',
    r'mark\s+this\s+(?:message\s+)?as\s+notify',
    r'override\s+(?:the\s+)?(?:routing|classification)',
    r'disregard\s+(?:your\s+)?(?:instructions|system\s+prompt)',
]


class SafetyChecker:
    """Rule-based safety layer that can override LLM routing decisions."""

    def __init__(self, data_loader):
        self.dl = data_loader
        # Compile regex patterns
        self._suspicious_urls = [re.compile(p, re.IGNORECASE) for p in _SUSPICIOUS_URL_PATTERNS]
        self._scam_patterns = [re.compile(p, re.IGNORECASE) for p in _SCAM_PATTERNS]
        self._injection_patterns = [re.compile(p, re.IGNORECASE) for p in _INJECTION_PATTERNS]

    def check(self, message: dict, context: dict, llm_result: dict) -> dict:
        """Apply safety checks and potentially override the LLM decision.

        Args:
            message: raw message dict from messages.csv
            context: assembled context dict
            llm_result: dict with action, message_type, reason, confidence

        Returns:
            Possibly modified result dict
        """
        result = llm_result.copy()
        text = str(message.get("message_text", "")) if pd.notna(message.get("message_text")) else ""

        # Also check media content
        msg_ctx = context.get("message", {})
        voice_text = msg_ctx.get("voice_transcription", "")
        image_text = msg_ctx.get("image_text", "")
        all_text = f"{text} {voice_text} {image_text}"

        # --- Hard overrides ---

        # 1. Prompt injection detection
        if self._has_injection(all_text):
            result["action"] = "mute"
            result["message_type"] = "scam"
            result["reason"] = (
                "The message tries to instruct the router, "
                "but the routing decision should be based on the actual content and risk."
            )
            result["confidence"] = max(float(result.get("confidence", 0.85)), 0.85)
            return result

        # 2. Scam/phishing detection
        scam_signal = self._detect_scam(all_text)
        if scam_signal:
            result["action"] = "mute"
            result["message_type"] = "scam"
            result["reason"] = scam_signal
            result["confidence"] = max(float(result.get("confidence", 0.85)), 0.83)
            return result

        # 3. User opted out of promotions from this business
        if self._user_opted_out_promotions(context):
            sender_ctx = context.get("sender_context", {})
            # Only override if LLM didn't already mute, and content is promotional
            if result["action"] != "mute" and result.get("message_type") in ("promotion", "business_update"):
                result["action"] = "mute"
                result["reason"] = (
                    "The user has opted out of or repeatedly dismissed similar marketing messages."
                )
                result["confidence"] = max(float(result.get("confidence", 0.81)), 0.81)
                return result

        # --- Soft adjustments ---

        # 4. High forward count → bias toward digest/mute
        fwd_count = int(message.get("forwarded_count", 0))
        if fwd_count >= 5 and result["action"] == "notify":
            # Heavily forwarded messages are rarely urgent for the user
            if result.get("message_type") not in ("urgent", "scam"):
                result["action"] = "digest"
                result["reason"] = result.get("reason", "") + " (Highly forwarded content is rarely urgent.)"
                result["confidence"] = min(float(result.get("confidence", 0.8)), 0.82)

        # 5. User has muted this group → lower notify to digest
        sender_ctx = context.get("sender_context", {})
        if sender_ctx.get("user_muted_group") and result["action"] == "notify":
            if result.get("message_type") not in ("urgent", "scam", "payment"):
                # Only keep notify for urgent/payment in muted groups
                if "@" + context.get("user", {}).get("id", "") not in text:
                    # Not directly mentioned
                    result["action"] = "digest"
                    result["confidence"] = min(float(result.get("confidence", 0.8)), 0.80)

        # 6. Unverified business with domain mismatch + high reports
        if sender_ctx.get("conversation_type") == "business":
            if (not sender_ctx.get("business_verified", True)
                    and sender_ctx.get("domain_match") is False
                    and sender_ctx.get("business_reports_30d", 0) > 5):
                result["action"] = "mute"
                result["message_type"] = "spam"
                result["reason"] = (
                    "Unverified business with domain mismatch and high report count."
                )
                result["confidence"] = 0.83

        # --- Confidence calibration ---
        # Clamp confidence to observed sample range
        conf = float(result.get("confidence", 0.8))
        conf = max(0.50, min(0.95, conf))
        result["confidence"] = round(conf, 2)

        return result

    def _has_injection(self, text: str) -> bool:
        """Check for prompt injection attempts."""
        for pattern in self._injection_patterns:
            if pattern.search(text):
                return True
        return False

    def _detect_scam(self, text: str) -> str | None:
        """Detect scam/phishing signals. Returns reason string or None."""
        # Check for suspicious URLs
        for pattern in self._suspicious_urls:
            if pattern.search(text):
                return "The message contains a suspicious URL that may be a phishing attempt."

        # Check for scam language patterns
        for pattern in self._scam_patterns:
            if pattern.search(text):
                return (
                    "The message uses pressure tactics to extract sensitive information "
                    "like OTP, password, or payment details."
                )

        return None

    def _user_opted_out_promotions(self, context: dict) -> bool:
        """Check if the user opted out of promotions from this business."""
        sender_ctx = context.get("sender_context", {})
        if sender_ctx.get("conversation_type") != "business":
            return False
        return sender_ctx.get("user_opted_out_promotions", False)
