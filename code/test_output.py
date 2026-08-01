"""
Test & Validation Script for Message Notification Router
Run: python3 code/test_output.py
"""

import csv
import sys
from pathlib import Path
from collections import Counter

DATASET_DIR = Path(__file__).resolve().parent.parent / "dataset"
OUTPUT_PATH = DATASET_DIR / "output.csv"
MESSAGES_PATH = DATASET_DIR / "messages.csv"
SAMPLE_PATH = DATASET_DIR / "sample_messages.csv"

ALLOWED_ACTIONS = {"notify", "digest", "mute"}
ALLOWED_TYPES = {
    "personal", "urgent", "event", "payment", "business_update",
    "promotion", "greeting", "forward", "spam", "scam", "unknown"
}

passed = 0
failed = 0


def check(condition: bool, label: str, detail: str = ""):
    global passed, failed
    if condition:
        passed += 1
        print(f"  ✅ {label}")
    else:
        failed += 1
        print(f"  ❌ {label}")
        if detail:
            print(f"     → {detail}")


def main():
    global passed, failed
    print("=" * 60)
    print("  Message Notification Router — Test Suite")
    print("=" * 60)

    # ── Test 1: File exists ──
    print("\n📄 Test 1: Output file exists")
    check(OUTPUT_PATH.exists(), f"output.csv exists at {OUTPUT_PATH}")
    if not OUTPUT_PATH.exists():
        print("\n💀 Cannot continue — output.csv not found. Run: python3 code/main.py")
        sys.exit(1)

    # ── Test 2: Load & column validation ──
    print("\n📋 Test 2: Column structure")
    with open(OUTPUT_PATH) as f:
        reader = csv.DictReader(f)
        output_rows = list(reader)
        columns = reader.fieldnames

    expected_cols = ["message_id", "action", "message_type", "reason", "confidence", "evidence_message_ids"]
    check(columns == expected_cols, "Columns match expected order",
          f"Got: {columns}" if columns != expected_cols else "")

    # ── Test 3: Row count matches messages.csv ──
    print("\n🔢 Test 3: Row count")
    with open(MESSAGES_PATH) as f:
        messages = list(csv.DictReader(f))
    check(len(output_rows) == len(messages),
          f"Row count: {len(output_rows)} output rows for {len(messages)} messages",
          f"Expected {len(messages)}, got {len(output_rows)}")

    # ── Test 4: Every message_id from messages.csv is in output ──
    print("\n🔑 Test 4: Message ID coverage")
    expected_ids = {m["message_id"] for m in messages}
    output_ids = {r["message_id"] for r in output_rows}
    missing = expected_ids - output_ids
    extra = output_ids - expected_ids
    check(len(missing) == 0, f"No missing message IDs", f"Missing: {missing}" if missing else "")
    check(len(extra) == 0, f"No extra message IDs", f"Extra: {extra}" if extra else "")
    check(len(output_ids) == len(output_rows), "No duplicate message IDs",
          f"Duplicates found" if len(output_ids) != len(output_rows) else "")

    # ── Test 5: Valid action values ──
    print("\n🎯 Test 5: Valid action values")
    invalid_actions = [(r["message_id"], r["action"]) for r in output_rows if r["action"] not in ALLOWED_ACTIONS]
    check(len(invalid_actions) == 0, f"All actions valid ({ALLOWED_ACTIONS})",
          f"Invalid: {invalid_actions[:5]}" if invalid_actions else "")

    # ── Test 6: Valid message_type values ──
    print("\n🏷️  Test 6: Valid message_type values")
    invalid_types = [(r["message_id"], r["message_type"]) for r in output_rows
                     if r["message_type"] not in ALLOWED_TYPES]
    check(len(invalid_types) == 0, f"All message_types valid",
          f"Invalid: {invalid_types[:5]}" if invalid_types else "")

    # ── Test 7: Confidence range ──
    print("\n📊 Test 7: Confidence values")
    confidences = []
    bad_conf = []
    for r in output_rows:
        try:
            c = float(r["confidence"])
            confidences.append(c)
            if not (0 <= c <= 1):
                bad_conf.append((r["message_id"], c))
        except ValueError:
            bad_conf.append((r["message_id"], r["confidence"]))
    check(len(bad_conf) == 0, "All confidence values in [0, 1]",
          f"Invalid: {bad_conf[:5]}" if bad_conf else "")

    if confidences:
        avg = sum(confidences) / len(confidences)
        min_c, max_c = min(confidences), max(confidences)
        check(0.5 <= avg <= 0.95, f"Mean confidence reasonable: {avg:.3f}",
              f"Mean {avg:.3f} seems unusual")
        print(f"     ℹ️  Range: {min_c:.2f}–{max_c:.2f}, Mean: {avg:.3f}")

    # ── Test 8: Reason is non-empty ──
    print("\n💬 Test 8: Reason field")
    empty_reasons = [r["message_id"] for r in output_rows if not r.get("reason", "").strip()]
    check(len(empty_reasons) == 0, "All rows have a non-empty reason",
          f"Empty reasons: {empty_reasons[:5]}" if empty_reasons else "")

    # ── Test 9: Evidence format ──
    print("\n🔍 Test 9: Evidence format")
    bad_evidence = []
    has_evidence = 0
    for r in output_rows:
        ev = r.get("evidence_message_ids", "").strip()
        if not ev:
            bad_evidence.append((r["message_id"], "empty"))
        elif ev == "none":
            pass  # Valid
        else:
            has_evidence += 1
            # Should be semicolon-separated message IDs
            ids = ev.split(";")
            for eid in ids:
                if not eid.startswith("message_"):
                    bad_evidence.append((r["message_id"], f"bad id: {eid}"))
    check(len(bad_evidence) == 0, "All evidence_message_ids properly formatted",
          f"Issues: {bad_evidence[:5]}" if bad_evidence else "")
    print(f"     ℹ️  {has_evidence}/{len(output_rows)} rows have historical evidence")

    # ── Test 10: Distribution sanity ──
    print("\n📈 Test 10: Distribution sanity")
    action_dist = Counter(r["action"] for r in output_rows)
    type_dist = Counter(r["message_type"] for r in output_rows)

    check(action_dist["notify"] >= 5, f"Enough notify actions: {action_dist['notify']}",
          "Too few notify actions — may be under-notifying")
    check(action_dist["digest"] >= 5, f"Enough digest actions: {action_dist['digest']}",
          "Too few digest actions")
    check(action_dist["mute"] >= 5, f"Enough mute actions: {action_dist['mute']}",
          "Too few mute actions")
    check(len(type_dist) >= 5, f"Using {len(type_dist)} different message types",
          "Too few message types — might be under-classifying")

    print(f"\n     Action distribution:")
    for a in ["notify", "digest", "mute"]:
        count = action_dist[a]
        pct = count / len(output_rows) * 100
        bar = "█" * int(pct / 2)
        print(f"       {a:8s}: {count:3d} ({pct:5.1f}%) {bar}")

    print(f"\n     Type distribution:")
    for t, count in type_dist.most_common():
        print(f"       {t:18s}: {count:3d}")

    # ── Test 11: Sample message accuracy ──
    print("\n🎯 Test 11: Sample message accuracy (vs sample_messages.csv)")
    with open(SAMPLE_PATH) as f:
        samples = list(csv.DictReader(f))

    # Build lookup for output
    output_by_id = {r["message_id"]: r for r in output_rows}

    # Note: sample messages have different IDs (sample_msg_XXX) than actual messages (msg_XXX)
    # So we can't directly compare. Instead, let's check quality of our predictions.
    print(f"     ℹ️  Sample file has {len(samples)} examples with IDs like '{samples[0]['message_id']}'")
    print(f"     ℹ️  Output file has IDs like '{output_rows[0]['message_id']}'")
    print(f"     ℹ️  Sample messages are separate from prediction target — used as format/style reference")

    # ── Test 12: Scam detection quality ──
    print("\n🛡️  Test 12: Scam/safety detection")
    # Load messages to check text content
    msg_by_id = {m["message_id"]: m for m in messages}
    # Use combined signals: scam keyword + pressure tactic (not just one keyword)
    scam_keywords = ["otp", "login code", "verify your account", "confirm password"]
    pressure_phrases = ["blocked", "suspended", "deactivated", "act now", "expire", "penalty",
                        "reply with", "share your", "send your", "enter your"]
    
    scam_messages = []
    for r in output_rows:
        msg = msg_by_id.get(r["message_id"], {})
        text = str(msg.get("message_text", "")).lower()
        has_scam_kw = any(kw in text for kw in scam_keywords)
        has_pressure = any(p in text for p in pressure_phrases)
        if has_scam_kw and has_pressure:
            scam_messages.append({
                "id": r["message_id"],
                "action": r["action"],
                "type": r["message_type"],
                "text_preview": text[:80]
            })

    if scam_messages:
        muted_scams = sum(1 for s in scam_messages if s["action"] == "mute")
        check(muted_scams == len(scam_messages),
              f"Scam-like messages muted: {muted_scams}/{len(scam_messages)}",
              f"Not muted: {[s for s in scam_messages if s['action'] != 'mute'][:3]}")
    else:
        print("  ℹ️  No obvious scam-keyword messages found to verify")

    # ── Summary ──
    print(f"\n{'=' * 60}")
    total = passed + failed
    if failed == 0:
        print(f"  🎉 ALL {total} TESTS PASSED!")
    else:
        print(f"  ⚠️  {passed}/{total} tests passed, {failed} failed")
    print(f"{'=' * 60}")

    return failed == 0


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
