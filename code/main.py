"""
Message Notification Router — Main Entry Point

Orchestrates the full pipeline:
1. Load all data
2. Pre-process media (images + voice notes)
3. For each message: retrieve evidence → assemble context → route via LLM → safety check
4. Write output.csv
"""

import csv
import sys
import time
from pathlib import Path

# Ensure code/ is on the path
sys.path.insert(0, str(Path(__file__).resolve().parent))

from config import OUTPUT_PATH, OUTPUT_COLUMNS, OPENAI_API_KEY
from data_loader import DataLoader
from media_processor import MediaProcessor
from evidence_retriever import EvidenceRetriever
from context_assembler import ContextAssembler
from router import Router
from safety import SafetyChecker


def main():
    start_time = time.time()
    print("=" * 60)
    print("  Message Notification Router")
    print("=" * 60)

    # --- Validate API key ---
    if not OPENAI_API_KEY:
        print("❌ Error: OPEN_AI_API_KEY not found in environment or .env file")
        sys.exit(1)
    print(f"✅ API key loaded (ends with ...{OPENAI_API_KEY[-4:]})")

    # --- Step 1: Load all data ---
    print("\n📂 Step 1: Loading dataset...")
    dl = DataLoader()
    messages = dl.get_all_messages()
    print(f"   Loaded {len(messages)} messages to route")
    print(f"   {len(dl.users_df)} users, {len(dl.groups_df)} groups, "
          f"{len(dl.business_accounts_df)} businesses")
    print(f"   {len(dl.message_history_df)} historical messages with events")

    # --- Step 2: Pre-process media ---
    print("\n🎨 Step 2: Processing media files...")
    mp = MediaProcessor()
    media_results = mp.process_all(dl)

    # --- Step 3: Initialize components ---
    print("\n🔧 Step 3: Initializing routing components...")
    er = EvidenceRetriever(dl)
    ca = ContextAssembler(dl, media_results)
    router = Router()
    safety = SafetyChecker(dl)
    print("   Evidence retriever, context assembler, router, safety checker ready")

    # --- Step 4: Route all messages ---
    print(f"\n🚀 Step 4: Routing {len(messages)} messages...")
    results = []
    action_counts = {"notify": 0, "digest": 0, "mute": 0}
    type_counts = {}

    for i, msg in enumerate(messages):
        msg_id = msg["message_id"]
        progress = f"[{i+1}/{len(messages)}]"

        try:
            # 4a. Retrieve evidence
            evidence = er.find_evidence(msg)

            # 4b. Assemble context
            context = ca.assemble(msg, evidence)
            formatted_context = ca.format_for_prompt(context)

            # 4c. Route via LLM
            llm_result = router.route(formatted_context, evidence)

            # 4d. Safety post-check
            final_result = safety.check(msg, context, llm_result)

            # Ensure evidence_message_ids is set
            if "evidence_message_ids" not in final_result or not final_result["evidence_message_ids"]:
                final_result["evidence_message_ids"] = er.format_evidence_ids(evidence)

            result = {
                "message_id": msg_id,
                "action": final_result["action"],
                "message_type": final_result["message_type"],
                "reason": final_result["reason"],
                "confidence": final_result["confidence"],
                "evidence_message_ids": final_result["evidence_message_ids"]
            }

            action_counts[result["action"]] = action_counts.get(result["action"], 0) + 1
            type_counts[result["message_type"]] = type_counts.get(result["message_type"], 0) + 1

            # Print progress
            emoji = {"notify": "🔔", "digest": "📋", "mute": "🔇"}.get(result["action"], "❓")
            print(f"   {progress} {emoji} {msg_id} → {result['action']} "
                  f"({result['message_type']}, conf={result['confidence']})")

        except Exception as e:
            print(f"   {progress} ❌ {msg_id} → ERROR: {e}")
            # Fallback for any error
            evidence = er.find_evidence(msg)
            result = {
                "message_id": msg_id,
                "action": "digest",
                "message_type": "unknown",
                "reason": f"Routing error: {str(e)[:100]}",
                "confidence": 0.5,
                "evidence_message_ids": er.format_evidence_ids(evidence)
            }
            action_counts["digest"] += 1
            type_counts["unknown"] = type_counts.get("unknown", 0) + 1

        results.append(result)

        # Small delay between API calls to avoid rate limiting
        if i < len(messages) - 1:
            time.sleep(0.3)

    # --- Step 5: Write output.csv ---
    print(f"\n💾 Step 5: Writing output.csv...")
    write_output(results)

    # --- Summary ---
    elapsed = time.time() - start_time
    print(f"\n{'=' * 60}")
    print(f"  ✅ COMPLETE — {len(results)} messages routed in {elapsed:.1f}s")
    print(f"{'=' * 60}")
    print(f"\n  Action distribution:")
    for action, count in sorted(action_counts.items()):
        pct = count / len(results) * 100
        bar = "█" * int(pct / 2)
        print(f"    {action:8s}: {count:3d} ({pct:5.1f}%) {bar}")

    print(f"\n  Message type distribution:")
    for mtype, count in sorted(type_counts.items(), key=lambda x: -x[1]):
        print(f"    {mtype:18s}: {count:3d}")

    print(f"\n  Output saved to: {OUTPUT_PATH}")


def write_output(results: list[dict]):
    """Write results to output.csv with exact column order."""
    with open(OUTPUT_PATH, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=OUTPUT_COLUMNS)
        writer.writeheader()
        for row in results:
            writer.writerow(row)
    print(f"   ✅ Wrote {len(results)} rows to {OUTPUT_PATH}")


def validate_output():
    """Quick validation of the output file."""
    print("\n🔍 Validating output.csv...")
    with open(OUTPUT_PATH) as f:
        reader = csv.DictReader(f)
        rows = list(reader)

    errors = []
    if len(rows) != 110:
        errors.append(f"Expected 110 rows, got {len(rows)}")

    if reader.fieldnames != OUTPUT_COLUMNS:
        errors.append(f"Column mismatch: {reader.fieldnames}")

    for i, row in enumerate(rows):
        if row["action"] not in ("notify", "digest", "mute"):
            errors.append(f"Row {i+1}: invalid action '{row['action']}'")
        if row["message_type"] not in (
            "personal", "urgent", "event", "payment", "business_update",
            "promotion", "greeting", "forward", "spam", "scam", "unknown"
        ):
            errors.append(f"Row {i+1}: invalid message_type '{row['message_type']}'")
        try:
            conf = float(row["confidence"])
            if not (0 <= conf <= 1):
                errors.append(f"Row {i+1}: confidence {conf} out of range")
        except ValueError:
            errors.append(f"Row {i+1}: invalid confidence '{row['confidence']}'")

    if errors:
        print(f"   ❌ {len(errors)} validation errors:")
        for e in errors[:10]:
            print(f"      • {e}")
    else:
        print("   ✅ All validations passed!")

    return len(errors) == 0


if __name__ == "__main__":
    main()
    validate_output()
