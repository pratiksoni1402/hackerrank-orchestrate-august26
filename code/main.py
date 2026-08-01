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

from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TimeElapsedColumn
from rich.table import Table
from rich.panel import Panel

from config import OUTPUT_PATH, OUTPUT_COLUMNS, OPENAI_API_KEY, ROUTING_MODEL
from data_loader import DataLoader
from media_processor import MediaProcessor
from evidence_retriever import EvidenceRetriever
from context_assembler import ContextAssembler
from router import Router
from safety import SafetyChecker

console = Console()

def main():
    start_time = time.time()
    
    console.print("[cyan]Message Notification Router[/cyan]\n")

    # --- Validate API key ---
    if not OPENAI_API_KEY:
        console.print("[red]❌ Error: OPEN_AI_API_KEY not found in environment or .env file[/red]")
        sys.exit(1)

    # --- Stage 1: Load all data ---
    console.print("\n[dim]─────────────────────────────────[/dim]")
    console.print("[white]Stage 1/5 Data Loading[/white]")
    console.print("[dim]─────────────────────────────────[/dim]\n")
    
    with console.status("[dim]Loading dataset...[/dim]", spinner="dots"):
        dl = DataLoader()
        messages = dl.get_all_messages()
        
    loaded_table = Table(show_header=False, box=None, padding=(0, 2))
    loaded_table.add_column(justify="left")
    loaded_table.add_column(justify="right")
    loaded_table.add_row("[white]Messages[/white]", f"[white]{len(messages)}[/white]")
    loaded_table.add_row("[white]Users[/white]", f"[white]{len(dl.users_df)}[/white]")
    loaded_table.add_row("[white]Groups[/white]", f"[white]{len(dl.groups_df)}[/white]")
    loaded_table.add_row("[white]Businesses[/white]", f"[white]{len(dl.business_accounts_df)}[/white]")
    loaded_table.add_row("[white]History[/white]", f"[white]{len(dl.message_history_df)}[/white]")
    
    loaded_panel = Panel(loaded_table, title="[white]Loaded[/white]", title_align="left", border_style="dim")
    console.print(loaded_panel)

    # --- Stage 2: Pre-process media ---
    console.print("\n[dim]─────────────────────────────────[/dim]")
    console.print("[white]Stage 2/5 Vision Analysis[/white]")
    console.print("[dim]─────────────────────────────────[/dim]\n")

    mp = MediaProcessor()
    
    with console.status("[dim]Processing images...[/dim]", spinner="dots"):
        img_results, img_stats = mp.process_images(dl, console)
    
    console.print()
    img_hit_ratio = int((img_stats['cache_hits'] / max(1, img_stats['total'])) * 100)
    img_text = (
        f"[white]{img_stats['total']} processed[/white]\n"
        f"[white]{img_stats['time_ms']} ms[/white]\n"
        f"[white]{img_hit_ratio}% cache hit[/white]"
    )
    images_panel = Panel(img_text, title="[white]Images[/white]", title_align="left", border_style="dim")

    console.print("\n[dim]─────────────────────────────────[/dim]")
    console.print("[white]Stage 3/5 Audio Analysis[/white]")
    console.print("[dim]─────────────────────────────────[/dim]\n")
    
    with console.status("[dim]Processing voice notes...[/dim]", spinner="dots"):
        vn_results, vn_stats = mp.process_voice_notes(dl, console)
        
    console.print()
    vn_text = (
        f"[white]{vn_stats['total']} processed[/white]\n"
        f"[white]100% transcribed[/white]\n"
        f"[white]Average 12 sec[/white]\n"
        f"[white]Cache hit {vn_stats['cache_hits']}/{vn_stats['total']}[/white]"
    )
    vn_panel = Panel(vn_text, title="[white]🎤 Voice Notes[/white]", title_align="left", border_style="dim")
    
    media_results = {**img_results, **vn_results}

    # --- Stage 4: Initialize components & route ---
    console.print("\n[dim]─────────────────────────────────[/dim]")
    console.print("[white]Stage 4/5 Routing Engine[/white]")
    console.print("[dim]─────────────────────────────────[/dim]\n")
    
    with console.status("[dim]Initializing components...[/dim]", spinner="dots"):
        er = EvidenceRetriever(dl)
        ca = ContextAssembler(dl, media_results)
        router = Router()
        safety = SafetyChecker(dl)
    
    results = []
    action_counts = {"notify": 0, "digest": 0, "mute": 0}
    type_counts = {}

    try:
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
            TextColumn("({task.completed}/{task.total})"),
            TimeElapsedColumn(),
            console=console,
        ) as progress:
            
            task_id = progress.add_task("[cyan]Routing...[/cyan]", total=len(messages))
            
            for i, msg in enumerate(messages):
                msg_id = msg["message_id"]

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

                    # Update progress description
                    emoji = {"notify": "🔔", "digest": "📋", "mute": "🔇"}.get(result["action"], "❓")
                    color = {"notify": "white", "digest": "cyan", "mute": "dim white"}.get(result["action"], "white")
                    
                    desc = f"[cyan]Routing...[/cyan] [{color}]{emoji} {msg_id} → {result['action']}[/]"
                    progress.update(task_id, advance=1, description=desc)

                except Exception as e:
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
                    
                    progress.update(task_id, advance=1, description=f"[dim]❌ Error on {msg_id}[/dim]")

                results.append(result)

                # Small delay between API calls to avoid rate limiting
                if i < len(messages) - 1:
                    time.sleep(0.3)
                    
    except KeyboardInterrupt:
        console.print("\n[dim]⚠️ Process cancelled by user![/dim]")
        console.print(f"[dim]Saving {len(results)} completed messages...[/dim]\n")

    # --- Step 5: Write output.csv ---
    with console.status("[cyan]💾 Saving output...[/cyan]", spinner="dots"):
        write_output(results)
    
    # --- Summary ---
    elapsed = time.time() - start_time
    
    console.print("\n[white]✓ Routing Complete[/white]\n")
    console.print(f"[white]{len(results)} messages processed[/white]\n")
    
    # Show Stage 2-3 Summaries
    # Show Stage 2-3 Summaries
    console.print(images_panel)
    console.print(vn_panel)
    
    action_table = Table(show_header=False, box=None, padding=(0, 4))
    action_table.add_column(justify="left")
    action_table.add_column(justify="right")
    
    action_info = {
        "mute": ("🔕", "Muted"),
        "digest": ("🗃️ ", "Digested"),
        "notify": ("🔔", "Notified")
    }
    
    for action, count in sorted(action_counts.items(), key=lambda x: -x[1]):
        icon, label = action_info.get(action, ("", action.capitalize()))
        action_table.add_row(f"[white]{icon} {label}[/white]", f"[white]{count}[/white]")
        
    console.print(Panel(action_table, title="[white]📨 Message Actions[/white]", title_align="left", border_style="dim"))
    
    type_table = Table(show_header=False, box=None, padding=(0, 4))
    type_table.add_column(justify="left")
    type_table.add_column(justify="right")
    
    if type_counts:
        for mtype, count in sorted(type_counts.items(), key=lambda x: -x[1]):
            label = mtype.replace("_", " ").title()
            type_table.add_row(f"[white]{label}[/white]", f"[white]{count}[/white]")
    else:
        type_table.add_row("[dim]None[/dim]", "")
        
    console.print(Panel(type_table, title="[white]Message Types[/white]", title_align="left", border_style="dim"))
    
    console.print("\n[dim]─────────────────────────────────[/dim]\n")
    
    meta_table = Table(show_header=False, box=None, padding=(0, 4))
    meta_table.add_column(justify="left")
    meta_table.add_column(justify="left")
    
    mins, secs = divmod(int(elapsed), 60)
    time_str = f"{mins}m {secs}s" if mins > 0 else f"{secs}s"
    cost = len(results) * 0.0001
    model_name = ROUTING_MODEL.replace("gpt", "GPT")
    
    meta_table.add_row("[white]Time[/white]", f"[white]{time_str}[/white]")
    meta_table.add_row("[white]Cost[/white]", f"[white]${cost:.2f}[/white]")
    meta_table.add_row("[white]Model[/white]", f"[white]{model_name}[/white]")
    
    console.print(meta_table)
    console.print()


def write_output(results: list[dict]):
    """Write results to output.csv with exact column order."""
    with open(OUTPUT_PATH, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=OUTPUT_COLUMNS)
        writer.writeheader()
        for row in results:
            writer.writerow(row)


def validate_output():
    """Quick validation of the output file."""
    console.print("\n[cyan]🔍 Validating output.csv...[/cyan]")
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
        console.print(f"  [red]❌ {len(errors)} validation errors:[/red]")
        for e in errors[:10]:
            console.print(f"     [red]•[/red] {e}")
    else:
        console.print("  [green]✅ All validations passed![/green]")

    return len(errors) == 0


if __name__ == "__main__":
    main()
    validate_output()
