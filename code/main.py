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
from rich.panel import Panel
from rich.table import Table
from rich.style import Style

from config import OUTPUT_PATH, OUTPUT_COLUMNS, OPENAI_API_KEY
from data_loader import DataLoader
from media_processor import MediaProcessor
from evidence_retriever import EvidenceRetriever
from context_assembler import ContextAssembler
from router import Router
from safety import SafetyChecker

console = Console()

def main():
    start_time = time.time()
    
    console.print(Panel.fit(
        "[bold cyan]Message Notification Router[/bold cyan]\n[dim]Hackathon Edition[/dim]",
        border_style="cyan"
    ))

    # --- Validate API key ---
    if not OPENAI_API_KEY:
        console.print("[bold red]❌ Error: OPEN_AI_API_KEY not found in environment or .env file[/bold red]")
        sys.exit(1)
    console.print(f"[green]✅ API key loaded[/green] [dim](ends with ...{OPENAI_API_KEY[-4:]})[/dim]\n")

    # --- Step 1: Load all data ---
    with console.status("[bold blue]📂 Step 1: Loading dataset...", spinner="dots"):
        dl = DataLoader()
        messages = dl.get_all_messages()
    
    console.print(f"  [green]✓[/green] Loaded {len(messages)} messages to route")
    console.print(f"  [green]✓[/green] {len(dl.users_df)} users, {len(dl.groups_df)} groups, {len(dl.business_accounts_df)} businesses")
    console.print(f"  [green]✓[/green] {len(dl.message_history_df)} historical messages with events\n")

    # --- Step 2: Pre-process media ---
    with console.status("[bold magenta]🎨 Step 2: Processing media files...", spinner="dots"):
        mp = MediaProcessor()
        media_results = mp.process_all(dl)
    console.print("  [green]✓[/green] Media processing complete: images and voice notes cached\n")

    # --- Step 3: Initialize components ---
    with console.status("[bold yellow]🔧 Step 3: Initializing routing components...", spinner="dots"):
        er = EvidenceRetriever(dl)
        ca = ContextAssembler(dl, media_results)
        router = Router()
        safety = SafetyChecker(dl)
    console.print("  [green]✓[/green] Evidence retriever, context assembler, router, safety checker ready\n")

    # --- Step 4: Route all messages ---
    console.print(f"[bold green]🚀 Step 4: Routing {len(messages)} messages...[/bold green]")
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
            
            task_id = progress.add_task("[cyan]Routing messages...", total=len(messages))
            
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
                    color = {"notify": "bold red", "digest": "bold yellow", "mute": "dim white"}.get(result["action"], "white")
                    
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
                    
                    progress.update(task_id, advance=1, description=f"[red]❌ Error on {msg_id}[/red]")

                results.append(result)

                # Small delay between API calls to avoid rate limiting
                if i < len(messages) - 1:
                    time.sleep(0.3)
                    
    except KeyboardInterrupt:
        console.print(Panel(
            f"[bold yellow]⚠️ Process cancelled by user![/bold yellow]\n[white]Saving {len(results)} completed messages...[/white]",
            border_style="yellow"
        ))

    # --- Step 5: Write output.csv ---
    with console.status("[bold cyan]💾 Step 5: Writing output.csv...", spinner="dots"):
        write_output(results)
    
    # --- Summary ---
    elapsed = time.time() - start_time
    
    console.print(Panel.fit(
        f"[bold green]✅ COMPLETE[/bold green] — {len(results)} messages routed in {elapsed:.1f}s",
        border_style="green"
    ))

    # Action Table
    action_table = Table(title="Action Distribution", title_style="bold blue")
    action_table.add_column("Action", style="cyan", no_wrap=True)
    action_table.add_column("Count", justify="right", style="magenta")
    action_table.add_column("Percentage", justify="right", style="green")
    action_table.add_column("Bar", justify="left")

    for action, count in sorted(action_counts.items()):
        pct = (count / len(results) * 100) if results else 0.0
        bar = "[blue]" + "█" * int(pct / 2) + "[/blue]"
        action_table.add_row(action, str(count), f"{pct:.1f}%", bar)

    # Type Table
    type_table = Table(title="Message Type Distribution", title_style="bold magenta")
    type_table.add_column("Type", style="cyan")
    type_table.add_column("Count", justify="right", style="magenta")

    for mtype, count in sorted(type_counts.items(), key=lambda x: -x[1]):
        type_table.add_row(mtype, str(count))

    console.print("\n")
    console.print(action_table)
    console.print("\n")
    console.print(type_table)
    console.print(f"\n[dim]Output saved to: {OUTPUT_PATH}[/dim]")


def write_output(results: list[dict]):
    """Write results to output.csv with exact column order."""
    with open(OUTPUT_PATH, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=OUTPUT_COLUMNS)
        writer.writeheader()
        for row in results:
            writer.writerow(row)


def validate_output():
    """Quick validation of the output file."""
    console.print("\n[bold cyan]🔍 Validating output.csv...[/bold cyan]")
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
