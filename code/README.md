# HackerRank Orchestrate: Message Notification Router

This is the solution for the HackerRank Orchestrate Message Notification Router hackathon.

## System Architecture

The solution uses a multi-agent architecture with **gpt-4o-mini** and **whisper-1**:
1. **Screening Agent**: A fast LLM pass to filter obvious scams/spam.
2. **Main Routing Agent**: A robust routing engine that handles complex contextual rules.
3. **Safety Checker**: A deterministic rule-based layer that overrides the LLM on prompt injections, verified scams, and business promotion opt-outs.
4. **Context Assembler & Evidence Retriever**: Builds comprehensive prompt context by matching historical user reactions, group behaviors, and content similarity.

## Setup Instructions

1. Ensure you have Python 3.10+ installed.
2. Create and activate a virtual environment (recommended):
   ```bash
   # If you are stuck in an old or broken environment, deactivate it first:
   # deactivate
   
   python3 -m venv .venv
   source .venv/bin/activate
   ```
3. Install the required dependencies:
   ```bash
   pip install -r requirements.txt
   ```
   *Note: `pillow-avif-plugin` is included to support transcoding of AVIF WhatsApp images to JPEG natively.*
4. Set up your environment variables. You must provide an OpenAI API key. Create a `.env` file in the project root or export it directly:
   ```bash
   export OPEN_AI_API_KEY="sk-your-openai-api-key"
   ```

## Running the Pipeline

To run the full pipeline and generate predictions for all messages:

```bash
python3 main.py
```

The script will:
1. Load all dataset CSVs from `../dataset`.
2. Process images and voice notes concurrently (caching results to `.cache/` to save API costs on repeated runs).
3. Route all 110 messages in parallel using `asyncio` and `AsyncOpenAI`.
4. Output the detailed logs and progress to the terminal using `rich`.
5. Save the final predictions to `../dataset/output.csv`.

## Validating Output

The `main.py` script automatically runs a quick validation check (`validate_output()`) at the end of execution to ensure:
- Exactly 110 rows are generated.
- The output columns strictly match the requested order.
- Action and message types belong to the allowed sets.
- Confidence scores fall within [0.0, 1.0].

To run evaluation scripts separately (if any are provided in `evaluation/`), refer to the scripts in that directory.
