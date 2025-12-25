# LLM Subtitle Translator

A robust, context-aware subtitle translation tool powered by Google's Gemini 3 models (Flash & Pro).

This tool is designed to translate VTT and SRT subtitles while maintaining conversational context, specific terminology, and tone. It supports specific show configurations (like "Jet Lag: The Game") and uses a hybrid model approach to balance cost and quality.

## Features

-   **Context-Aware**: Mainains a sliding window of previous lines to ensure consistent context (e.g., "back" direction vs. body part).
-   **Hybrid Model Strategy**: Uses `gemini-3-flash` for speed/cost and automatically falls back to `gemini-3-pro` for low-confidence translations (interactive mode).
-   **Show-Specific Configurations**: Define system prompts and glossaries in simple YAML files (see `contexts/jetlag.yaml`).
-   **Resumable**: Automatically detects existing progress and resumes translation if interrupted.
-   **Batch API Support**: Option to use Gemini Batch API for large bulk translations at 50% cost.
-   **Statistics**: Detailed reporting on token usage, fallback rates, and confidence scores.

## Installation

1.  **Clone the repository**:
    ```bash
    git clone https://github.com/lahdekorpi/LLM-Subtitle-Translator.git
    cd LLM-Subtitle-Translator
    ```

2.  **Create a virtual environment**:
    ```bash
    python3 -m venv venv
    source venv/bin/activate
    ```

3.  **Install dependencies**:
    ```bash
    pip install -r requirements.txt
    ```

4.  **Configure Environment**:
    Copy the example environment file and add your Gemini API Key.
    ```bash
    cp .env.example .env
    # Edit .env and paste your key from https://aistudio.google.com/app/apikey
    ```

## Usage

### 1. Interactive Mode (Default)
Best for single episodes or when you need immediate results. The tool translates line-by-line in real-time.

*   **Real-time Progress**: Shows a progress bar.
*   **Auto-Fallback**: If the primary model (Flash) produces low confidence, it **automatically** retries with the Pro model.
*   **Resumable**: Can be stopped and restarted without losing progress.

```bash
# Translates immediately
python3 main.py --inputs "episode.en.vtt" --show-config contexts/jetlag.yaml
```

### 2. Batch Mode (Cost Saving)
Best for bulk processing entire seasons. Submits jobs to Google's queue (50% cheaper) but takes longer (hours).

```bash
# Submit job
python3 main.py --inputs "season1/" --batch --show-config contexts/jetlag.yaml

# Check status later
python3 main.py --check-batch <JOB_ID>
```

### Options
-   `--inputs`: **(Required)** One or more files (`.vtt`, `.srt`) or directories to process.
-   `--show-config`: Path to the YAML file containing the system prompt and glossary.
-   `--config`: Path to the global config (defaults to `config.yaml`).
-   `--batch`: If set, submits jobs to the Gemini Batch API (50% cheaper, async). **Default: Interactive Mode** (real-time).
-   `--check-batch <ID>`: Check status of a submitted batch job.
-   `--dry-run`: Preview what files will be processed without running.

## Configuration

**Global Settings (`config.yaml`)**:
Control models, temperature, batch size, and fallback thresholds.

**Show Contexts (`contexts/*.yaml`)**:
Define the persona, tone, and specific glossary terms.
```yaml
system_prompt: |
  You are translating a travel show. Tone: Witty, fast-paced.
glossary:
  game_mechanics:
    "Tag": { "translations": {"fi": "Hippa"}, "guidance": "The game name." }
```

## License

MIT License
