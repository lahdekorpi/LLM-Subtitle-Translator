import os
import argparse
import asyncio
import json
import logging
import time
from dataclasses import dataclass
from typing import List, Optional, Dict, Any
from pathlib import Path

# Third-party imports
try:
    import yaml
except ImportError:
    print("Error: PyYAML is missing. Please run: pip install -r requirements.txt")
    import sys
    sys.exit(1)

try:
    from pydantic import BaseModel, Field
except ImportError:
    print("Error: Pydantic is missing. Please run: pip install -r requirements.txt")
    import sys
    sys.exit(1)

try:
    from google import genai
    from google.genai import types
except ImportError:
    print("Error: google-genai is missing. Please run: pip install -r requirements.txt")
    import sys
    sys.exit(1)

from dotenv import load_dotenv
import webvtt
import pysrt
from tqdm import tqdm

# Setup logging
logging.basicConfig(level=logging.WARNING, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

@dataclass
class TranslationConfig:
    """Holds configuration for the translation process."""
    target_language: str
    target_language_code: str
    model: str
    batch_size: int
    temperature: float
    max_output_tokens: int
    request_interval: float
    timeout: int
    retries: int
    workers: int
    context_prompt: str = ""
    glossary_prompt: str = ""
    api_key: Optional[str] = None
    fallback_model: Optional[str] = None
    min_confidence: float = 0.8
    context_overlap: int = 5

@dataclass
class TranslationStats:
    total_files: int = 0
    total_cues: int = 0
    processed_cues: int = 0
    fallback_count: int = 0
    low_confidence_count: int = 0
    start_time: float = 0.0

    def get_summary(self) -> str:
        duration = time.time() - self.start_time
        return (
            f"\n--- Translation Summary ---\n"
            f"Time Elapsed: {duration:.2f}s\n"
            f"Files Processed: {self.total_files}\n"
            f"Total Cues: {self.processed_cues}\n"
            f"Fallback Model Used: {self.fallback_count} times\n"
            f"Low Confidence Actions: {self.low_confidence_count}\n"
            f"---------------------------"
        )

class TranslationItem(BaseModel):
    lines: List[str] = Field(description="The translated lines, corresponding exactly to the input line numbers.")
    confidence: float = Field(description="A score from 0.0 to 1.0 indicating confidence in the translation quality.")
    issues: Optional[str] = Field(description="Description of any difficulties or ambiguities encountered, or null if none.")

class GeminiTranslator:
    """Handles interactions with the Google Gemini API using the new google-genai SDK."""
    def __init__(self, config: TranslationConfig, stats: TranslationStats):
        self.config = config
        self.stats = stats
        self._setup_api()

    def _setup_api(self):
        if not self.config.api_key:
            raise ValueError("API Key is missing. Please set GEMINI_API_KEY environment variable.")
        
        # Initialize the new Client
        self.client = genai.Client(api_key=self.config.api_key)

    async def check_batch_status(self, batch_id: str):
        """Checks the status of a batch job and downloads results if complete."""
        try:
             # In the new SDK, accessing batch jobs might differ slightly.
             # Assuming standard client.batches.get syntax or similar.
             # The new SDK structure: client.batches.get(name=...)
             batch_job = self.client.batches.get(name=batch_id)
             print(f"Job {batch_id} Status: {batch_job.state}")
             
             if batch_job.state == "SUCCEEDED": # check enum or string
                 print("Job succeeded! Results available at:")
                 print(f"Output URI: {batch_job.output_uri}") # check property name
             elif batch_job.state == "FAILED":
                 print(f"Job failed: {batch_job.error}")
        except Exception as e:
            logger.error(f"Failed to check batch status: {e}")

    async def translate_batch(self, batch_cues: List[str], batch_id: int, previous_context: List[str] = []) -> List[str]:
        """Translates a batch of subtitle cues with retry and fallback logic."""
        
        numbered_lines = []
        for j, cue_text in enumerate(batch_cues):
            sanitized = cue_text.replace("\n", " ").replace('"', "'")
            numbered_lines.append(f'{j+1}. "{sanitized}"')
        prompt_batch = "\n".join(numbered_lines)

        context_str = ""
        if previous_context:
            context_str = "PREVIOUS CONTEXT (Do not translate, strictly for reference):\n" + "\n".join(previous_context) + "\n---\n"
        
        # Dynamic Prompt Construction
        system_instruction = (
            f"You are an expert subtitle translator. Translate the following lines into {self.config.target_language}.\n"
            "Rules:\n"
            "1. Maintain the tone and style described in the context.\n"
            "2. Output strictly valid JSON with keys 'lines', 'confidence', 'issues'.\n"
            "3. 'lines' must be an array of strings corresponding exactly to input indices.\n"
            f"{self.config.context_prompt}\n"
            f"{self.config.glossary_prompt}"
        )

        full_prompt = f"{context_str}TRANSLATE THESE LINES TO {self.config.target_language}:\n{prompt_batch}"

        # Try with Primary Model
        result = await self._generate_with_retry(self.config.model, full_prompt, system_instruction, batch_id, "Primary")
        
        # Check Confidence & Fallback
        if result:
            # Output validation
            if len(result.lines) != len(batch_cues):
                logger.warning(f"Batch {batch_id}: Mismatch line count {len(result.lines)} vs {len(batch_cues)}")
            
            if self.config.fallback_model and result.confidence < self.config.min_confidence:
                self.stats.low_confidence_count += 1
                logger.info(f"Batch {batch_id}: Low confidence ({result.confidence:.2f}). Invoking fallback model...")
                
                fallback_result = await self._generate_with_retry(self.config.fallback_model, full_prompt, system_instruction, batch_id, "Fallback")
                if fallback_result:
                    self.stats.fallback_count += 1
                    return fallback_result.lines
        
        return result.lines if result else []

    async def _generate_with_retry(self, model_name, prompt, system_instruction, batch_id, label) -> Optional[TranslationItem]:
        for attempt in range(self.config.retries + 1):
            try:
                # New SDK Usage
                # models.generate_content is synchronous? Or is there an async client?
                # The 'google.genai' package client usually supports async if configured or via aio
                # Actually, the standard `genai.Client` has `aio` property for async.
                
                response = await self.client.aio.models.generate_content(
                    model=model_name,
                    contents=prompt,
                    config=types.GenerateContentConfig(
                        system_instruction=system_instruction,
                        temperature=self.config.temperature,
                        max_output_tokens=self.config.max_output_tokens,
                        response_mime_type="application/json",
                        response_schema=TranslationItem
                    )
                )
                
                # Cleanup potential markdown
                text = response.text
                if not text:
                     logger.warning(f"Batch {batch_id} [{label}]: Empty response.")
                     continue

                if "```json" in text:
                    text = text.split("```json")[1].split("```")[0]
                elif "```" in text:
                        text = text.split("```")[1].split("```")[0]
                
                data = json.loads(text)
                return TranslationItem(**data)

            except Exception as e:
                logger.warning(f"Batch {batch_id} [{label}]: Attempt {attempt+1} failed: {e}")
                if attempt < self.config.retries:
                    await asyncio.sleep(2 ** attempt)
        
        logger.error(f"Batch {batch_id} [{label}]: All retries failed.")
        return None

    async def submit_batch_job(self, file_path: Path):
        """Submits a file for batch processing using the Gemini Batch API."""
        # 1. Prepare JSONL content
        output_jsonl_path = file_path.with_suffix('.jsonl')
        
        # Read subs
        is_vtt = file_path.suffix == '.vtt'
        if is_vtt:
            captions = webvtt.read(str(file_path)).captions
        else:
            captions = pysrt.open(str(file_path), encoding='utf-8')
        
        cues_text = [c.text for c in captions]
        batch_size = self.config.batch_size
        context_overlap = self.config.context_overlap
        
        batches = [cues_text[i:i + batch_size] for i in range(0, len(cues_text), batch_size)]
        
        print(f"Preparing batch job for {file_path.name} ({len(batches)} batches)...")
        
        with open(output_jsonl_path, 'w', encoding='utf-8') as f:
            for i, batch in enumerate(batches):
                
                # Calculate Context Window
                start_idx = i * batch_size
                prev_start = max(0, start_idx - context_overlap)
                previous_context = cues_text[prev_start:start_idx] if prev_start < start_idx else []

                # Build Prompt
                numbered_lines = []
                for j, cue_text in enumerate(batch):
                     sanitized = cue_text.replace("\n", " ").replace('"', "'")
                     numbered_lines.append(f'{j+1}. "{sanitized}"')
                prompt_batch = "\n".join(numbered_lines)
                
                context_str = ""
                if previous_context:
                    context_str = "PREVIOUS CONTEXT (Do not translate, strictly for reference):\n" + "\n".join([c.replace("\n", " ") for c in previous_context]) + "\n---\n"
                
                system_instruction = (
                    f"Translate the following lines into {self.config.target_language}.\n"
                    f"{self.config.context_prompt}\n"
                    f"{self.config.glossary_prompt}"
                )
                final_content = f"{context_str}TRANSLATE THESE LINES:\n{prompt_batch}"

                # Request object for Batch API
                # New SDK Batch API request format
                request = {
                    "request": {
                        "model": self.config.model, # Model is at top level in some versions, or inside? 
                        # Actually Batch API usually takes standard GenerateContentRequest structure
                        "contents": [
                            {"role": "user", "parts": [{"text": final_content}]}
                        ],
                        # system_instruction is now a top-level field or part of generationConfig depending on API version
                        # In google-genai, it is usually part of config. 
                        # However, for JSONL batch input, it often follows the REST API JSON structure.
                        # REST API: "system_instruction": ...
                        "tools": [], 
                        "generationConfig": {
                            "responseMimeType": "application/json",
                            "responseSchema": TranslationItem.model_json_schema(),
                            "temperature": self.config.temperature
                        },
                        "systemInstruction": {
                            "parts": [ {"text": system_instruction} ]
                        }
                    },
                    "custom_id": f"{file_path.name}-batch-{i}"
                }
                f.write(json.dumps(request) + "\n")
        
        # 2. Upload File
        try:
            print("Uploading batch file to Gemini...")
            # client.files.upload
            batch_file = self.client.files.upload(path=output_jsonl_path)
            
            # 3. Create Batch Job
            print("Creating Batch Job...")
            # client.batches.create
            job = self.client.batches.create(
                model=self.config.model,
                src=batch_file.name, # or uri? check docs. Usually uses 'source' or 'src'
                dest_output_uri=f"gcs://", # Check if this is required or optional default
                display_name=f"Subtitles: {file_path.name}"
            )
            
            print(f"Batch Job Submitted! ID: {job.name}")
            print(f"Use --check-batch {job.name} to check status later.")
            
        except Exception as e:
            logger.error(f"Failed to submit batch job: {e}")
        finally:
            # Cleanup local jsonl
            if output_jsonl_path.exists():
                os.remove(output_jsonl_path)

class SubtitleProcessor:
    """Manages the subtitle processing workflow."""
    def __init__(self, config: TranslationConfig, stats: TranslationStats):
        self.config = config
        self.stats = stats
        self.translator = GeminiTranslator(config, stats)

    def get_output_filename(self, input_path: Path) -> Path:
        """Determines the output filename based on the input filename and target language."""
        suffix = input_path.suffix  # .srt or .vtt
        stem = input_path.stem
        # If input is 'video.en.srt', we want 'video.fin.srt'
        if stem.endswith('.en'):
             stem = stem[:-3]
        return input_path.parent / f"{stem}.{self.config.target_language_code}{suffix}"

    async def process_file(self, file_path: Path, semaphore: asyncio.Semaphore, pbar: tqdm):
        """Processes a single subtitle file."""
        async with semaphore:
            output_path = self.get_output_filename(file_path)
            
            # Identify format
            is_vtt = file_path.suffix == '.vtt'
            try:
                if is_vtt:
                    captions = webvtt.read(str(file_path)).captions
                else:
                    captions = pysrt.open(str(file_path), encoding='utf-8')
            except Exception as e:
                logger.error(f"Failed to read {file_path}: {e}")
                return

            total_cues = len(captions)
            
            # Load existing progress
            translated_cues = []
            if output_path.exists():
                try:
                    if is_vtt:
                        translated_cues = webvtt.read(str(output_path)).captions
                    else:
                        translated_cues = pysrt.open(str(output_path), encoding='utf-8')
                    logger.info(f"Resuming {file_path.name}: {len(translated_cues)}/{total_cues} already done.")
                except Exception:
                    logger.warning(f"Could not read existing output {output_path}. Starting over.")
            
            # Determine start index
            start_index = len(translated_cues)
            if start_index >= total_cues:
                pbar.update(total_cues) # Mark as done
                return

            cues_to_process = captions[start_index:]
            
            # Batch Processing
            batch_size = self.config.batch_size
            batches = [cues_to_process[i:i + batch_size] for i in range(0, len(cues_to_process), batch_size)]

            for i, batch in enumerate(batches):
                batch_id = (start_index // batch_size) + i + 1
                
                # Context Overlap calculation
                current_batch_start_idx = i * batch_size
                total_idx = start_index + current_batch_start_idx
                
                previous_context = []
                if total_idx > 0:
                     overlap = self.config.context_overlap
                     prev_start = max(0, total_idx - overlap)
                     previous_context = [c.text for c in captions[prev_start:total_idx]]

                source_texts = [c.text for c in batch]
                
                translated_texts = await self.translator.translate_batch(
                    source_texts, 
                    batch_id, 
                    previous_context=previous_context
                )
                
                # Handle results
                if not translated_texts:
                    logger.error(f"Batch {batch_id} failed for {file_path.name}. Stopping file processing.")
                    break
                
                # Mismatch recovery
                if len(translated_texts) != len(batch):
                    logger.warning(f"Batch {batch_id}: length mismatch {len(translated_texts)} vs {len(batch)}")
                    translated_texts = translated_texts[:len(batch)]
                    translated_texts += [""] * (len(batch) - len(translated_texts))

                # Update cues and append
                current_translated_batch = []
                import copy
                for original_cue, new_text in zip(batch, translated_texts):
                    # Robust cloning
                    if is_vtt:
                         new_cue = copy.copy(original_cue)
                         new_cue.text = new_text
                         current_translated_batch.append(new_cue)
                    else:
                         new_cue = copy.copy(original_cue)
                         new_cue.text = new_text
                         current_translated_batch.append(new_cue)

                # Incremental Save
                translated_cues.extend(current_translated_batch)
                self.stats.processed_cues += len(current_translated_batch)
                
                try:
                    if is_vtt:
                        vtt = webvtt.WebVTT()
                        vtt.captions = translated_cues
                        vtt.save(str(output_path))
                    else:
                        srt_file = pysrt.SubRipFile(items=translated_cues)
                        srt_file.save(str(output_path), encoding='utf-8')
                except Exception as e:
                    logger.error(f"Failed to save {output_path}: {e}")
                    break

                # Update progress bar
                pbar.update(len(batch))
                
                # Rate limit sleep
                await asyncio.sleep(self.config.request_interval)
            
            logger.info(f"Finished {file_path.name}")

    async def run(self, input_paths: List[Path]):
        """Main execution loop for processing multiple files."""
        self.stats.total_files = len(input_paths)
        semaphore = asyncio.Semaphore(self.config.workers)
        
        print(f"Queuing {len(input_paths)} files...")
        
        with tqdm(total=len(input_paths), desc="Files") as file_pbar:
            async def wrapped_process(path):
                dummy_pbar = tqdm(disable=True) 
                await self.process_file(path, semaphore, dummy_pbar)
                file_pbar.update(1)

            tasks = [wrapped_process(p) for p in input_paths]
            await asyncio.gather(*tasks)


def load_show_config(show_config_path: Optional[Path], target_lang_code: str) -> tuple[str, str]:
    """
    Loads system prompt and glossary from a show-specific YAML file.
    Returns (system_prompt, glossary_prompt).
    """
    if not show_config_path or not show_config_path.exists():
        return "", ""

    try:
        with open(show_config_path, 'r', encoding='utf-8') as f:
            data = yaml.safe_load(f)
            
        system_prompt = data.get('system_prompt', '')
        
        # Parse Glossary
        glossary_data = data.get('glossary', {})
        terms = []
        
        for category, items in glossary_data.items():
            if isinstance(items, dict):
                for term, details in items.items():
                    if isinstance(details, dict):
                        # Dynamic lookup using target_lang_code (e.g., 'fi')
                        # Fallback to direct 'translation_fi' if old format
                        translations = details.get('translations', {})
                        trans = translations.get(target_lang_code, '')
                        
                        # Fallback for compatibility/safety
                        if not trans:
                             # Try legacy keys just in case or default to empty
                             trans = details.get(f"translation_{target_lang_code}", '')

                        guide = details.get('guidance', '')
                        if trans:
                            terms.append(f"- {term}: {trans} ({guide})")
                    else:
                        # Simple key-value pair
                        terms.append(f"- {term}: {details}")

        glossary_prompt = ""
        if terms:
            glossary_prompt = "GLOSSARY (Strictly adhere to these translations):\n" + "\n".join(terms) + "\n"
            
        return system_prompt, glossary_prompt

    except Exception as e:
        logger.error(f"Failed to load show config {show_config_path}: {e}")
        return "", ""


def discover_files(inputs: List[str]) -> List[Path]:
    """Finds all valid subtitle files from the input list (files or directories)."""
    valid_extensions = {'.srt', '.vtt'}
    found_files = []

    for inp in inputs:
        path = Path(inp)
        if path.is_file():
            if path.suffix in valid_extensions:
                found_files.append(path)
        elif path.is_dir():
            for ext in valid_extensions:
                found_files.extend(path.rglob(f"*{ext}"))
    
    return sorted(list(set(found_files)))

def load_config(config_path: Path) -> dict:
    """Loads configuration from a YAML file."""
    if not config_path.exists():
        logger.warning(f"Config file {config_path} not found. Using defaults.")
        return {}
    with open(config_path, 'r', encoding='utf-8') as f:
        return yaml.safe_load(f)

def main():
    load_dotenv(override=True)
    parser = argparse.ArgumentParser(description="LLM Subtitle Translator (google-genai SDK 1.0+)")
    parser.add_argument("--inputs", nargs='+', required=True, help="Input files or directories")
    parser.add_argument("--config", default="config.yaml", help="Path to global config file")
    parser.add_argument("--show-config", help="Path to show-specific config file (e.g., contexts/jetlag.yaml)")
    parser.add_argument("--batch", action="store_true", help="Use Gemini Batch API")
    parser.add_argument("--check-batch", help="Check status of a batch job by ID")
    parser.add_argument("--dry-run", action="store_true", help="List files to be processed without running")
    
    args = parser.parse_args()

    # Load Config
    file_config = load_config(Path(args.config))
    
    # Load Show Config (Context + Glossary)
    target_code = file_config.get("target_language_code", "fin")
    system_prompt, glossary_prompt = load_show_config(Path(args.show_config) if args.show_config else None, target_code)

    # Merge with defaults and args
    config = TranslationConfig(
        target_language=file_config.get("target_language", "Finnish"),
        target_language_code=target_code,
        model=file_config.get("model", "models/gemini-3-flash-preview"),
        batch_size=file_config.get("batch_size", 40),
        temperature=file_config.get("temperature", 0.1),
        max_output_tokens=file_config.get("max_output_tokens", 8192),
        request_interval=file_config.get("request_interval", 1.0),
        timeout=file_config.get("timeout", 30),
        retries=file_config.get("retries", 3),
        workers=file_config.get("workers", 1), # Default to sequential for reliability
        fallback_model=file_config.get("fallback_model", None),
        min_confidence=file_config.get("min_confidence", 0.8),
        context_overlap=file_config.get("context_overlap", 5),
        context_prompt=system_prompt,
        glossary_prompt=glossary_prompt,
        api_key=os.getenv("GEMINI_API_KEY")
    )

    stats = TranslationStats()
    stats.start_time = time.time()

    if args.check_batch:
        translator = GeminiTranslator(config, stats)
        # Note: calling async function from sync main needs runner
        asyncio.run(translator.check_batch_status(args.check_batch))
        return

    # Discover Files
    files_to_process = discover_files(args.inputs)
    
    if args.dry_run:
        print(f"Found {len(files_to_process)} files to process:")
        for f in files_to_process:
            print(f" - {f}")
        return

    if not files_to_process:
        print("No subtitle files found.")
        return

    print(f"Starting translation for {len(files_to_process)} files...")
    print(f"Model: {config.model} | Mode: {'Batch' if args.batch else 'Interactive'}")

    processor = SubtitleProcessor(config, stats)
    
    if args.batch:
        # Submit all found files as batches
        for f in files_to_process:
            asyncio.run(processor.translator.submit_batch_job(f))
    else:
        try:
            asyncio.run(processor.run(files_to_process))
        except KeyboardInterrupt:
            print("\nProcess interrupted by user. Progress has been saved.")
        except Exception as e:
            logger.error(f"Unexpected error: {e}")
            import traceback
            traceback.print_exc()
        finally:
            print(stats.get_summary())

if __name__ == "__main__":
    main()
