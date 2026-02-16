"""
Gemini API verification: spot-check that preprocessed WAVs are accepted by Gemini.
Option A: Run manually (e.g. python scripts/verify_gemini_audio.py).
Option B: Run as optional pipeline task (skippable via RUN_GEMINI_VERIFICATION=false).

Picks one or a few files from data/processed/staged/ or test/, sends to Gemini
with a simple prompt (e.g. transcribe or describe). Returns exit code (0/1) and
full API response details for each call.
"""
import argparse
import json
import os
import sys
from pathlib import Path

from scripts.utils import get_logger, PROCESSED_DIR

logger = get_logger("verify_gemini_audio")

# Env: set to "true" (or "1") to run verification; otherwise skip (no-op success)
RUN_VERIFICATION_ENV = "RUN_GEMINI_VERIFICATION"
# Env: API key (Gemini / Google AI)
GEMINI_API_KEY_ENV = "GEMINI_API_KEY"
GOOGLE_API_KEY_ENV = "GOOGLE_API_KEY"

# Default: check up to this many files
DEFAULT_MAX_FILES = 2
# Timeout per API call (seconds)
GEMINI_TIMEOUT_SEC = 30
# Retries per file
GEMINI_RETRIES = 2


def _is_verification_enabled() -> bool:
    v = os.environ.get(RUN_VERIFICATION_ENV, "").strip().lower()
    return v in ("1", "true", "yes")


def _get_api_key() -> str | None:
    return os.environ.get(GEMINI_API_KEY_ENV) or os.environ.get(GOOGLE_API_KEY_ENV) or None


def _pick_wavs(data_dir: Path, max_files: int, prefer: str) -> list[Path]:
    """Pick up to max_files WAV paths from processed dir. Prefer 'staged' or 'test'."""
    candidates: list[Path] = []
    for split in (prefer, "staged", "test", "dev", "holdout"):
        split_dir = data_dir / split
        if not split_dir.is_dir():
            continue
        wavs = sorted(split_dir.glob("*.wav"))[: max_files - len(candidates)]
        for w in wavs:
            if w not in candidates:
                candidates.append(w)
            if len(candidates) >= max_files:
                return candidates
    return candidates


# Exit codes for pipeline / CLI
EXIT_SUCCESS = 0
EXIT_FAILURE = 1


def _serialize_gemini_response(response) -> dict:
    """Build a JSON-serializable dict with all available details from a Gemini generate_content response."""
    out = {}
    try:
        out["text"] = getattr(response, "text", None) or ""
        out["candidates"] = []
        for c in getattr(response, "candidates", []) or []:
            cand = {
                "finish_reason": getattr(c, "finish_reason", None),
                "content": None,
            }
            content = getattr(c, "content", None)
            if content is not None:
                parts = getattr(content, "parts", None) or []
                cand["content"] = {"parts": [getattr(p, "text", None) or str(p) for p in parts]}
            out["candidates"].append(cand)
        out["usage_metadata"] = None
        um = getattr(response, "usage_metadata", None)
        if um is not None:
            out["usage_metadata"] = {
                "prompt_token_count": getattr(um, "prompt_token_count", None),
                "candidates_token_count": getattr(um, "candidates_token_count", None),
                "total_token_count": getattr(um, "total_token_count", None),
            }
        out["prompt_feedback"] = None
        pf = getattr(response, "prompt_feedback", None)
        if pf is not None:
            out["prompt_feedback"] = {
                "block_reason": getattr(pf, "block_reason", None),
            }
    except Exception as e:
        out["_serialize_error"] = str(e)
    return out


def _call_gemini_with_audio(
    audio_path: Path, api_key: str, timeout: int = GEMINI_TIMEOUT_SEC, retries: int = GEMINI_RETRIES
) -> dict:
    """
    Send one WAV to Gemini (inline audio), ask for a brief description/transcript.
    Returns dict with: success (bool), status_code (0|1), api_response (full details), error (if failed).
    """
    try:
        from google import genai
        from google.genai import types
    except ImportError:
        return {
            "success": False,
            "status_code": EXIT_FAILURE,
            "api_response": None,
            "error": "google-genai not installed; pip install google-genai",
        }

    with open(audio_path, "rb") as f:
        audio_bytes = f.read()

    prompt = "Listen to this short audio. Reply with one sentence: what is said or what you hear (e.g. speech content or sound description)."
    model_id = "gemini-2.0-flash"

    for attempt in range(retries):
        try:
            client = genai.Client(api_key=api_key)
            part = types.Part.from_bytes(data=audio_bytes, mime_type="audio/wav")
            response = client.models.generate_content(
                model=model_id,
                contents=[prompt, part],
            )
            api_details = _serialize_gemini_response(response)
            api_details["model"] = model_id

            if not getattr(response, "candidates", None):
                return {
                    "success": False,
                    "status_code": EXIT_FAILURE,
                    "api_response": api_details,
                    "error": "Gemini returned no candidates",
                }
            text = (getattr(response, "text", None) or "").strip()
            if not text:
                return {
                    "success": False,
                    "status_code": EXIT_FAILURE,
                    "api_response": api_details,
                    "error": "Gemini response was empty",
                }
            return {
                "success": True,
                "status_code": EXIT_SUCCESS,
                "api_response": api_details,
                "error": None,
            }
        except Exception as e:
            logger.warning("Gemini call attempt %s failed: %s", attempt + 1, e)
            if attempt == retries - 1:
                return {
                    "success": False,
                    "status_code": EXIT_FAILURE,
                    "api_response": None,
                    "error": str(e),
                }
    return {
        "success": False,
        "status_code": EXIT_FAILURE,
        "api_response": None,
        "error": "Max retries exceeded",
    }


def run_verification(
    data_dir: Path | None = None,
    max_files: int = DEFAULT_MAX_FILES,
    prefer_split: str = "staged",
    timeout_sec: int = GEMINI_TIMEOUT_SEC,
    retries: int = GEMINI_RETRIES,
    *,
    force_run: bool = False,
) -> dict:
    """
    Run Gemini verification on a few preprocessed WAVs.
    Returns dict with exit_code (0=success/skipped, 1=failure), success, and full api_response details per file.
    If RUN_GEMINI_VERIFICATION is not enabled and force_run is False, returns skip result without calling API.
    """
    if data_dir is None:
        data_dir = PROCESSED_DIR
    data_dir = Path(data_dir)

    if not force_run and not _is_verification_enabled():
        logger.info("Verification skipped (set %s=true to enable)", RUN_VERIFICATION_ENV)
        return {
            "skipped": True,
            "reason": "RUN_GEMINI_VERIFICATION not enabled",
            "checked": 0,
            "exit_code": EXIT_SUCCESS,
            "success": True,
            "results": [],
        }

    api_key = _get_api_key()
    if not api_key:
        logger.warning("No API key found (%s or %s); skipping verification", GEMINI_API_KEY_ENV, GOOGLE_API_KEY_ENV)
        return {
            "skipped": True,
            "reason": "no API key",
            "checked": 0,
            "exit_code": EXIT_SUCCESS,
            "success": True,
            "results": [],
        }

    wavs = _pick_wavs(data_dir, max_files, prefer_split)
    if not wavs:
        logger.warning("No WAV files found under %s (staged/test/dev/holdout)", data_dir)
        return {
            "skipped": False,
            "success": False,
            "exit_code": EXIT_FAILURE,
            "reason": "no WAV files",
            "checked": 0,
            "results": [],
        }

    results = []
    all_ok = True
    for wav in wavs:
        call_result = _call_gemini_with_audio(wav, api_key, timeout_sec, retries)
        results.append({
            "file": str(wav),
            "success": call_result["success"],
            "status_code": call_result["status_code"],
            "error": call_result.get("error"),
            "api_response": call_result.get("api_response"),
        })
        if not call_result["success"]:
            all_ok = False
            logger.error("Verification failed for %s: %s", wav, call_result.get("error"))

    if all_ok:
        logger.info("Gemini verification passed for %d file(s)", len(wavs))
    return {
        "skipped": False,
        "success": all_ok,
        "exit_code": EXIT_SUCCESS if all_ok else EXIT_FAILURE,
        "checked": len(wavs),
        "results": results,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Verify preprocessed audio with Gemini API (optional stage).")
    parser.add_argument(
        "data_dir",
        nargs="?",
        default=None,
        help="Processed data root (default: data/processed)",
    )
    parser.add_argument(
        "--max-files",
        type=int,
        default=DEFAULT_MAX_FILES,
        help="Max number of WAVs to check (default: %s)" % DEFAULT_MAX_FILES,
    )
    parser.add_argument(
        "--prefer",
        choices=("staged", "test", "dev", "holdout"),
        default="staged",
        help="Preferred split to pick files from (default: staged)",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Run verification even if RUN_GEMINI_VERIFICATION is not set",
    )
    args = parser.parse_args()
    data_dir = Path(args.data_dir) if args.data_dir else None
    if data_dir is None:
        data_dir = PROCESSED_DIR

    result = run_verification(
        data_dir=data_dir,
        max_files=args.max_files,
        prefer_split=args.prefer,
        force_run=args.force,
    )

    exit_code = result.get("exit_code", EXIT_SUCCESS if result.get("success", result.get("skipped")) else EXIT_FAILURE)
    if exit_code != EXIT_SUCCESS and not result.get("skipped"):
        print(json.dumps(result, indent=2), file=sys.stderr)
    return exit_code


if __name__ == "__main__":
    sys.exit(main())
