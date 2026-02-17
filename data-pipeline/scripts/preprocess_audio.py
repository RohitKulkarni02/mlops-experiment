"""
Inference-style audio preprocessing: 16 kHz mono WAV, loudness normalization, silence trimming.
Same spec as when sending audio to Gemini/APIs at inference (pipeline and backend use this).
Reads from data/raw, writes to staging; stratified_split then builds evaluation sets (dev/test/holdout).
Supports .mp4 (e.g. MELD raw) by extracting audio via ffmpeg.
"""
import sys
import subprocess
import tempfile
from pathlib import Path

# Allow running as script from repo root or data-pipeline: ensure pipeline root is on path
_PIPELINE_ROOT = Path(__file__).resolve().parent.parent
if str(_PIPELINE_ROOT) not in sys.path:
    sys.path.insert(0, str(_PIPELINE_ROOT))

import numpy as np
import soundfile as sf

from scripts.utils import get_logger, load_config, RAW_DIR, PROCESSED_DIR

logger = get_logger("preprocess_audio")

# One-time warning for ffmpeg/video failures so we don't spam thousands of lines
_ffmpeg_fail_warned = False

try:
    import librosa
except ImportError:
    librosa = None  # type: ignore


def _load_audio_from_video(path: Path, sr: int | None = None) -> tuple[np.ndarray, int]:
    """Extract audio from .mp4/.mkv etc. using ffmpeg; return (samples, sample_rate)."""
    path = Path(path).resolve()
    sr = sr or 16000
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
        wav_path = Path(f.name)
    try:
        cmd = [
            "ffmpeg", "-y", "-i", str(path),
            "-vn", "-acodec", "pcm_s16le", "-ar", str(sr), "-ac", "1",
            str(wav_path),
        ]
        try:
            out = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
        except (FileNotFoundError, OSError):
            raise RuntimeError(
                "ffmpeg not found or not runnable. Install it for .mp4 (MELD), e.g.: winget install ffmpeg"
            )
        if out.returncode != 0:
            err = (out.stderr or out.stdout or "").strip()
            if err:
                err = err.split("\n")[-1].strip() or err[:120]
            logger.debug("ffmpeg failed for %s: %s", path.name, err)
            raise RuntimeError("ffmpeg extraction failed")
        data, rate = sf.read(wav_path)
        if data.ndim > 1:
            data = data.mean(axis=1)
        return np.asarray(data, dtype=np.float32), int(rate)
    finally:
        wav_path.unlink(missing_ok=True)


def load_audio(path: Path, sr: int | None = None) -> tuple[np.ndarray, int]:
    """Load audio; return (samples, sample_rate). Supports .mp4 via ffmpeg; else soundfile then librosa."""
    path = Path(path).resolve()
    suf = path.suffix.lower()
    if suf in (".mp4", ".m4v", ".mkv", ".avi", ".mov"):
        return _load_audio_from_video(path, sr=sr)
    try:
        data, rate = sf.read(path)
    except Exception:
        if librosa is not None:
            data, rate = librosa.load(path, sr=sr, mono=True)
            data = np.asarray(data, dtype=np.float32)
            rate = int(rate)
            return data, rate
        raise
    if data.ndim > 1:
        data = data.mean(axis=1)
    if sr and rate != sr and librosa is not None:
        resampled = librosa.resample(data.astype(np.float64), orig_sr=rate, target_sr=sr)
        if isinstance(resampled, tuple):
            data = np.asarray(resampled[0], dtype=np.float32)
        else:
            data = np.asarray(resampled, dtype=np.float32)
        rate = sr
    elif sr and rate != sr:
        from scipy import signal as scipy_signal
        if rate != sr:
            num = int(len(data) * sr / rate)
            data = np.asarray(scipy_signal.resample(data, num), dtype=np.float32)
            rate = sr
    arr = np.asarray(data, dtype=np.float32)
    return arr, rate


def normalize_loudness(data: np.ndarray, target_db: float = -20.0) -> np.ndarray:
    """Peak-normalize then scale to target dB (relative to full scale)."""
    if data.size == 0:
        return data
    peak = np.abs(data).max()
    if peak <= 0:
        return data
    data = data / peak
    # RMS-based loudness approx: scale so that RMS gives ~target_db
    rms = np.sqrt(np.mean(data ** 2))
    if rms > 0:
        target_linear = 10 ** (target_db / 20.0)
        data = data * (target_linear / rms)
    return np.clip(data, -1.0, 1.0).astype(np.float32)


def trim_silence(data: np.ndarray, sr: int, top_db: float = 25.0) -> np.ndarray:
    """Trim leading/trailing silence using energy threshold."""
    if librosa is not None:
        # librosa.effects.trim(y, top_db=...) — sr not used in recent versions
        trimmed, _ = librosa.effects.trim(data, top_db=top_db)
        return np.asarray(trimmed, dtype=np.float32)
    # Fallback: simple threshold on RMS in windows
    win = min(int(0.02 * sr), len(data) // 4)
    if win < 1:
        return data
    energy = np.convolve(data ** 2, np.ones(win) / win, mode="same")
    thresh = np.max(energy) * (10 ** (-top_db / 10))
    idx = np.where(energy >= thresh)[0]
    if len(idx) == 0:
        return data
    return data[idx[0] : idx[-1] + 1].astype(np.float32)


def process_one(
    in_path: Path,
    out_path: Path,
    target_sr: int = 16000,
    mono: bool = True,
    normalize: bool = True,
    trim: bool = True,
) -> bool:
    """Convert one file to target format and save."""
    try:
        data, sr = load_audio(in_path, sr=target_sr)
        if mono and data.ndim > 1:
            data = data.mean(axis=1)
        if normalize:
            data = normalize_loudness(data)
        if trim:
            data = trim_silence(data, sr)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        sf.write(out_path, data, sr, subtype="PCM_16")
        return True
    except Exception as e:
        global _ffmpeg_fail_warned
        try:
            if in_path.suffix.lower() in (".mp4", ".m4v", ".mkv", ".avi", ".mov"):
                if not _ffmpeg_fail_warned:
                    _ffmpeg_fail_warned = True
                    logger.warning(
                        "Video file failed (ffmpeg not found?). .mp4 (MELD) will be skipped. "
                        "Install ffmpeg and add to PATH (restart terminal after winget install ffmpeg). First file: %s",
                        in_path.name,
                    )
                    print(
                        "Video files (.mp4) will be skipped: ffmpeg not found. Install ffmpeg and restart terminal, then re-run.",
                        flush=True,
                    )
                else:
                    logger.debug("Skipped video %s: %s", in_path.name, e)
            else:
                logger.warning("Failed %s: %s", in_path, e)
        except Exception:
            pass
        return False


def collect_audio_files(root: Path, exts: tuple[str, ...] = (".wav", ".mp3", ".flac", ".ogg", ".m4a")) -> list[Path]:
    """Collect audio files under root. MELD (.mp4) is skipped by default; add .mp4 to exts if ffmpeg is available.
    Skips macOS resource-fork files (._*) which are not real media."""
    root = Path(root).resolve()
    files = []
    for ext in exts:
        files.extend(root.rglob(f"*{ext}"))
        files.extend(root.rglob(f"*{ext.upper()}"))
    # Exclude ._* (macOS resource forks / metadata) - ffmpeg fails on them and they're not real media
    files = [f.resolve() for f in files if not f.name.startswith("._")]
    return sorted(set(files))


def run_preprocessing(
    raw_subdir: str | Path | None = None,
    out_subdir: Path | None = None,
) -> tuple[int, int]:
    """Run preprocessing on raw data. Returns (success_count, fail_count)."""
    cfg = load_config()
    preproc = cfg.get("preprocessing", {})
    target_sr = preproc.get("target_sr", 16000)
    mono = preproc.get("mono", True)
    normalize = preproc.get("normalize_loudness", True)
    trim = preproc.get("trim_silence", True)
    include_video = preproc.get("include_video", False)
    exts = (".wav", ".mp3", ".flac", ".ogg", ".m4a")
    if include_video:
        exts = exts + (".mp4", ".m4v", ".mkv", ".avi", ".mov")

    if raw_subdir:
        raw_root = Path(raw_subdir) if isinstance(raw_subdir, str) else raw_subdir
    else:
        raw_root = RAW_DIR
    if out_subdir is None:
        out_subdir = PROCESSED_DIR / "staged"
    out_subdir = Path(out_subdir)
    out_subdir.mkdir(parents=True, exist_ok=True)

    raw_root = Path(raw_root).resolve()
    files = collect_audio_files(raw_root, exts=exts)
    # Log per-folder counts so we can see if MELD/TESS/RAVDESS are found
    try:
        by_folder = {}
        for f in files:
            try:
                rel = f.relative_to(raw_root)
                top = rel.parts[0] if rel.parts else "."
                by_folder[top] = by_folder.get(top, 0) + 1
            except ValueError:
                by_folder["."] = by_folder.get(".", 0) + 1
        for folder, count in sorted(by_folder.items()):
            logger.info("  %s: %d files", folder, count)
    except Exception:
        pass
    logger.info("Found %d audio files under %s", len(files), raw_root)
    ok, fail = 0, 0
    total = len(files)
    ffmpeg_warned = False
    for i, fp in enumerate(files):
        if total > 100 and (i + 1) % 500 == 0:
            logger.info("  progress: %d / %d (%.0f%%)", i + 1, total, 100.0 * (i + 1) / total)
            print("  progress: %d / %d" % (i + 1, total), flush=True)
        rel = fp.relative_to(raw_root)
        out_path = out_subdir / rel.with_suffix(".wav")
        try:
            if process_one(fp, out_path, target_sr=target_sr, mono=mono, normalize=normalize, trim=trim):
                ok += 1
            else:
                fail += 1
        except Exception as e:
            fail += 1
            if not ffmpeg_warned and fp.suffix.lower() in (".mp4", ".m4v", ".mkv", ".avi", ".mov"):
                ffmpeg_warned = True
                msg = (
                    "Video file failed (likely ffmpeg). .mp4 (MELD) will be skipped. "
                    "If you installed ffmpeg with winget, close and reopen the terminal so PATH updates, then re-run."
                )
                logger.warning("%s First failure: %s", msg, e)
                print(msg, flush=True)
            else:
                logger.debug("Skipped %s: %s", fp.name, e)
    logger.info("Preprocessing done: %d ok, %d failed", ok, fail)
    return ok, fail


def main() -> None:
    import sys
    raw_subdir = sys.argv[1] if len(sys.argv) > 1 else None
    run_preprocessing(raw_subdir=raw_subdir)


if __name__ == "__main__":
    main()
