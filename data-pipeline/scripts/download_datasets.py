"""
Download emotion and speech datasets; validate checksums; store in data/raw/ with DVC tracking.
Uses curl (subprocess) for downloads when available to avoid worker hangs; falls back to requests.
"""
import sys
import hashlib
import shutil
import subprocess
import tarfile
import zipfile
from pathlib import Path
from typing import Dict, List, Optional

# Allow running as script from repo root or data-pipeline: ensure pipeline root is on path
_PIPELINE_ROOT = Path(__file__).resolve().parent.parent
if str(_PIPELINE_ROOT) not in sys.path:
    sys.path.insert(0, str(_PIPELINE_ROOT))

import requests
from tqdm import tqdm

from scripts.utils import get_logger, load_config, RAW_DIR

logger = get_logger("download_datasets")

# Optional: for MELD/TESS via Hugging Face (pip install datasets)
def _audio_item_to_wav(audio_item, out_path: Path, sr: int = 16000) -> bool:
    """Write one audio item (dict with array/path/bytes, sampling_rate) to out_path as WAV."""
    try:
        if audio_item is None:
            return False
        if isinstance(audio_item, dict):
            path = audio_item.get("path")
            if path and Path(path).exists():
                out_path.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(path, out_path)
                return True
            arr = audio_item.get("array")
            if arr is None:
                b = audio_item.get("bytes")
                if isinstance(b, bytes):
                    out_path.parent.mkdir(parents=True, exist_ok=True)
                    out_path.write_bytes(b)
                    return True
                return False
            import numpy as np
            import soundfile as sf
            arr = np.asarray(arr, dtype=np.float32)
            if arr.ndim > 1:
                arr = arr.mean(axis=1)
            sample_rate = int(audio_item.get("sampling_rate", sr))
            out_path.parent.mkdir(parents=True, exist_ok=True)
            sf.write(out_path, arr, sample_rate, subtype="PCM_16")
            return True
        return False
    except Exception as e:
        logger.exception("Failed to write %s: %s", out_path, e)
        return False


def _fetch_meld_via_hf(raw_dir: Path) -> bool:
    """Fetch MELD from Hugging Face and save WAVs under raw_dir/MELD/. Returns True if any file written."""
    try:
        from datasets import load_dataset
    except ImportError:
        logger.warning("MELD via Hugging Face requires: pip install datasets")
        return False
    out_dir = raw_dir / "MELD"
    out_dir.mkdir(parents=True, exist_ok=True)
    # Prefer audio-specific repos; then BLOSSOM (large); then declare-lab
    for repo in ("ajyy/MELD_audio", "EdwardLin2023/MELD-Audio", "BLOSSOM-framework/MELD", "declare-lab/MELD"):
        try:
            _log_and_print("MELD: loading %s ..." % repo)
            ds = load_dataset(repo, trust_remote_code=True)
            break
        except Exception as e:
            logger.warning("%s failed: %s", repo, e)
            continue
    else:
        logger.error("Could not load MELD from Hugging Face. Place WAVs manually in %s", out_dir)
        return False
    count = 0
    for split_name in list(ds.keys()):
        rows = ds[split_name]
        audio_col = next((c for c in ("Utterance_Audio", "audio", "Audio", "audio_path", "path", "wav", "speech") if c in rows.column_names), None)
        if not audio_col:
            logger.warning("MELD split %s has no audio column (columns: %s)", split_name, rows.column_names)
            continue
        split_dir = out_dir / split_name
        split_dir.mkdir(parents=True, exist_ok=True)
        for i, row in enumerate(rows):
            if _audio_item_to_wav(row.get(audio_col), split_dir / f"meld_{split_name}_{i:05d}.wav"):
                count += 1
    _log_and_print("MELD: %d WAVs -> %s" % (count, out_dir))
    return count > 0


def _try_extract_archives_in(dir_path: Path) -> bool:
    """If dir has no .wav/.mp3/.flac but has files that look like zip/gzip, extract them. Returns True if extracted."""
    dir_path = Path(dir_path)
    if not dir_path.is_dir():
        return False
    has_audio = any(dir_path.rglob(f"*{e}") for e in (".wav", ".mp3", ".flac", ".WAV", ".MP3", ".FLAC"))
    if has_audio:
        return False
    extracted = False
    # Also try one level of subdirs (e.g. TESS/E8H2MF/archive.zip)
    to_scan = [dir_path]
    for sub in dir_path.iterdir():
        if sub.is_dir() and not any(sub.rglob("*.wav")) and not any(sub.rglob("*.WAV")):
            to_scan.append(sub)
    for scan_dir in to_scan:
        for f in list(scan_dir.iterdir()):
            if not f.is_file():
                continue
            try:
                with open(f, "rb") as fp:
                    magic = fp.read(4)
            except Exception:
                continue
            if magic[:2] == b"\x1f\x8b":
                try:
                    out = scan_dir / "extracted"
                    out.mkdir(parents=True, exist_ok=True)
                    with tarfile.open(f, "r:gz") as tf:
                        tf.extractall(out)
                    logger.info("Extracted gzip %s -> %s", f.name, out)
                    extracted = True
                except Exception as e:
                    logger.warning("Could not extract %s as tar.gz: %s", f.name, e)
            elif magic[:2] == b"PK":
                try:
                    out = scan_dir / "extracted"
                    out.mkdir(parents=True, exist_ok=True)
                    with zipfile.ZipFile(f, "r") as zf:
                        zf.extractall(out)
                    logger.info("Extracted zip %s -> %s", f.name, out)
                    extracted = True
                except Exception as e:
                    logger.warning("Could not extract %s as zip: %s", f.name, e)
    return extracted


def _fetch_meld_via_kaggle(raw_dir: Path) -> bool:
    """Fetch MELD from Kaggle via kagglehub (zaber666/meld-dataset) and copy audio to raw_dir/MELD/. Returns True if any file copied."""
    try:
        import kagglehub
    except ImportError:
        logger.warning("MELD via Kaggle requires: pip install kagglehub (and Kaggle API credentials)")
        return False
    raw_dir = Path(raw_dir).resolve()
    out_dir = raw_dir / "MELD"
    out_dir.mkdir(parents=True, exist_ok=True)
    try:
        _log_and_print("MELD: downloading via kagglehub (zaber666/meld-dataset) ...")
        path = kagglehub.dataset_download("zaber666/meld-dataset")
        kaggle_path = Path(path).resolve()
        _log_and_print("MELD: Kaggle path: %s" % kaggle_path)
        exts = ("*.wav", "*.WAV", "*.mp3", "*.MP3", "*.flac", "*.FLAC", "*.ogg", "*.OGG", "*.m4a", "*.M4A")
        all_audio = []
        for ext in exts:
            all_audio.extend(kaggle_path.rglob(ext))
        all_audio = sorted(set(Path(f).resolve() for f in all_audio))
        if not all_audio:
            logger.warning("MELD: no audio files (wav/mp3/flac/ogg/m4a) found under %s", kaggle_path)
            return False
        for src in all_audio:
            try:
                rel = src.relative_to(kaggle_path)
            except ValueError:
                continue
            dest = out_dir / rel
            dest.parent.mkdir(parents=True, exist_ok=True)
            if src != dest and src.resolve() != dest.resolve():
                shutil.copy2(src, dest)
        _log_and_print("MELD: %d audio files -> %s" % (len(all_audio), out_dir))
        _log_and_print("MELD: preprocess will read from: %s" % out_dir)
        return True
    except Exception as e:
        logger.exception("MELD via kagglehub failed: %s", e)
        return False


def _fetch_tess_via_kaggle(raw_dir: Path) -> bool:
    """Fetch TESS from Kaggle via kagglehub and copy WAVs to raw_dir/TESS/. Returns True if any file copied."""
    try:
        import kagglehub
    except ImportError:
        logger.warning("TESS via Kaggle requires: pip install kagglehub (and Kaggle API credentials)")
        return False
    out_dir = raw_dir / "TESS"
    out_dir.mkdir(parents=True, exist_ok=True)
    try:
        _log_and_print("TESS: downloading via kagglehub (ejlok1/toronto-emotional-speech-set-tess) ...")
        path = kagglehub.dataset_download("ejlok1/toronto-emotional-speech-set-tess")
        kaggle_path = Path(path)
        _log_and_print("TESS: Kaggle path: %s" % kaggle_path)
        wavs = list(kaggle_path.rglob("*.wav")) + list(kaggle_path.rglob("*.WAV"))
        if not wavs:
            logger.warning("TESS: no WAV files found under %s", kaggle_path)
            return False
        for src in wavs:
            rel = src.relative_to(kaggle_path)
            dest = out_dir / rel
            dest.parent.mkdir(parents=True, exist_ok=True)
            if dest != src and src.resolve() != dest.resolve():
                shutil.copy2(src, dest)
        _log_and_print("TESS: %d WAVs -> %s" % (len(wavs), out_dir))
        return True
    except Exception as e:
        logger.exception("TESS via kagglehub failed: %s", e)
        return False


def _fetch_tess_via_hf(raw_dir: Path) -> bool:
    """Fetch TESS from Hugging Face (if available) and save WAVs under raw_dir/TESS/. Returns True if any file written."""
    try:
        from datasets import load_dataset
    except ImportError:
        logger.warning("TESS via Hugging Face requires: pip install datasets")
        _try_extract_archives_in(raw_dir / "TESS")
        return False
    out_dir = raw_dir / "TESS"
    out_dir.mkdir(parents=True, exist_ok=True)
    for repo in ("DagsHub-Datasets/Toronto-emotional-speech-set-TESS", "ehcalabres/tess"):
        try:
            ds = load_dataset(repo, trust_remote_code=True)
            break
        except Exception as e:
            logger.warning("%s failed: %s", repo, e)
            continue
    else:
        logger.warning("TESS not on Hugging Face. Trying to extract any archive in %s", out_dir)
        return _try_extract_archives_in(out_dir)
    count = 0
    for split_name in list(ds.keys()):
        rows = ds[split_name]
        audio_col = next((c for c in ("audio", "Audio", "file", "path", "wav") if c in rows.column_names), rows.column_names[0] if rows.column_names else None)
        if not audio_col:
            continue
        split_dir = out_dir / split_name
        split_dir.mkdir(parents=True, exist_ok=True)
        for i, row in enumerate(rows):
            if _audio_item_to_wav(row.get(audio_col), split_dir / f"tess_{i:05d}.wav"):
                count += 1
    _log_and_print("TESS: %d WAVs -> %s" % (count, out_dir))
    return count > 0

# Session with headers so GitHub/Zenodo serve the file (fallback when curl not used)
DOWNLOAD_HEADERS = {
    "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
    "Accept": "application/octet-stream, application/zip, */*",
}


def compute_sha256(path: Path, chunk_size: int = 8192) -> str:
    """Compute SHA256 checksum of a file."""
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while chunk := f.read(chunk_size):
            h.update(chunk)
    return h.hexdigest()


def _download_with_curl(
    url: str,
    dest: Path,
    connect_timeout: int = 30,
    max_time: int = 7200,
) -> bool:
    """Download using curl (subprocess). Reliable in Airflow workers; avoids Python requests hang."""
    dest.parent.mkdir(parents=True, exist_ok=True)
    curl_cmd = [
        "curl",
        "-L",  # follow redirects
        "-f",  # fail on HTTP 4xx/5xx
        "--connect-timeout", str(connect_timeout),
        "--max-time", str(max_time),
        "-o", str(dest),
        "--",  # end options
        url,
    ]
    logger.info("Downloading with curl: %s -> %s", url, dest)
    print(f"Running curl (progress below)...", flush=True)
    try:
        result = subprocess.run(
            curl_cmd,
            timeout=max_time + 60,
        )
        if result.returncode != 0:
            logger.warning("curl exited with code %s", result.returncode)
            return False
        size = dest.stat().st_size
        logger.info("Downloaded %s (%s bytes)", dest.name, size)
        return True
    except FileNotFoundError:
        logger.info("curl not found, will try requests")
        return False
    except subprocess.TimeoutExpired:
        logger.exception("curl timed out")
        return False
    except Exception as e:
        logger.exception("curl failed: %s", e)
        return False


def _download_with_requests(
    url: str,
    dest: Path,
    connect_timeout: int = 30,
    read_timeout: int = 7200,
) -> bool:
    """Fallback: download with requests (can hang in some Airflow worker environments)."""
    dest.parent.mkdir(parents=True, exist_ok=True)
    timeout_tuple = (connect_timeout, read_timeout)
    logger.info("Downloading with requests: %s -> %s", url, dest)
    try:
        session = requests.Session()
        session.headers.update(DOWNLOAD_HEADERS)
        resp = session.get(url, stream=True, timeout=timeout_tuple, allow_redirects=True)
        resp.raise_for_status()
        total = int(resp.headers.get("content-length", 0))
        with open(dest, "wb") as f:
            for chunk in resp.iter_content(chunk_size=8192):
                if chunk:
                    f.write(chunk)
        logger.info("Downloaded %s (%s bytes)", dest.name, dest.stat().st_size)
        return True
    except Exception as e:
        logger.exception("requests download failed: %s", e)
        return False


def download_file(
    url: str,
    dest: Path,
    validate_checksum: Optional[str] = None,
    connect_timeout: int = 30,
    read_timeout: int = 7200,
) -> bool:
    """Download a file. Prefers curl (reliable in workers); falls back to requests."""
    dest.parent.mkdir(parents=True, exist_ok=True)
    print(f"Downloading: {url} -> {dest}", flush=True)
    logger.info("Downloading %s -> %s", url, dest)

    # Prefer curl (avoids hang in Airflow worker); fall back to requests if curl fails or is missing
    ok = False
    if shutil.which("curl"):
        ok = _download_with_curl(url, dest, connect_timeout=connect_timeout, max_time=read_timeout)
    if not ok:
        ok = _download_with_requests(url, dest, connect_timeout=connect_timeout, read_timeout=read_timeout)

    if not ok:
        return False

    size = dest.stat().st_size
    if size < 100 and dest.suffix.lower() == ".zip":
        with open(dest, "rb") as f:
            head = f.read(4)
        if head != b"PK\x03\x04" and head != b"PK\x05\x06":
            logger.warning("File does not look like a zip (got %r). Server may have returned an error page.", head)
            return False
    # For .tar.gz / .tgz, check gzip magic bytes (server may return HTML 404/redirect)
    if dest.suffix.lower() == ".gz" or dest.name.lower().endswith(".tgz"):
        with open(dest, "rb") as f:
            magic = f.read(2)
        if magic != b"\x1f\x8b":
            logger.warning(
                "Downloaded file is not a valid gzip (got %r). Server may have returned HTML. Remove %s and try manual download.",
                magic,
                dest,
            )
            dest.unlink(missing_ok=True)
            return False
    if validate_checksum:
        actual = compute_sha256(dest)
        if actual != validate_checksum:
            logger.warning("Checksum mismatch for %s: expected %s, got %s", dest, validate_checksum, actual)
            return False
    return True


def unzip_if_needed(path: Path, out_dir: Optional[Path] = None) -> Path:
    """Unzip if path is a zip file; return directory containing contents."""
    if not path.suffix.lower() == ".zip":
        return path.parent
    out_dir = out_dir or path.parent / path.stem
    out_dir.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(path, "r") as zf:
        zf.extractall(out_dir)
    logger.info("Extracted %s -> %s", path.name, out_dir)
    return out_dir


def extract_tar_gz(path: Path, out_dir: Optional[Path] = None) -> Path:
    """Extract .tar.gz or .tgz to out_dir; return directory containing contents."""
    if not (path.suffix.lower() == ".gz" and path.stem.endswith(".tar")) and path.suffix.lower() != ".tgz":
        return path.parent
    with open(path, "rb") as f:
        magic = f.read(2)
    if magic != b"\x1f\x8b":
        raise ValueError(
            f"File is not a valid gzip archive (magic {magic!r}). "
            "Server may have returned HTML. Download the dataset manually and place it in data/raw."
        )
    out_dir = out_dir or path.parent
    out_dir.mkdir(parents=True, exist_ok=True)
    with tarfile.open(path, "r:gz") as tf:
        tf.extractall(out_dir)
    logger.info("Extracted %s -> %s", path.name, out_dir)
    return out_dir


def _extract_nested_tar_gz(dir_path: Path) -> None:
    """Extract all .tar.gz / .tgz under dir_path (recursively) into their parent dir. MELD has train/dev/test.tar.gz inside MELD.Raw/."""
    dir_path = Path(dir_path).resolve()
    if not dir_path.is_dir():
        return
    archives = []
    for f in dir_path.rglob("*.tar.gz"):
        if f.is_file():
            archives.append(f)
    for f in dir_path.rglob("*.tgz"):
        if f.is_file() and f not in archives:
            archives.append(f)
    total = len(archives)
    for i, arc in enumerate(archives, 1):
        try:
            with open(arc, "rb") as fp:
                if fp.read(2) != b"\x1f\x8b":
                    continue
            _log_and_print("MELD: extracting %d/%d %s (can take 5–20 min each for train/test) ..." % (i, total, arc.name))
            extract_tar_gz(arc, arc.parent)
        except Exception as e:
            logger.warning("Skipping %s: %s", arc, e)


def _log_and_print(msg: str, level: str = "info") -> None:
    """Log and print so both logger and Airflow task stdout show the message."""
    getattr(logger, level)(msg)
    print(msg, flush=True)


def _download_ravdess(url: str, raw_root: Optional[Path] = None) -> bool:
    """
    RAVDESS download + extract: zip at raw_root/RAVDESS.zip, extract to raw_root/RAVDESS/.
    """
    root = Path(raw_root or RAW_DIR).resolve()
    zip_path = root / "RAVDESS.zip"
    dataset_dir = root / "RAVDESS"
    if dataset_dir.exists():
        _log_and_print("RAVDESS already extracted. Skipping download.")
        return True
    _log_and_print(f"Downloading RAVDESS from: {url}")
    if not download_file(url, zip_path, connect_timeout=30, read_timeout=7200):
        return False
    _log_and_print("Extracting RAVDESS...")
    dataset_dir.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(zip_path, "r") as zf:
        zf.extractall(dataset_dir)
    _log_and_print("RAVDESS download + extraction complete.")
    return True


def download_datasets(datasets: Optional[List[str]] = None) -> Dict[str, bool]:
    """Download configured datasets into data/raw/<name>/."""
    from scripts.utils import CONFIG_DIR

    raw_root = Path(RAW_DIR).resolve()
    _log_and_print(f"[Step 0] Raw data directory (preprocess reads here): {raw_root}")

    config_path = CONFIG_DIR / "datasets.yaml"
    _log_and_print(f"[Step 1] Loading config from: {config_path}")
    cfg = load_config()
    if not cfg:
        _log_and_print("ERROR: Config is empty or file not found. No datasets will be downloaded.", "error")
        return {}

    emotion = cfg.get("emotion_datasets", {})
    speech = cfg.get("multilingual_speech", {})
    all_ds = {**emotion, **speech}
    _log_and_print(f"[Step 2] Config loaded. emotion_datasets: {list(emotion.keys())}, multilingual_speech: {list(speech.keys())}")

    if datasets:
        all_ds = {k: v for k, v in all_ds.items() if k in datasets}
        _log_and_print(f"[Step 3] Filtered to requested datasets: {list(all_ds.keys())}")
    else:
        _log_and_print(f"[Step 3] No filter (datasets=None). Using all configured datasets: {list(all_ds.keys())}")

    # Log every dataset and its URL before doing any downloads
    _log_and_print("[Step 4] Dataset URLs from config:")
    for name, meta in all_ds.items():
        if not isinstance(meta, dict):
            _log_and_print(f"  - {name}: (invalid meta, skipping)")
            continue
        url = meta.get("url")
        if url:
            _log_and_print(f"  - {name}: URL={url}")
        elif name == "MELD":
            _log_and_print(f"  - {name}: via Kaggle (kagglehub) or URL/Hugging Face")
        elif name == "TESS":
            _log_and_print(f"  - {name}: via Kaggle (pip install kagglehub) or Hugging Face")
        else:
            _log_and_print(f"  - {name}: no URL configured (will skip)")

    results = {}
    for name, meta in all_ds.items():
        if not isinstance(meta, dict):
            continue
        # TESS: always use Kaggle (kagglehub) then HF then archive extract — ignore Borealis URL so we get real WAVs
        if name == "TESS":
            _log_and_print("[Fetch] TESS: Kaggle (kagglehub) then Hugging Face, then extract archives")
            tess_dir = raw_root / "TESS"
            if tess_dir.exists():
                _try_extract_archives_in(tess_dir)
            ok = _fetch_tess_via_kaggle(raw_root)
            if not ok:
                ok = _fetch_tess_via_hf(raw_root)
            results[name] = ok
            continue
        # MELD: prefer Kaggle (kagglehub) so WAVs land in data/raw/MELD/ for preprocessing; then URL or HF
        if name == "MELD":
            _log_and_print("[Fetch] MELD: Kaggle (kagglehub) then URL/Hugging Face")
            meld_dir = raw_root / "MELD"
            _log_and_print("MELD: target dir (preprocess reads here): %s" % meld_dir.resolve())
            if meld_dir.exists():
                _log_and_print("MELD: extracting any nested .tar.gz (train/dev/test) ...")
                _extract_nested_tar_gz(meld_dir)
                # Quick count so user can confirm preprocess will see them
                wavs = list(meld_dir.rglob("*.wav")) + list(meld_dir.rglob("*.WAV"))
                mp4s = list(meld_dir.rglob("*.mp4")) + list(meld_dir.rglob("*.MP4"))
                _log_and_print("MELD: after extraction: %d .wav, %d .mp4 (preprocess uses both)" % (len(wavs), len(mp4s)))
            ok = _fetch_meld_via_kaggle(raw_root)
            if not ok:
                url = meta.get("url")
                if url:
                    raw_dir = raw_root / "MELD"
                    raw_dir.mkdir(parents=True, exist_ok=True)
                    fname = url.rstrip("/").split("/")[-1].split("?")[0] or "MELD.Raw.tar.gz"
                    dest = raw_dir / fname
                    _log_and_print("[Download] MELD: %s -> %s" % (url, dest))
                    ok = download_file(url, dest, meta.get("checksum"))
                    if ok and (fname.endswith(".tar.gz") or fname.endswith(".tgz")):
                        extract_tar_gz(dest, raw_dir)
                        _extract_nested_tar_gz(raw_dir)
                if not ok:
                    _log_and_print("[Fetch] MELD: trying Hugging Face dataset ...")
                    ok = _fetch_meld_via_hf(raw_root)
            if ok and meld_dir.exists():
                _extract_nested_tar_gz(meld_dir)
                wavs = list(meld_dir.rglob("*.wav")) + list(meld_dir.rglob("*.WAV"))
                mp4s = list(meld_dir.rglob("*.mp4")) + list(meld_dir.rglob("*.MP4"))
                _log_and_print("MELD: ready for preprocess: %d .wav, %d .mp4 under %s" % (len(wavs), len(mp4s), meld_dir.resolve()))
            results[name] = ok
            continue
        url = meta.get("url")
        if not url:
            _log_and_print(f"[Skip] {name}: no URL configured (skipped, pipeline continues)", "warning")
            results[name] = True
            continue
        # RAVDESS: use exact flow that works when run manually (zip at raw root, extract to raw_root/RAVDESS)
        if name == "RAVDESS":
            results[name] = _download_ravdess(url, raw_root=raw_root)
            continue
        raw_dir = raw_root / name
        raw_dir.mkdir(parents=True, exist_ok=True)
        fname = url.rstrip("/").split("/")[-1].split("?")[0] or f"{name}.zip"
        dest = raw_dir / fname
        if dest.exists() and meta.get("checksum"):
            if compute_sha256(dest) == meta["checksum"]:
                _log_and_print(f"[OK] {name}: already present and valid: {dest}")
                results[name] = True
                continue
        _log_and_print(f"[Download] {name}: {url} -> {dest}")
        ok = download_file(url, dest, meta.get("checksum"))
        if ok:
            if fname.endswith(".zip"):
                unzip_if_needed(dest, raw_dir / "extracted")
            elif fname.endswith(".tar.gz") or fname.endswith(".tgz"):
                extract_tar_gz(dest, raw_dir)
        results[name] = ok

    _log_and_print(f"[Step 5] Done. Results: {results}")
    return results


def main() -> None:
    import sys
    datasets = sys.argv[1:] if len(sys.argv) > 1 else None
    r = download_datasets(datasets)
    failed = [k for k, v in r.items() if not v]
    if failed:
        logger.warning("Some downloads failed or skipped: %s", failed)
        sys.exit(1)
    logger.info("Download results: %s", r)


if __name__ == "__main__":
    main()
