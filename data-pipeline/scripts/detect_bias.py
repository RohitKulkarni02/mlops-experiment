"""
Bias detection via data slicing: by demographics (gender, accent), emotion, language, audio quality.
Compute metrics per slice and document disparities. Output report for mitigation (re-sampling, stratified eval).
"""
import json
from pathlib import Path
from typing import Dict, List, Optional

from scripts.utils import get_logger, load_config, PROCESSED_DIR

logger = get_logger("detect_bias")


def load_manifests(data_dir: Path) -> List[dict]:
    """Load all manifest.json entries from dev/test/holdout."""
    entries = []
    for split in ("dev", "test", "holdout"):
        manifest_path = data_dir / split / "manifest.json"
        if not manifest_path.exists():
            continue
        with open(manifest_path, encoding="utf-8") as f:
            data = json.load(f)
        for item in (data if isinstance(data, list) else data.get("items", [])):
            item["split"] = split
            entries.append(item)
    return entries


def slice_by_emotion(entries: List[dict]) -> Dict[str, List[dict]]:
    """Slice by emotion label."""
    by_emotion: Dict[str, List[dict]] = {}
    for e in entries:
        label = e.get("emotion") or e.get("label") or "unknown"
        by_emotion.setdefault(label, []).append(e)
    return by_emotion


def slice_by_speaker(entries: List[dict]) -> Dict[str, List[dict]]:
    """Slice by speaker_id (proxy for demographics when no explicit gender/accent)."""
    by_speaker: Dict[str, List[dict]] = {}
    for e in entries:
        sid = e.get("speaker_id", "unknown")
        by_speaker.setdefault(sid, []).append(e)
    return by_speaker


def compute_counts_per_slice(slices: Dict[str, List[dict]]) -> Dict[str, int]:
    """Return count per slice for imbalance check."""
    return {k: len(v) for k, v in slices.items()}


def run_bias_analysis(
    data_dir: Optional[Path] = None,
    report_path: Optional[Path] = None,
) -> dict:
    """Run slicing, compute counts, and write bias report."""
    if data_dir is None:
        data_dir = PROCESSED_DIR
    data_dir = Path(data_dir)
    if report_path is None:
        report_path = data_dir / "bias_report.json"

    entries = load_manifests(data_dir)
    if not entries:
        logger.warning("No manifest entries found under %s", data_dir)
        return {"slices": {}, "disparities": [], "recommendations": []}

    by_emotion = slice_by_emotion(entries)
    by_speaker = slice_by_speaker(entries)
    emotion_counts = compute_counts_per_slice(by_emotion)
    speaker_counts = compute_counts_per_slice(by_speaker)

    total = len(entries)
    expected_per_emotion = total / len(by_emotion) if by_emotion else 0
    disparities = []
    for label, count in emotion_counts.items():
        if expected_per_emotion and abs(count - expected_per_emotion) / expected_per_emotion > 0.5:
            disparities.append({
                "slice": "emotion",
                "value": label,
                "count": count,
                "expected_approx": round(expected_per_emotion, 1),
                "note": "Imbalanced emotion class; consider re-sampling or stratified evaluation.",
            })

    report = {
        "total_entries": total,
        "by_emotion": emotion_counts,
        "by_speaker_count": len(by_speaker),
        "speaker_sample_counts": dict(list(speaker_counts.items())[:20]),  # sample
        "disparities": disparities,
        "recommendations": [
            "Use stratified evaluation and report metrics per emotion slice.",
            "If deploying, consider confidence thresholding for under-represented classes.",
        ],
    }
    report_path.parent.mkdir(parents=True, exist_ok=True)
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    logger.info("Bias report written to %s; %d disparities noted", report_path, len(disparities))
    return report


def main() -> None:
    import sys
    data_dir = Path(sys.argv[1]) if len(sys.argv) > 1 else None
    run_bias_analysis(data_dir=data_dir)


if __name__ == "__main__":
    main()
