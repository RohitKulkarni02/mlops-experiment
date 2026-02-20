"""
Legal glossary preparation: process 500+ legal terms for translation validation.
Expects source glossary (e.g. from repo data/legal_glossary/legal_terms.json); writes normalized version for pipeline.
"""
import json
from pathlib import Path
from typing import List, Optional

from scripts.utils import get_logger, load_config, LEGAL_GLOSSARY_DIR, PIPELINE_ROOT

logger = get_logger("legal_glossary_prep")


def load_glossary(path: Optional[Path] = None) -> List[dict]:
    """Load legal terms from JSON. Accepts repo root data/legal_glossary or pipeline data/legal_glossary."""
    if path is None:
        for base in (LEGAL_GLOSSARY_DIR, PIPELINE_ROOT.parent / "data" / "legal_glossary"):
            p = base / "legal_terms.json"
            if p.exists():
                path = p
                break
    if path is None or not path.exists():
        logger.warning("No legal_terms.json found")
        return []
    with open(path, encoding="utf-8") as f:
        data = json.load(f)
    terms = data.get("terms", data) if isinstance(data, dict) else data
    return terms if isinstance(terms, list) else []


def run_glossary_prep(
    source_path: Optional[Path] = None,
    out_dir: Optional[Path] = None,
) -> int:
    """Normalize and write glossary for translation validation; return term count."""
    terms = load_glossary(source_path)
    if out_dir is None:
        out_dir = LEGAL_GLOSSARY_DIR
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "legal_terms_processed.json"
    normalized = []
    for t in terms:
        if isinstance(t, dict):
            normalized.append({
                "term": t.get("term", "").strip(),
                "pronunciation": t.get("pronunciation", "").strip(),
                "category": t.get("category", "").strip(),
                "definition": t.get("definition", "").strip(),
            })
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump({"terms": normalized, "count": len(normalized)}, f, indent=2)
    logger.info("Wrote %d terms to %s", len(normalized), out_path)
    return len(normalized)


def main() -> None:
    import sys
    src = Path(sys.argv[1]) if len(sys.argv) > 1 else None
    run_glossary_prep(source_path=src)


if __name__ == "__main__":
    main()
