#!/usr/bin/env python3
"""Initialize non-overlapping v7 expansion shards from the shared cache."""

import json
import os
import re
from pathlib import Path


ROOT = Path(
    os.environ.get("PAPER_MINING_ROOT", Path(__file__).resolve().parents[1])
).expanduser().resolve()
CACHE = ROOT / "discovery" / "all_papers_cache.json"
OLD_SHARDS = range(8)
NEW_SHARDS = range(8, 16)
PIPELINE_VERSION = 7


def safe_output_stem(paper: dict) -> str:
    value = (
        paper.get("arxiv_id")
        or paper.get("doi")
        or paper.get("pmcid")
        or paper.get("title")
        or "paper"
    )
    value = re.sub(r"[^A-Za-z0-9._-]+", "_", value).strip("._")
    return (value or "paper")[:120]


def main() -> None:
    with CACHE.open() as handle:
        papers = json.load(handle)

    existing_stems = {
        path.stem
        for shard in OLD_SHARDS
        for path in (ROOT / "shards" / f"shard_{shard}" / "parsed").glob("*.json")
    }
    existing_ids = []
    matched_stems = set()
    for paper in papers:
        stem = safe_output_stem(paper)
        if stem not in existing_stems:
            continue
        existing_ids.append(paper.get("arxiv_id") or paper.get("title", "unknown"))
        matched_stems.add(stem)

    unmatched = sorted(existing_stems - matched_stems)
    if unmatched:
        preview = ", ".join(unmatched[:10])
        raise RuntimeError(
            f"Could not match {len(unmatched)} existing outputs to the candidate cache: {preview}"
        )

    progress = {
        "searches_done": [],
        "papers_processed": 0,
        "total_collected": 0,
        "pipeline_version": PIPELINE_VERSION,
        "processed_ids": existing_ids,
        "ok": 0,
        "fail": 0,
        "arxiv_source_ok": 0,
        "pmc_xml_ok": 0,
        "gateway_ok": 0,
        "arxiv_ok": 0,
        "open_access_ok": 0,
        "gateway_fail": 0,
        "failure_reasons": {},
    }

    for shard in NEW_SHARDS:
        shard_dir = ROOT / "shards" / f"shard_{shard}"
        (shard_dir / "papers").mkdir(parents=True, exist_ok=True)
        (shard_dir / "parsed").mkdir(parents=True, exist_ok=True)
        progress_path = shard_dir / "bulk_progress.json"
        if progress_path.exists():
            raise RuntimeError(f"Refusing to overwrite existing progress: {progress_path}")
        with progress_path.open("w") as handle:
            json.dump(progress, handle, ensure_ascii=False, indent=2)

    print(
        f"Initialized shards 8-15; excluded {len(existing_ids)} existing papers "
        f"from {len(papers)} cached candidates."
    )


if __name__ == "__main__":
    main()
