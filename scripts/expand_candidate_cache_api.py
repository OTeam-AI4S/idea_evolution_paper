#!/usr/bin/env python3
"""Expand the candidate cache with rate-limited monthly arXiv API slices."""

from __future__ import annotations

import calendar
import json
import os
import re
import time
from pathlib import Path

from paper_mining.downloader.arxiv_downloader import ArxivDownloader


ROOT = Path(
    os.environ.get("PAPER_MINING_ROOT", Path(__file__).resolve().parents[1])
).expanduser().resolve()
CACHE = ROOT / "discovery" / "all_papers_cache.json"
NEW_RECORDS = ROOT / "discovery" / "api_new_records.jsonl"
PROGRESS = ROOT / "discovery" / "api_harvest_progress.json"
TARGET_CANDIDATES = int(os.environ.get("PAPER_CANDIDATE_TARGET", "320000"))
ALL_CATS = [
    "cs.AI", "cs.CL", "cs.CV", "cs.LG", "cs.NE", "cs.IR", "cs.RO",
    "cs.CR", "cs.DS", "cs.DB", "cs.HC", "cs.SE", "cs.GT", "cs.IT",
    "cs.MA", "cs.MM", "cs.SI", "cs.CC", "cs.CG", "cs.DC", "cs.DM",
    "cs.ET", "cs.FL", "cs.GR", "cs.MS", "cs.NA", "cs.OH", "cs.OS",
    "cs.PF", "cs.PL", "cs.SC", "cs.SY", "cs.SD",
    "q-bio.BM", "q-bio.CB", "q-bio.GN", "q-bio.MN", "q-bio.NC",
    "q-bio.OT", "q-bio.PE", "q-bio.QM", "q-bio.SC", "q-bio.TO",
]


def normalize_id(value: str) -> str:
    return re.sub(r"(?i)v\d+$", "", value.strip()).lower()


def normalize_title(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", " ", value.lower()).strip()


def as_dict(paper) -> dict:
    return {
        "title": paper.title,
        "authors": paper.authors,
        "year": paper.year,
        "doi": paper.doi,
        "pmid": paper.pmid,
        "pmcid": paper.pmcid,
        "arxiv_id": paper.arxiv_id,
        "field": paper.field,
        "venue": paper.venue,
        "abstract": paper.abstract,
        "url": paper.url,
        "publisher_url": paper.publisher_url,
        "pdf_url": paper.pdf_url,
        "source": paper.source,
        "keywords": paper.keywords,
        "citation_count": paper.citation_count,
    }


def build_tasks() -> list[tuple[str, str]]:
    tasks = []
    for year in range(2026, 2014, -1):
        last_month = 7 if year == 2026 else 12
        for month in range(last_month, 0, -1):
            last_day = calendar.monthrange(year, month)[1]
            date_range = (
                f"submittedDate:[{year}{month:02d}010000 TO "
                f"{year}{month:02d}{last_day:02d}2359]"
            )
            for category in ALL_CATS:
                tasks.append((category, date_range))
    return tasks


def merge_cache() -> int:
    with CACHE.open() as handle:
        papers = json.load(handle)
    by_id = {
        normalize_id(p.get("arxiv_id") or "")
        for p in papers if p.get("arxiv_id")
    }
    by_title = {
        normalize_title(p.get("title") or "")
        for p in papers if p.get("title")
    }
    with NEW_RECORDS.open() as handle:
        for line in handle:
            paper = json.loads(line)
            paper_id = normalize_id(paper.get("arxiv_id") or "")
            title = normalize_title(paper.get("title") or "")
            if paper_id in by_id or title in by_title:
                continue
            papers.append(paper)
            if paper_id:
                by_id.add(paper_id)
            if title:
                by_title.add(title)

    temporary = CACHE.with_suffix(".json.tmp")
    with temporary.open("w") as handle:
        json.dump(papers, handle, ensure_ascii=False, indent=2)
    os.replace(temporary, CACHE)
    return len(papers)


def main() -> None:
    with CACHE.open() as handle:
        existing = json.load(handle)
    known_ids = {
        normalize_id(p.get("arxiv_id") or "")
        for p in existing if p.get("arxiv_id")
    }
    known_titles = {
        normalize_title(p.get("title") or "")
        for p in existing if p.get("title")
    }
    del existing

    new_count = 0
    if NEW_RECORDS.exists():
        with NEW_RECORDS.open() as handle:
            for line in handle:
                paper = json.loads(line)
                paper_id = normalize_id(paper.get("arxiv_id") or "")
                title = normalize_title(paper.get("title") or "")
                if paper_id:
                    known_ids.add(paper_id)
                if title:
                    known_titles.add(title)
                new_count += 1

    state = {}
    if PROGRESS.exists():
        with PROGRESS.open() as handle:
            state = json.load(handle)
    next_task = int(state.get("next_task", 0))
    failed_tasks = list(state.get("failed_tasks", []))
    tasks = build_tasks()
    downloader = ArxivDownloader(
        output_dir=str(ROOT / "discovery" / "papers"),
        request_delay=3.5,
    )

    while next_task < len(tasks) and len(known_ids) < TARGET_CANDIDATES:
        category, date_range = tasks[next_task]
        results = None
        last_error = None
        for attempt in range(3):
            try:
                results = downloader.search(
                    query=date_range,
                    max_results=300,
                    categories=[category],
                    sort_by="submittedDate",
                )
                break
            except Exception as exc:
                last_error = str(exc)
                time.sleep(min(15 * (attempt + 1), 45))

        added = []
        if results is None:
            failed_tasks.append({
                "task": next_task,
                "category": category,
                "query": date_range,
                "error": last_error,
            })
        else:
            for result in results:
                paper = as_dict(result)
                paper_id = normalize_id(paper.get("arxiv_id") or "")
                title = normalize_title(paper.get("title") or "")
                if paper_id in known_ids or title in known_titles:
                    continue
                if paper_id:
                    known_ids.add(paper_id)
                if title:
                    known_titles.add(title)
                added.append(paper)

        if added:
            with NEW_RECORDS.open("a") as handle:
                for paper in added:
                    handle.write(json.dumps(paper, ensure_ascii=False) + "\n")
            new_count += len(added)

        next_task += 1
        with PROGRESS.open("w") as handle:
            json.dump(
                {
                    "next_task": next_task,
                    "total_tasks": len(tasks),
                    "new_records": new_count,
                    "candidate_count": len(known_ids),
                    "failed_tasks": failed_tasks[-100:],
                },
                handle,
                indent=2,
            )
        print(
            f"task={next_task}/{len(tasks)} category={category} "
            f"returned={len(results or [])} added={len(added)} "
            f"candidates={len(known_ids)}/{TARGET_CANDIDATES}",
            flush=True,
        )

    total = merge_cache()
    print(f"Merged candidate cache: {total} papers", flush=True)


if __name__ == "__main__":
    main()
