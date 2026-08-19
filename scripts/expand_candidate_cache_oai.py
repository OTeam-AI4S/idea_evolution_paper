#!/usr/bin/env python3
"""Expand the paper candidate cache with official arXiv OAI-PMH metadata."""

from __future__ import annotations

import json
import os
import re
import time
import xml.etree.ElementTree as ET
from pathlib import Path

import requests

from paper_mining.downloader.arxiv_downloader import ArxivDownloader


ROOT = Path(
    os.environ.get("PAPER_MINING_ROOT", Path(__file__).resolve().parents[1])
).expanduser().resolve()
CACHE = ROOT / "discovery" / "all_papers_cache.json"
NEW_RECORDS = ROOT / "discovery" / "oai_new_records.jsonl"
PROGRESS = ROOT / "discovery" / "oai_harvest_progress.json"
TARGET_CANDIDATES = int(os.environ.get("PAPER_CANDIDATE_TARGET", "320000"))
OAI_URL = "https://oaipmh.arxiv.org/oai"
OAI_NS = "http://www.openarchives.org/OAI/2.0/"
ARXIV_NS = "http://arxiv.org/OAI/arXiv/"
CATEGORIES = [
    "cs.AI", "cs.CL", "cs.CV", "cs.LG", "cs.NE", "cs.IR", "cs.RO",
    "cs.CR", "cs.DS", "cs.DB", "cs.HC", "cs.SE", "cs.GT", "cs.IT",
    "cs.MA", "cs.MM", "cs.SI", "cs.CC", "cs.CG", "cs.DC", "cs.DM",
    "cs.ET", "cs.FL", "cs.GR", "cs.MS", "cs.NA", "cs.OH", "cs.OS",
    "cs.PF", "cs.PL", "cs.SC", "cs.SY", "cs.SD",
    "q-bio.BM", "q-bio.CB", "q-bio.GN", "q-bio.MN", "q-bio.NC",
    "q-bio.OT", "q-bio.PE", "q-bio.QM", "q-bio.SC", "q-bio.TO",
]


def oai_set(category: str) -> str:
    archive, suffix = category.split(".", 1)
    return f"{archive}:{archive}:{suffix}"


def text(parent: ET.Element, name: str) -> str:
    node = parent.find(f"{{{ARXIV_NS}}}{name}")
    if node is None:
        return ""
    return re.sub(r"\s+", " ", "".join(node.itertext())).strip()


def normalize_id(value: str) -> str:
    return re.sub(r"(?i)v\d+$", "", value.strip()).lower()


def normalize_title(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", " ", value.lower()).strip()


def parse_record(record: ET.Element) -> dict | None:
    header = record.find(f"{{{OAI_NS}}}header")
    if header is None or header.attrib.get("status") == "deleted":
        return None
    metadata = record.find(f"{{{OAI_NS}}}metadata")
    if metadata is None:
        return None
    arxiv = metadata.find(f"{{{ARXIV_NS}}}arXiv")
    if arxiv is None:
        return None

    arxiv_id = text(arxiv, "id")
    title = text(arxiv, "title")
    if not arxiv_id or not title:
        return None

    authors = []
    authors_node = arxiv.find(f"{{{ARXIV_NS}}}authors")
    if authors_node is not None:
        for author in authors_node.findall(f"{{{ARXIV_NS}}}author"):
            keyname = text(author, "keyname")
            forenames = text(author, "forenames")
            name = " ".join(part for part in (forenames, keyname) if part)
            if name:
                authors.append(name)

    categories = text(arxiv, "categories").split()
    primary = categories[0] if categories else ""
    label = ArxivDownloader.CATEGORIES.get(primary, primary)
    field = ArxivDownloader._category_field(primary, label)
    journal_ref = text(arxiv, "journal-ref")
    comment = text(arxiv, "comments")
    venue = journal_ref or ArxivDownloader._venue_from_comment(comment)
    created = text(arxiv, "created")
    year_match = re.search(r"(19|20)\d{2}", created)
    year = int(year_match.group(0)) if year_match else None

    return {
        "title": title,
        "authors": authors,
        "year": year,
        "doi": text(arxiv, "doi") or None,
        "pmid": None,
        "pmcid": None,
        "arxiv_id": arxiv_id,
        "field": field or "Unknown",
        "venue": venue or "arXiv preprint",
        "abstract": text(arxiv, "abstract"),
        "url": f"https://arxiv.org/abs/{arxiv_id}",
        "publisher_url": None,
        "pdf_url": f"https://arxiv.org/pdf/{arxiv_id}",
        "source": "arxiv",
        "keywords": categories,
        "citation_count": 0,
    }


def request_page(
    session: requests.Session,
    set_spec: str,
    token: str | None,
) -> ET.Element:
    if token:
        params = {"verb": "ListRecords", "resumptionToken": token}
    else:
        params = {
            "verb": "ListRecords",
            "metadataPrefix": "arXiv",
            "set": set_spec,
            "from": "2020-01-01",
        }
    last_error = None
    for attempt in range(8):
        try:
            response = session.get(OAI_URL, params=params, timeout=(20, 120))
        except requests.RequestException as exc:
            last_error = exc
            time.sleep(min(2 ** attempt, 60))
            continue
        if response.status_code == 200:
            return ET.fromstring(response.content)
        delay = int(response.headers.get("Retry-After", 0) or 0)
        time.sleep(max(delay, min(2 ** attempt, 60)))
    if last_error is not None:
        raise last_error
    response.raise_for_status()
    raise RuntimeError("unreachable")


def merge_cache() -> int:
    with CACHE.open() as handle:
        papers = json.load(handle)
    by_id = {normalize_id(p.get("arxiv_id") or "") for p in papers}
    by_title = {normalize_title(p.get("title") or "") for p in papers}
    with NEW_RECORDS.open() as handle:
        for line in handle:
            paper = json.loads(line)
            paper_id = normalize_id(paper.get("arxiv_id") or "")
            title = normalize_title(paper.get("title") or "")
            if paper_id in by_id or title in by_title:
                continue
            papers.append(paper)
            by_id.add(paper_id)
            by_title.add(title)

    temporary = CACHE.with_suffix(".json.tmp")
    with temporary.open("w") as handle:
        json.dump(papers, handle, ensure_ascii=False, indent=2)
    os.replace(temporary, CACHE)
    return len(papers)


def main() -> None:
    with CACHE.open() as handle:
        existing = json.load(handle)
    known_ids = {normalize_id(p.get("arxiv_id") or "") for p in existing}
    known_titles = {normalize_title(p.get("title") or "") for p in existing}
    del existing

    new_count = 0
    if NEW_RECORDS.exists():
        with NEW_RECORDS.open() as handle:
            for line in handle:
                paper = json.loads(line)
                known_ids.add(normalize_id(paper.get("arxiv_id") or ""))
                known_titles.add(normalize_title(paper.get("title") or ""))
                new_count += 1

    state = {}
    if PROGRESS.exists():
        with PROGRESS.open() as handle:
            state = json.load(handle)
    token = state.get("resumption_token")
    pages = int(state.get("pages", 0))
    set_index = int(state.get("set_index", 0))

    session = requests.Session()
    contact = os.environ.get("PAPER_MINING_CONTACT_EMAIL", "research@example.com")
    session.headers["User-Agent"] = f"PaperMining/0.1 (mailto:{contact})"
    while len(known_ids) < TARGET_CANDIDATES and set_index < len(CATEGORIES):
        category = CATEGORIES[set_index]
        set_spec = oai_set(category)
        root = request_page(session, set_spec, token)
        error = root.find(f"{{{OAI_NS}}}error")
        if error is not None:
            raise RuntimeError(f"OAI error {error.attrib.get('code')}: {error.text}")
        list_records = root.find(f"{{{OAI_NS}}}ListRecords")
        if list_records is None:
            raise RuntimeError("OAI response has no ListRecords element")

        added = []
        for record in list_records.findall(f"{{{OAI_NS}}}record"):
            paper = parse_record(record)
            if paper is None:
                continue
            paper_id = normalize_id(paper["arxiv_id"])
            title = normalize_title(paper["title"])
            if paper_id in known_ids or title in known_titles:
                continue
            known_ids.add(paper_id)
            known_titles.add(title)
            added.append(paper)

        if added:
            with NEW_RECORDS.open("a") as handle:
                for paper in added:
                    handle.write(json.dumps(paper, ensure_ascii=False) + "\n")
            new_count += len(added)

        token_node = list_records.find(f"{{{OAI_NS}}}resumptionToken")
        token = (token_node.text or "").strip() if token_node is not None else ""
        if not token:
            set_index += 1
        pages += 1
        with PROGRESS.open("w") as handle:
            json.dump(
                {
                    "resumption_token": token or None,
                    "set_index": set_index,
                    "last_category": category,
                    "pages": pages,
                    "new_records": new_count,
                    "candidate_count": len(known_ids),
                },
                handle,
                indent=2,
            )
        print(
            f"page={pages} category={category} added={len(added)} new={new_count} "
            f"candidates={len(known_ids)}/{TARGET_CANDIDATES}",
            flush=True,
        )
        time.sleep(5)

    total = merge_cache()
    print(f"Merged candidate cache: {total} papers", flush=True)


if __name__ == "__main__":
    main()
