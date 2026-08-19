"""Local-first DOI enrichment for publication venue metadata."""

from __future__ import annotations

import json
import logging
import os
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Dict, Iterable, List, Optional
from urllib.parse import quote

import requests

from ..parser.reference_abstracts import normalize_doi, normalize_whitespace


logger = logging.getLogger(__name__)


class PublicationMetadataResolver:
    """Resolve DOI publication venues, persisting results as append-only JSONL."""

    BASE_URL = "https://api.crossref.org/works/{doi}"

    def __init__(
        self,
        cache_path: str,
        allow_remote: bool = True,
        max_workers: int = 6,
    ):
        self.cache_path = Path(cache_path)
        self.allow_remote = allow_remote
        self.max_workers = max_workers
        self.records: Dict[str, dict] = {}
        contact = os.environ.get("PAPER_MINING_CONTACT_EMAIL", "research@example.com")
        self.session_headers = {
            "User-Agent": f"PaperMining/0.4 (mailto:{contact})"
        }
        self._load()

    def enrich(self, papers: List[dict]) -> List[dict]:
        missing = set()
        for paper in papers:
            doi = normalize_doi(paper.get("doi"))
            if doi and doi not in self.records and self.allow_remote:
                missing.add(doi)

        if missing:
            fetched = []
            with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
                futures = {
                    executor.submit(self._fetch, doi): doi
                    for doi in sorted(missing)
                }
                for future in as_completed(futures):
                    record = future.result()
                    if record:
                        self.records[record["doi"]] = record
                        fetched.append(record)
            self._append(fetched)

        for paper in papers:
            record = self.records.get(normalize_doi(paper.get("doi")))
            if not record:
                continue
            current_venue = normalize_whitespace(paper.get("venue"))
            if not current_venue or current_venue == "arXiv preprint":
                paper["venue"] = record.get("venue") or current_venue
            if (
                (not paper.get("field") or paper.get("field") == "Unknown")
                and record.get("field")
            ):
                paper["field"] = record["field"]
        return papers

    def _fetch(self, doi: str) -> Optional[dict]:
        session = requests.Session()
        session.headers.update(self.session_headers)
        try:
            response = session.get(
                self.BASE_URL.format(doi=quote(doi, safe="")),
                timeout=(10, 30),
            )
            response.raise_for_status()
            item = response.json().get("message", {})
            containers = item.get("container-title") or []
            venue = normalize_whitespace(
                containers[0] if containers else item.get("publisher")
            )
            subjects = item.get("subject") or []
            field = normalize_whitespace(subjects[0] if subjects else "")
            return {
                "doi": doi,
                "venue": venue or None,
                "field": field or None,
            }
        except Exception as exc:
            logger.debug("Publication lookup failed for %s: %s", doi, exc)
            return None

    def _load(self) -> None:
        if not self.cache_path.exists():
            return
        try:
            with open(self.cache_path, encoding="utf-8") as handle:
                for line in handle:
                    if not line.strip():
                        continue
                    record = json.loads(line)
                    doi = normalize_doi(record.get("doi"))
                    if doi:
                        self.records[doi] = record
        except (OSError, json.JSONDecodeError) as exc:
            logger.warning(
                "Cannot load publication metadata cache %s: %s",
                self.cache_path,
                exc,
            )

    def _append(self, records: Iterable[dict]) -> None:
        records = list(records)
        if not records:
            return
        self.cache_path.parent.mkdir(parents=True, exist_ok=True)
        with open(self.cache_path, "a", encoding="utf-8") as handle:
            for record in records:
                handle.write(json.dumps(record, ensure_ascii=False) + "\n")
        self.cache_path.chmod(0o644)
