"""Resolve cited-paper identities to complete abstracts.

Resolution is deliberately identifier-first.  Local metadata is consulted
before any network endpoint, and title-only references are accepted only when
they exactly match a normalized title in the local metadata index.  The
resolver never substitutes citation context or truncates an abstract.
"""

from __future__ import annotations

import html
import json
import logging
import os
import re
import time
import xml.etree.ElementTree as ET
from difflib import SequenceMatcher
from pathlib import Path
from typing import Dict, Iterable, List, Optional
from urllib.parse import quote

import requests


logger = logging.getLogger(__name__)

ARXIV_API_URL = "https://export.arxiv.org/api/query"
PUBMED_EFETCH_URL = (
    "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/efetch.fcgi"
)
CROSSREF_WORK_URL = "https://api.crossref.org/works/{identifier}"
CROSSREF_SEARCH_URL = "https://api.crossref.org/works"
SEMANTIC_SCHOLAR_BATCH_URL = (
    "https://api.semanticscholar.org/graph/v1/paper/batch"
)
DEFAULT_SEMANTIC_SCHOLAR_KEY_PATH = (
    "~/.config/paper_mining/semantic_scholar_api_key"
)


def normalize_whitespace(value: Optional[str]) -> str:
    """Change whitespace layout only; preserve every non-whitespace character."""
    return re.sub(r"\s+", " ", value or "").strip()


def normalize_title(value: Optional[str]) -> str:
    value = (value or "").translate(str.maketrans({
        "ﬁ": "fi", "ﬂ": "fl", "ﬀ": "ff", "ﬃ": "ffi", "ﬄ": "ffl",
        "\u00ad": "",
    }))
    value = re.sub(r"(?<=[a-z])-\s+(?=[a-z])", "", value)
    return re.sub(r"[^a-z0-9]+", " ", value.lower()).strip()


def normalize_arxiv_id(value: Optional[str]) -> str:
    value = re.sub(r"(?i)^(?:arxiv\s*:|https?://arxiv\.org/(?:abs|pdf)/)", "", value or "")
    return re.sub(r"(?i)v\d+$", "", value).strip().lower()


def normalize_doi(value: Optional[str]) -> str:
    value = re.sub(
        r"(?i)^(?:doi\s*:|https?://(?:dx\.)?doi\.org/)",
        "",
        value or "",
    )
    return value.strip().rstrip(".,;:)]}'\"").lower()


def normalize_pmid(value: Optional[str]) -> str:
    value = re.sub(r"(?i)^pmid\s*:\s*", "", value or "").strip()
    return value if value.isdigit() else ""


def clean_abstract(value: Optional[str]) -> str:
    """Remove markup and an optional label, but never shorten prose."""
    text = html.unescape(value or "")
    text = re.sub(r"<[^>]+>", " ", text)
    text = normalize_whitespace(text)
    return re.sub(r"(?i)^abstract\s*[:.-]?\s*", "", text).strip()


def abstract_is_complete(value: Optional[str]) -> bool:
    """Conservative guard against empty metadata and obvious snippets."""
    text = clean_abstract(value)
    if len(text) < 80:
        return False
    if text.endswith(("…", "...")):
        return False
    return True


class ReferenceAbstractResolver:
    """Resolve reference candidates from local indexes, then optional endpoints."""

    def __init__(
        self,
        metadata_paths: Optional[Iterable[str]] = None,
        cache_path: Optional[str] = None,
        allow_remote: bool = False,
        request_timeout: int = 30,
        max_title_lookups: int = 30,
        allow_crossref_title_search: bool = True,
        semantic_scholar_api_key: Optional[str] = None,
        semantic_scholar_key_path: Optional[str] = None,
    ):
        self.allow_remote = allow_remote
        self.request_timeout = request_timeout
        self.max_title_lookups = max_title_lookups
        self.allow_crossref_title_search = allow_crossref_title_search
        self.semantic_scholar_api_key = self._load_semantic_scholar_api_key(
            semantic_scholar_api_key,
            semantic_scholar_key_path,
        )
        self.cache_path = Path(cache_path) if cache_path else None
        self.session = requests.Session()
        # Use the environment selected by the launcher. The Slurm script tests
        # the configured proxy and unsets it only when it is actually dead.
        self.session.trust_env = True
        contact = os.environ.get("PAPER_MINING_CONTACT_EMAIL", "research@example.com")
        self.session.headers.update({
            "User-Agent": f"PaperMining/0.4 (mailto:{contact})"
        })

        self.by_arxiv: Dict[str, dict] = {}
        self.by_doi: Dict[str, dict] = {}
        self.by_pmid: Dict[str, dict] = {}
        self.by_title: Dict[str, dict] = {}
        self.ambiguous_titles = set()
        self._cache_records: Dict[str, dict] = {}
        self._pending_cache: Dict[str, dict] = {}

        for path in metadata_paths or []:
            self._load_metadata(Path(path))
        if self.cache_path:
            self._load_metadata(self.cache_path, is_cache=True)

    def resolve_many(self, candidates: Iterable[dict]) -> List[dict]:
        """Return only references whose cited-paper abstract was verified."""
        candidates = list(candidates)
        resolved: Dict[int, dict] = {}
        pending: List[tuple[int, dict]] = []

        for index, candidate in enumerate(candidates):
            record = self._local_match(candidate)
            if record:
                final = self._final_reference(candidate, record)
                if final:
                    resolved[index] = final
                    continue
            pending.append((index, candidate))

        if self.allow_remote and pending:
            self._fetch_semantic_scholar([
                candidate for _, candidate in pending
                if not self._local_match(candidate)
            ])
            self._fetch_arxiv([
                candidate.get("arxiv_id", "")
                for _, candidate in pending
                if candidate.get("arxiv_id")
            ])
            self._fetch_pubmed([
                candidate.get("pmid", "")
                for _, candidate in pending
                if candidate.get("pmid")
            ])
            for _, candidate in pending:
                if candidate.get("doi") and not self._local_match(candidate):
                    self._fetch_crossref(candidate["doi"])

            unresolved = [
                candidate for _, candidate in pending
                if not self._local_match(candidate)
                and self._title_candidate_is_usable(candidate)
            ]
            unresolved.sort(
                key=lambda candidate: (
                    self._year(candidate.get("year")) or 0,
                    len(normalize_title(candidate.get("title")).split()),
                ),
                reverse=True,
            )
            unresolved = unresolved[:self.max_title_lookups]
            self._fetch_arxiv_by_titles(unresolved)
            if self.allow_crossref_title_search:
                for candidate in unresolved:
                    if not self._local_match(candidate):
                        self._fetch_crossref_by_title(candidate)

            for index, candidate in pending:
                record = self._local_match(candidate)
                final = self._final_reference(candidate, record) if record else None
                if final:
                    resolved[index] = final

        if self.cache_path:
            self._save_cache()
        return [resolved[index] for index in sorted(resolved)]

    def _local_match(self, candidate: dict) -> Optional[dict]:
        arxiv_id = normalize_arxiv_id(candidate.get("arxiv_id"))
        doi = normalize_doi(candidate.get("doi"))
        pmid = normalize_pmid(candidate.get("pmid"))
        title = normalize_title(candidate.get("title"))
        if arxiv_id and arxiv_id in self.by_arxiv:
            return self.by_arxiv[arxiv_id]
        if doi and doi in self.by_doi:
            return self.by_doi[doi]
        if pmid and pmid in self.by_pmid:
            return self.by_pmid[pmid]
        # Exact normalized title matching is intentional. Fuzzy title matching
        # can silently attach the abstract of a different paper.
        if (
            title
            and title not in self.ambiguous_titles
            and title in self.by_title
        ):
            return self.by_title[title]
        return None

    def _final_reference(self, candidate: dict, record: dict) -> Optional[dict]:
        abstract = clean_abstract(record.get("abstract"))
        if not abstract_is_complete(abstract):
            return None
        title = normalize_whitespace(record.get("title") or candidate.get("title"))
        if not title:
            return None

        identifier = self._best_identifier(record) or candidate.get("id")
        if not identifier:
            return None
        contexts = candidate.get("citation_contexts")
        if not isinstance(contexts, list):
            contexts = []
        return {
            "id": identifier,
            "title": title,
            # Kept for schema compatibility. Its value is the complete
            # cited-paper abstract, never an in-paper citation context.
            "summary": abstract,
            "citation_contexts": [
                dict(context) for context in contexts
                if isinstance(context, dict)
            ],
        }

    @staticmethod
    def _load_semantic_scholar_api_key(
        explicit_key: Optional[str],
        key_path: Optional[str],
    ) -> str:
        key = (explicit_key or os.environ.get(
            "SEMANTIC_SCHOLAR_API_KEY", ""
        )).strip()
        if key:
            return key
        path = Path(
            key_path or os.environ.get(
                "SEMANTIC_SCHOLAR_API_KEY_FILE",
                DEFAULT_SEMANTIC_SCHOLAR_KEY_PATH,
            )
        ).expanduser()
        try:
            return path.read_text(encoding="utf-8").strip()
        except OSError:
            return ""

    @staticmethod
    def _semantic_scholar_identifier(candidate: dict) -> str:
        arxiv_id = normalize_arxiv_id(candidate.get("arxiv_id"))
        if arxiv_id:
            return f"ARXIV:{arxiv_id}"
        doi = normalize_doi(candidate.get("doi"))
        if doi:
            return f"DOI:{doi}"
        pmid = normalize_pmid(candidate.get("pmid"))
        if pmid:
            return f"PMID:{pmid}"
        return ""

    def _fetch_semantic_scholar(self, candidates: Iterable[dict]) -> None:
        """Batch-resolve stable IDs before slower source-specific lookups."""
        if not self.semantic_scholar_api_key:
            return
        identifiers = list(dict.fromkeys(
            identifier
            for candidate in candidates
            if (identifier := self._semantic_scholar_identifier(candidate))
        ))
        fields = "title,abstract,authors,year,externalIds"
        for start in range(0, len(identifiers), 500):
            batch = identifiers[start:start + 500]
            for attempt in range(4):
                try:
                    response = self.session.post(
                        SEMANTIC_SCHOLAR_BATCH_URL,
                        params={"fields": fields},
                        json={"ids": batch},
                        headers={"x-api-key": self.semantic_scholar_api_key},
                        timeout=self.request_timeout,
                    )
                    if response.status_code == 429:
                        retry_after = response.headers.get("Retry-After", "")
                        try:
                            delay = float(retry_after)
                        except ValueError:
                            delay = min(2 ** attempt, 30)
                        if attempt < 3:
                            time.sleep(max(0.0, min(delay, 60.0)))
                            continue
                    response.raise_for_status()
                    for item in response.json():
                        if not isinstance(item, dict):
                            continue
                        external = item.get("externalIds") or {}
                        self._register({
                            "title": item.get("title"),
                            "abstract": item.get("abstract"),
                            "arxiv_id": external.get("ArXiv"),
                            "doi": external.get("DOI"),
                            "pmid": external.get("PubMed"),
                            "authors": [
                                author.get("name", "")
                                for author in item.get("authors") or []
                                if isinstance(author, dict)
                            ],
                            "year": item.get("year"),
                        }, persist=True)
                    break
                except Exception as exc:
                    if attempt == 3:
                        logger.warning(
                            "Semantic Scholar batch lookup failed: %s", exc
                        )

    def _register(
        self,
        record: dict,
        cache_key: Optional[str] = None,
        persist: bool = False,
    ) -> None:
        abstract = clean_abstract(record.get("abstract"))
        title = normalize_whitespace(record.get("title"))
        if not title or not abstract_is_complete(abstract):
            return
        normalized = {
            "title": title,
            "abstract": abstract,
            "arxiv_id": normalize_arxiv_id(record.get("arxiv_id")) or None,
            "doi": normalize_doi(record.get("doi")) or None,
            "pmid": normalize_pmid(record.get("pmid")) or None,
            "authors": record.get("authors") or [],
            "year": str(record.get("year") or ""),
        }
        if normalized["arxiv_id"]:
            self.by_arxiv[normalized["arxiv_id"]] = normalized
        if normalized["doi"]:
            self.by_doi[normalized["doi"]] = normalized
        if normalized["pmid"]:
            self.by_pmid[normalized["pmid"]] = normalized
        title_key = normalize_title(title)
        existing = self.by_title.get(title_key)
        if existing and self._best_identifier(existing) != self._best_identifier(normalized):
            self.ambiguous_titles.add(title_key)
            self.by_title.pop(title_key, None)
        elif title_key not in self.ambiguous_titles:
            self.by_title[title_key] = normalized
        key = cache_key or self._best_identifier(normalized)
        if key and persist:
            self._cache_records[key] = normalized
            self._pending_cache[key] = normalized

    @staticmethod
    def _best_identifier(record: dict) -> str:
        arxiv_id = normalize_arxiv_id(record.get("arxiv_id"))
        if arxiv_id:
            return f"arxiv:{arxiv_id}"
        doi = normalize_doi(record.get("doi"))
        if doi:
            return f"doi:{doi}"
        pmid = normalize_pmid(record.get("pmid"))
        if pmid:
            return f"pmid:{pmid}"
        title = normalize_title(record.get("title"))
        return f"title:{title}" if title else ""

    def _load_metadata(self, path: Path, is_cache: bool = False) -> None:
        if not path.exists() or path.stat().st_size == 0:
            return
        try:
            with open(path, encoding="utf-8") as handle:
                if path.suffix.lower() == ".jsonl":
                    records = [json.loads(line) for line in handle if line.strip()]
                else:
                    payload = json.load(handle)
                    records = (
                        list(payload.values())
                        if isinstance(payload, dict)
                        else payload
                    )
            for record in records:
                if isinstance(record, dict):
                    self._register(record)
                    if is_cache:
                        key = self._best_identifier(record)
                        if key:
                            self._cache_records[key] = record
        except (OSError, json.JSONDecodeError) as exc:
            logger.warning("Cannot load reference metadata %s: %s", path, exc)

    def _fetch_arxiv(self, identifiers: Iterable[str]) -> None:
        identifiers = list(dict.fromkeys(
            value for value in map(normalize_arxiv_id, identifiers) if value
        ))
        missing = [value for value in identifiers if value not in self.by_arxiv]
        for start in range(0, len(missing), 100):
            batch = missing[start:start + 100]
            try:
                response = self.session.get(
                    ARXIV_API_URL,
                    params={"id_list": ",".join(batch), "max_results": len(batch)},
                    timeout=self.request_timeout,
                )
                response.raise_for_status()
                root = ET.fromstring(response.content)
                namespace = {"atom": "http://www.w3.org/2005/Atom"}
                for entry in root.findall("atom:entry", namespace):
                    raw_id = entry.findtext("atom:id", "", namespace)
                    self._register({
                        "title": entry.findtext("atom:title", "", namespace),
                        "abstract": entry.findtext("atom:summary", "", namespace),
                        "arxiv_id": raw_id.rsplit("/", 1)[-1],
                        "authors": [
                            node.findtext("atom:name", "", namespace)
                            for node in entry.findall("atom:author", namespace)
                        ],
                        "year": entry.findtext(
                            "atom:published", "", namespace
                        )[:4],
                    }, persist=True)
            except Exception as exc:
                logger.warning("arXiv reference metadata lookup failed: %s", exc)

    def _fetch_pubmed(self, identifiers: Iterable[str]) -> None:
        identifiers = list(dict.fromkeys(
            value for value in map(normalize_pmid, identifiers) if value
        ))
        missing = [value for value in identifiers if value not in self.by_pmid]
        for start in range(0, len(missing), 200):
            batch = missing[start:start + 200]
            try:
                response = self.session.get(
                    PUBMED_EFETCH_URL,
                    params={
                        "db": "pubmed",
                        "id": ",".join(batch),
                        "retmode": "xml",
                        "tool": "paper_mining",
                        "email": os.environ.get(
                            "PAPER_MINING_CONTACT_EMAIL", "research@example.com"
                        ),
                    },
                    timeout=self.request_timeout,
                )
                response.raise_for_status()
                root = ET.fromstring(response.content)
                for article in root.findall(".//PubmedArticle"):
                    pmid = article.findtext(".//PMID", "")
                    title = " ".join(
                        article.find(".//ArticleTitle").itertext()
                    ) if article.find(".//ArticleTitle") is not None else ""
                    abstract = " ".join(
                        " ".join(node.itertext())
                        for node in article.findall(".//Abstract/AbstractText")
                    )
                    doi = ""
                    for node in article.findall(".//ArticleId"):
                        if node.attrib.get("IdType") == "doi":
                            doi = node.text or ""
                    self._register({
                        "title": title,
                        "abstract": abstract,
                        "pmid": pmid,
                        "doi": doi,
                    }, persist=True)
            except Exception as exc:
                logger.warning("PubMed reference metadata lookup failed: %s", exc)

    def _fetch_crossref(self, identifier: str) -> None:
        doi = normalize_doi(identifier)
        if not doi or doi in self.by_doi:
            return
        try:
            response = self.session.get(
                CROSSREF_WORK_URL.format(identifier=quote(doi, safe="")),
                timeout=self.request_timeout,
            )
            response.raise_for_status()
            item = response.json().get("message", {})
            titles = item.get("title") or []
            self._register({
                "title": titles[0] if titles else "",
                "abstract": item.get("abstract"),
                "doi": item.get("DOI") or doi,
            }, persist=True)
        except Exception as exc:
            logger.debug("Crossref abstract lookup failed for %s: %s", doi, exc)

    def _fetch_arxiv_by_titles(self, candidates: List[dict]) -> None:
        """Resolve title/author/year candidates in small arXiv API batches."""
        unique = {}
        for candidate in candidates:
            key = normalize_title(candidate.get("title"))
            if key and key not in self.by_title:
                unique.setdefault(key, candidate)
        values = list(unique.values())

        for start in range(0, len(values), 10):
            batch = values[start:start + 10]
            clauses = []
            for candidate in batch:
                title = normalize_whitespace(candidate.get("title"))
                title = title.replace('"', " ")
                clauses.append(f'ti:"{title}"')
            try:
                response = self.session.get(
                    ARXIV_API_URL,
                    params={
                        "search_query": " OR ".join(clauses),
                        "start": 0,
                        "max_results": min(100, len(batch) * 5),
                    },
                    timeout=self.request_timeout,
                )
                response.raise_for_status()
                root = ET.fromstring(response.content)
                namespace = {"atom": "http://www.w3.org/2005/Atom"}
                records = []
                for entry in root.findall("atom:entry", namespace):
                    raw_id = entry.findtext("atom:id", "", namespace)
                    records.append({
                        "title": entry.findtext(
                            "atom:title", "", namespace
                        ),
                        "abstract": entry.findtext(
                            "atom:summary", "", namespace
                        ),
                        "arxiv_id": raw_id.rsplit("/", 1)[-1],
                        "authors": [
                            node.findtext("atom:name", "", namespace)
                            for node in entry.findall(
                                "atom:author", namespace
                            )
                        ],
                        "year": entry.findtext(
                            "atom:published", "", namespace
                        )[:4],
                    })

                for candidate in batch:
                    match = self._best_identity_match(candidate, records)
                    if match:
                        self._register(match, persist=True)
            except Exception as exc:
                logger.warning(
                    "arXiv title reference lookup failed: %s",
                    exc,
                )

    def _fetch_crossref_by_title(self, candidate: dict) -> None:
        title = normalize_whitespace(candidate.get("title"))
        if not title:
            return
        params = {
            "query.title": title,
            "rows": 5,
            "select": "DOI,title,abstract,author,published,issued",
        }
        surname = self._first_author_surname(candidate.get("authors"))
        if surname:
            params["query.author"] = surname
        try:
            response = self.session.get(
                CROSSREF_SEARCH_URL,
                params=params,
                timeout=self.request_timeout,
            )
            response.raise_for_status()
            records = []
            for item in response.json().get("message", {}).get("items", []):
                titles = item.get("title") or []
                authors = [
                    " ".join(filter(None, (
                        author.get("given"), author.get("family")
                    )))
                    for author in item.get("author") or []
                ]
                date_parts = (
                    (item.get("published") or item.get("issued") or {})
                    .get("date-parts") or []
                )
                year = (
                    str(date_parts[0][0])
                    if date_parts and date_parts[0] else ""
                )
                records.append({
                    "title": titles[0] if titles else "",
                    "abstract": item.get("abstract"),
                    "doi": item.get("DOI"),
                    "authors": authors,
                    "year": year,
                })
            match = self._best_identity_match(candidate, records)
            if match and abstract_is_complete(match.get("abstract")):
                self._register(match, persist=True)
        except Exception as exc:
            logger.debug(
                "Crossref title lookup failed for %s: %s",
                title,
                exc,
            )

    def _best_identity_match(
        self,
        candidate: dict,
        records: Iterable[dict],
    ) -> Optional[dict]:
        best = None
        best_score = 0.0
        candidate_title = normalize_title(candidate.get("title"))
        candidate_year = self._year(candidate.get("year"))
        candidate_author = self._first_author_surname(
            candidate.get("authors")
        )
        for record in records:
            record_title = normalize_title(record.get("title"))
            if not record_title:
                continue
            similarity = SequenceMatcher(
                None, candidate_title, record_title
            ).ratio()
            if similarity < 0.94:
                continue
            record_year = self._year(record.get("year"))
            if (
                candidate_year and record_year
                and abs(candidate_year - record_year) > 3
            ):
                continue
            record_author = self._first_author_surname(
                record.get("authors")
            )
            if (
                candidate_author and record_author
                and candidate_author != record_author
            ):
                continue
            score = similarity
            if candidate_year and record_year:
                score += 0.03
            if candidate_author and record_author:
                score += 0.05
            if score > best_score:
                best = record
                best_score = score
        return best

    @staticmethod
    def _title_candidate_is_usable(candidate: dict) -> bool:
        title = normalize_title(candidate.get("title"))
        words = title.split()
        return (
            3 <= len(words) <= 40
            and len(title) >= 15
            and (
                bool(candidate.get("authors") or candidate.get("year"))
                or (len(words) >= 5 and len(title) >= 25)
            )
        )

    @staticmethod
    def _first_author_surname(authors) -> str:
        if isinstance(authors, list):
            first = authors[0] if authors else ""
        else:
            first = re.split(r"(?i)\s+and\s+", str(authors or ""))[0]
        first = normalize_whitespace(first)
        if not first:
            return ""
        surname = first.split(",", 1)[0] if "," in first else first.split()[-1]
        return re.sub(r"[^a-z0-9]+", "", surname.lower())

    @staticmethod
    def _year(value) -> Optional[int]:
        match = re.search(r"\b(?:19|20)\d{2}\b", str(value or ""))
        return int(match.group(0)) if match else None

    def _save_cache(self) -> None:
        if not self.cache_path or not self._pending_cache:
            return
        self.cache_path.parent.mkdir(parents=True, exist_ok=True)
        if self.cache_path.suffix.lower() == ".jsonl":
            with open(self.cache_path, "a", encoding="utf-8") as handle:
                for record in self._pending_cache.values():
                    handle.write(json.dumps(record, ensure_ascii=False) + "\n")
            self.cache_path.chmod(0o644)
            self._pending_cache.clear()
            return
        temporary = self.cache_path.with_suffix(self.cache_path.suffix + ".tmp")
        with open(temporary, "w", encoding="utf-8") as handle:
            json.dump(self._cache_records, handle, ensure_ascii=False, indent=2)
        temporary.chmod(0o644)
        temporary.replace(self.cache_path)
        self._pending_cache.clear()
