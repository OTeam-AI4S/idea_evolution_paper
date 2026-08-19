"""
arXiv API downloader — best for CS, math, physics, and related fields.
Uses the official arXiv API (free, no key required).
"""

import logging
import re
import time
import xml.etree.ElementTree as ET
from typing import List, Optional

from .base import BaseDownloader, PaperInfo

logger = logging.getLogger(__name__)


class ArxivDownloader(BaseDownloader):
    """Download papers from arXiv.org via the official API."""

    name = "arxiv"

    # arXiv categories mapped to readable labels
    CATEGORIES = {
        "cs.AI": "Artificial Intelligence",
        "cs.CL": "Computation and Language (NLP)",
        "cs.CV": "Computer Vision",
        "cs.LG": "Machine Learning",
        "cs.IR": "Information Retrieval",
        "cs.NE": "Neural and Evolutionary Computing",
        "cs.RO": "Robotics",
        "cs.SD": "Sound",
        "stat.ML": "Machine Learning (Statistics)",
        "math.OC": "Optimization and Control",
        "physics": "Physics",
        "q-bio": "Quantitative Biology",
        "q-fin": "Quantitative Finance",
        "q-bio.BM": "Biomolecules",
        "q-bio.CB": "Cell Behavior",
        "q-bio.GN": "Genomics",
        "q-bio.MN": "Molecular Networks",
        "q-bio.NC": "Neurons and Cognition",
        "q-bio.OT": "Other Quantitative Biology",
        "q-bio.PE": "Populations and Evolution",
        "q-bio.QM": "Quantitative Methods",
        "q-bio.SC": "Subcellular Processes",
        "q-bio.TO": "Tissues and Organs",
    }
    API_URL = "https://export.arxiv.org/api/query"
    ATOM_NS = "http://www.w3.org/2005/Atom"
    ARXIV_NS = "http://arxiv.org/schemas/atom"

    def search(
        self,
        query: str,
        max_results: int = 50,
        categories: Optional[List[str]] = None,
        sort_by: str = "relevance",  # "relevance", "lastUpdatedDate", "submittedDate"
        year_from: Optional[int] = None,
        year_to: Optional[int] = None,
        **kwargs,
    ) -> List[PaperInfo]:
        """
        Search arXiv for papers.

        Args:
            query: Search query string (supports arXiv API syntax).
            max_results: Maximum number of results.
            categories: List of arXiv categories (e.g., ["cs.AI", "cs.CL"]).
            sort_by: Sort order.
            year_from: Filter papers from this year onwards.
            year_to: Filter papers up to this year.
        """
        cat_list = categories if categories else None
        search_query = query
        if cat_list:
            cat_filter = " OR ".join(f"cat:{c}" for c in cat_list)
            search_query = f"({query}) AND ({cat_filter})"

        logger.info(f"Searching arXiv: '{search_query}' (max {max_results})")

        papers = []
        for paper in self._query(
            search_query=search_query,
            max_results=max_results,
            sort_by=sort_by,
        ):
            pub_year = paper.year
            if year_from and (pub_year is None or pub_year < year_from):
                continue
            if year_to and (pub_year is None or pub_year > year_to):
                continue
            papers.append(paper)

        logger.info(f"arXiv returned {len(papers)} papers for query: '{query}'")
        return papers

    def search_by_venue(
        self,
        venue: str,
        year: Optional[int] = None,
        max_results: int = 50,
        **kwargs,
    ) -> List[PaperInfo]:
        """
        Search for papers from a specific venue (conference/journal).
        Many top CS papers are on arXiv with venue noted in comments.

        Args:
            venue: Venue name (e.g., "NeurIPS", "ICLR", "CVPR").
            year: Specific year.
            max_results: Max results.
        """
        if year:
            query = f'"{venue}" AND {year}'
        else:
            query = f'"{venue}"'
        return self.search(query, max_results=max_results, **kwargs)

    def download_by_id(self, arxiv_id: str) -> Optional[PaperInfo]:
        """
        Download a specific paper by arXiv ID.

        Args:
            arxiv_id: arXiv identifier (e.g., "2604.28158" or "2604.28158v2").
        """
        clean_id = re.sub(r"(?i)v\d+$", "", arxiv_id)
        papers = self._query(id_list=[clean_id], max_results=1)
        if not papers:
            logger.error(f"Paper not found on arXiv: {arxiv_id}")
            return None
        paper = papers[0]
        self.download_pdf(paper)
        return paper

    def _query(
        self,
        *,
        search_query: Optional[str] = None,
        id_list: Optional[List[str]] = None,
        max_results: int = 50,
        sort_by: str = "relevance",
    ) -> List[PaperInfo]:
        """Query Atom directly, avoiding the optional arxiv/lxml dependency."""
        sort_map = {
            "relevance": "relevance",
            "lastUpdatedDate": "lastUpdatedDate",
            "submittedDate": "submittedDate",
        }
        papers: List[PaperInfo] = []
        page_size = 100
        for start in range(0, max_results, page_size):
            requested = min(page_size, max_results - start)
            params = {
                "search_query": search_query or "",
                "id_list": ",".join(id_list or []),
                "sortBy": sort_map.get(sort_by, "relevance"),
                "sortOrder": "descending",
                "start": start,
                "max_results": requested,
            }
            response = None
            for attempt in range(4):
                try:
                    self._rate_limit()
                    response = self.session.get(
                        self.API_URL,
                        params=params,
                        timeout=(10, 60),
                    )
                    response.raise_for_status()
                    break
                except Exception:
                    if attempt == 3:
                        raise
                    time.sleep(min(2 ** attempt, 8))

            page = self._parse_feed(response.content)
            papers.extend(page)
            if len(page) < requested or id_list:
                break
        return papers[:max_results]

    def _parse_feed(self, payload: bytes) -> List[PaperInfo]:
        root = ET.fromstring(payload)
        atom = f"{{{self.ATOM_NS}}}"
        arxiv_ns = f"{{{self.ARXIV_NS}}}"
        papers = []
        for entry in root.findall(f"{atom}entry"):
            entry_id = self._text(entry.find(f"{atom}id"))
            title = self._text(entry.find(f"{atom}title"))
            abstract = self._text(entry.find(f"{atom}summary"))
            published = self._text(entry.find(f"{atom}published"))
            authors = [
                self._text(author.find(f"{atom}name"))
                for author in entry.findall(f"{atom}author")
            ]
            categories = [
                node.attrib.get("term", "")
                for node in entry.findall(f"{atom}category")
                if node.attrib.get("term")
            ]
            primary_node = entry.find(f"{arxiv_ns}primary_category")
            primary = (
                primary_node.attrib.get("term", "")
                if primary_node is not None else
                (categories[0] if categories else "")
            )
            doi = self._text(entry.find(f"{arxiv_ns}doi")) or None
            journal_ref = self._text(
                entry.find(f"{arxiv_ns}journal_ref")
            )
            comment = self._text(entry.find(f"{arxiv_ns}comment"))
            pdf_url = next(
                (
                    node.attrib.get("href")
                    for node in entry.findall(f"{atom}link")
                    if node.attrib.get("title") == "pdf"
                ),
                None,
            )
            short_id = entry_id.rstrip("/").rsplit("/", 1)[-1]
            year_match = re.match(r"(\d{4})", published)
            year = int(year_match.group(1)) if year_match else None
            cat_label = self.CATEGORIES.get(primary, primary)
            field = self._category_field(primary, cat_label)
            venue = journal_ref or self._venue_from_comment(comment)
            papers.append(PaperInfo(
                title=title,
                authors=[name for name in authors if name],
                year=year,
                doi=doi,
                arxiv_id=short_id,
                abstract=abstract,
                url=entry_id,
                pdf_url=pdf_url or f"https://arxiv.org/pdf/{short_id}",
                field=field,
                venue=venue or "arXiv preprint",
                source="arxiv",
                keywords=list(dict.fromkeys(
                    [primary] + categories if primary else categories
                )),
            ))
        return papers

    @staticmethod
    def _category_field(primary: str, label: str) -> str:
        if primary.startswith("cs."):
            return f"Computer Science / {label}"
        if primary.startswith("q-bio."):
            return f"Quantitative Biology / {label}"
        if primary.startswith("stat."):
            return f"Statistics / {label}"
        if primary.startswith("math."):
            return f"Mathematics / {label}"
        if primary.startswith("physics.") or primary == "physics":
            return f"Physics / {label}"
        if primary.startswith("q-fin."):
            return f"Quantitative Finance / {label}"
        return label or "Unknown"

    @staticmethod
    def _venue_from_comment(comment: str) -> str:
        """Use only explicit publication/acceptance statements."""
        if not comment:
            return ""
        patterns = (
            r"(?i)\baccepted\s+(?:at|to|for)\s+(.+)",
            r"(?i)\bto\s+appear\s+(?:at|in)\s+(.+)",
            r"(?i)\bpublished\s+(?:at|in)\s+(.+)",
        )
        for pattern in patterns:
            match = re.search(pattern, comment)
            if match:
                venue = match.group(1).strip()
                venue = re.split(
                    r"(?i)\s*[.;]\s*(?:\d+\s+pages?|code|camera-ready|"
                    r"supplement|arxiv)\b",
                    venue,
                    maxsplit=1,
                )[0]
                return venue.rstrip(" .;")
        return ""

    @staticmethod
    def _text(node: Optional[ET.Element]) -> str:
        if node is None:
            return ""
        return re.sub(r"\s+", " ", "".join(node.itertext())).strip()
