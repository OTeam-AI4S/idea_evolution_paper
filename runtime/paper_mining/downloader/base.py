"""
Base classes for paper downloaders.
"""

import os
import time
import logging
from abc import ABC
from dataclasses import dataclass, field as dataclass_field
from typing import List, Optional, Dict, Any
from pathlib import Path

import requests
from tqdm import tqdm

from ..utils import sanitize_filename, get_paper_id, ensure_dir

logger = logging.getLogger(__name__)


@dataclass
class PaperInfo:
    """Structured information about a paper."""
    title: str
    authors: List[str] = dataclass_field(default_factory=list)
    year: Optional[int] = None
    doi: Optional[str] = None
    pmid: Optional[str] = None
    pmcid: Optional[str] = None
    arxiv_id: Optional[str] = None
    field: Optional[str] = None
    venue: Optional[str] = None  # verified journal/conference or preprint
    abstract: Optional[str] = None
    url: Optional[str] = None
    publisher_url: Optional[str] = None
    pdf_url: Optional[str] = None
    source: Optional[str] = None  # which downloader found this
    citation_count: Optional[int] = None
    bibtex: Optional[str] = None
    keywords: List[str] = dataclass_field(default_factory=list)

    @property
    def id(self) -> str:
        return get_paper_id(self.doi or self.title)

    @property
    def safe_title(self) -> str:
        return sanitize_filename(self.title)


class BaseDownloader(ABC):
    """Abstract base class for paper downloaders."""

    name: str = "base"

    def __init__(
        self,
        output_dir: str = "data/pdfs",
        request_delay: float = 1.0,
        max_pdf_bytes: Optional[int] = None,
    ):
        self.output_dir = ensure_dir(output_dir)
        self.request_delay = request_delay
        self.max_pdf_bytes = max_pdf_bytes
        self.session = requests.Session()
        contact = os.environ.get("PAPER_MINING_CONTACT_EMAIL", "research@example.com")
        self.session.headers.update({
            "User-Agent": f"PaperMining/0.1 (mailto:{contact})"
        })

    def _rate_limit(self):
        """Sleep to respect rate limits."""
        time.sleep(self.request_delay)

    def search(self, query: str, max_results: int = 50, **kwargs) -> List[PaperInfo]:
        """Search for papers matching a query. Override in subclasses."""
        raise NotImplementedError(
            f"{self.name} downloader does not support search. "
            f"Use it for PDF download only."
        )

    def download_pdf(self, paper: PaperInfo, filename: Optional[str] = None) -> Optional[Path]:
        """Download a single paper PDF. Returns path or None on failure."""
        if not paper.pdf_url:
            logger.warning(f"No PDF URL for: {paper.title[:80]}")
            return None

        if filename is None:
            filename = f"{paper.id}_{paper.safe_title}.pdf"

        output_path = self.output_dir / filename

        if output_path.exists():
            with open(output_path, "rb") as handle:
                starts_as_pdf = handle.read(4) == b"%PDF"
                handle.seek(max(output_path.stat().st_size - 2048, 0))
                has_eof = b"%%EOF" in handle.read()
            if starts_as_pdf and has_eof:
                logger.info(f"Already downloaded: {output_path.name[:60]}")
                return output_path
            output_path.unlink()

        self._rate_limit()

        try:
            response = self.session.get(paper.pdf_url, stream=True, timeout=60)
            response.raise_for_status()

            # Check if it's actually a PDF
            content_type = response.headers.get("Content-Type", "")
            if "pdf" not in content_type and "octet-stream" not in content_type:
                # Check first bytes for PDF magic number
                if len(response.content) < 100 or not response.content.startswith(b"%PDF"):
                    logger.warning(f"Not a PDF: {paper.pdf_url}")
                    return None

            total = int(response.headers.get("Content-Length", 0))
            if self.max_pdf_bytes and total > self.max_pdf_bytes:
                logger.warning(
                    f"Skipping oversized trial PDF ({total / 1024 / 1024:.1f} MB): "
                    f"{paper.title[:60]}"
                )
                return None

            temporary_path = output_path.with_suffix(output_path.suffix + ".part")
            with open(temporary_path, "wb") as f:
                if total:
                    with tqdm(
                        total=total, unit="B", unit_scale=True,
                        desc=paper.safe_title[:50]
                    ) as pbar:
                        for chunk in response.iter_content(chunk_size=8192):
                            f.write(chunk)
                            pbar.update(len(chunk))
                else:
                    f.write(response.content)
            temporary_path.replace(output_path)

            logger.info(f"Downloaded: {output_path.name[:60]}")
            return output_path

        except Exception as e:
            temporary_path = output_path.with_suffix(output_path.suffix + ".part")
            temporary_path.unlink(missing_ok=True)
            logger.error(f"Failed to download {paper.title[:80]}: {e}")
            return None

    def download_batch(
        self,
        papers: List[PaperInfo],
        max_concurrent: int = 5,
    ) -> List[Path]:
        """
        Download multiple papers sequentially with progress bar.

        Returns list of successfully downloaded paths.
        """
        downloaded = []
        for paper in tqdm(papers, desc="Downloading papers"):
            path = self.download_pdf(paper)
            if path:
                downloaded.append(path)
        return downloaded

    def search_and_download(
        self,
        query: str,
        max_results: int = 50,
        **kwargs,
    ) -> List[Path]:
        """Convenience: search and download in one call."""
        papers = self.search(query, max_results=max_results, **kwargs)
        logger.info(f"Found {len(papers)} papers for query: {query}")
        return self.download_batch(papers)
