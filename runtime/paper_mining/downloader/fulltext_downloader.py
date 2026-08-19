"""Download structured full text before falling back to PDFs."""

from __future__ import annotations

import hashlib
import logging
import os
import re
import time
from pathlib import Path
from typing import Optional

from .base import BaseDownloader, PaperInfo


logger = logging.getLogger(__name__)


class FullTextDownloader(BaseDownloader):
    """Fetch arXiv source bundles and PMC JATS XML using public endpoints."""

    name = "structured_fulltext"
    ARXIV_SOURCE_URL = "https://export.arxiv.org/e-print/{identifier}"
    PMC_EFETCH_URL = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/efetch.fcgi"
    PMC_IDCONV_URL = "https://www.ncbi.nlm.nih.gov/pmc/utils/idconv/v1.0/"

    def __init__(
        self,
        output_dir: str = "data/fulltext",
        request_delay: float = 1.0,
        max_source_bytes: Optional[int] = 100 * 1024 * 1024,
        max_download_seconds: Optional[float] = 180.0,
    ):
        super().__init__(output_dir=output_dir, request_delay=request_delay)
        self.max_source_bytes = max_source_bytes
        self.max_download_seconds = max_download_seconds
        self._pmcid_cache = {}

    def download_arxiv_source(self, paper: PaperInfo) -> Optional[Path]:
        if not paper.arxiv_id:
            return None
        identifier = re.sub(
            r"^(?:arxiv:)?", "", paper.arxiv_id, flags=re.IGNORECASE
        ).strip()
        candidates = [identifier]
        unversioned = re.sub(r"v\d+$", "", identifier, flags=re.IGNORECASE)
        if unversioned != identifier:
            candidates.append(unversioned)

        digest = hashlib.sha1(identifier.encode("utf-8")).hexdigest()[:16]
        output_path = self.output_dir / f"arxiv_{digest}.src"
        for candidate in candidates:
            result = self._download(
                self.ARXIV_SOURCE_URL.format(identifier=candidate),
                output_path,
                kind="arXiv source",
            )
            if result:
                return result
        return None

    def download_pmc_xml(self, paper: PaperInfo) -> Optional[Path]:
        pmcid = self._normalize_pmcid(paper.pmcid)
        if not pmcid and paper.pmid:
            pmcid = self.resolve_pmcid(paper.pmid)
        if not pmcid:
            return None

        output_path = self.output_dir / f"{pmcid}.xml"
        return self._download(
            self.PMC_EFETCH_URL,
            output_path,
            kind="PMC XML",
            params={
                "db": "pmc",
                "id": pmcid,
                "retmode": "xml",
            },
            expected_prefix=b"<?xml",
        )

    def resolve_pmcid(self, pmid: str) -> Optional[str]:
        normalized = re.sub(r"^pmid:\s*", "", str(pmid), flags=re.IGNORECASE)
        normalized = normalized.strip()
        if not normalized:
            return None
        if normalized in self._pmcid_cache:
            return self._pmcid_cache[normalized]

        self._rate_limit()
        try:
            response = self.session.get(
                self.PMC_IDCONV_URL,
                params={
                    "ids": normalized,
                    "format": "json",
                    "tool": "paper_mining",
                    "email": os.environ.get(
                        "PAPER_MINING_CONTACT_EMAIL", "research@example.com"
                    ),
                },
                timeout=30,
            )
            response.raise_for_status()
            records = response.json().get("records", [])
            pmcid = self._normalize_pmcid(
                records[0].get("pmcid") if records else None
            )
        except Exception as exc:
            logger.debug("PMID to PMCID conversion failed for %s: %s", normalized, exc)
            pmcid = None
        self._pmcid_cache[normalized] = pmcid
        return pmcid

    @staticmethod
    def _normalize_pmcid(value: Optional[str]) -> Optional[str]:
        if not value:
            return None
        normalized = re.sub(r"^pmc", "", str(value), flags=re.IGNORECASE).strip()
        return f"PMC{normalized}" if normalized.isdigit() else None

    def _download(
        self,
        url: str,
        output_path: Path,
        *,
        kind: str,
        params: Optional[dict] = None,
        expected_prefix: Optional[bytes] = None,
    ) -> Optional[Path]:
        if output_path.exists() and output_path.stat().st_size > 100:
            return output_path

        temporary_path = output_path.with_suffix(output_path.suffix + ".part")
        self._rate_limit()
        try:
            started = time.monotonic()
            read_timeout = min(
                120.0,
                self.max_download_seconds or 120.0,
            )
            response = self.session.get(
                url,
                params=params,
                stream=True,
                timeout=(15, read_timeout),
            )
            response.raise_for_status()
            total = int(response.headers.get("Content-Length", 0))
            if self.max_source_bytes and total > self.max_source_bytes:
                logger.warning("%s is too large: %.1f MB", kind, total / 1024 / 1024)
                return None

            written = 0
            first = b""
            with open(temporary_path, "wb") as handle:
                for chunk in response.iter_content(chunk_size=64 * 1024):
                    if (
                        self.max_download_seconds
                        and time.monotonic() - started > self.max_download_seconds
                    ):
                        raise TimeoutError(
                            f"{kind} exceeded {self.max_download_seconds:g}s"
                        )
                    if not chunk:
                        continue
                    if not first:
                        first = chunk[:512].lstrip()
                    written += len(chunk)
                    if self.max_source_bytes and written > self.max_source_bytes:
                        raise ValueError(f"{kind} exceeded the size limit")
                    handle.write(chunk)

            if written < 100:
                raise ValueError(f"{kind} response is empty")
            if first.lower().startswith((b"<!doctype html", b"<html")):
                raise ValueError(f"{kind} endpoint returned HTML")
            if expected_prefix and not first.startswith(expected_prefix):
                if b"<article" not in first and b"<pmc-articleset" not in first:
                    raise ValueError(f"{kind} response has an unexpected format")

            temporary_path.replace(output_path)
            logger.info("Downloaded %s: %s", kind, output_path.name)
            return output_path
        except Exception as exc:
            temporary_path.unlink(missing_ok=True)
            logger.debug("%s download failed: %s", kind, exc)
            return None
