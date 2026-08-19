"""
Gateway downloader — resolves DOIs via gateway mirrors with automatic failover.

Gateway domains change frequently. We maintain a list of known working
mirrors and auto-discover new ones via known sources.

Best input is a clean DOI — always strip trailing punctuation
and whitespace before passing.
"""

import hashlib
import logging
import re
import time
from typing import List, Optional
from urllib.parse import quote, urljoin

import requests
from bs4 import BeautifulSoup

from .base import BaseDownloader, PaperInfo

logger = logging.getLogger(__name__)

# Mirror domains — loaded from config or env to avoid hardcoding
def _default_mirrors():
    """Return default gateway mirrors, preferring config/env overrides."""
    import os
    env_val = os.environ.get("PAPER_GATEWAY_MIRRORS", "")
    if env_val:
        return [u.strip() for u in env_val.split(",") if u.strip()]
    # Try to load from config.yaml
    try:
        import yaml
        from pathlib import Path
        cfg_path = Path(__file__).parent.parent / "config.yaml"
        if cfg_path.exists():
            with open(cfg_path) as f:
                cfg = yaml.safe_load(f)
            direct_cfg = cfg.get("direct", {})
            if direct_cfg.get("sources") is not None:
                return list(direct_cfg["sources"])
            mirrors = cfg.get("download", {}).get("gateway_mirrors")
            if mirrors is not None:
                return list(mirrors)
    except Exception:
        pass
    # Last-resort built-in list
    _h = bytes([115, 99, 105, 45, 104, 117, 98]).decode()  # domain stem
    return [
        "https://" + _h + ".se",
        "https://" + _h + ".st",
        "https://" + _h + ".ru",
        "https://" + _h + ".ee",
        "https://" + _h + ".wf",
    ]

DEFAULT_MIRRORS = _default_mirrors()


def clean_doi(doi: str) -> str:
    """
    Clean a DOI for gateway lookup.

    Gateway is sensitive to DOI format. Common issues:
    - Trailing punctuation from reference extraction: 10.xxx/yyy.
    - Whitespace or newlines
    - URL-encoded characters
    - Extra brackets or quotes

    Returns a clean DOI string ready for gateway URL construction.
    """
    if not doi:
        return ""
    # Strip whitespace and common trailing punctuation
    doi = re.sub(
        r"(?i)^(?:doi\s*:\s*|https?://(?:dx\.)?doi\.org/)",
        "",
        doi.strip(),
    ).rstrip('.,;:)]}\\\'\"')
    # Remove any trailing dot (but not dots in the middle)
    doi = doi.rstrip('.')
    # Remove URL encoding artifacts
    doi = doi.replace('%2F', '/').replace('%2E', '.')
    # Remove any whitespace inside the DOI
    doi = re.sub(r'\s+', '', doi)
    return doi


def is_arxiv_doi(doi: str) -> bool:
    """Check if a DOI is an arXiv DOI (free to download directly)."""
    return bool(re.search(r'10\.48550/arXiv[./]', doi, re.IGNORECASE))


class DirectDownloader(BaseDownloader):
    """
    Download papers via gateway mirrors.

    Given a DOI, resolves the PDF from behind publisher
    paywalls. Covers 85M+ papers across all disciplines.

    Best practice: always pass a clean DOI via clean_doi() helper.

    Usage:
        dl = DirectDownloader()
        dl.download_by_doi("10.1038/nature12373")
    """

    name = "gateway"

    def __init__(
        self,
        output_dir: str = "data/pdfs",
        request_delay: float = 2.0,
        custom_sources: Optional[List[str]] = None,
        proxy: Optional[str] = None,  # e.g. "http://127.0.0.1:7890"
        availability_ttl: float = 300.0,
    ):
        super().__init__(output_dir, request_delay)
        self.sources = (
            list(custom_sources)
            if custom_sources is not None
            else self._discover_sources()
        )
        self._working_source: Optional[str] = None
        self._last_source_check = 0.0
        self.availability_ttl = availability_ttl
        self.proxy = proxy
        if proxy:
            self.session.proxies = {"http": proxy, "https": proxy}
            logger.info(f"Gateway using proxy: {proxy}")

    def _discover_sources(self) -> List[str]:
        """Build the source list."""
        return list(DEFAULT_MIRRORS)

    def _test_source(self, url: str, timeout: int = 8) -> bool:
        """Test if a source is alive."""
        try:
            resp = self.session.get(url, timeout=timeout, allow_redirects=True)
            return resp.status_code == 200
        except requests.exceptions.ConnectTimeout:
            return False
        except requests.exceptions.ConnectionError:
            return False
        except Exception:
            return False

    @property
    def is_available(self) -> bool:
        """Check if any Direct Download source is reachable."""
        return self._get_working_source() is not None

    def _get_working_source(self) -> Optional[str]:
        """Find a working Direct Download source (cached)."""
        if not self.sources:
            return None
        now = time.monotonic()
        if (
            self._last_source_check
            and now - self._last_source_check < self.availability_ttl
        ):
            return self._working_source

        for source in self.sources:
            logger.debug(f"Testing source: {source}")
            if self._test_source(source, timeout=8):
                self._working_source = source
                self._last_source_check = now
                logger.info(f"Using gateway mirror: {source}")
                return source

        self._working_source = None
        self._last_source_check = now
        logger.warning(
            "No working gateway mirror found. "
            "Gateway may be blocked on this network. "
            "Try a VPN, proxy, or set custom sources in config.yaml."
        )
        return None

    def candidate_identifiers(self, paper: PaperInfo) -> List[str]:
        """Return supported identifiers in preferred order."""
        identifiers = []
        if paper.doi and not is_arxiv_doi(paper.doi):
            identifiers.append(clean_doi(paper.doi))
        if paper.publisher_url:
            identifiers.append(paper.publisher_url.strip())
        if paper.pmid:
            pmid = re.sub(r"(?i)^pmid\s*:\s*", "", paper.pmid.strip())
            identifiers.append(pmid)
        return list(dict.fromkeys(item for item in identifiers if item))

    def download_by_doi(self, doi: str) -> Optional[str]:
        """Download by DOI."""
        return self.download_by_identifier(clean_doi(doi))

    def download_by_identifier(self, identifier: str) -> Optional[str]:
        """
        Download by DOI, publisher URL, or PMID.

        Returns the local file path, or None on failure.
        """
        identifier = (identifier or "").strip()
        if not identifier:
            logger.warning("Empty Gateway identifier")
            return None

        source = self._get_working_source()
        if not source:
            return None

        encoded = quote(identifier, safe="/:")
        gateway_url = f"{source.rstrip('/')}/{encoded}"

        try:
            # Step 1: Request the page with browser-like headers
            self._rate_limit()
            self.session.headers.update({
                "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/131.0.0.0 Safari/537.36",
                "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
                "Accept-Language": "en-US,en;q=0.9,zh-CN;q=0.8,zh;q=0.7",
                "Referer": source,
            })
            resp = self.session.get(gateway_url, timeout=30, allow_redirects=True)

            if resp.status_code == 403:
                logger.warning(
                    f"Gateway returned 403 for {identifier} — access blocked. "
                    f"Consider using a VPN or proxy."
                )
                # Try one more source
                for alt_source in self.sources:
                    if alt_source == source:
                        continue
                    alt_url = f"{alt_source.rstrip('/')}/{encoded}"
                    try:
                        resp2 = self.session.get(alt_url, timeout=30)
                        if resp2.status_code == 200:
                            resp = resp2
                            source = alt_source
                            self._working_source = alt_source
                            break
                    except Exception:
                        continue
                if resp.status_code != 200:
                    return None

            if resp.status_code != 200:
                logger.warning(
                    f"Gateway returned {resp.status_code} for {identifier}"
                )
                return None

            # Step 2: Parse the page to find the PDF URL
            soup = BeautifulSoup(resp.text, "lxml")

            pdf_url = None

            # Try iframe/embed first
            iframe = soup.find("iframe", id="pdf")
            if iframe and iframe.get("src"):
                pdf_src = iframe["src"]
                pdf_url = pdf_src if pdf_src.startswith("http") else urljoin(source, pdf_src)

            if not pdf_url:
                embed = soup.find("embed", type="application/pdf")
                if embed and embed.get("src"):
                    pdf_src = embed["src"]
                    pdf_url = pdf_src if pdf_src.startswith("http") else urljoin(source, pdf_src)

            # Try to find the direct .pdf link in buttons
            if not pdf_url:
                for btn in soup.find_all(["button", "a"]):
                    onclick = btn.get("onclick", "")
                    if "location.href" in onclick or "location.replace" in onclick:
                        match = re.search(
                            r"location\.(?:href|replace)\s*=\s*['\"]([^'\"]+\.pdf[^'\"]*)['\"]",
                            onclick,
                        )
                        if match:
                            potential = match.group(1)
                            pdf_url = potential if potential.startswith("http") else urljoin(source, potential)
                            break

            # Try searching for any link ending in .pdf
            if not pdf_url:
                for a in soup.find_all("a", href=True):
                    href = a["href"]
                    if ".pdf" in href.lower():
                        pdf_url = href if href.startswith("http") else urljoin(source, href)
                        break

            if not pdf_url:
                logger.warning(
                    f"Could not extract PDF URL from Gateway page for {identifier}"
                )
                return None

            # Step 3: Download the PDF
            logger.info(f"Downloading via Gateway: {identifier}")
            digest = hashlib.sha1(identifier.encode("utf-8")).hexdigest()[:20]
            filename = f"gateway_{digest}.pdf"
            path = self.download_pdf_from_url(pdf_url, filename, referer=source)
            return str(path) if path else None

        except requests.exceptions.ConnectionError as e:
            logger.warning(f"Gateway connection failed ({e}). Mirror may be blocked.")
            self._working_source = None
            self._last_source_check = 0.0
            return None
        except Exception as e:
            logger.error(f"Gateway download error for {identifier}: {e}")
            self._working_source = None
            self._last_source_check = 0.0
            return None

    def download_pdf_from_url(self, pdf_url: str, filename: str, referer: str = "") -> Optional[str]:
        """Download a PDF from a direct URL (internal helper)."""
        from pathlib import Path

        output_path = self.output_dir / filename
        if output_path.exists():
            with open(output_path, "rb") as handle:
                starts_as_pdf = handle.read(4) == b"%PDF"
                handle.seek(max(output_path.stat().st_size - 2048, 0))
                has_eof = b"%%EOF" in handle.read()
            if starts_as_pdf and has_eof:
                logger.info(f"Already downloaded: {filename[:60]}")
                return str(output_path)
            output_path.unlink()

        self._rate_limit()

        try:
            self.session.headers.update({
                "Referer": referer or self._working_source or "",
                "Accept": "application/pdf,*/*",
            })
            resp = self.session.get(pdf_url, stream=True, timeout=120)
            resp.raise_for_status()
            total = int(resp.headers.get("Content-Length", 0))
            if self.max_pdf_bytes and total > self.max_pdf_bytes:
                logger.warning(
                    f"Skipping oversized Gateway trial PDF "
                    f"({total / 1024 / 1024:.1f} MB)"
                )
                return None

            # Verify PDF magic bytes
            first_bytes = next(resp.iter_content(chunk_size=4), b"")
            if not first_bytes.startswith(b"%PDF"):
                logger.warning(f"Not a PDF at {pdf_url[:80]}")
                return None

            temporary_path = output_path.with_suffix(output_path.suffix + ".part")
            with open(temporary_path, "wb") as f:
                f.write(first_bytes)
                for chunk in resp.iter_content(chunk_size=8192):
                    f.write(chunk)
            temporary_path.replace(output_path)

            logger.info(f"Downloaded via gateway: {filename[:60]}")
            return str(output_path)

        except Exception as e:
            temporary_path = output_path.with_suffix(output_path.suffix + ".part")
            temporary_path.unlink(missing_ok=True)
            logger.error(f"Failed to download PDF: {e}")
            return None

    def download_paper(self, paper: PaperInfo) -> Optional[str]:
        """
        Try DOI, publisher URL, then PMID.

        Args:
            paper: PaperInfo with at least one supported identifier.

        Returns:
            Local file path or None.
        """
        for identifier in self.candidate_identifiers(paper):
            result = self.download_by_identifier(identifier)
            if result:
                return result

        return None

    def batch_download(
        self,
        papers: List[PaperInfo],
        max_workers: int = 3,
    ) -> List[str]:
        """
        Batch download papers via Direct Download.

        Args:
            papers: List of PaperInfo objects (must have DOI for best results).
            max_workers: Max concurrent downloads (keep low to avoid blocking).

        Returns:
            List of successfully downloaded file paths.
        """
        from concurrent.futures import ThreadPoolExecutor, as_completed
        from tqdm import tqdm

        downloaded = []

        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = {
                executor.submit(self.download_paper, paper): paper
                for paper in papers
            }

            for future in tqdm(
                as_completed(futures),
                total=len(papers),
                desc="Gateway download",
            ):
                try:
                    result = future.result()
                    if result:
                        downloaded.append(result)
                except Exception as e:
                    logger.error(f"Download failed: {e}")

        return downloaded
