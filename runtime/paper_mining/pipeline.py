"""
End-to-end pipeline: search → download → parse → structure.
"""

import json
import logging
import os
from pathlib import Path
from typing import List, Optional, Dict, Any
from dataclasses import dataclass, field
from datetime import datetime

from .downloader.base import PaperInfo
from .parser import PDFParser, SectionSplitter
from .parser.pdf_parser import ParsedDocument
from .utils import load_config, ensure_dir

logger = logging.getLogger(__name__)


@dataclass
class PipelineResult:
    """Result of processing one paper through the full pipeline."""
    paper: PaperInfo
    pdf_path: Optional[str] = None
    parsed: Optional[ParsedDocument] = None
    sections: Dict[str, str] = field(default_factory=dict)
    status: str = "pending"  # pending, downloaded, parsed, done, error
    error: Optional[str] = None


class PaperMiningPipeline:
    """
    End-to-end pipeline for paper mining.

    Usage:
        pipeline = PaperMiningPipeline("config.yaml")
        results = pipeline.run(
            queries=["large language model reasoning"],
            max_papers_per_query=10,
        )
    """

    def __init__(
        self,
        config_path: str = "config.yaml",
        enabled_downloaders: Optional[set] = None,
    ):
        self.config = load_config(config_path)

        download_cfg = self.config.get("download", {})
        output_dir = os.environ.get(
            "PAPER_MINING_ARTIFACT_DIR",
            download_cfg.get("output_dir", "data/pdfs"),
        )
        delay = download_cfg.get("request_delay", 1.0)

        parsing_cfg = self.config.get("parsing", {})
        parsed_dir = parsing_cfg.get("output_dir", "data/parsed")
        self.parsed_dir = ensure_dir(parsed_dir)
        self.output_format = parsing_cfg.get("output_format", "json")

        direct_cfg = self.config.get("direct", {})
        enabled = enabled_downloaders or {
            "structured", "direct", "arxiv",
            "semantic_scholar", "scholarly",
        }
        self.downloaders = {}
        if "structured" in enabled:
            from .downloader.fulltext_downloader import FullTextDownloader
            self.downloaders["structured"] = FullTextDownloader(
                output_dir=output_dir,
                request_delay=delay,
                max_download_seconds=float(os.environ.get(
                    "PAPER_MINING_SOURCE_DOWNLOAD_TIMEOUT", "180"
                )),
            )
        if "direct" in enabled:
            from .downloader.direct_downloader import DirectDownloader
            self.downloaders["direct"] = DirectDownloader(
                output_dir=output_dir,
                request_delay=delay,
                custom_sources=direct_cfg.get("sources"),
                proxy=direct_cfg.get("proxy"),
            )
        if "arxiv" in enabled:
            from .downloader.arxiv_downloader import ArxivDownloader
            self.downloaders["arxiv"] = ArxivDownloader(
                output_dir=output_dir,
                request_delay=delay,
            )
        if "semantic_scholar" in enabled:
            from .downloader.semantic_scholar import SemanticScholarDownloader
            self.downloaders["semantic_scholar"] = SemanticScholarDownloader(
                output_dir=output_dir,
                request_delay=delay,
            )
        if "scholarly" in enabled:
            from .downloader.scholarly_downloader import ScholarlyDownloader
            self.downloaders["scholarly"] = ScholarlyDownloader(
                output_dir=output_dir,
                request_delay=delay,
            )

        # Sources for search/discovery
        self.discovery_sources = download_cfg.get(
            "discovery_sources", ["arxiv", "semantic_scholar"]
        )

        # PDF download priority: known PDF URL first, then Gateway candidate
        self.download_priority = download_cfg.get(
            "download_priority", ["arxiv", "direct", "semantic_scholar", "scholarly"]
        )

        # Initialize parser
        engine = parsing_cfg.get("engine", "hybrid")
        self.parser = PDFParser(engine=engine)
        self.splitter = SectionSplitter()

        self.results: List[PipelineResult] = []

    def run(
        self,
        queries: List[str],
        max_papers_per_query: int = 50,
        years: Optional[tuple] = None,
        venues: Optional[List[str]] = None,
        categories: Optional[List[str]] = None,
        download: bool = True,
        parse: bool = True,
        resume: bool = True,
    ) -> List[PipelineResult]:
        """
        Run the full pipeline.

        Args:
            queries: List of search queries.
            max_papers_per_query: Max papers to fetch per query.
            years: Optional (year_from, year_to) tuple.
            venues: Optional venue filter (journal/conference names).
            categories: arXiv categories (e.g., ["cs.AI", "cs.CL"]).
            download: Whether to download PDFs.
            parse: Whether to parse PDFs.
            resume: Skip already-downloaded papers.

        Returns:
            List of PipelineResult objects.
        """
        # Phase 1: Search and collect papers
        all_papers = []
        for query in queries:
            logger.info(f"Processing query: '{query}'")
            papers = self._search_all_sources(
                query=query,
                max_results=max_papers_per_query,
                years=years,
                venues=venues,
                categories=categories,
            )
            all_papers.extend(papers)
            logger.info(f"  -> Found {len(papers)} papers")

        # Deduplicate by DOI or title
        all_papers = self._deduplicate(all_papers)
        logger.info(f"Total unique papers: {len(all_papers)}")

        # Create PipelineResult for each
        self.results = [PipelineResult(paper=p) for p in all_papers]

        # Phase 2: Download PDFs
        if download:
            for result in self.results:
                try:
                    pdf_path = self._download_paper(result.paper, resume=resume)
                    if pdf_path:
                        result.pdf_path = str(pdf_path)
                        result.status = "downloaded"
                    else:
                        result.status = "error"
                        result.error = "Download failed"
                except Exception as e:
                    result.status = "error"
                    result.error = str(e)
                    logger.error(f"Error downloading {result.paper.title[:80]}: {e}")

        # Phase 3: Parse PDFs
        if parse:
            for result in self.results:
                if result.pdf_path:
                    try:
                        parsed = self.parser.parse(result.pdf_path)

                        # Extract sections
                        sections = self.splitter.split(parsed)

                        # Try to extract abstract if not found in sections
                        if "abstract" not in sections:
                            abstract = self.splitter.extract_abstract(parsed.full_text)
                            if abstract:
                                sections["abstract"] = abstract

                        parsed.sections = sections
                        result.parsed = parsed
                        result.sections = sections
                        result.status = "done"

                        # Save parsed output
                        self._save_result(result)

                    except Exception as e:
                        result.status = "error"
                        result.error = f"Parse error: {e}"
                        logger.error(f"Error parsing {result.paper.title[:80]}: {e}")

        # Print summary
        self._print_summary()

        return self.results

    def _search_all_sources(
        self,
        query: str,
        max_results: int,
        years: Optional[tuple] = None,
        venues: Optional[List[str]] = None,
        categories: Optional[List[str]] = None,
    ) -> List[PaperInfo]:
        """Try all configured discovery sources in order, collecting results."""
        all_papers = []
        year_from, year_to = years if years else (None, None)

        for source_name in self.discovery_sources:
            downloader = self.downloaders.get(source_name)
            if downloader is None:
                logger.warning(f"Unknown source: {source_name}")
                continue

            try:
                remaining = max_results - len(all_papers)
                if remaining <= 0:
                    break

                papers = downloader.search(
                    query=query,
                    max_results=remaining,
                    year_from=year_from,
                    year_to=year_to,
                    categories=categories if source_name == "arxiv" else None,
                    venue_filter=venues if source_name == "semantic_scholar" else None,
                )
                all_papers.extend(papers)
                logger.info(f"  {source_name}: {len(papers)} results")

            except Exception as e:
                logger.warning(f"Source {source_name} failed: {e}")
                continue

        return all_papers

    def _download_paper(
        self, paper: PaperInfo, resume: bool = True
    ) -> Optional[Path]:
        """
        Try downloading from configured sources.
        Priority is configured; known PDF URLs should precede Gateway candidates.

        Direct Download is automatically skipped if no mirror is reachable.
        """
        direct = self.downloaders.get("direct")

        for source_name in self.download_priority:
            # Skip Direct Download if blocked/unreachable
            if source_name == "direct":
                if (
                    direct is None
                    or not direct.candidate_identifiers(paper)
                    or not direct.is_available
                ):
                    continue

            downloader = self.downloaders.get(source_name)
            if downloader is None:
                continue

            try:
                # Gateway uses DOI, publisher URL, or PMID.
                if source_name == "direct" and direct:
                    path_str = direct.download_paper(paper)
                    if path_str:
                        return Path(path_str)
                else:
                    path = downloader.download_pdf(paper)
                    if path:
                        return path
            except Exception as e:
                logger.debug(f"Source {source_name} failed: {e}")
                continue

        return None

    def _deduplicate(self, papers: List[PaperInfo]) -> List[PaperInfo]:
        """Remove duplicate papers (by DOI first, then by title similarity)."""
        seen_dois = set()
        seen_titles = set()
        unique = []

        for paper in papers:
            # Check DOI
            if paper.doi:
                key = paper.doi.lower()
                if key in seen_dois:
                    continue
                seen_dois.add(key)

            # Check title (normalized)
            title_key = paper.title.lower().strip().rstrip(".")
            if title_key in seen_titles:
                continue
            seen_titles.add(title_key)

            unique.append(paper)

        return unique

    def _save_result(self, result: PipelineResult):
        """Save pipeline result to JSON."""
        if not result.parsed:
            return

        output = {
            "paper": {
                "title": result.paper.title,
                "authors": result.paper.authors,
                "year": result.paper.year,
                "doi": result.paper.doi,
                "arxiv_id": result.paper.arxiv_id,
                "field": result.paper.field,
                "venue": result.paper.venue,
                "abstract": result.paper.abstract,
                "citation_count": result.paper.citation_count,
            },
            "pdf_path": result.pdf_path,
            "parsed_title": result.parsed.title,
            "sections": result.sections,
            "metadata": result.parsed.metadata,
            "parse_errors": result.parsed.parse_errors,
            "processed_at": datetime.now().isoformat(),
        }

        filename = f"{result.paper.id}.json"
        output_path = self.parsed_dir / filename
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(output, f, ensure_ascii=False, indent=2)
        output_path.chmod(0o644)

    def _print_summary(self):
        """Print pipeline execution summary."""
        total = len(self.results)
        downloaded = sum(1 for r in self.results if r.pdf_path)
        parsed = sum(1 for r in self.results if r.parsed)
        errors = sum(1 for r in self.results if r.status == "error")

        print("\n" + "=" * 60)
        print(f"  Pipeline Summary")
        print("=" * 60)
        print(f"  Papers found:      {total}")
        print(f"  PDFs downloaded:   {downloaded}")
        print(f"  Successfully parsed: {parsed}")
        print(f"  Errors:            {errors}")
        print(f"  Output directory:  {self.parsed_dir}")
        print("=" * 60)

    def get_section_texts(self, section_name: str) -> List[str]:
        """
        Get text for a specific section across all successfully parsed papers.

        Args:
            section_name: e.g., "introduction", "method", "related_work".

        Returns:
            List of section texts from all papers.
        """
        texts = []
        for result in self.results:
            if result.sections and section_name in result.sections:
                texts.append(result.sections[section_name])
        return texts

    def export_corpus(
        self,
        output_path: str = "data/corpus.jsonl",
        sections: Optional[List[str]] = None,
    ):
        """
        Export all parsed papers as a JSONL corpus file,
        suitable for LLM training or analysis.

        Args:
            output_path: Path to output JSONL file.
            sections: Sections to include (default: all).
        """
        with open(output_path, "w", encoding="utf-8") as f:
            for result in self.results:
                if not result.parsed:
                    continue

                entry = {
                    "title": result.paper.title,
                    "authors": result.paper.authors,
                    "year": result.paper.year,
                    "field": result.paper.field,
                    "venue": result.paper.venue,
                    "sections": {},
                }

                for name, text in result.sections.items():
                    if sections is None or name in sections:
                        entry["sections"][name] = text

                f.write(json.dumps(entry, ensure_ascii=False) + "\n")
        Path(output_path).chmod(0o644)

        logger.info(f"Exported corpus to {output_path}")
