"""Compact local full-text pipeline for PDF, arXiv source, and PMC XML."""

from __future__ import annotations

import json
import logging
import re
import tempfile
from pathlib import Path
from typing import Iterable, List, Optional

from .downloader.arxiv_downloader import ArxivDownloader
from .downloader.base import PaperInfo
from .parser.local_references import LocalReferenceExtractor
from .parser.reference_abstracts import (
    ReferenceAbstractResolver,
    abstract_is_complete,
    normalize_whitespace,
)
from .parser.pdf_parser import PDFParser, ParsedDocument
from .parser.section_splitter import SectionSplitter
from .parser.structured_parser import (
    StructuredDocument,
    StructuredFullTextParser,
)


logger = logging.getLogger(__name__)


def _clean_text(text: Optional[str]) -> str:
    """Backward-compatible alias for whitespace-only normalization."""
    return normalize_whitespace(text)


def _looks_like_table_line(line: str) -> bool:
    words = re.findall(r"[A-Za-z]+", line)
    numbers = re.findall(r"(?<![A-Za-z])[-+]?\d+(?:\.\d+)?%?", line)
    metric_tokens = re.findall(
        r"(?i)\b(?:AP|AP50|AP75|FLOPs?|top-?[15]|mAP|AUC|F1|params?)\b",
        line,
    )
    return (
        len(numbers) >= 8
        and len(numbers) >= max(4, len(words) // 3)
    ) or len(metric_tokens) >= 5


def _clean_method(text: Optional[str]) -> str:
    """Remove common PDF front matter and table rows from a method section."""
    text = text or ""
    prefix = text[:800]
    email = re.search(
        r"(?i)(?:\{[^{}]{0,300}\}\s*)?@[A-Za-z0-9.-]+\.[A-Za-z]{2,}",
        prefix,
    )
    if email:
        text = text[email.end():]

    kept = []
    skip_caption_continuation = False
    for raw_line in text.splitlines():
        line = _clean_text(raw_line)
        if not line:
            continue
        if re.match(r"(?i)^(?:table|tab\.)\s+[A-Z]?\d+\b", line):
            skip_caption_continuation = True
            continue
        if _looks_like_table_line(line):
            continue
        if skip_caption_continuation:
            if len(line) < 180 and not re.search(r"[.!?]\s*$", line):
                continue
            skip_caption_continuation = False
        kept.append(line)

    cleaned = _clean_text(" ".join(kept))
    cleaned = re.sub(r"(?:\s*\[equation\]\s*){2,}", " [equation] ", cleaned)
    return cleaned


def _method_quality_errors(method: str) -> List[str]:
    errors = []
    if len(method) > 40000:
        errors.append("method is implausibly long")
    if "\x00" in method:
        errors.append("method contains binary/OCR artifacts")
    if re.search(r"(?i)\btable\s+[A-Z]?\d+\b", method):
        errors.append("method contains table/result content")
    if re.search(r"(?i)\bfig(?:ure)?\.?\s*\d+\s*:", method):
        errors.append("method contains figure-caption content")
    if re.search(
        r"(?i)copyright|all rights reserved|corresponding author",
        method,
    ):
        errors.append("method contains publication front matter")

    prefix = method[:1200]
    numbers = re.findall(
        r"(?<![A-Za-z])[-+]?\d+(?:\.\d+)?%?",
        prefix,
    )
    words = re.findall(r"[A-Za-z]+", prefix)
    if len(numbers) >= 20 and len(numbers) >= len(words) * 0.25:
        errors.append("method starts with a numeric result matrix")
    return errors


def _reference_summary_is_noisy(summary: str) -> bool:
    numbers = re.findall(
        r"(?<![A-Za-z])[-+]?\d+(?:\.\d+)?%?",
        summary,
    )
    words = re.findall(r"[A-Za-z]+", summary)
    return bool(
        "\x00" in summary
        or re.search(r"(?i)\btable\s+[A-Z]?\d+\s*[:.]", summary)
        or re.search(r"(?i)\bfig(?:ure)?\.?\s*\d+\s*:", summary)
        or re.search(r"(?i)\[t\]\s*input\s*:\s*output\s*:", summary)
        or re.search(r"(?i)copyright|all rights reserved", summary)
        or re.search(r"(?:#\d+|\bc\|)", summary)
        or (
            len(numbers) >= 10
            and len(numbers) >= max(5, len(words) // 3)
        )
    )


def _reference_is_noisy(reference: dict) -> bool:
    title = reference.get("title", "")
    return bool(
        _reference_summary_is_noisy(reference.get("summary", ""))
        or re.search(
            r"(?i)frobnicat|lorem ipsum|placeholder",
            title,
        )
    )


def _safe_name(value: str) -> str:
    value = re.sub(r"[^A-Za-z0-9._-]+", "_", value).strip("._")
    return (value or "paper")[:120]


class LocalPDFPipeline:
    """Parse local full text and write only the compact training schema."""

    def __init__(
        self,
        output_dir: str,
        engine: str = "hybrid",
        max_references: int = 50,
        reference_metadata_paths: Optional[Iterable[str]] = None,
        reference_cache_path: Optional[str] = None,
        resolve_reference_abstracts_remotely: bool = False,
        min_references: int = 1,
        max_reference_title_lookups: int = 30,
        allow_crossref_title_search: bool = True,
        max_citation_contexts_per_reference: int = 2,
        citation_context_sentences_before: int = 1,
        citation_context_sentences_after: int = 1,
        reference_resolver: Optional[ReferenceAbstractResolver] = None,
    ):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.parser = PDFParser(engine=engine)
        self.splitter = SectionSplitter()
        self.min_references = max(1, int(min_references))
        self.references = LocalReferenceExtractor(
            max_references=max_references,
            max_contexts_per_reference=max_citation_contexts_per_reference,
            require_context=True,
            context_sentences_before=citation_context_sentences_before,
            context_sentences_after=citation_context_sentences_after,
        )
        self.reference_resolver = (
            reference_resolver
            or ReferenceAbstractResolver(
                metadata_paths=reference_metadata_paths,
                cache_path=reference_cache_path,
                allow_remote=resolve_reference_abstracts_remotely,
                max_title_lookups=max_reference_title_lookups,
                allow_crossref_title_search=allow_crossref_title_search,
            )
        )
        self.structured = StructuredFullTextParser()

    def process_pdf(
        self,
        pdf_path: str,
        *,
        title: Optional[str] = None,
        abstract: Optional[str] = None,
        field: Optional[str] = None,
        venue: Optional[str] = None,
        year: Optional[int] = None,
        paper_id: Optional[str] = None,
    ) -> dict:
        """Parse one existing PDF. The caller retains ownership of the file."""
        doc = self.parser.parse(pdf_path)
        sections = self.splitter.split(doc)
        return self._process_text(
            full_text=doc.full_text,
            sections=sections,
            title=title or doc.metadata.get("title") or doc.title,
            abstract=abstract,
            field=field,
            venue=venue,
            year=year,
            paper_id=paper_id or Path(pdf_path).stem,
        )

    def process_arxiv_source(
        self,
        source_path: str,
        *,
        title: Optional[str] = None,
        abstract: Optional[str] = None,
        field: Optional[str] = None,
        venue: Optional[str] = None,
        year: Optional[int] = None,
        paper_id: Optional[str] = None,
    ) -> dict:
        """Parse one arXiv TeX source bundle and write compact JSON."""
        document = self.structured.parse_arxiv_source(source_path)
        return self._process_structured(
            document,
            title=title,
            abstract=abstract,
            field=field,
            venue=venue,
            year=year,
            paper_id=paper_id or Path(source_path).stem,
        )

    def process_pmc_xml(
        self,
        xml_path: str,
        *,
        title: Optional[str] = None,
        abstract: Optional[str] = None,
        field: Optional[str] = None,
        venue: Optional[str] = None,
        year: Optional[int] = None,
        paper_id: Optional[str] = None,
    ) -> dict:
        """Parse one PMC JATS XML document and write compact JSON."""
        document = self.structured.parse_pmc_xml(xml_path)
        return self._process_structured(
            document,
            title=title,
            abstract=abstract,
            field=field,
            venue=venue,
            year=year,
            paper_id=paper_id or Path(xml_path).stem,
        )

    def _process_structured(
        self,
        document: StructuredDocument,
        *,
        title: Optional[str],
        abstract: Optional[str],
        field: Optional[str],
        venue: Optional[str],
        year: Optional[int],
        paper_id: str,
    ) -> dict:
        sections = dict(document.sections)
        detected = self.splitter.split(
            ParsedDocument(
                file_path="structured-fulltext",
                full_text=document.full_text,
            )
        )
        for name, text in detected.items():
            sections.setdefault(name, text)
        return self._process_text(
            full_text=document.full_text,
            sections=sections,
            title=title,
            abstract=abstract,
            field=field,
            venue=venue,
            year=year,
            paper_id=paper_id,
            structured_references=document.references,
        )

    def _process_text(
        self,
        *,
        full_text: str,
        sections: dict,
        title: Optional[str],
        abstract: Optional[str],
        field: Optional[str],
        venue: Optional[str],
        year: Optional[int],
        paper_id: str,
        structured_references: Optional[Iterable[object]] = None,
    ) -> dict:
        resolved_title = _clean_text(title)
        resolved_abstract = _clean_text(
            abstract
            or sections.get("abstract")
            or self.splitter.extract_abstract(full_text)
        )
        introduction = _clean_text(sections.get("introduction"))
        method = _clean_method(sections.get("method"))
        # Method is optional in the compact schema. If the parser found only a
        # fragment or a contaminated section, retain the introduction and leave
        # method empty instead of rejecting the whole paper.
        if len(method) < 200 or _method_quality_errors(method):
            method = ""
        reference_candidates = self.references.extract_candidates(
            full_text,
            sections.get("references"),
            structured_references=structured_references,
            sections=sections,
        )
        references = self.reference_resolver.resolve_many(reference_candidates)
        references = [
            reference for reference in references
            if not _reference_is_noisy(reference)
        ]

        output = {
            "title": resolved_title,
            "abstract": resolved_abstract,
            "introduction": introduction,
            "field": _clean_text(field) or "Unknown",
            "venue": _clean_text(venue) or "Unknown",
            "year": year,
            "method": method,
            "references": references,
        }
        validation_errors = self.validate_output(output)
        if validation_errors:
            raise ValueError("; ".join(validation_errors))

        output_name = _safe_name(paper_id or resolved_title)
        self._write_json(output, self.output_dir / f"{output_name}.json")
        return output

    def run_arxiv_trial(
        self,
        query: str,
        max_papers: int = 10,
        categories: Optional[List[str]] = None,
        exclude_dir: Optional[str] = None,
    ) -> dict:
        """Download, locally parse, and immediately clean up an arXiv trial."""
        excluded = self._existing_arxiv_ids(exclude_dir)
        search_limit = max(max_papers * 5, max_papers)
        searcher = ArxivDownloader(
            output_dir=tempfile.gettempdir(),
            request_delay=1.0,
        )
        candidates = searcher.search(
            query=query,
            max_results=search_limit,
            categories=categories,
            sort_by="submittedDate",
        )

        candidates = [
            paper for paper in candidates
            if self._clean_arxiv_id(paper.arxiv_id) not in excluded
        ]

        stats = {
            "requested": max_papers,
            "selected": 0,
            "succeeded": 0,
            "failed": 0,
            "outputs": [],
            "errors": [],
        }
        for paper in candidates:
            if stats["succeeded"] >= max_papers:
                break

            stats["selected"] += 1
            output_path = self.output_dir / (
                _safe_name(paper.arxiv_id or paper.id) + ".json"
            )
            if output_path.exists():
                existing = self._load_json(output_path)
                existing_errors = self.validate_output(existing)
                if not existing_errors:
                    stats["succeeded"] += 1
                    stats["outputs"].append(str(output_path))
                    continue
                self._quarantine(output_path)
                stats["failed"] += 1
                stats["errors"].append({
                    "id": paper.arxiv_id or paper.id,
                    "error": "; ".join(existing_errors),
                })
                continue

            try:
                with tempfile.TemporaryDirectory(
                    prefix="paper_mining_local_"
                ) as temporary_dir:
                    downloader = ArxivDownloader(
                        output_dir=temporary_dir,
                        request_delay=1.0,
                    )
                    pdf_path = downloader.download_pdf(paper)
                    if not pdf_path:
                        raise RuntimeError("PDF download failed")
                    self.process_pdf(
                        str(pdf_path),
                        title=paper.title,
                        abstract=paper.abstract,
                        field=paper.field,
                        venue=paper.venue,
                        paper_id=paper.arxiv_id or paper.id,
                    )

                stats["succeeded"] += 1
                stats["outputs"].append(str(output_path))
            except Exception as exc:
                stats["failed"] += 1
                stats["errors"].append({
                    "id": paper.arxiv_id or paper.id,
                    "error": str(exc),
                })
                logger.exception("Failed local trial paper: %s", paper.title)

        return stats

    def validate_output(self, output: object) -> List[str]:
        """Return quality errors for the compact schema."""
        if not isinstance(output, dict):
            return ["output is not a JSON object"]
        if set(output) != {
            "title", "abstract", "introduction", "field", "venue",
            "year", "method", "references",
        }:
            return ["output keys do not match the compact schema"]

        errors = []
        if len(output.get("title", "")) < 5:
            errors.append("title is empty or too short")
        if len(output.get("abstract", "")) < 100:
            errors.append("abstract is empty or too short")
        if len(output.get("introduction", "")) < 100:
            errors.append("introduction is empty or too short")
        if not output.get("field"):
            errors.append("field is empty")
        if not output.get("venue"):
            errors.append("venue is empty")
        if output.get("method"):
            errors.extend(_method_quality_errors(output["method"]))

        references = output.get("references")
        if not isinstance(references, list) or not references:
            errors.append("no cited-paper abstracts resolved")
        elif len(references) < self.min_references:
            errors.append(
                f"too few cited-paper abstracts resolved "
                f"({len(references)} < {self.min_references})"
            )
        elif any(
            not isinstance(reference, dict)
            or set(reference) != {
                "id", "title", "summary", "citation_contexts"
            }
            or not all(reference.get(key) for key in ("id", "title", "summary"))
            or not abstract_is_complete(reference.get("summary"))
            or not isinstance(reference.get("citation_contexts"), list)
            or not reference.get("citation_contexts")
            or any(
                not isinstance(context, dict)
                or set(context) != {
                    "source_section", "context_before",
                    "citation_sentence", "context_after",
                }
                or not context.get("source_section")
                or not context.get("citation_sentence")
                or not isinstance(context.get("context_before"), list)
                or not isinstance(context.get("context_after"), list)
                for context in reference.get("citation_contexts", [])
            )
            for reference in references
        ):
            errors.append(
                "one or more references lack a complete abstract or citation context"
            )
        return errors

    def _existing_arxiv_ids(self, directory: Optional[str]) -> set:
        if not directory:
            return set()
        path = Path(directory)
        if not path.exists():
            return set()
        return {
            self._clean_arxiv_id(item.stem)
            for item in path.glob("*.json")
        }

    def _clean_arxiv_id(self, value: Optional[str]) -> str:
        if not value:
            return ""
        value = re.sub(r"^arxiv:", "", value, flags=re.IGNORECASE)
        return re.sub(r"v\d+$", "", value, flags=re.IGNORECASE).lower()

    def _write_json(self, output: dict, path: Path) -> None:
        temporary_path = path.with_suffix(path.suffix + ".tmp")
        with open(temporary_path, "w", encoding="utf-8") as handle:
            json.dump(output, handle, ensure_ascii=False, indent=2)
        temporary_path.chmod(0o644)
        temporary_path.replace(path)

    def _load_json(self, path: Path) -> object:
        try:
            with open(path, encoding="utf-8") as handle:
                return json.load(handle)
        except (OSError, json.JSONDecodeError):
            return None

    def _quarantine(self, path: Path) -> None:
        rejected_dir = self.output_dir / "_rejected"
        rejected_dir.mkdir(parents=True, exist_ok=True)
        path.replace(rejected_dir / path.name)
