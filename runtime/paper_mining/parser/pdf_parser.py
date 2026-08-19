"""
PDF parser using multiple backends: PyMuPDF (primary) and pdfplumber (fallback).

PyMuPDF (fitz) is fast and good for born-digital PDFs.
pdfplumber excels at table extraction and scanned documents.
"""

import json
import logging
from pathlib import Path
from typing import Optional, Dict, Any, List, Tuple
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)


@dataclass
class ParsedPage:
    """A single page of parsed content."""
    page_num: int
    text: str
    blocks: List[Dict[str, Any]] = field(default_factory=list)
    tables: List[List[List[str]]] = field(default_factory=list)
    images: List[Dict[str, Any]] = field(default_factory=list)


@dataclass
class ParsedDocument:
    """Complete parsed PDF document."""
    file_path: str
    title: Optional[str] = None
    authors: Optional[List[str]] = None
    abstract: Optional[str] = None
    full_text: str = ""
    pages: List[ParsedPage] = field(default_factory=list)
    sections: Dict[str, str] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)
    parse_errors: List[str] = field(default_factory=list)


class PDFParser:
    """
    Parse PDF files into structured text.

    Supports two engines:
    - "pymupdf": Fast, good for born-digital PDFs (default primary).
    - "pdfplumber": Better for scanned docs, tables, complex layouts.
    - "hybrid": Try pymupdf first, fall back to pdfplumber.
    """

    def __init__(self, engine: str = "hybrid"):
        self.engine = engine

    def parse(self, pdf_path: str) -> ParsedDocument:
        """
        Parse a single PDF file.

        Args:
            pdf_path: Path to the PDF file.

        Returns:
            ParsedDocument with extracted text, metadata, and structure.
        """
        path = Path(pdf_path)
        if not path.exists():
            raise FileNotFoundError(f"PDF not found: {pdf_path}")

        logger.info(f"Parsing: {path.name}")

        if self.engine == "pymupdf":
            return self._parse_pymupdf(path)
        elif self.engine == "pdfplumber":
            return self._parse_pdfplumber(path)
        else:  # hybrid
            try:
                return self._parse_pymupdf(path)
            except Exception as e:
                logger.warning(f"PyMuPDF failed ({e}), falling back to pdfplumber")
                return self._parse_pdfplumber(path)

    def _parse_pymupdf(self, path: Path) -> ParsedDocument:
        """Parse using PyMuPDF (fitz)."""
        import fitz  # PyMuPDF

        doc = ParsedDocument(file_path=str(path))
        pdf = fitz.open(str(path))
        if pdf.page_count <= 0:
            pdf.close()
            raise ValueError(f"PDF contains no readable pages: {path}")

        # Extract metadata
        meta = pdf.metadata
        doc.metadata = {
            "title": meta.get("title", ""),
            "author": meta.get("author", ""),
            "subject": meta.get("subject", ""),
            "keywords": meta.get("keywords", ""),
            "page_count": pdf.page_count,
            "format": meta.get("format", "PDF"),
        }

        all_text = []
        for page_num in range(pdf.page_count):
            page = pdf[page_num]

            # Extract text blocks with positions
            blocks = page.get_text("dict")["blocks"]

            page_blocks = []
            page_text_parts = []
            for block in blocks:
                if block["type"] == 0:  # text block
                    block_text = ""
                    for line in block.get("lines", []):
                        line_text = " ".join(
                            span["text"] for span in line.get("spans", [])
                        )
                        block_text += line_text + "\n"

                    page_text_parts.append(block_text)
                    page_blocks.append({
                        "type": "text",
                        "bbox": block["bbox"],
                        "text": block_text.strip(),
                    })
                elif block["type"] == 1:  # image block
                    page_blocks.append({
                        "type": "image",
                        "bbox": block["bbox"],
                        "size": block.get("size", 0),
                    })

            page_text = "\n".join(page_text_parts)
            all_text.append(page_text)

            doc.pages.append(ParsedPage(
                page_num=page_num + 1,
                text=page_text.strip(),
                blocks=page_blocks,
            ))

        pdf.close()

        doc.full_text = "\n\n".join(all_text)

        # Try to extract title from first page (usually largest font on page 1)
        doc.title = self._extract_title_pymupdf(doc)

        return doc

    def _parse_pdfplumber(self, path: Path) -> ParsedDocument:
        """Parse using pdfplumber (better for complex layouts)."""
        import pdfplumber

        doc = ParsedDocument(file_path=str(path))

        with pdfplumber.open(path) as pdf:
            if not pdf.pages:
                raise ValueError(f"PDF contains no readable pages: {path}")
            doc.metadata = {
                "page_count": len(pdf.pages),
                "metadata": pdf.metadata or {},
            }

            all_text = []
            for page_num, page in enumerate(pdf.pages):
                text = page.extract_text() or ""
                all_text.append(text)

                # Extract tables
                tables = page.extract_tables() or []
                parsed_tables = []
                for table in tables:
                    if table:
                        parsed_tables.append(table)

                doc.pages.append(ParsedPage(
                    page_num=page_num + 1,
                    text=text.strip(),
                    tables=parsed_tables,
                ))

        doc.full_text = "\n\n".join(all_text)
        return doc

    def _extract_title_pymupdf(self, doc: ParsedDocument) -> Optional[str]:
        """
        Heuristic: title is often the first substantive text block,
        with the largest font size on the first page.
        """
        if not doc.pages:
            return None

        first_page = doc.pages[0]
        text_blocks = [b for b in first_page.blocks if b["type"] == "text"]

        if not text_blocks:
            # Fall back to first non-empty lines
            lines = first_page.text.strip().split("\n")
            for line in lines[:5]:
                candidate = line.strip()
                if len(candidate) > 10 and len(candidate) < 300:
                    return candidate
            return None

        # Return the first text block that looks like a title (not author list, not affiliation)
        for block in text_blocks[:3]:
            text = block["text"].strip()
            # Skip blocks that look like author lists (contain commas at end, emails, etc.)
            if not text:
                continue
            if "@" in text:
                continue
            if len(text) > 10 and len(text) < 400:
                # Check it's not all names (too many commas)
                comma_ratio = text.count(",") / max(len(text), 1)
                if comma_ratio < 0.05:
                    return text

        return text_blocks[0]["text"].strip() if text_blocks else None

    def parse_batch(
        self,
        pdf_paths: List[str],
        output_dir: Optional[str] = None,
        output_format: str = "json",
    ) -> List[ParsedDocument]:
        """
        Parse multiple PDFs.

        Args:
            pdf_paths: List of PDF file paths.
            output_dir: If provided, save parsed results to this directory.
            output_format: "json" or "txt".

        Returns:
            List of ParsedDocument objects.
        """
        from tqdm import tqdm

        results = []
        for pdf_path in tqdm(pdf_paths, desc="Parsing PDFs"):
            try:
                parsed = self.parse(pdf_path)
                results.append(parsed)

                if output_dir:
                    self._save_parsed(parsed, output_dir, output_format)

            except Exception as e:
                logger.error(f"Failed to parse {pdf_path}: {e}")

        return results

    def _save_parsed(
        self,
        doc: ParsedDocument,
        output_dir: str,
        output_format: str = "json",
    ):
        """Save parsed document to disk."""
        out_path = Path(output_dir)
        out_path.mkdir(parents=True, exist_ok=True)

        stem = Path(doc.file_path).stem

        if output_format == "json":
            output = {
                "file_path": doc.file_path,
                "title": doc.title,
                "authors": doc.authors,
                "abstract": doc.abstract,
                "full_text": doc.full_text,
                "sections": doc.sections,
                "metadata": doc.metadata,
                "parse_errors": doc.parse_errors,
            }
            output_path = out_path / f"{stem}.json"
            with open(output_path, "w", encoding="utf-8") as f:
                json.dump(output, f, ensure_ascii=False, indent=2)

        elif output_format == "txt":
            output_path = out_path / f"{stem}.txt"
            with open(output_path, "w", encoding="utf-8") as f:
                f.write(f"Title: {doc.title or 'N/A'}\n")
                f.write(f"Authors: {', '.join(doc.authors or ['N/A'])}\n")
                f.write("=" * 80 + "\n\n")
                f.write(doc.full_text)

        output_path.chmod(0o644)

        logger.info(f"Saved parsed output: {output_path}")
