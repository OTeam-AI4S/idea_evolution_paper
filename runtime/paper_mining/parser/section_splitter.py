"""
Section splitter — segments parsed PDF text into academic sections
(Abstract, Introduction, Method, Related Work, Experiments, etc.)
"""

import re
import logging
from typing import Dict, List, Tuple, Optional, Pattern

from .pdf_parser import ParsedDocument

logger = logging.getLogger(__name__)


class SectionSplitter:
    """
    Split academic paper text into standard sections using regex patterns.

    The splitter works by:
    1. Detecting section headers via compiled regex patterns.
    2. Assigning text between headers to the corresponding section.
    3. Handling variations in numbering and formatting.
    """

    # Default section detection patterns (ordered by typical appearance)
    DEFAULT_PATTERNS: List[Tuple[str, List[str]]] = [
        ("abstract", [
            r"(?i)^\s*abstract\s*$",
            r"(?i)^\s*a\s*b\s*s\s*t\s*r\s*a\s*c\s*t\s*$",
        ]),
        ("introduction", [
            r"(?i)^\s*(?:(?:\d+(?:\.\d+)*)|[IVXLC]+)[\.\)]?\s+introduction\s*$",
            r"(?i)^\s*introduction\s*$",
            r"(?i)^\s*i\.?\s+introduction\s*$",
        ]),
        ("related_work", [
            r"(?i)^\s*(?:(?:(?:\d+(?:\.\d+)*)|[IVXLC]+)[\.\)]?\s+)?related\s+work\s*$",
            r"(?i)^\s*(?:(?:(?:\d+(?:\.\d+)*)|[IVXLC]+)[\.\)]?\s+)?related\s+research\s*$",
            r"(?i)^\s*(?:(?:(?:\d+(?:\.\d+)*)|[IVXLC]+)[\.\)]?\s+)?background(\s+and\s+related\s+work)?\s*$",
            r"(?i)^\s*(?:(?:(?:\d+(?:\.\d+)*)|[IVXLC]+)[\.\)]?\s+)?literature\s+review\s*$",
            r"(?i)^\s*(?:(?:(?:\d+(?:\.\d+)*)|[IVXLC]+)[\.\)]?\s+)?previous\s+work\s*$",
            r"(?i)^\s*(?:(?:(?:\d+(?:\.\d+)*)|[IVXLC]+)[\.\)]?\s+)?state\s+of\s+the\s+art\s*$",
        ]),
        ("background", [
            r"(?i)^\s*(?:\d+[\.\)]\s*)?background\s*$",
            r"(?i)^\s*(?:\d+[\.\)]\s*)?preliminaries?\s*$",
            r"(?i)^\s*(?:\d+[\.\)]\s*)?problem\s+(statement|formulation|definition)\s*$",
        ]),
        ("method", [
            r"(?i)^\s*(?:(?:(?:\d+(?:\.\d+)*)|[IVXLC]+)[\.\)]?\s+)?(?:our\s+)?(?:proposed\s+)?(?:method|methodology|approach|model|framework|architecture|system|algorithm)s?\s*$",
            r"(?i)^\s*(?:(?:(?:\d+(?:\.\d+)*)|[IVXLC]+)[\.\)]?\s+)?(?:problem\s+formulation|technical\s+approach|model\s+design|training\s+(?:procedure|strategy|methodology)|system\s+implementation|implementation\s+details|optimization)\s*$",
            r"(?i)^\s*(?:(?:(?:\d+(?:\.\d+)*)|[IVXLC]+)[\.\)]?\s+)?(?:the\s+)?[A-Za-z0-9][A-Za-z0-9\- ]{1,60}\s+(?:method|approach|model|framework|architecture|algorithm)\s*$",
        ]),
        ("experiments", [
            r"(?i)^\s*(?:(?:(?:\d+(?:\.\d+)*)|[IVXLC]+)[\.\)]?\s+)?experiments?(?:\s+and\s+results?)?\s*$",
            r"(?i)^\s*(?:(?:(?:\d+(?:\.\d+)*)|[IVXLC]+)[\.\)]?\s+)?experimental\s+(setup|results?|evaluation|design)\s*$",
            r"(?i)^\s*(?:(?:(?:\d+(?:\.\d+)*)|[IVXLC]+)[\.\)]?\s+)?evaluation\s*$",
            r"(?i)^\s*(?:(?:(?:\d+(?:\.\d+)*)|[IVXLC]+)[\.\)]?\s+)?results?(?:\s+and\s+(analysis|discussion))?\s*$",
            r"(?i)^\s*(?:(?:(?:\d+(?:\.\d+)*)|[IVXLC]+)[\.\)]?\s+)?empirical\s+(study|evaluation|results)\s*$",
        ]),
        ("discussion", [
            r"(?i)^\s*(?:\d+[\.\)]\s*)?discussion\s*$",
            r"(?i)^\s*(?:\d+[\.\)]\s*)?(general\s+)?discussion\s*$",
            r"(?i)^\s*(?:\d+[\.\)]\s*)?analysis(\s+and\s+discussion)?\s*$",
        ]),
        ("limitations", [
            r"(?i)^\s*(?:\d+(?:\.\d+)*[\.\)]?\s+)?limitations?\s*$",
            r"(?i)^\s*(?:\d+(?:\.\d+)*[\.\)]?\s+)?(?:study\s+)?limitations?\s+and\s+future\s+work\s*$",
        ]),
        ("conclusion", [
            r"(?i)^\s*(?:\d+[\.\)]\s*)?conclusions?(?:\s+and\s+(future\s+)?(work|outlook|directions))?\s*$",
            r"(?i)^\s*(?:\d+[\.\)]\s*)?summary(\s+and\s+conclusions?)?\s*$",
            r"(?i)^\s*(?:\d+[\.\)]\s*)?concluding\s+remarks\s*$",
            r"(?i)^\s*(?:\d+[\.\)]\s*)?future\s+work\s*$",
        ]),
        ("acknowledgments", [
            r"(?i)^\s*(?:\d+[\.\)]\s*)?acknowledg?ments?\s*$",
            r"(?i)^\s*(?:\d+[\.\)]\s*)?acknowledgements?\s*$",
        ]),
        ("references", [
            r"(?i)^\s*(?:(?:\d+(?:\.\d+)*)|[IVXLC]+)[\.\)]?\s+references?\s*$",
            r"(?i)^\s*references?\s*$",
            r"(?i)^\s*(?:(?:(?:\d+(?:\.\d+)*)|[IVXLC]+)[\.\)]?\s+)?bibliography\s*$",
            r"(?i)^\s*(?:(?:(?:\d+(?:\.\d+)*)|[IVXLC]+)[\.\)]?\s+)?works\s+cited\s*$",
        ]),
    ]

    def __init__(self, custom_patterns: Optional[List[Tuple[str, List[str]]]] = None):
        """
        Args:
            custom_patterns: Optional custom section patterns.
                             List of (section_name, [regex_patterns]).
        """
        self.patterns = custom_patterns or self.DEFAULT_PATTERNS
        self._compiled: List[Tuple[str, List[Pattern]]] = [
            (name, [re.compile(p) for p in patterns])
            for name, patterns in self.patterns
        ]

    def split(self, doc: ParsedDocument) -> Dict[str, str]:
        """
        Split a parsed document into sections.

        Args:
            doc: ParsedDocument from PDFParser.

        Returns:
            Dict mapping section names to their text content.
        """
        text = doc.full_text
        if not text:
            return {}

        lines = text.split("\n")
        return self._split_lines(lines)

    def _split_lines(self, lines: List[str]) -> Dict[str, str]:
        """
        Scan lines for section headers and accumulate text between them.

        Strategy:
        1. Find all potential section header line numbers.
        2. Choose the best matching sequence.
        3. Extract text between consecutive headers.
        """
        n = len(lines)

        # Find all candidate section boundaries
        # boundary: (line_index, section_name, score)
        candidates: List[Tuple[int, str, float]] = []

        for i, line in enumerate(lines):
            line_stripped = line.strip()
            if not line_stripped:
                continue
            if len(line_stripped) > 100:  # Headers are usually short
                continue

            for section_name, patterns in self._compiled:
                for pattern in patterns:
                    if pattern.match(line_stripped):
                        # Score: shorter match = more likely a header
                        score = 1.0 / max(len(line_stripped), 1) * 10
                        candidates.append((i, section_name, score))
                        break  # one match per section type per line

        if not candidates:
            logger.warning("No section headers detected in document")
            return {"full_text": "\n".join(lines)}

        # Sort by line number
        candidates.sort(key=lambda x: x[0])

        # Deduplicate duplicate interpretations of the same header.  Different
        # sections can legitimately be only two lines apart in short papers,
        # so they must not replace each other.
        filtered = []
        for i, (line_num, name, score) in enumerate(candidates):
            same_line = filtered and line_num == filtered[-1][0]
            repeated_nearby = (
                filtered
                and name == filtered[-1][1]
                and line_num - filtered[-1][0] <= 2
            )
            if same_line or repeated_nearby:
                if score > filtered[-1][2]:
                    filtered[-1] = (line_num, name, score)
            else:
                filtered.append((line_num, name, score))

        # Only keep sections that appear in typical order
        # (Abstract -> Introduction -> Related Work -> Method -> Experiments -> Conclusion)
        sections: Dict[str, str] = {}

        for idx, (line_num, section_name, _) in enumerate(filtered):
            start = line_num + 1  # skip header line itself

            # Determine end: next section header or end of document
            if idx + 1 < len(filtered):
                end = filtered[idx + 1][0]
            else:
                end = n

            # Extract text
            section_text = "\n".join(lines[start:end]).strip()

            # If we already saw this section, merge
            if section_name in sections:
                sections[section_name] += "\n\n" + section_text
            else:
                sections[section_name] = section_text

        logger.info(
            f"Found {len(sections)} sections: {list(sections.keys())}"
        )
        return sections

    def extract_abstract(self, text: str) -> Optional[str]:
        """
        Extract abstract text heuristically — looks for the block between
        "Abstract" header and the next section header or blank-line gap.
        """
        lines = text.split("\n")

        abstract_start = None
        for i, line in enumerate(lines):
            if re.match(r"(?i)^\s*abstract\s*$", line.strip()):
                abstract_start = i + 1
                break

        if abstract_start is None:
            return None

        # Find end: next section header pattern
        abstract_end = len(lines)
        for i in range(abstract_start, len(lines)):
            line = lines[i].strip()
            for section_name, patterns in self._compiled:
                if section_name == "abstract":
                    continue
                for pattern in patterns:
                    if pattern.match(line):
                        abstract_end = i
                        break
                if abstract_end < len(lines):
                    break
            if abstract_end < len(lines):
                break

        return "\n".join(lines[abstract_start:abstract_end]).strip()

    def extract_by_keyword(
        self, text: str, keyword: str, context_lines: int = 3
    ) -> List[str]:
        """
        Find paragraphs containing a keyword (useful for methodology detection).

        Args:
            text: Full text to search.
            keyword: Keyword to search for.
            context_lines: Number of surrounding lines to include.

        Returns:
            List of matching text snippets.
        """
        lines = text.split("\n")
        matches = []

        for i, line in enumerate(lines):
            if keyword.lower() in line.lower():
                start = max(0, i - context_lines)
                end = min(len(lines), i + context_lines + 1)
                snippet = "\n".join(lines[start:end])
                matches.append(snippet)

        return matches
