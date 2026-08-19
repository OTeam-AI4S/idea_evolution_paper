"""
Paper downloaders supporting multiple sources.

Discovery: arXiv API, Semantic Scholar API, Google Scholar
PDF download: Direct (primary), arXiv, OA links
"""

from importlib import import_module

from .base import BaseDownloader, PaperInfo

__all__ = [
    "BaseDownloader",
    "PaperInfo",
    "ArxivDownloader",
    "SemanticScholarDownloader",
    "ScholarlyDownloader",
    "DirectDownloader",
    "FullTextDownloader",
    "CrossRefResolver",
]

_LAZY_IMPORTS = {
    "ArxivDownloader": (".arxiv_downloader", "ArxivDownloader"),
    "SemanticScholarDownloader": (
        ".semantic_scholar", "SemanticScholarDownloader"
    ),
    "ScholarlyDownloader": (".scholarly_downloader", "ScholarlyDownloader"),
    "DirectDownloader": (".direct_downloader", "DirectDownloader"),
    "FullTextDownloader": (".fulltext_downloader", "FullTextDownloader"),
    "CrossRefResolver": (".crossref_resolver", "CrossRefResolver"),
}


def __getattr__(name):
    """Load optional downloader dependencies only when they are requested."""
    if name not in _LAZY_IMPORTS:
        raise AttributeError(name)
    module_name, attribute = _LAZY_IMPORTS[name]
    value = getattr(import_module(module_name, __name__), attribute)
    globals()[name] = value
    return value
