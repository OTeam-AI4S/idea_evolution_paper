"""
Utility functions for paper mining.
"""

import os
import re
import yaml
import hashlib
import logging
from pathlib import Path
from typing import Optional, Dict, Any

logger = logging.getLogger(__name__)


def load_config(config_path: str = "config.yaml") -> Dict[str, Any]:
    """Load YAML configuration file."""
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def sanitize_filename(name: str, max_length: int = 150) -> str:
    """Sanitize a string to be safe as a filename."""
    # Remove or replace unsafe characters
    name = re.sub(r'[<>:"/\\|?*]', "_", name)
    # Collapse whitespace
    name = re.sub(r"\s+", " ", name).strip()
    # Truncate
    if len(name) > max_length:
        name = name[:max_length].rsplit(" ", 1)[0]
    return name


def get_paper_id(identifier: str) -> str:
    """Generate a short hash ID for a paper from its DOI or title."""
    return hashlib.md5(identifier.encode("utf-8")).hexdigest()[:12]


def ensure_dir(path: str) -> Path:
    """Ensure a directory exists and return it as Path."""
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p


def project_root() -> Path:
    """Get the project root directory."""
    return Path(__file__).parent.parent
