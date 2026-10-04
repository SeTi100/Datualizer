"""Source identification: label and SHA-256 fingerprint of an input (Datenkatalog P2)."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
from pathlib import Path
from typing import TextIO


@dataclass(frozen=True)
class SourceInfo:
    """A read source: display label, content hash and the input to hand to the pre-scanner."""

    label: str
    sha256: str
    scan_input: str | Path


def is_file_path(source: object) -> bool:
    """Same rule as the pre-scanner: a Path, or a one-line string naming an existing file."""
    if isinstance(source, Path):
        return True
    return isinstance(source, str) and "\n" not in source and "\r" not in source and Path(source).is_file()


def read_source(source: str | Path | TextIO, index: int = 0) -> SourceInfo:
    """Fingerprint a source. Files are hashed byte by byte, text and streams as UTF-8."""
    if is_file_path(source):
        path = Path(source)
        return SourceInfo(str(path), hashlib.sha256(path.read_bytes()).hexdigest(), path)
    if isinstance(source, str):
        text = source
        label = f"<text #{index}>"
    elif hasattr(source, "read"):
        text = source.read()
        label = str(getattr(source, "name", f"<stream #{index}>"))
    else:
        raise TypeError(f"Unsupported source type: {type(source)}")
    return SourceInfo(label, hashlib.sha256(text.encode("utf-8")).hexdigest(), text)
