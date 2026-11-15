"""Extract PDF text in memory and preserve source page numbers."""
from io import BytesIO
from typing import List, Tuple

from pypdf import PdfReader


def load_pdf(file_bytes: bytes) -> List[Tuple[str, int]]:
    """Return nonempty (text, one-based page) pairs without a shared temp file."""
    reader = PdfReader(BytesIO(file_bytes))
    pages: List[Tuple[str, int]] = []
    for number, page in enumerate(reader.pages, start=1):
        text = (page.extract_text() or "").strip()
        if text:
            pages.append((text, number))
    return pages
