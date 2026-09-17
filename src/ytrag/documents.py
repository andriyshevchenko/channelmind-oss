"""Extract plain text from uploaded reference documents (txt, pdf, docx).

Produces "doc" dicts shaped like transcript docs (id/title/url/text) so the same
word-based chunker can index them. Documents have no timecodes or video URL, so
their chunks carry empty url/start — the assistant cites them by source name.
"""
from __future__ import annotations

from pathlib import Path

SUPPORTED_EXTENSIONS = {".txt", ".md", ".pdf", ".docx"}


def is_supported(filename: str) -> bool:
    return Path(filename).suffix.lower() in SUPPORTED_EXTENSIONS


def extract_text(path: Path) -> str:
    suffix = path.suffix.lower()
    if suffix in (".txt", ".md"):
        return path.read_text(encoding="utf-8", errors="ignore")
    if suffix == ".pdf":
        return _extract_pdf(path)
    if suffix == ".docx":
        return _extract_docx(path)
    raise ValueError(f"Unsupported document type: {suffix}")


def _extract_pdf(path: Path) -> str:
    from pypdf import PdfReader

    reader = PdfReader(str(path))
    pages = [(page.extract_text() or "") for page in reader.pages]
    return "\n\n".join(pages).strip()


def _extract_docx(path: Path) -> str:
    import docx

    document = docx.Document(str(path))
    return "\n".join(p.text for p in document.paragraphs).strip()


def document_to_doc(source_id: str, path: Path, title: str) -> dict:
    """Build a transcript-shaped doc dict from a reference document file."""
    text = extract_text(path)
    return {
        "id": f"doc_{source_id}_{path.stem}",
        "title": title,
        "url": "",
        "text": text,
    }
