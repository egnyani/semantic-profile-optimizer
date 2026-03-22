"""
Convert a DOCX file to PDF via the LibreOffice microservice hosted on Render.com.

All DOCX pre-processing (font substitution + line-spacing adjustment) happens
here on Vercel, so Render is purely a dumb LibreOffice converter. This means
Render rebuilds never affect formatting.

Set env var PDF_SERVICE_URL to the Render service URL.
"""

from __future__ import annotations

import io
import os
import re
import urllib.request
import urllib.error
import zipfile
from pathlib import Path

# ── Font substitution ──────────────────────────────────────────────────────
# Metric-compatible Microsoft → LibreOffice equivalents (same character widths).
# Carlito ≡ Calibri | Liberation Sans ≡ Arial | Liberation Serif ≡ Times New Roman
_FONT_MAP = {
    "Calibri": "Carlito",
    "Arial": "Liberation Sans",
    "Times New Roman": "Liberation Serif",
}
_FONT_XML_FILES = {
    "word/document.xml",
    "word/styles.xml",
    "word/settings.xml",
    "word/theme/theme1.xml",
    "word/fontTable.xml",
}

# ── Line-spacing scales to try ─────────────────────────────────────────────
# LibreOffice renders Carlito with tighter vertical metrics than Word renders
# Calibri, leaving extra whitespace at the bottom. We scale w:line values to
# compensate. Scales are tried in descending order; the first that produces a
# 1-page PDF is used. The Render service returns HTTP 422 if the PDF > 1 page.
_SCALES = [1.11, 1.08, 1.05, 1.0]


def _build_docx(docx_path: Path, line_scale: float) -> bytes:
    """Patch fonts and scale line spacing in the DOCX, return modified bytes."""

    def _scale_spacing(xml: str) -> str:
        def _replace(m: re.Match) -> str:
            tag = m.group(0)
            if 'w:lineRule="exact"' in tag:
                return tag
            def _scale_val(vm: re.Match) -> str:
                return f'w:line="{int(int(vm.group(1)) * line_scale)}"'
            return re.sub(r'w:line="(\d+)"', _scale_val, tag)
        return re.sub(r'<w:spacing\b[^/]*/>', _replace, xml)

    out = io.BytesIO()
    with zipfile.ZipFile(docx_path, "r") as zin, \
         zipfile.ZipFile(out, "w", zipfile.ZIP_DEFLATED) as zout:
        for item in zin.infolist():
            data = zin.read(item.filename)
            if item.filename in _FONT_XML_FILES:
                text = data.decode("utf-8")
                for src, dst in _FONT_MAP.items():
                    text = text.replace(src, dst)
                if item.filename in ("word/document.xml", "word/styles.xml"):
                    text = _scale_spacing(text)
                data = text.encode("utf-8")
            zout.writestr(item, data)
    return out.getvalue()


def _post_to_render(docx_bytes: bytes, filename: str, base_url: str) -> bytes | None:
    """POST DOCX bytes to Render. Returns PDF bytes, or None if PDF > 1 page (422)."""
    boundary = "PdfServiceBoundary"
    body = (
        f"--{boundary}\r\n"
        f'Content-Disposition: form-data; name="file"; filename="{filename}"\r\n'
        f"Content-Type: application/vnd.openxmlformats-officedocument.wordprocessingml.document\r\n\r\n"
    ).encode() + docx_bytes + f"\r\n--{boundary}--\r\n".encode()

    req = urllib.request.Request(
        f"{base_url}/convert",
        data=body,
        headers={"Content-Type": f"multipart/form-data; boundary={boundary}"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=90) as resp:
            return resp.read()
    except urllib.error.HTTPError as e:
        if e.code == 422:
            return None  # PDF overflowed — caller will retry with smaller scale
        raise RuntimeError(f"PDF service error {e.code}: {e.read().decode('utf-8', errors='ignore')}")


def convert_docx_to_pdf(docx_path: Path) -> Path:
    """Pre-process DOCX, send to Render for conversion, return local PDF path."""
    base_url = os.environ.get("PDF_SERVICE_URL", "").strip().rstrip("/")
    if not base_url:
        raise RuntimeError("PDF_SERVICE_URL env var is not set.")

    for scale in _SCALES:
        docx_bytes = _build_docx(docx_path, scale)
        pdf_bytes = _post_to_render(docx_bytes, docx_path.name, base_url)
        if pdf_bytes is not None:
            pdf_path = docx_path.with_suffix(".pdf")
            pdf_path.write_bytes(pdf_bytes)
            return pdf_path

    # All scales overflowed — last resort: send unscaled
    docx_bytes = _build_docx(docx_path, 1.0)
    pdf_bytes = _post_to_render(docx_bytes, docx_path.name, base_url) or b""
    pdf_path = docx_path.with_suffix(".pdf")
    pdf_path.write_bytes(pdf_bytes)
    return pdf_path
