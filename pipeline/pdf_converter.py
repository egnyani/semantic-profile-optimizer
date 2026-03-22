"""
Convert a DOCX file to PDF via the LibreOffice microservice hosted on Render.com.

Set the env var PDF_SERVICE_URL to the Render service URL, e.g.:
  PDF_SERVICE_URL=https://resume-pdf-converter.onrender.com
"""

from __future__ import annotations

import io
import os
import re
import urllib.request
import urllib.error
import zipfile
from pathlib import Path

# LibreOffice renders Carlito with ~6.5% tighter vertical line height than
# Word renders Calibri, causing content to end ~0.6" higher on the page.
# Scaling all explicit w:line spacing values by this factor compensates.
_LINE_SCALE = 1.09

# Metric-compatible substitutes for Microsoft fonts.
# Carlito  ≡ Calibri  (same char widths)
# Liberation Sans  ≡ Arial
# Liberation Serif ≡ Times New Roman
_FONT_MAP = {
    "Calibri": "Carlito",
    "Arial": "Liberation Sans",
    "Times New Roman": "Liberation Serif",
}

# DOCX XML files that contain font references
_FONT_XML_FILES = {
    "word/document.xml",
    "word/styles.xml",
    "word/settings.xml",
    "word/theme/theme1.xml",
}


def _scale_line_spacing(xml: str) -> str:
    """Scale w:line values in w:spacing elements to compensate for LibreOffice rendering."""
    def _replace(m: re.Match) -> str:
        tag = m.group(0)
        # Don't touch exact line rules — they're already absolute
        if 'w:lineRule="exact"' in tag:
            return tag
        def _scale_val(vm: re.Match) -> str:
            val = int(vm.group(1))
            return f'w:line="{int(val * _LINE_SCALE)}"'
        return re.sub(r'w:line="(\d+)"', _scale_val, tag)
    return re.sub(r'<w:spacing\b[^/]*/>', _replace, xml)


def _patch_fonts(docx_path: Path) -> bytes:
    """Return the DOCX bytes with fonts swapped and line spacing adjusted for LibreOffice."""
    out_buf = io.BytesIO()
    with zipfile.ZipFile(docx_path, "r") as zin, \
         zipfile.ZipFile(out_buf, "w", zipfile.ZIP_DEFLATED) as zout:
        for item in zin.infolist():
            data = zin.read(item.filename)
            if item.filename in _FONT_XML_FILES:
                text = data.decode("utf-8")
                for src, dst in _FONT_MAP.items():
                    text = text.replace(src, dst)
                if item.filename in ("word/document.xml", "word/styles.xml"):
                    text = _scale_line_spacing(text)
                data = text.encode("utf-8")
            zout.writestr(item, data)
    return out_buf.getvalue()


def convert_docx_to_pdf(docx_path: Path) -> Path:
    """POST *docx_path* to the PDF microservice, save result, return local PDF path."""
    base_url = os.environ.get("PDF_SERVICE_URL", "").strip().rstrip("/")
    if not base_url:
        raise RuntimeError("PDF_SERVICE_URL env var is not set.")

    url = f"{base_url}/convert"
    # Patch fonts so LibreOffice uses metric-compatible equivalents
    docx_bytes = _patch_fonts(docx_path)
    filename = docx_path.name

    # Build multipart/form-data body
    boundary = "PdfServiceBoundary"
    body = (
        f"--{boundary}\r\n"
        f'Content-Disposition: form-data; name="file"; filename="{filename}"\r\n'
        f"Content-Type: application/vnd.openxmlformats-officedocument.wordprocessingml.document\r\n\r\n"
    ).encode() + docx_bytes + f"\r\n--{boundary}--\r\n".encode()

    req = urllib.request.Request(
        url,
        data=body,
        headers={"Content-Type": f"multipart/form-data; boundary={boundary}"},
        method="POST",
    )

    try:
        with urllib.request.urlopen(req, timeout=90) as resp:
            pdf_bytes = resp.read()
    except urllib.error.HTTPError as e:
        raise RuntimeError(f"PDF service error {e.code}: {e.read().decode('utf-8', errors='ignore')}")

    pdf_path = docx_path.with_suffix(".pdf")
    pdf_path.write_bytes(pdf_bytes)
    return pdf_path
