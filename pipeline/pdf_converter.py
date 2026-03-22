"""
Convert a DOCX file to PDF via the LibreOffice microservice hosted on Render.com.

Set the env var PDF_SERVICE_URL to the Render service URL, e.g.:
  PDF_SERVICE_URL=https://resume-pdf-converter.onrender.com
"""

from __future__ import annotations

import os
import urllib.request
import urllib.error
from pathlib import Path


def convert_docx_to_pdf(docx_path: Path) -> Path:
    """POST *docx_path* to the PDF microservice, save result, return local PDF path."""
    base_url = os.environ.get("PDF_SERVICE_URL", "").strip().rstrip("/")
    if not base_url:
        raise RuntimeError("PDF_SERVICE_URL env var is not set.")

    url = f"{base_url}/convert"
    docx_bytes = docx_path.read_bytes()
    filename = docx_path.name

    # Build a minimal multipart/form-data body
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
