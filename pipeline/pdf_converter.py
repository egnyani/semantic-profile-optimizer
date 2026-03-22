"""
Convert a DOCX file to PDF using the Google Drive REST API directly.

Uses only google-auth (lightweight) + stdlib urllib — no google-api-python-client.

Flow:
  1. Upload the DOCX to Google Drive (converting to Google Docs format)
  2. Export the Google Doc as PDF
  3. Delete the file from Drive
  4. Return the local PDF path

Setup:
  - Create a Google Cloud project and enable the Google Drive API
  - Create a service account and download the JSON key
  - Set the env var GOOGLE_SERVICE_ACCOUNT_JSON to the full contents of that JSON file
"""

from __future__ import annotations

import io
import json
import os
import urllib.request
import urllib.error
from pathlib import Path


_DRIVE_UPLOAD = "https://www.googleapis.com/upload/drive/v3/files?uploadType=multipart&fields=id"
_DRIVE_EXPORT  = "https://www.googleapis.com/drive/v3/files/{id}/export?mimeType=application%2Fpdf"
_DRIVE_DELETE  = "https://www.googleapis.com/drive/v3/files/{id}"


def _get_access_token(sa_info: dict) -> str:
    """Exchange service-account credentials for a short-lived OAuth2 access token."""
    import time
    import json as _json

    # google-auth is a small package — import here to keep startup fast
    from google.oauth2 import service_account

    creds = service_account.Credentials.from_service_account_info(
        sa_info,
        scopes=["https://www.googleapis.com/auth/drive.file"],
    )
    creds.refresh(google.auth.transport.requests.Request())
    return creds.token


def _get_token(sa_info: dict) -> str:
    """Get OAuth2 bearer token via google-auth."""
    import google.auth.transport.requests
    from google.oauth2 import service_account

    creds = service_account.Credentials.from_service_account_info(
        sa_info,
        scopes=["https://www.googleapis.com/auth/drive.file"],
    )
    creds.refresh(google.auth.transport.requests.Request())
    return creds.token


def _api(method: str, url: str, token: str, body: bytes | None = None,
         content_type: str | None = None) -> bytes:
    """Make an authorized Drive API call and return raw response bytes."""
    headers = {"Authorization": f"Bearer {token}"}
    if content_type:
        headers["Content-Type"] = content_type
    req = urllib.request.Request(url, data=body, headers=headers, method=method)
    try:
        with urllib.request.urlopen(req) as resp:
            return resp.read()
    except urllib.error.HTTPError as e:
        raise RuntimeError(f"HTTP {e.code} from Google API: {e.read().decode('utf-8', errors='ignore')}")


def convert_docx_to_pdf(docx_path: Path) -> Path:
    """Upload *docx_path* to Google Drive, export as PDF, delete, return local PDF path.

    Raises RuntimeError if GOOGLE_SERVICE_ACCOUNT_JSON is not set or conversion fails.
    """
    sa_json = os.environ.get("GOOGLE_SERVICE_ACCOUNT_JSON", "").strip()
    if not sa_json:
        raise RuntimeError(
            "GOOGLE_SERVICE_ACCOUNT_JSON env var is not set."
        )

    sa_info = json.loads(sa_json)
    token = _get_token(sa_info)

    # 1. Upload DOCX as multipart — tell Drive to convert to Google Docs format
    boundary = "ResumeConverterBoundary"
    metadata = json.dumps({
        "name": docx_path.stem,
        "mimeType": "application/vnd.google-apps.document",
    }).encode()
    docx_bytes = docx_path.read_bytes()

    body = (
        f"--{boundary}\r\n"
        f"Content-Type: application/json; charset=UTF-8\r\n\r\n"
    ).encode() + metadata + (
        f"\r\n--{boundary}\r\n"
        f"Content-Type: application/vnd.openxmlformats-officedocument.wordprocessingml.document\r\n\r\n"
    ).encode() + docx_bytes + f"\r\n--{boundary}--".encode()

    upload_resp = _api(
        "POST", _DRIVE_UPLOAD, token,
        body=body,
        content_type=f"multipart/related; boundary={boundary}",
    )
    file_id = json.loads(upload_resp)["id"]

    try:
        # 2. Export as PDF
        pdf_bytes = _api("GET", _DRIVE_EXPORT.format(id=file_id), token)
    finally:
        # 3. Always delete from Drive
        try:
            _api("DELETE", _DRIVE_DELETE.format(id=file_id), token)
        except Exception:
            pass

    # 4. Save PDF next to the DOCX
    pdf_path = docx_path.with_suffix(".pdf")
    pdf_path.write_bytes(pdf_bytes)
    return pdf_path
