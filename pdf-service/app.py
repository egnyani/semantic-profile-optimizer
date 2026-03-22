import io
import os
import re
import subprocess
import tempfile
import zipfile
from pathlib import Path

from flask import Flask, request, send_file, jsonify
from pypdf import PdfReader

app = Flask(__name__)

# Font substitutions: Microsoft → LibreOffice metric-compatible equivalents
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
}

# Try descending line scales until the PDF fits on 1 page
_SCALE_ATTEMPTS = [1.11, 1.08, 1.05, 1.0]


def _patch_docx(docx_bytes: bytes, line_scale: float) -> bytes:
    """Swap fonts and scale line spacing in DOCX XML."""

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
    with zipfile.ZipFile(io.BytesIO(docx_bytes), "r") as zin, \
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


def _pdf_page_count(pdf_path: Path) -> int:
    return len(PdfReader(str(pdf_path)).pages)


def _convert(docx_path: Path, out_dir: str) -> Path:
    result = subprocess.run(
        ["libreoffice", "--headless", "--convert-to", "pdf", "--outdir", out_dir, str(docx_path)],
        capture_output=True, text=True, timeout=60,
    )
    if result.returncode != 0:
        raise RuntimeError(result.stderr)
    pdf_path = docx_path.with_suffix(".pdf")
    if not pdf_path.exists():
        raise RuntimeError("PDF not generated")
    return pdf_path


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/convert")
def convert():
    if "file" not in request.files:
        return jsonify({"error": "No file uploaded"}), 400

    f = request.files["file"]
    if not f.filename.endswith(".docx"):
        return jsonify({"error": "Only .docx files accepted"}), 400

    raw_bytes = f.read()

    with tempfile.TemporaryDirectory() as tmp:
        # Try each scale in descending order; stop at the first that fits 1 page
        for scale in _SCALE_ATTEMPTS:
            patched = _patch_docx(raw_bytes, scale)
            docx_path = Path(tmp) / f.filename
            docx_path.write_bytes(patched)

            try:
                pdf_path = _convert(docx_path, tmp)
            except RuntimeError as e:
                return jsonify({"error": str(e)}), 500

            if _pdf_page_count(pdf_path) == 1:
                return send_file(
                    str(pdf_path),
                    mimetype="application/pdf",
                    as_attachment=True,
                    download_name=pdf_path.name,
                )

            # Overflowed — remove PDF and try a smaller scale
            pdf_path.unlink(missing_ok=True)

        # All scales overflowed — return the unscaled version as fallback
        patched = _patch_docx(raw_bytes, 1.0)
        docx_path.write_bytes(patched)
        pdf_path = _convert(docx_path, tmp)
        return send_file(str(pdf_path), mimetype="application/pdf",
                         as_attachment=True, download_name=pdf_path.name)


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", 8000)))
