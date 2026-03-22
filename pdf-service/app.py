import os
import subprocess
import tempfile
from pathlib import Path

from flask import Flask, request, send_file, jsonify

app = Flask(__name__)

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

    with tempfile.TemporaryDirectory() as tmp:
        docx_path = Path(tmp) / f.filename
        f.save(str(docx_path))

        result = subprocess.run(
            ["libreoffice", "--headless", "--convert-to", "pdf", "--outdir", tmp, str(docx_path)],
            capture_output=True, text=True, timeout=60,
        )

        if result.returncode != 0:
            return jsonify({"error": result.stderr}), 500

        pdf_path = docx_path.with_suffix(".pdf")
        if not pdf_path.exists():
            return jsonify({"error": "PDF not generated"}), 500

        return send_file(str(pdf_path), mimetype="application/pdf", as_attachment=True,
                         download_name=pdf_path.name)


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", 8000)))
