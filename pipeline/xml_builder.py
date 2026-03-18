"""
XML-patching DOCX builder.

Instead of rebuilding from scratch (builder.py), this module patches the
original resume DOCX in-place: opens the ZIP, rewrites only the matching
bullet paragraph XML runs, and saves a new ZIP.  All fonts, colours, bullet
styles, margins, headers, and footers are preserved exactly.
"""

from __future__ import annotations

import copy
import re
import zipfile
from pathlib import Path
from typing import Any

from lxml import etree


_NS = "http://schemas.openxmlformats.org/wordprocessingml/2006/main"
_W = f"{{{_NS}}}"
_XML_SPACE = "{http://www.w3.org/XML/1998/namespace}space"


def _normalize(text: str) -> str:
    """Lowercase, strip bullets and extra whitespace for fuzzy matching."""
    text = re.sub(r"^[\-\*\u2022\u2013]\s*", "", text)
    return re.sub(r"\s+", " ", text).strip().lower()


def _para_text(p: etree._Element) -> str:
    """Return concatenated run text of a paragraph element."""
    return "".join(
        "".join(t.text or "" for t in r.findall(f"{_W}t"))
        for r in p.findall(f".//{_W}r")
    )


def _replace_para_text(p: etree._Element, new_text: str) -> None:
    """Replace all runs in *p* with a single run carrying *new_text*."""
    runs = p.findall(f"{_W}r")
    # Find a template run to copy formatting (rPr)
    template_run = next(
        (r for r in runs if any(t.text for t in r.findall(f"{_W}t"))),
        None,
    )
    if template_run is None:
        return  # can't patch a run-less paragraph

    # Remove all runs
    for r in list(p.findall(f"{_W}r")):
        p.remove(r)

    # Build replacement run
    new_run = copy.deepcopy(template_run)
    for t in new_run.findall(f"{_W}t"):
        new_run.remove(t)
    t_el = etree.SubElement(new_run, f"{_W}t")
    t_el.text = new_text
    t_el.set(_XML_SPACE, "preserve")
    p.append(new_run)


def xml_patch_docx(
    source_path: str | Path,
    rewrites: list[dict[str, str]],
    output_path: str | Path,
) -> Path:
    """
    Copy *source_path* DOCX and patch bullet text via XML rewriting.

    Parameters
    ----------
    source_path:
        Original resume DOCX (preserved as template).
    rewrites:
        List of ``{"original": str, "rewritten": str}`` dicts from the pipeline.
    output_path:
        Where to save the tailored DOCX.

    Returns
    -------
    Path of the written output DOCX.
    """
    source_path = Path(source_path)
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Build normalised-key → new-text lookup
    rewrite_map: dict[str, str] = {}
    for rw in rewrites:
        orig = rw.get("original", "")
        new = rw.get("rewritten", "")
        if orig and new:
            key = _normalize(orig)
            rewrite_map[key] = new
            # Also add a 60-char prefix key for robustness
            rewrite_map[key[:60]] = new

    # Read all files from the source ZIP
    all_files: dict[str, bytes] = {}
    with zipfile.ZipFile(source_path, "r") as zin:
        for name in zin.namelist():
            all_files[name] = zin.read(name)

    # Parse and patch document.xml
    xml_bytes = all_files["word/document.xml"]
    root = etree.fromstring(xml_bytes)
    body = root.find(f"{_W}body")
    if body is None:
        raise RuntimeError("Malformed DOCX: no <w:body> found")

    paras = body.findall(f".//{_W}p")
    patched = 0
    for p in paras:
        raw = _para_text(p)
        if not raw.strip():
            continue

        norm = _normalize(raw)
        # Try full key, then 60-char prefix
        new_text = rewrite_map.get(norm) or rewrite_map.get(norm[:60])
        if new_text:
            _replace_para_text(p, new_text)
            patched += 1

    new_xml = etree.tostring(root, xml_declaration=True, encoding="UTF-8", standalone=True)
    all_files["word/document.xml"] = new_xml

    # Write new DOCX
    with zipfile.ZipFile(output_path, "w", zipfile.ZIP_DEFLATED) as zout:
        for name, data in all_files.items():
            zout.writestr(name, data)

    return output_path
