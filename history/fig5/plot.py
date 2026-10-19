"""Render this figure using only files within this folder."""
import argparse
import importlib
import os
from pathlib import Path
import sys
import tempfile

ROOT = Path(__file__).resolve().parent

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=ROOT / "output")
    args = parser.parse_args()
    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("MPLBACKEND", "Agg")
    os.environ.setdefault("MPLCONFIGDIR", tempfile.mkdtemp(prefix="figure-mpl-"))
    sys.path.insert(0, str(ROOT / "code"))
    renderer = importlib.import_module("render_fig5_evidence_layout")
    layout = renderer.layout
    if "fig5" == "fig5":
        layout.OUTPUT_PATH = output / "fig5_nhb_evidence_layout.png"
        layout.DATA_DIR = output
    else:
        layout.OUTPUT_DIR = output
        layout.MANIFEST_DIR = output
        renderer.base.AUDIT = output
    renderer.main()

if __name__ == "__main__":
    main()
