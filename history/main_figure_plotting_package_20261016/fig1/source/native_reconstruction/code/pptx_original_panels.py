"""Render and reuse the complete supplied PowerPoint panels without re-layout."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import subprocess
import tempfile
from xml.sax.saxutils import escape

import matplotlib.image as mpimg

ASSETS = Path(__file__).resolve().parents[1] / "assets/figure1_pptx_original"
PANEL_WIDTH_MM = 83.
SLIDE_WIDTH_MM = 338.71077777777776
CROP_HEIGHT_MM = 474.
PANEL_HEIGHT_MM = PANEL_WIDTH_MM * CROP_HEIGHT_MM / SLIDE_WIDTH_MM


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def prepare():
    """Keep the PPTX intact; render vector/text at twice the final PNG resolution."""
    from PIL import Image
    from pptx import Presentation

    source = ASSETS / "source.pptx"
    with tempfile.TemporaryDirectory(prefix="fig1_pptx_render_") as temp:
        work = Path(temp)
        config = work / "fonts.conf"
        config.write_text(
            '<?xml version="1.0"?><!DOCTYPE fontconfig SYSTEM "fonts.dtd">'
            '<fontconfig><include ignore_missing="yes">/etc/fonts/fonts.conf</include>'
            f'<dir>{escape(str(ASSETS / "fonts"))}</dir>'
            f'<cachedir>{escape(str(work / "fontcache"))}</cachedir>'
            '<alias binding="strong"><family>Calibri</family><prefer><family>Carlito</family></prefer></alias>'
            '<alias binding="strong"><family>Calibri Light</family><prefer><family>Carlito</family></prefer></alias>'
            '<alias binding="strong"><family>-apple-system</family><prefer><family>Carlito</family></prefer></alias>'
            '</fontconfig>')
        environment = {**os.environ, "FONTCONFIG_FILE": str(config),
                       "GSETTINGS_BACKEND": "memory", "SAL_USE_VCLPLUGIN": "svp"}
        subprocess.run(["libreoffice", f"-env:UserInstallation={(work / 'profile').as_uri()}",
                        "--headless", "--convert-to", "pdf", "--outdir", str(work),
                        str(source)], env=environment, check=True, timeout=90)
        document = Presentation(source)
        assert len(document.slides) == 2
        width_mm, height_mm = document.slide_width / 36000, document.slide_height / 36000
        pdf = work / "source.pdf"
        raster_dpi = 1200 * PANEL_WIDTH_MM / width_mm
        width_px = math.ceil(PANEL_WIDTH_MM / 25.4 * 1200)
        height_px = math.ceil(PANEL_HEIGHT_MM / 25.4 * 1200)
        subprocess.run(["pdftoppm", "-r", str(raster_dpi), "-x", "0", "-y", "0",
                        "-W", str(width_px), "-H", str(height_px), "-png",
                        str(pdf), str(work / "slide")], check=True, timeout=90)
        text_pages = subprocess.check_output(["pdftotext", str(pdf), "-"], text=True).split("\f")
        font_report = subprocess.check_output(["pdffonts", str(pdf)], text=True)
        manifest = {
            "pptx_name": "figure1 panal ab.pptx", "pptx_sha256": digest(source),
            "source_category": "Illustrative task and group-design slides supplied by the author",
            "method": "Complete original slides, original geometry and text; only blank lower slide area cropped",
            "panel_width_mm": PANEL_WIDTH_MM, "panel_height_mm": PANEL_HEIGHT_MM,
            "raster_effective_dpi": 1200, "final_figure_dpi": 600,
            "source_font_substitutions": {"Calibri": "Carlito (metric compatible)",
                "Calibri Light": "Carlito", "Arial": "Liberation Sans (metric compatible)",
                "-apple-system": "Carlito"},
            "typography_scope": "PPTX typography is preserved by user request; native 5/6/8-pt audit covers only surrounding figure and quantitative panels",
            "powerpoint_pixel_identity_claimed": False,
            "pdf_font_report": font_report,
            "font_files_sha256": {p.name: digest(p) for p in (ASSETS / "fonts").glob("*.ttf")},
            "images": {},
        }
        for index in range(2):
            target = ASSETS / f"slide_{index + 1}.png"
            with Image.open(work / f"slide-{index + 1}.png") as raster:
                raster.convert("RGB").save(target, dpi=(1200, 1200))
                pixel_size = list(raster.size)
            manifest["images"][target.name] = {
                "slide": index + 1, "sha256": digest(target),
                "pixel_size": pixel_size, "crop_mm": [0, 0, width_mm, CROP_HEIGHT_MM],
                "source_page_mm": [width_mm, height_mm],
                "source_text": text_pages[index],
            }
        (ASSETS / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    return manifest


def check_assets():
    manifest = json.loads((ASSETS / "manifest.json").read_text())
    assert digest(ASSETS / "source.pptx") == manifest["pptx_sha256"]
    assert len(manifest["images"]) == 2
    for name, spec in manifest["images"].items():
        assert digest(ASSETS / name) == spec["sha256"], name
    return manifest


def draw(ax, slide):
    image = mpimg.imread(ASSETS / f"slide_{slide}.png")
    ax.imshow(image, extent=(0, PANEL_WIDTH_MM, PANEL_HEIGHT_MM, 0),
              aspect="equal", interpolation="hanning", zorder=1)
    ax.set(xlim=(0, PANEL_WIDTH_MM), ylim=(PANEL_HEIGHT_MM, 0))
    ax.set_axis_off()


def task1(ax):
    draw(ax, 1)


def task2(ax):
    draw(ax, 2)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prepare", action="store_true")
    args = parser.parse_args()
    print(json.dumps(prepare() if args.prepare else check_assets(), indent=2))
