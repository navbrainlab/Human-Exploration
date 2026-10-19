"""Lossless, native-resolution assembly of approved scientific figure panels."""
from __future__ import annotations

import argparse
import hashlib
import io
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "source"
OUTPUT = ROOT / "output/fig1.png"
DPI = 600
# The original PNG has a completely blank separator at rows 2895..3095.
# Its results start below this cut; no marks, letters or labels are cropped.
RESULTS_TOP_PX = 3024


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read_sources():
    images = {}
    for name in ("F1ab.png", "fig1_rebuilt_bar_box.png"):
        with Image.open(SOURCE / name) as im:
            assert im.mode == "RGB"
            assert all(abs(d - DPI) < .01 for d in im.info["dpi"])
            images[name] = np.array(im)
    top, previous = images.values()
    assert top.shape == (1913, 4251, 3)
    assert previous.shape == (4410, 4251, 3)
    assert np.all(previous[RESULTS_TOP_PX-5:RESULTS_TOP_PX+5] == 255)
    task_qa = json.loads((SOURCE / "F1ab_verification.json").read_text())
    old_qa = json.loads((SOURCE / "fig1_rebuilt_bar_box_verification.json").read_text())
    assert sha(SOURCE / "F1ab.png") == task_qa["output_sha256"]
    assert sha(SOURCE / "fig1_rebuilt_bar_box.png") == old_qa["sha256"]
    return top, previous


def pixel_audit(path):
    top, previous = read_sources()
    lower = lower_results(previous)
    with Image.open(path) as im:
        actual = np.asarray(im)
        assert im.mode == "RGB"
        assert all(abs(d - DPI) < .01 for d in im.info["dpi"])
    assert actual.shape == (len(top)+len(lower), top.shape[1], 3)
    np.testing.assert_array_equal(actual[:len(top)], top)
    np.testing.assert_array_equal(actual[len(top):], lower)
    return dict(top_changed_pixels=0, results_changed_pixels=0,
                pixel_dimensions=[actual.shape[1], actual.shape[0]],
                dpi=DPI, public_letters=["a", "b", "c", "e"])


def lower_results(previous):
    updated=SOURCE/'results_axis_spacing.png'
    if not updated.exists():
        return previous[RESULTS_TOP_PX:]
    report=json.loads((SOURCE/'results_axis_spacing.json').read_text())
    assert sha(updated)==report['output_sha256']
    assert sha(SOURCE/'fig1_rebuilt_bar_box.png')==report['original_results_sha256']
    with Image.open(updated) as image:
        assert image.mode=='RGB' and all(abs(d-DPI)<.01 for d in image.info['dpi'])
        result=np.array(image)
    assert result.shape==previous[RESULTS_TOP_PX:].shape
    return result


def source_manifest():
    return {p.name: sha(p) for p in sorted(SOURCE.iterdir()) if p.is_file()}


def render(replace=False):
    if OUTPUT.exists() and not replace:
        raise FileExistsError(f"Refusing to overwrite {OUTPUT}")
    top, previous = read_sources()
    lower = lower_results(previous)
    height, width = len(top)+len(lower), top.shape[1]
    fig = plt.figure(figsize=(width/DPI, height/DPI), dpi=DPI, facecolor="white")
    # figimage uses native pixels, unlike an imshow axes that can resample panels.
    fig.figimage(lower, xo=0, yo=0, origin="upper", resize=False)
    fig.figimage(top, xo=0, yo=len(lower), origin="upper", resize=False)
    buffer = io.BytesIO()
    fig.savefig(buffer, format="png", dpi=DPI, facecolor="white")
    plt.close(fig)
    buffer.seek(0)
    with Image.open(buffer) as im:
        im.convert("RGB").save(OUTPUT, dpi=(DPI, DPI))
    report = pixel_audit(OUTPUT)
    native = json.loads((SOURCE / "fig1_rebuilt_bar_box_native_style.json").read_text())
    report.update(
        status="PASS_lossless_panel_assembly",
        assembly="600-dpi Matplotlib figimage; no resampling, no image synthesis",
        old_results_crop=[0, RESULTS_TOP_PX, previous.shape[1], len(previous)],
        result_translation_px=[0, len(top)-RESULTS_TOP_PX],
        canvas_mm=[width*25.4/DPI, height*25.4/DPI],
        source_hashes=source_manifest(),
        output_sha256=sha(OUTPUT), renderer_sha256=sha(__file__),
        original_files_untouched=True,
        no_analysis_recomputed=True,
        inherited_font_sizes_pt=[5, 6, 8],
        inherited_tests=native["tests"],
        inherited_sample_counts=native["personal_best_counts"],
        qa_scope="Pixel identity with approved 600-dpi sources; source native QA retained, no new PDF audit",
    )
    if (SOURCE/'results_axis_spacing.json').exists():
        report['native_results_revision']=json.loads((SOURCE/'results_axis_spacing.json').read_text())
    (ROOT / "qa").mkdir(exist_ok=True)
    (ROOT / "qa/verification.json").write_text(json.dumps(report, indent=2)+"\n")
    with Image.open(OUTPUT) as im:
        im.thumbnail((1800, 1800))
        im.save(ROOT / "qa/preview.png")
    verify()
    print(OUTPUT)


def verify():
    saved = json.loads((ROOT / "qa/verification.json").read_text())
    assert sha(OUTPUT) == saved["output_sha256"]
    assert sha(__file__) == saved["renderer_sha256"]
    assert source_manifest() == saved["source_hashes"]
    for key, value in pixel_audit(OUTPUT).items():
        assert saved[key] == value, key
    # Verify the two original files if this package is still in the project.
    for name in ("F1ab.png", "fig1_rebuilt_bar_box.png"):
        original = ROOT.parent / name
        if original.exists():
            assert sha(original) == saved["source_hashes"][name], name
    print("PASS: a/b and lower results match approved sources pixel-for-pixel; original source files unchanged")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--verify", action="store_true")
    parser.add_argument("--replace", action="store_true",
                        help="Overwrite this variant only after explicit author approval")
    args = parser.parse_args()
    verify() if args.verify else render(replace=args.replace)
