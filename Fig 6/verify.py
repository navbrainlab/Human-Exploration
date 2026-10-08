"""Verify all supplied files, then compare a re-render to the reference PNG."""
import argparse
import csv
from hashlib import sha256
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--inputs-only", action="store_true")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "output")
    args = parser.parse_args()
    with (ROOT / "manifest.csv").open(newline="", encoding="utf-8") as handle:
        records = list(csv.DictReader(handle))
    failures = [record["path"] for record in records
                if not (ROOT / record["path"]).is_file()
                or sha256((ROOT / record["path"]).read_bytes()).hexdigest() != record["sha256"]]
    report = {"manifest_file_count": len(records), "file_errors": failures}
    if failures:
        print(json.dumps(report, indent=2))
        raise SystemExit(1)
    if not args.inputs_only:
        import numpy as np
        from PIL import Image
        reference = ROOT / "reference/fig6_full_joint90.png"
        rendered = args.output_dir.resolve() / "fig6_full_joint90.png"
        with Image.open(reference) as image:
            expected = np.array(image.convert("RGB"), dtype=np.int16)
        with Image.open(rendered) as image:
            actual = np.array(image.convert("RGB"), dtype=np.int16)
            report["dpi"] = image.info.get("dpi")
        report["shape_equal"] = expected.shape == actual.shape
        report["pixel_equal"] = bool(report["shape_equal"] and np.array_equal(expected, actual))
        if report["shape_equal"]:
            difference = np.abs(expected - actual)
            report["changed_pixels"] = int(np.any(difference, axis=2).sum())
            report["max_channel_difference"] = int(difference.max())
        report["png_sha256_equal"] = sha256(reference.read_bytes()).hexdigest() == sha256(rendered.read_bytes()).hexdigest()
        qa = json.loads((args.output_dir / "fig6_full_joint90_render_qa.json").read_text(encoding="utf-8"))
        report["text_collisions"] = qa["text_collisions"]
        report["clipped_text"] = qa["clipped_text"]
        alignment = json.loads((args.output_dir / "fig6_full_joint90_alignment.json").read_text(encoding="utf-8"))
        report["alignment"] = alignment["verdict"]
        (args.output_dir / "reproduction_check.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))
    if not args.inputs_only and (not report["pixel_equal"] or report["text_collisions"]
                                 or report["clipped_text"] or report["alignment"] != "PASS"):
        raise SystemExit(1)

if __name__ == "__main__":
    main()
