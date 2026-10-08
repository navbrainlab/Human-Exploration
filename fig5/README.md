# fig5: standalone reproduction

Reproduces `fig5_nhb_evidence_layout.png`. Python 3.11 is recommended (Windows, macOS or Linux).
No mini_DG checkout, Codex skills, GPU, Jupyter, or original absolute paths are needed.
Each figure folder is independent; it may be copied without its sibling folder.

## Run

From this folder, in a Python 3.11 virtual environment:

```bash
python -m pip install -r requirements-lock.txt
python verify.py --inputs-only
python plot.py
python verify.py
```

The result and render/audit files are written to `output/`, never to `reference/`.
You may run `plot.py` by its full path from any working directory. To select a
different output directory, pass `--output-dir PATH` to both `plot.py` and `verify.py`.
Only PNG is exported, using the original 600 dpi, dimensions, palette and layout.
`reference/` contains the exact requested original PNG for comparison.

## Contents and provenance

- `data/`: exact plotting tables, copied without changing values or precision.
- `code/`: editable renderer/layout/helpers and a local alignment auditor.
- `assets/`: required image assets (Fig. 5 only).
- `caption.md`, `methods.md`: retained scientific interpretation and limitations.
- `results_and_caption.md`: current manuscript Results and matching caption.
- `manifest.csv`: relative file paths, SHA-256 hashes and source provenance.
- `provenance/`: original audit/manifest and the tested dependency versions.
- `reproduction_verified.json`: isolated re-render verification, when supplied.

Fig. 5 includes the gaze composites in both PNG and SVG, their background, and the sampled-trial CSV. SVG files are input assets, not new exports. The saved density display is qualitative, as in the original figure.

The original provenance JSON may mention historical absolute source locations;
these are documentary records, not files accessed by the portable renderer.
`verify.py` checks both file integrity and pixel equality to the reference; the
output contains `reproduction_check.json`. Different operating systems/rendering
libraries can introduce small antialiasing differences even with the same data.
Use the pinned environment to minimize them; the data hashes must always match.
If deliberately editing the source or data, its integrity check will fail until
the manifest is regenerated. Rendering itself does not require matching hashes.
