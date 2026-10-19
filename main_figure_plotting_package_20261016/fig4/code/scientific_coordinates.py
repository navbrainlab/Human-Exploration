"""Fingerprint quantitative data-space marks independently of publication style."""
from hashlib import sha256
import numpy as np
from matplotlib.collections import PathCollection, PolyCollection
from matplotlib.container import ErrorbarContainer
from matplotlib.patches import Rectangle


def array_digest(array):
    values = np.asarray(array, dtype=np.float64)
    return {"shape": list(values.shape), "sha256": sha256(values.tobytes()).hexdigest()}


def quantitative_coordinates(axes):
    result = {}
    for name, axis in axes.items():
        traces = []
        for line in axis.lines:
            xy = line.get_xydata()
            if len(xy) == 4 and np.allclose(xy[:, 0], [0, 0, 1, 1]):
                continue  # Significance brackets are annotation geometry.
            if len(xy) >= 4:
                traces.append(array_digest(xy))
        errorbars = []
        for container in axis.containers:
            if isinstance(container, ErrorbarContainer):
                mean_artist, _, error_collections = container.lines
                errorbars.append({
                    "mean": array_digest(mean_artist.get_xydata()) if mean_artist is not None else None,
                    "intervals": [array_digest(collection.get_segments()) for collection in error_collections],
                })
        result[name] = {
            "points": [array_digest(c.get_offsets()) for c in axis.collections if isinstance(c, PathCollection)],
            "bands": [[array_digest(p.vertices) for p in c.get_paths()]
                      for c in axis.collections if isinstance(c, PolyCollection)],
            "traces": traces, "errorbars": errorbars,
            "bar_heights": [float(p.get_height()) for p in axis.patches if isinstance(p, Rectangle)],
        }
    return result
