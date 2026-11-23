"""Reproduce narrative Fig. 6."""

from plotting import core as renderer
from plotting import experience_relayout_20261117


def figure_rows():
    return experience_relayout_20261117.rows(renderer)


if __name__ == "__main__":
    renderer.main(6, figure_rows, gap=4)
