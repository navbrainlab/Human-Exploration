"""Reproduce narrative Fig. 2."""

from plotting import core as renderer


def figure_rows():
    original = renderer.current(2)
    return [renderer.row(original, name="old2all")]


if __name__ == "__main__":
    renderer.main(2, figure_rows, gap=4)
