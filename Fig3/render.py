"""Reproduce narrative Fig. 3."""

from plotting import core as renderer


def figure_rows():
    original = renderer.current(3)
    return [renderer.row(original, name="old3all")]


if __name__ == "__main__":
    renderer.main(3, figure_rows, gap=4)
