"""Reproduce narrative Fig. 7."""

from plotting import core as renderer


def figure_rows():
    original = renderer.current(6)
    return [
        renderer.row(original, 0, 4500, name="old6a_g"),
        renderer.row(original, 4500, None, name="old6hi_reference"),
    ]


if __name__ == "__main__":
    renderer.main(7, figure_rows, gap=4)
