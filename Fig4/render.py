"""Reproduce narrative Fig. 4."""

from plotting import core as renderer


def figure_rows():
    letters = {"a": "f", "b": "g", "c": "a", "d": "b", "e": "c",
               "f": "d", "g": "e", "h": "h", "i": "i"}
    original = renderer.current(5, letters)
    return [
        renderer.row(original, 2550, 3635, name="old5cd"),
        renderer.row(original, 3635, 4620, name="old5efg"),
        renderer.row(original, 0, 2550, name="old5ab"),
    ]


if __name__ == "__main__":
    renderer.main(4, figure_rows, gap=4)
