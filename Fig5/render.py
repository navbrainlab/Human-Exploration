"""Reproduce narrative Fig. 5."""

from plotting import core as renderer


def figure_rows():
    letters = {"g": "c", "h": "d", "i": "e"}
    parts = ["ab__upper", "g__g", "hi__hi"]
    original = renderer.current(4, letters, parts=parts)
    return [
        renderer.row(original, 0, 3000, name="old4ab"),
        renderer.row(original, 3000, 4450, name="old4g"),
        renderer.row(original, 4450, None, name="old4hi"),
    ]


if __name__ == "__main__":
    renderer.main(5, figure_rows, gap=4)
