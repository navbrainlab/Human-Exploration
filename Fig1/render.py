"""Reproduce narrative Fig. 1."""

from plotting import core as renderer


def figure_rows():
    original = renderer.current(1, parts=["top"])
    task_centers = renderer.QA["fig1_task_column_centers_px"]
    return [
        renderer.row(original, name="old1ab"),
        renderer.task_learning_row(task_centers),
    ]


if __name__ == "__main__":
    renderer.main(1, figure_rows, gap=1)
