"""Exact layout classifier for auditing the illustrative action."""
import numpy as np

def is_dis_action(action) -> bool:
    """Return whether an action has a row-constant dimension with levels 1, 2, 3."""
    array = np.asarray(action, dtype=int)
    if array.ndim != 3 or array.shape[:2] != (3, 3):
        raise ValueError(f"action must have shape (3, 3, dims), got {array.shape}")
    for dim in range(array.shape[2]):
        rows = array[:, :, dim]
        if np.all(rows == rows[:, :1]) and set(rows[:, 0].tolist()) == {1, 2, 3}:
            return True
    return False
