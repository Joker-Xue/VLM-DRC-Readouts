"""Region helpers used to build the CNN evidence cache (copied from the research workspace)."""
import numpy as np


def _box_mean_argmax(arr: np.ndarray, radius: int) -> tuple[int, int, float]:
    """Return ``(cx, cy, peak_smoothed_value)`` of the box-mean argmax."""
    h, w = arr.shape
    pad = np.pad(arr, radius, mode="edge")
    cum = pad.cumsum(0).cumsum(1)
    cum = np.pad(cum, ((1, 0), (1, 0)))
    side = 2 * radius + 1
    s = (
        cum[side:, side:]
        - cum[:-side, side:]
        - cum[side:, :-side]
        + cum[:-side, :-side]
    )
    smoothed = s / (side * side)
    flat = int(smoothed.argmax())
    cy, cx = divmod(flat, w)
    return int(cx), int(cy), float(smoothed[cy, cx])


def _crop_mean(arr: np.ndarray, cx: int, cy: int, radius: int) -> float:
    x0, x1 = max(0, cx - radius), min(arr.shape[-1], cx + radius + 1)
    y0, y1 = max(0, cy - radius), min(arr.shape[-2], cy + radius + 1)
    return float(arr[..., y0:y1, x0:x1].mean())
