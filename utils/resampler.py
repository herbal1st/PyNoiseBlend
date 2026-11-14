"""Zero-dependency vectorized 2D bilinear resampling utilities."""

from dataclasses import dataclass
from functools import lru_cache
import numpy as np
from numpy.typing import NDArray


@dataclass(frozen=True, slots=True)
class ResamplePlan:
    """Cached separable bilinear resampling indices and work buffers."""

    x_low: NDArray[np.int32]
    x_high: NDArray[np.int32]
    x_weight: NDArray[np.float32]
    y_low: NDArray[np.int32]
    y_high: NDArray[np.int32]
    y_weight: NDArray[np.float32]
    horiz_low: NDArray[np.float32]
    horiz_high: NDArray[np.float32]


def bilinear_resample_2d(
    source: NDArray[np.float32],
    target_height: int,
    target_width: int,
) -> NDArray[np.float32]:
    """Resamples a 2D float32 array to target dimensions bilinearly."""
    src_h, src_w = source.shape
    if src_h == target_height and src_w == target_width:
        return source.copy()

    out = np.empty((target_height, target_width), dtype=np.float32)
    work = np.empty((target_height, target_width), dtype=np.float32)
    bilinear_resample_into(source, target_height, target_width, out, work)
    return out


def bilinear_resample_into(
    source: NDArray[np.float32],
    target_height: int,
    target_width: int,
    out: NDArray[np.float32],
    work: NDArray[np.float32],
) -> None:
    """Resamples 2D float32 array bilinearly into preallocated buffers."""
    src_h, src_w = source.shape
    if src_h == target_height and src_w == target_width:
        np.copyto(out, source)
        return

    plan = _get_resample_plan(src_h, src_w, target_height, target_width)
    np.take(source, plan.x_low, axis=1, out=plan.horiz_low)
    np.take(source, plan.x_high, axis=1, out=plan.horiz_high)
    np.subtract(plan.horiz_high, plan.horiz_low, out=plan.horiz_high)
    np.multiply(plan.horiz_high, plan.x_weight, out=plan.horiz_high)
    np.add(plan.horiz_low, plan.horiz_high, out=plan.horiz_low)

    np.take(plan.horiz_low, plan.y_low, axis=0, out=out)
    np.take(plan.horiz_low, plan.y_high, axis=0, out=work)
    np.subtract(work, out, out=work)
    np.multiply(work, plan.y_weight, out=work)
    np.add(out, work, out=out)


@lru_cache(maxsize=16)
def _get_resample_plan(
    src_h: int,
    src_w: int,
    tgt_h: int,
    tgt_w: int,
) -> ResamplePlan:
    """Builds and caches separable index and weight arrays for a pair."""
    x_ratio = float(src_w - 1) / float(tgt_w)
    x_idx = np.arange(tgt_w, dtype=np.float32) * x_ratio
    x_low = np.floor(x_idx).astype(np.int32)
    x_high = np.clip(x_low + 1, 0, src_w - 1)
    x_weight = (x_idx - x_low)[np.newaxis, :].astype(np.float32)

    y_ratio = float(src_h - 1) / float(tgt_h)
    y_idx = np.arange(tgt_h, dtype=np.float32) * y_ratio
    y_low = np.floor(y_idx).astype(np.int32)
    y_high = np.clip(y_low + 1, 0, src_h - 1)
    y_weight = (y_idx - y_low)[:, np.newaxis].astype(np.float32)

    horiz_low = np.empty((src_h, tgt_w), dtype=np.float32)
    horiz_high = np.empty((src_h, tgt_w), dtype=np.float32)

    return ResamplePlan(
        x_low=x_low,
        x_high=x_high,
        x_weight=x_weight,
        y_low=y_low,
        y_high=y_high,
        y_weight=y_weight,
        horiz_low=horiz_low,
        horiz_high=horiz_high,
    )
