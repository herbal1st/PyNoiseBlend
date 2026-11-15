"""Pure NumPy vectorized topographical contour net (isoline) generator."""

from dataclasses import dataclass
from functools import lru_cache
from typing import Tuple
import numpy as np
from numpy.typing import NDArray
import pygame


@dataclass(frozen=True, slots=True)
class TileContourEdges:
    """Compact directional contour colors and masks for coarse tile seams."""

    v_colors: NDArray[np.uint8]
    v_active: NDArray[np.bool_]
    h_colors: NDArray[np.uint8]
    h_active: NDArray[np.bool_]


@dataclass(slots=True)
class ContourScratch:
    """Reusable (W, H) buffers eliminating per-frame contour allocations."""

    field_buf: NDArray[np.float32]
    resample_work: NDArray[np.float32]
    work_a: NDArray[np.float32]
    work_b: NDArray[np.float32]
    mask: NDArray[np.bool_]


@dataclass(frozen=True, slots=True)
class CellResamplePlan:
    """Cached coarse-cell-to-screen mapping and sub-pixel weights."""

    cell_sx: NDArray[np.int32]
    cell_wx: NDArray[np.float32]
    cell_sy: NDArray[np.int32]
    cell_wy: NDArray[np.float32]


def create_contour_scratch(width: int, height: int) -> ContourScratch:
    """Allocates reusable (W, H) scratch buffers matching SDL layout."""
    shape = (width, height)
    return ContourScratch(
        field_buf=np.empty(shape, dtype=np.float32),
        resample_work=np.empty(shape, dtype=np.float32),
        work_a=np.empty(shape, dtype=np.float32),
        work_b=np.empty(shape, dtype=np.float32),
        mask=np.empty(shape, dtype=bool),
    )


def bake_contour_net(
    heightfield: NDArray[np.float32],
    lut: NDArray[np.uint8],
    intervals: float,
    bandwidth: float,
    tile_stepping: bool,
    bg_color: Tuple[int, int, int],
) -> NDArray[np.uint8]:
    """Synthesizes contour isolines via crossing or tolerance bands."""
    h_dim, w_dim = heightfield.shape
    bg_rgb = np.array(bg_color, dtype=np.uint8)
    buffer = np.full((h_dim, w_dim, 3), bg_rgb, dtype=np.uint8)

    if tile_stepping:
        count = max(1, int(round(intervals))) + 1
        levels = np.linspace(0.0, 1.0, count, dtype=np.float32)
        _bake_tile_contours(heightfield, lut, levels, bandwidth, buffer)
    else:
        _bake_band_contours(heightfield, lut, intervals, bandwidth, buffer)

    return buffer


def bake_tile_contour_edges(
    heightfield: NDArray[np.float32],
    lut: NDArray[np.uint8],
    intervals: float,
) -> TileContourEdges:
    """Computes seam-crossing colors and masks on an unexpanded tile grid."""
    h_dim, w_dim = heightfield.shape
    v_cols = max(0, w_dim - 1)
    h_rows = max(0, h_dim - 1)

    v_colors = np.zeros((h_dim, v_cols, 3), dtype=np.uint8)
    v_active = np.zeros((h_dim, v_cols), dtype=bool)
    h_colors = np.zeros((h_rows, w_dim, 3), dtype=np.uint8)
    h_active = np.zeros((h_rows, w_dim), dtype=bool)

    if h_dim < 2 and w_dim < 2:
        return TileContourEdges(
            v_colors=v_colors,
            v_active=v_active,
            h_colors=h_colors,
            h_active=h_active,
        )

    count = max(1, int(round(intervals))) + 1
    levels = np.linspace(0.0, 1.0, count, dtype=np.float32)

    has_v = w_dim > 1
    has_h = h_dim > 1
    if has_v:
        v_min = np.minimum(heightfield[:, :-1], heightfield[:, 1:])
        v_max = np.maximum(heightfield[:, :-1], heightfield[:, 1:])
    if has_h:
        h_min = np.minimum(heightfield[:-1, :], heightfield[1:, :])
        h_max = np.maximum(heightfield[:-1, :], heightfield[1:, :])

    for lvl in levels:
        lvl_idx = int(np.clip(lvl * 1023.0, 0, 1023))
        lvl_color = lut[lvl_idx]

        if has_v:
            v_cross = (v_min <= lvl) & (lvl <= v_max)
            v_active |= v_cross
            v_colors[v_cross] = lvl_color

        if has_h:
            h_cross = (h_min <= lvl) & (lvl <= h_max)
            h_active |= h_cross
            h_colors[h_cross] = lvl_color

    return TileContourEdges(
        v_colors=v_colors,
        v_active=v_active,
        h_colors=h_colors,
        h_active=h_active,
    )


def draw_tile_contour_edges_to_surface(
    surface: pygame.Surface,
    heightfield: NDArray[np.float32],
    lut: NDArray[np.uint8],
    intervals: float,
    tile_size: int,
    bg_color: Tuple[int, int, int],
) -> None:
    """Stamps 1px tile-boundary isoline contours onto target surface."""
    surface.fill(bg_color)
    edges = bake_tile_contour_edges(
        heightfield=heightfield,
        lut=lut,
        intervals=intervals,
    )
    tgt_w, tgt_h = surface.get_size()

    v_cols_y = np.repeat(edges.v_colors, tile_size, axis=0)[:tgt_h]
    v_mask_y = np.repeat(edges.v_active, tile_size, axis=0)[:tgt_h]
    v_cols_xy = np.transpose(v_cols_y, (1, 0, 2))
    v_mask_xy = np.transpose(v_mask_y, (1, 0))

    h_cols_x = np.repeat(edges.h_colors, tile_size, axis=1)[:, :tgt_w]
    h_mask_x = np.repeat(edges.h_active, tile_size, axis=1)[:, :tgt_w]
    h_cols_xy = np.transpose(h_cols_x, (1, 0, 2))
    h_mask_xy = np.transpose(h_mask_x, (1, 0))

    surf_view = pygame.surfarray.pixels3d(surface)
    try:
        v_slots = surf_view[tile_size::tile_size].shape[0]
        v_count = min(v_cols_xy.shape[0], v_slots)
        if v_count > 0:
            v_stop = tile_size * (v_count + 1)
            v_target = surf_view[tile_size:v_stop:tile_size]
            v_src = v_cols_xy[:v_count]
            v_where = v_mask_xy[:v_count, :, None]
            np.copyto(v_target, v_src, where=v_where)

        h_slots = surf_view[:, tile_size::tile_size].shape[1]
        h_count = min(h_cols_xy.shape[1], h_slots)
        if h_count > 0:
            h_stop = tile_size * (h_count + 1)
            h_target = surf_view[:, tile_size:h_stop:tile_size]
            h_src = h_cols_xy[:, :h_count]
            h_where = h_mask_xy[:, :h_count, None]
            np.copyto(h_target, h_src, where=h_where)
    finally:
        del surf_view


def draw_contour_net_to_surface(
    surface: pygame.Surface,
    heightfield_wh: NDArray[np.float32],
    lut: NDArray[np.uint8],
    intervals: float,
    bandwidth: float,
    bg_color: Tuple[int, int, int],
    scratch: ContourScratch,
) -> None:
    """Draws crisp isolines directly onto a (W, H) surface in-place."""
    surface.fill(bg_color)

    step_count = max(1, int(round(intervals)))
    step_f = float(step_count)
    limit = bandwidth * step_f

    np.multiply(heightfield_wh, step_f, out=scratch.work_a)
    np.rint(scratch.work_a, out=scratch.work_b)
    np.subtract(scratch.work_a, scratch.work_b, out=scratch.work_a)
    np.abs(scratch.work_a, out=scratch.work_a)
    np.less_equal(scratch.work_a, limit, out=scratch.mask)

    if not np.any(scratch.mask):
        return

    indices = np.clip(
        heightfield_wh[scratch.mask] * 1023.0, 0.0, 1023.0
    ).astype(np.int32)
    colors = lut[indices]

    surf_view = pygame.surfarray.pixels3d(surface)
    try:
        surf_view[scratch.mask] = colors
    finally:
        del surf_view


def draw_culled_contour_net_to_surface(
    surface: pygame.Surface,
    coarse_wh: NDArray[np.float32],
    lut: NDArray[np.uint8],
    intervals: float,
    bandwidth: float,
    bg_color: Tuple[int, int, int],
) -> None:
    """Draws 1px isolines from a coarse (W, H) grid via cell culling."""
    surface.fill(bg_color)
    src_w, src_h = coarse_wh.shape
    if src_w < 2 or src_h < 2:
        return

    step_f = float(max(1, int(round(intervals))))
    limit = bandwidth * step_f
    cx_idx, cy_idx = _find_active_cells(coarse_wh, step_f, limit)
    if cx_idx.size == 0:
        return

    tgt_w, tgt_h = surface.get_size()
    plan = _get_cell_plan(src_w, src_h, tgt_w, tgt_h)

    f00 = coarse_wh[cx_idx, cy_idx][:, None, None]
    f10 = coarse_wh[cx_idx + 1, cy_idx][:, None, None]
    f01 = coarse_wh[cx_idx, cy_idx + 1][:, None, None]
    f11 = coarse_wh[cx_idx + 1, cy_idx + 1][:, None, None]

    wx = plan.cell_wx[cx_idx][:, :, None]
    wy = plan.cell_wy[cy_idx][:, None, :]

    top = f00 + ((f10 - f00) * wx)
    bot = f01 + ((f11 - f01) * wx)
    vals = top + ((bot - top) * wy)

    scaled = vals * step_f
    mask = np.abs(scaled - np.rint(scaled)) <= limit
    hit_k, hit_bx, hit_by = np.nonzero(mask)
    if hit_k.size == 0:
        return

    screen_x = plan.cell_sx[cx_idx[hit_k], hit_bx]
    screen_y = plan.cell_sy[cy_idx[hit_k], hit_by]
    hit_vals = vals[hit_k, hit_bx, hit_by]

    indices = np.clip(hit_vals * 1023.0, 0.0, 1023.0).astype(np.int32)
    colors = lut[indices]

    surf_view = pygame.surfarray.pixels3d(surface)
    try:
        surf_view[screen_x, screen_y] = colors
    finally:
        del surf_view


def _find_active_cells(
    coarse_wh: NDArray[np.float32],
    step_f: float,
    limit: float,
) -> Tuple[NDArray[np.intp], NDArray[np.intp]]:
    """Identifies coarse cells whose value bounds cross a contour band."""
    f00 = coarse_wh[:-1, :-1]
    f10 = coarse_wh[1:, :-1]
    f01 = coarse_wh[:-1, 1:]
    f11 = coarse_wh[1:, 1:]

    c_min = np.minimum(np.minimum(f00, f10), np.minimum(f01, f11))
    c_max = np.maximum(np.maximum(f00, f10), np.maximum(f01, f11))

    u_min = c_min * step_f
    u_max = c_max * step_f
    active = np.ceil(u_min - limit) <= np.floor(u_max + limit)
    return np.nonzero(active)


@lru_cache(maxsize=8)
def _get_cell_plan(
    src_w: int,
    src_h: int,
    tgt_w: int,
    tgt_h: int,
) -> CellResamplePlan:
    """Precomputes per-cell screen coordinates and sub-pixel weights."""
    cell_sx, cell_wx = _build_axis_cells(src_w, tgt_w)
    cell_sy, cell_wy = _build_axis_cells(src_h, tgt_h)
    return CellResamplePlan(
        cell_sx=cell_sx,
        cell_wx=cell_wx,
        cell_sy=cell_sy,
        cell_wy=cell_wy,
    )


def _build_axis_cells(
    src_len: int,
    tgt_len: int,
) -> Tuple[NDArray[np.int32], NDArray[np.float32]]:
    """Maps each coarse cell interval along one axis to screen slots."""
    num_cells = src_len - 1
    ratio = float(num_cells) / float(tgt_len)
    coords = np.arange(tgt_len, dtype=np.float32) * ratio
    cell_px = np.clip(
        np.floor(coords).astype(np.int32), 0, num_cells - 1
    )
    weight_px = (coords - cell_px).astype(np.float32)

    counts = np.bincount(cell_px, minlength=num_cells)
    max_span = max(1, int(np.max(counts)))

    cell_pos = np.empty((num_cells, max_span), dtype=np.int32)
    cell_w = np.empty((num_cells, max_span), dtype=np.float32)

    cell_ids = np.arange(num_cells)
    starts = np.searchsorted(cell_px, cell_ids, side="left")
    ends = np.searchsorted(cell_px, cell_ids, side="right")

    for idx in range(num_cells):
        s_idx = int(starts[idx])
        e_idx = int(ends[idx])
        span = e_idx - s_idx
        if span <= 0:
            clamped = min(tgt_len - 1, s_idx)
            cell_pos[idx, :] = clamped
            cell_w[idx, :] = weight_px[clamped]
            continue

        cell_pos[idx, :span] = np.arange(s_idx, e_idx, dtype=np.int32)
        cell_w[idx, :span] = weight_px[s_idx:e_idx]
        if span < max_span:
            cell_pos[idx, span:] = e_idx - 1
            cell_w[idx, span:] = weight_px[e_idx - 1]

    return cell_pos, cell_w


def _bake_tile_contours(
    heightfield: NDArray[np.float32],
    lut: NDArray[np.uint8],
    levels: NDArray[np.float32],
    bandwidth: float,
    buffer: NDArray[np.uint8],
) -> None:
    """Marks cardinal neighbor cell crossings for full-res tile blocks."""
    h_left = np.roll(heightfield, 1, axis=1)
    h_right = np.roll(heightfield, -1, axis=1)
    h_up = np.roll(heightfield, 1, axis=0)
    h_down = np.roll(heightfield, -1, axis=0)

    h_min = np.minimum.reduce([heightfield, h_left, h_right, h_up, h_down])
    h_max = np.maximum.reduce([heightfield, h_left, h_right, h_up, h_down])

    for lvl in levels:
        lvl_idx = int(np.clip(lvl * 1023.0, 0, 1023))
        lvl_color = lut[lvl_idx]

        if lvl <= 0.001:
            mask = (heightfield <= bandwidth) | (h_min <= 0.001)
        elif lvl >= 0.999:
            mask = (heightfield >= 1.0 - bandwidth) | (h_max >= 0.999)
        else:
            mask = (h_min <= lvl) & (lvl <= h_max)

        buffer[mask] = lvl_color


def _bake_band_contours(
    heightfield: NDArray[np.float32],
    lut: NDArray[np.uint8],
    intervals: float,
    bandwidth: float,
    buffer: NDArray[np.uint8],
) -> None:
    """Marks continuous isolines via single-pass elevation distance."""
    step_count = max(1, int(round(intervals)))
    step_f = float(step_count)
    scaled = heightfield * step_f
    nearest = np.rint(scaled)
    limit = bandwidth * step_f
    contour_mask = np.abs(scaled - nearest) <= limit

    if not np.any(contour_mask):
        return

    indices = np.clip(
        heightfield[contour_mask] * 1023.0, 0.0, 1023.0
    ).astype(np.int32)
    buffer[contour_mask] = lut[indices]
