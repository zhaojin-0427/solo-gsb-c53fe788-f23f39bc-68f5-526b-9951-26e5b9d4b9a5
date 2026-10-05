"""配准后的辐亮度估计与运动/未知掩膜生成。"""
from __future__ import annotations

import numpy as np

from . import config as C


def _warp_nearest(
    image: np.ndarray,
    dy: int,
    dx: int,
    fill: float = 0.0,
) -> np.ndarray:
    """采样偏移 (dy, dx)：out[p] = image[p + (dy, dx)]，越界填 fill。

    使用最近邻（不插值），保证 16 位观测端点与有效/无效判断不被插值污染。
    """
    h, w = image.shape[:2]
    out = np.full_like(image, fill)
    # 输出坐标 p 满足 p+(dy,dx) 落在源图内：
    #   py ∈ [max(0,-dy), min(h,h-dy))，对应源行 py+dy ∈ [max(0,dy), min(h,h+dy))
    py0, py1 = max(0, -dy), min(h, h - dy)
    px0, px1 = max(0, -dx), min(w, w - dx)
    if py1 > py0 and px1 > px0:
        out[py0:py1, px0:px1] = image[py0 + dy:py1 + dy, px0 + dx:px1 + dx]
    return out


def merge_frames(
    frames: list[np.ndarray],
    exposures: list[float],
    shifts: list[tuple[int, int]],
    reference_index: int,
    tolerance: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """融合配准后的多曝光帧。

    Returns:
        radiance: float32 RGB，辐亮度 = raw/65535/曝光秒数；未知像素为 NaN。
        motion_mask: uint8，运动像素 255。
        unknown_mask: uint8，无法给出辐亮度的像素 255。
    """
    n = len(frames)
    h, w = frames[0].shape[:2]

    # 全部搬到参考帧坐标系（最近邻），再判断有效观测。
    raws = np.empty((n, h, w, 3), dtype=np.float64)
    valid = np.empty((n, h, w), dtype=bool)
    for i, (frame, (dy, dx)) in enumerate(zip(frames, shifts)):
        warped = _warp_nearest(frame.astype(np.float64), dy, dx, fill=0.0)
        raws[i] = warped
        # 越界区域填充 0，天然落入“无效”；剪切端点同样无效。
        valid[i] = np.all((warped > C.RAW_MIN) & (warped < C.RAW_MAX), axis=2)

    radiance_all = raws / C.RAW_SCALE / np.asarray(exposures)[:, None, None, None]

    # ---- 运动检测：任两帧（都有效时）任一通道辐亮度差超过容差即为运动 ----
    motion = np.zeros((h, w), dtype=bool)
    for i in range(n):
        for j in range(i + 1, n):
            both = valid[i] & valid[j]
            diff = np.max(np.abs(radiance_all[i] - radiance_all[j]), axis=2)
            motion |= both & (diff > tolerance)

    # 没有任何可比较帧的像素无法判定运动，按静态处理（随后由有效情况决定未知）。

    # ---- 辐亮度估计（曝光时长加权，使各帧权重按有效光子数规模变化）----
    weights = valid.astype(np.float64) * np.asarray(exposures)[:, None, None]
    weight_sum = weights.sum(axis=0)
    weighted_rad = (radiance_all * weights[..., None]).sum(axis=0)
    static_fuse = weight_sum > 0
    radiance = np.full((h, w, 3), np.nan, dtype=np.float64)
    np.divide(
        weighted_rad,
        weight_sum[..., None],
        where=static_fuse[..., None],
        out=radiance,
    )

    ref_valid = valid[reference_index]
    ref_rad = radiance_all[reference_index]

    # 运动区：只采用参考帧的有效观测；参考帧无效则未知。
    motion_ref_invalid = motion & ~ref_valid
    radiance = np.where(motion[..., None], ref_rad, radiance)
    radiance = np.where(motion_ref_invalid[..., None], np.nan, radiance)

    # 静态区：没有任何有效观测（含全部越界）则未知。
    static_all_invalid = ~motion & ~static_fuse

    unknown = motion_ref_invalid | static_all_invalid

    # 数值安全：无效位置上除法未定义，统一置 NaN。
    radiance[np.broadcast_to(unknown[..., None], radiance.shape)] = np.nan
    radiance = radiance.astype(np.float32)

    motion_png = (motion.astype(np.uint8)) * 255
    unknown_png = (unknown.astype(np.uint8)) * 255
    return radiance, motion_png, unknown_png
