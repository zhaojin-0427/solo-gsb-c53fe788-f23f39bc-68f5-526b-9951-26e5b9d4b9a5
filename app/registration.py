"""基于未剪切静态背景的整数平移配准。

设计要点
--------
* 参考帧不动，其余帧通过块级匹配对齐到参考帧；每轴只允许 [-8, 8] 的整数平移。
* 对帧 i 返回位移 (dy, dx) 的语义为：注册后参考坐标 (y, x) 处的内容取自该帧
  (y+dy, x+dx) 坐标（即融合模块的最近邻采样偏移）；参考帧自身为 (0, 0)。
* 只使用两帧“均未剪切”的像素（参考帧该块具备纹理）参与匹配，因此高对比度
  移动物体牵引不了背景位移：候选位移必须在遍布画面的多个背景块上取得一致票数。
* 整体流程：
    1. 在参考帧上均匀选取若干有纹理的背景块；
    2. 每个块在 17x17 个候选位移上计算“共同有效像素的平均绝对误差”（单位为
       辐亮度，即 raw/65535/曝光秒数）；
    3. 对所有块的最优位移做直方图表决，选出背景一致位移；
    4. 用全体背景块的 25 分位残差复核：位移必须足够好，且远离它的候选位移
       残差必须明显更差（防止大面积移动物体恰好形成另一组一致票数）。
"""
from __future__ import annotations

import numpy as np

from . import config as C


class RegistrationError(Exception):
    """静态背景无法可靠配准。"""


def _gray_radiance(frame: np.ndarray, exposure: float) -> tuple[np.ndarray, np.ndarray]:
    """返回线性灰度辐亮度（float64）与逐像素有效掩膜。

    任一通道为 0 或 65535 的观测无效。
    """
    valid = np.all((frame > C.RAW_MIN) & (frame < C.RAW_MAX), axis=2)
    gray = (frame.mean(axis=2).astype(np.float64) / C.RAW_SCALE) / exposure
    return gray, valid


def _gray_fraction(frame: np.ndarray) -> np.ndarray:
    """仅按 raw/65535 归一化的灰度，供与曝光无关的纹理检测使用。"""
    return frame.mean(axis=2).astype(np.float64) / C.RAW_SCALE


def _block_origins(size: int) -> list[int]:
    """在一条轴上均匀选取块原点，使两侧都留出至少 BLOCK_MARGIN 的搜索边距。"""
    usable = size - 2 * C.BLOCK_MARGIN - C.BLOCK_SIZE
    if usable < 0:
        return []
    count = int(np.ceil(usable / C.BLOCK_SIZE)) + 1
    count = min(max(count, C.BLOCKS_PER_AXIS_MIN // 2), C.BLOCKS_PER_AXIS_MAX)
    if count == 1:
        return [C.BLOCK_MARGIN + usable // 2]
    return [C.BLOCK_MARGIN + int(round(i * usable / (count - 1))) for i in range(count)]


def _is_textured(patch: np.ndarray) -> bool:
    """用归一化灰度的标准差和平均梯度判断背景块是否有纹理。"""
    if patch.std() < C.TEXTURE_STD:
        return False
    grad = np.concatenate([
        np.abs(np.diff(patch, axis=0)).ravel(),
        np.abs(np.diff(patch, axis=1)).ravel(),
    ])
    return float(np.mean(grad)) >= C.TEXTURE_GRAD


def _shift_patch(
    gray: np.ndarray,
    valid: np.ndarray,
    y0: int,
    x0: int,
    dy: int,
    dx: int,
    b: int,
) -> tuple[np.ndarray, np.ndarray]:
    """取候选帧贴片。

    (dy, dx) 即对外的采样偏移 d：registered(p) = src(p + d)，
    故参考块 [y0, y0+b) 对应源帧 [y0+dy, y0+dy+b)。
    """
    ys, xs = y0 + dy, x0 + dx
    return gray[ys:ys + b, xs:xs + b], valid[ys:ys + b, xs:xs + b]


def register_frames(
    frames: list[np.ndarray],
    exposures: list[float],
    reference_index: int,
) -> list[tuple[int, int]]:
    """估计每帧相对参考帧的整数位移，返回 [(dy, dx), ...]，参考帧为 (0, 0)。

    Raises:
        RegistrationError: 可用的静态背景不足，无法可靠配准。
    """
    h, w = frames[0].shape[:2]
    ref_gray, ref_valid = _gray_radiance(frames[reference_index], exposures[reference_index])
    ref_frac = _gray_fraction(frames[reference_index])

    ys = _block_origins(h)
    xs = _block_origins(w)
    if not ys or not xs:
        raise RegistrationError("图像过小，无法在搜索边距内容纳背景配准块")
    b = C.BLOCK_SIZE

    # 参考帧上的候选背景块：要求有纹理（纯色块无法确定位移）。
    ref_patches: list[tuple[np.ndarray, int, int]] = []
    for y0 in ys:
        for x0 in xs:
            rp = ref_frac[y0:y0 + b, x0:x0 + b]
            if rp.shape == (b, b) and _is_textured(rp):
                ref_patches.append((rp, y0, x0))
    if len(ref_patches) < C.MIN_TEXTURED_BLOCKS:
        raise RegistrationError(
            f"参考帧中具备纹理的未剪切背景块不足（{len(ref_patches)} < "
            f"{C.MIN_TEXTURED_BLOCKS}），无法依据静态背景配准"
        )

    shifts: list[tuple[int, int]] = [(0, 0)] * len(frames)
    m = C.MAX_SHIFT
    for fi, (frame, exposure) in enumerate(zip(frames, exposures)):
        if fi == reference_index:
            continue
        gray, valid = _gray_radiance(frame, exposure)

        # 每个块记录：最佳 MAD、最佳位移、每个位移上的 MAD（供整体复核）。
        best_mad: list[float] = []
        best_shift: list[tuple[int, int]] = []
        per_shift_mads: dict[tuple[int, int], list[float]] = {}

        for _rp, y0, x0 in ref_patches:
            ref_g = ref_gray[y0:y0 + b, x0:x0 + b]
            ref_v = ref_valid[y0:y0 + b, x0:x0 + b]
            block_best = np.inf
            block_shift = (0, 0)
            for dy in range(-m, m + 1):
                for dx in range(-m, m + 1):
                    cand_g, cand_v = _shift_patch(gray, valid, y0, x0, dy, dx, b)
                    both = ref_v & cand_v
                    frac = both.mean()
                    if frac < C.MIN_VALID_CELL_FRACTION:
                        continue
                    mad = float(np.abs(ref_g - cand_g)[both].mean())
                    per_shift_mads.setdefault((dy, dx), []).append(mad)
                    if mad < block_best:
                        block_best, block_shift = mad, (dy, dx)
            if np.isfinite(block_best):
                best_mad.append(block_best)
                best_shift.append(block_shift)

        if len(best_mad) < C.MIN_TEXTURED_BLOCKS:
            raise RegistrationError(
                f"第 {fi} 帧与参考帧共同有效的背景块不足（{len(best_mad)} < "
                f"{C.MIN_TEXTURED_BLOCKS}），无法配准"
            )

        # 块级表决：静态背景块的最优位移应当高度一致。
        votes: dict[tuple[int, int], int] = {}
        for sh in best_shift:
            votes[sh] = votes.get(sh, 0) + 1
        ranked = sorted(votes.items(), key=lambda kv: (-kv[1], kv[0]))
        win, win_count = ranked[0]
        second_count = ranked[1][1] if len(ranked) > 1 else 0
        if win_count < C.MIN_WIN_VOTES:
            raise RegistrationError(
                f"第 {fi} 帧背景块无法就位移达成一致（最高票数 {win_count} < "
                f"{C.MIN_WIN_VOTES}）"
            )
        if second_count > 0 and win_count < C.WIN_VOTE_RATIO * second_count:
            raise RegistrationError(
                f"第 {fi} 帧背景位移存在多个相互矛盾的候选（{win}: {win_count} 票，"
                f"次选 {second_count} 票），无法可靠配准"
            )

        # 整体复核：用全部背景块的 25 分位残差代表背景拟合质量。
        def q25(shift: tuple[int, int]) -> float:
            mads = per_shift_mads.get(shift, [])
            return float(np.percentile(mads, 25)) if mads else np.inf

        win_q25 = q25(win)
        if not np.isfinite(win_q25):
            raise RegistrationError(f"第 {fi} 帧胜出位移缺少有效残差数据，无法配准")

        distant = [
            q25(sh)
            for sh in per_shift_mads
            if sh != win
            and max(abs(sh[0] - win[0]), abs(sh[1] - win[1])) >= C.DISTANT_SHIFT
        ]
        distant = [v for v in distant if np.isfinite(v)]
        if distant and win_q25 > min(distant) / C.DISTANT_SCORE_RATIO:
            raise RegistrationError(
                f"第 {fi} 帧远处位移的背景残差与胜出位移相当，无法确认静态背景，"
                f"拒绝配准（胜出 {win}）"
            )

        # 胜出位移即采样偏移 d：registered(y,x) = src(y+dy, x+dx)，
        # 与融合模块的最近邻采样及 README 的位移定义一致。
        shifts[fi] = win

    return shifts
