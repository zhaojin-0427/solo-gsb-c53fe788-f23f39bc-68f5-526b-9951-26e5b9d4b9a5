"""端到端与单元测试：用合成多曝光场景验证配准、融合、运动/未知掩膜。"""
from __future__ import annotations

import numpy as np
import pytest
from fastapi.testclient import TestClient

from app import image_io
from app.fusion import merge_frames
from app.main import app
from app.registration import RegistrationError, register_frames

client = TestClient(app)


# --------------------------------------------------------------------------- #
# 合成场景
# --------------------------------------------------------------------------- #
def make_scene(
    size: int = 128,
    exposures=(0.25, 1.0, 4.0),
    shifts=((3, -4), (0, 0), (-5, 2)),
    reference_index: int = 1,
    seed: int = 0,
    moving: bool = True,
    object_rad: float = 0.6,
) -> tuple[list[np.ndarray], list[float], dict]:
    """构造同尺寸 16 位线性 RGB 场景：纹理静态背景 + 高对比度移动圆盘。

    返回 (frames(uint16 HxWx3), exposures, info)，info 中保存真值掩膜与背景辐亮度。
    """
    rng = np.random.default_rng(seed)
    yy, xx = np.meshgrid(np.arange(size), np.arange(size), indexing="ij")

    # 静态背景辐亮度：平滑渐变 + 高对比棋盘纹理，三通道略有差异。
    # 上限控制在 ~0.2，使最长曝光 4.0s 也不剪切（辐亮度*曝光 < 0.8）。
    base = 0.03 + 0.05 * (xx / size) + 0.03 * np.sin(yy / 7.0)
    checker = (((xx // 8) + (yy // 8)) % 2).astype(np.float64) * 0.03
    noise = rng.normal(0, 0.003, (size, size))
    bg = np.empty((size, size, 3))
    bg[..., 0] = base + checker + noise
    bg[..., 1] = base + checker * 0.8 + 0.01
    bg[..., 2] = base + checker * 0.6 + 0.02
    bg = np.clip(bg, 0.01, None)

    centers = [(34, 36), (86, 40), (60, 90)]
    radius = 13

    frames = []
    object_masks = []
    for fi, (exp, (dy, dx)) in enumerate(zip(exposures, shifts)):
        # 位移 d 约定（与 API/融合一致）：注册采样 out[p] = src[p+d]。
        # 故源帧由场景前向拷贝生成：src[p-d] = scene[p]。
        rad = np.zeros_like(bg)
        # scene 坐标 q 满足 q-d 在源帧内：
        qy0, qx0 = max(0, dy), max(0, dx)
        qy1, qx1 = min(size, size + dy), min(size, size + dx)
        rad[qy0 - dy:qy1 - dy, qx0 - dx:qx1 - dx] = bg[qy0:qy1, qx0:qx1]

        # 物体中心定义在“参考帧坐标系”；src[p]=scene[p+d]，故源帧中物体
        # 出现在 center - d。
        if moving:
            cy, cx = centers[fi]
            om = (yy - cy) ** 2 + (xx - cx) ** 2 <= radius**2
            om_src = (yy - (cy - dy)) ** 2 + (xx - (cx - dx)) ** 2 <= radius**2
            rad[om_src] = object_rad
        else:
            om = np.zeros((size, size), dtype=bool)
        object_masks.append(om)

        raw = np.clip(np.rint(rad * exp * 65535.0), 0, 65535).astype(np.uint16)
        frames.append(raw)

    info = {"bg": bg, "object_masks": object_masks, "size": size}
    return frames, list(exposures), info


def b64_png(frame: np.ndarray) -> str:
    import cv2

    ok, buf = cv2.imencode(".png", cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
    assert ok
    import base64

    return base64.b64encode(buf.tobytes()).decode()


def decode_tiff(payload: str) -> np.ndarray:
    import base64
    import cv2

    arr = cv2.imdecode(
        np.frombuffer(base64.b64decode(payload), np.uint8), cv2.IMREAD_UNCHANGED
    )
    return cv2.cvtColor(arr, cv2.COLOR_BGR2RGB)


def decode_mask(payload: str) -> np.ndarray:
    import base64
    import cv2

    return cv2.imdecode(
        np.frombuffer(base64.b64decode(payload), np.uint8), cv2.IMREAD_UNCHANGED
    )


# --------------------------------------------------------------------------- #
# 配准：移动前景不得牵引位移
# --------------------------------------------------------------------------- #
def test_registration_finds_true_shifts_despite_moving_object():
    frames, exposures, info = make_scene()
    shifts = register_frames(frames, exposures, reference_index=1)
    # API 位移即采样偏移 d：registered(p) = src(p+d)。
    # 生成器用“内容平移” shifts 渲染（src[p-d]=scene[p]），故 API 输出 = -shifts。
    assert shifts == [(-3, 4), (0, 0), (5, -2)]


def test_registration_static_scene_zero_shifts():
    frames, exposures, _ = make_scene(
        moving=False, shifts=((0, 0), (0, 0), (0, 0))
    )
    shifts = register_frames(frames, exposures, reference_index=0)
    assert shifts == [(0, 0), (0, 0), (0, 0)]


def test_registration_fails_on_unstructured_noise():
    rng = np.random.default_rng(1)
    size = 128
    frames = []
    for exp in (0.25, 1.0, 4.0):
        rad = rng.uniform(0.05, 0.3, (size, size, 3))
        raw = np.clip(rad * exp * 65535, 1, 65534).astype(np.uint16)
        frames.append(raw)  # 帧间无任何对应关系
    with pytest.raises(RegistrationError):
        register_frames(frames, [0.25, 1.0, 4.0], 1)


# --------------------------------------------------------------------------- #
# 融合：静态背景辐亮度准确、运动区只取参考帧、无重影
# --------------------------------------------------------------------------- #
def test_merge_static_background_radiance_and_motion_mask():
    frames, exposures, info = make_scene()
    shifts = register_frames(frames, exposures, 1)
    radiance, motion, unknown = merge_frames(frames, exposures, shifts, 1, tolerance=0.03)

    size = info["size"]
    om0, om1, om2 = info["object_masks"]
    object_union = om0 | om1 | om2

    # 运动掩膜必须覆盖“可检测”的运动轨迹：第三帧（4.0s）物体饱和无效，
    # 故只有存在至少一对都有效帧的物体区域才能被标记运动。
    detectable = np.zeros((size, size), dtype=bool)
    # 帧 0(0.25s)/帧 1(1.0s) 的物体都未剪切（0.6*exp<1）
    detectable |= om0 | om1
    covered = (motion > 0) & detectable
    assert covered.sum() >= 0.9 * detectable.sum()
    # 运动掩膜不得蔓延到物体并集之外（不能把静态背景误判为运动）。
    assert ((motion > 0) & ~object_union).sum() < 0.005 * (size * size)

    # 远离运动轨迹的内部静态区，重建辐亮度接近真值。
    yy, xx = np.meshgrid(np.arange(size), np.arange(size), indexing="ij")
    far = ~object_union
    d0 = np.minimum((yy - 34) ** 2 + (xx - 36) ** 2, (yy - 86) ** 2 + (xx - 40) ** 2)
    d0 = np.minimum(d0, (yy - 60) ** 2 + (xx - 90) ** 2)
    far = far & (d0 > 24**2)
    # 避开位移造成的越界边缘。
    far[:8, :] = far[-8:, :] = far[:, :8] = far[:, -8:] = False

    err = np.abs(radiance - info["bg"]).max(axis=2)
    assert np.nanmean(err[far]) < 0.02
    assert (unknown[far] > 0).sum() < 0.01 * far.sum()


def test_motion_region_uses_reference_only_and_no_ghosting():
    frames, exposures, info = make_scene()
    shifts = register_frames(frames, exposures, 1)
    radiance, motion, unknown = merge_frames(frames, exposures, shifts, 1, tolerance=0.03)

    om1 = info["object_masks"][1]
    # 参考帧物体内部（参考帧该区域未剪切：0.6*1.0*65535 < 65535）取参考辐亮度。
    inside = om1 & (motion > 0) & (unknown == 0)
    vals = radiance[inside]
    assert vals.shape[0] > 50
    assert np.abs(vals.mean() - 0.6) < 0.05

    # 运动轨迹上、但参考帧没有物体的像素，不能残留其它帧物体的高辐亮度（无重影）：
    # 这些位置若被判为运动，只可能取参考帧的背景值；不能是融合值。
    ghost_zone = (info["object_masks"][0] | info["object_masks"][2]) & ~om1
    motion_in_ghost = ghost_zone & (motion > 0)
    if motion_in_ghost.any():
        bg_vals = info["bg"][motion_in_ghost]
        got = radiance[motion_in_ghost]
        valid_got = ~np.isnan(got).any(axis=1)
        assert np.abs(got[valid_got] - bg_vals[valid_got]).max() < 0.05


def test_unknown_when_static_region_all_observations_invalid():
    """静态区所有帧的观测都无效（剪切端点 0/65535）-> 未知、NaN。"""
    size = 96
    yy, xx = np.meshgrid(np.arange(size), np.arange(size), indexing="ij")
    checker = (((xx // 8) + (yy // 8)) % 2).astype(float) * 0.04
    bg = np.clip(0.1 + checker + 0.02 * np.sin(yy / 5.0), 0.02, None)
    bg = np.repeat(bg[..., None], 3, axis=2)
    exposures = [1.0, 2.0, 4.0]
    frames = [
        np.clip(np.rint(bg * e * 65535), 0, 65535).astype(np.uint16) for e in exposures
    ]
    # 中央矩形：帧 0 全 0（欠曝无效），帧 1、2 全 65535（剪切无效）。
    for f, val in zip(frames, (0, 65535, 65535)):
        f[32:64, 32:64] = val

    shifts = [(0, 0)] * 3
    radiance, motion, unknown = merge_frames(frames, exposures, shifts, 1, tolerance=0.03)
    assert (unknown[32:64, 32:64] > 0).all()
    assert np.isnan(radiance[32:64, 32:64]).all()
    # 矩形之外的静态背景正常融合。
    assert not np.isnan(radiance[8, 8]).any()


def test_unknown_when_reference_invalid_in_motion_region():
    """融合不变量：被标记为运动且参考帧无效的像素必须未知、辐亮度为 NaN。

    高对比度物体只做小幅抖动（圆盘高度重叠），并在参考帧（1.0s）饱和：
    参考物体圆盘内部为无效像素，而短曝帧在相同坐标有有效且不同的辐亮度，
    从而在“参考无效”的位置检测出运动 -> 这些像素必须未知。
    """
    size = 96
    yy, xx = np.meshgrid(np.arange(size), np.arange(size), indexing="ij")
    checker = (((xx // 8) + (yy // 8)) % 2).astype(float) * 0.04
    bg = np.clip(0.1 + checker + 0.02 * np.sin(yy / 5.0), 0.02, None)
    bg = np.repeat(bg[..., None], 3, axis=2)
    cy, cx, radius = 48, 48, 16
    exposures = [0.1, 0.3, 1.0]
    jitters = [(3, -3), (-2, 2), (0, 0)]
    frames = []
    for e, (jdy, jdx) in zip(exposures, jitters):
        rad = bg.copy()
        om_j = (yy - (cy + jdy)) ** 2 + (xx - (cx + jdx)) ** 2 <= radius**2
        # 物体足够亮：短曝两帧未剪切，参考帧 1.0s 饱和。
        rad[om_j] = 2.0
        frames.append(np.clip(np.rint(rad * e * 65535), 0, 65535).astype(np.uint16))

    shifts = [(0, 0)] * 3
    radiance, motion, unknown = merge_frames(frames, exposures, shifts, 2, tolerance=0.03)
    om_ref = (yy - cy) ** 2 + (xx - cx) ** 2 <= radius**2

    # 参考盘内部（参考无效）与短曝帧（同一坐标有移动过来的物体）比较出运动。
    bad = (motion > 0) & om_ref
    assert bad.sum() > 50
    assert (unknown[bad] > 0).mean() > 0.95
    assert np.isnan(radiance[bad]).all()
    # 静态背景区没有未知、没有运动。
    far = (yy - cy) ** 2 + (xx - cx) ** 2 > (radius + 6) ** 2
    assert (unknown[far] > 0).sum() == 0
    assert (motion[far] > 0).sum() == 0


def test_warp_nearest_oob_regions():
    # warp 单元语义：out[p] = image[p+(dy,dx)]，越界填 0。
    from app.fusion import _warp_nearest

    img = np.arange(100, dtype=np.uint16).reshape(10, 10)
    w = _warp_nearest(img, 2, 3, fill=0)
    assert w[0, 0] == img[2, 3]
    assert w[7, 6] == img[9, 9]
    assert (w[8:, :] == 0).all() and (w[:, 7:] == 0).all()
    assert (w[:8, :7] != 0).all()


def test_out_of_bounds_is_unknown():
    # 端到端：三帧在场景生成器中按同一 d=(8,8) 渲染，warp 用相同 d 采样，
    # 渲染逆变换使右下 8 像素条带在源帧中本就无背景（全 0=无效），
    # 三帧均如此 -> 静态区无任何有效观测 -> 未知。
    frames, exposures, _ = make_scene(shifts=((8, 8), (8, 8), (8, 8)), moving=False)
    radiance, motion, unknown = merge_frames(
        frames, exposures, [(8, 8), (8, 8), (8, 8)], 1, tolerance=0.03
    )
    u = unknown > 0
    assert u[-8:, 16:].all()
    assert u[16:, -8:].all()
    assert np.isnan(radiance[-8:, 16:]).all()
    assert not np.isnan(radiance[64, 64]).any()


# --------------------------------------------------------------------------- #
# HTTP /merges
# --------------------------------------------------------------------------- #
def _post(frames, exposures, ref, tol):
    return client.post(
        "/merges",
        json={
            "frames": [b64_png(f) for f in frames],
            "exposures": exposures,
            "reference_index": ref,
            "tolerance": tol,
        },
    )


def test_endpoint_happy_path():
    frames, exposures, _ = make_scene()
    resp = _post(frames, exposures, 1, 0.03)
    assert resp.status_code == 200, resp.text
    data = resp.json()
    assert set(data) == {"radiance_tiff", "motion_mask_png", "unknown_mask_png", "shifts"}
    assert data["shifts"] == [
        {"dy": -3, "dx": 4},
        {"dy": 0, "dx": 0},
        {"dy": 5, "dx": -2},
    ]

    rad = decode_tiff(data["radiance_tiff"])
    assert rad.dtype == np.float32 and rad.shape == (128, 128, 3)
    m = decode_mask(data["motion_mask_png"])
    assert m.dtype == np.uint8 and set(np.unique(m)).issubset({0, 255})
    u = decode_mask(data["unknown_mask_png"])
    assert u.dtype == np.uint8


def test_endpoint_rejects_wrong_frame_count():
    frames, exposures, _ = make_scene()
    resp = client.post(
        "/merges",
        json={
            "frames": [b64_png(f) for f in frames[:2]],
            "exposures": exposures[:2],
            "reference_index": 0,
            "tolerance": 0.03,
        },
    )
    assert resp.status_code == 422


def test_endpoint_rejects_non_positive_exposure():
    frames, exposures, _ = make_scene()
    resp = client.post(
        "/merges",
        json={
            "frames": [b64_png(f) for f in frames],
            "exposures": [0.25, 0.0, 4.0],
            "reference_index": 1,
            "tolerance": 0.03,
        },
    )
    assert resp.status_code == 422


def test_endpoint_rejects_negative_tolerance():
    frames, exposures, _ = make_scene()
    resp = client.post(
        "/merges",
        json={
            "frames": [b64_png(f) for f in frames],
            "exposures": exposures,
            "reference_index": 1,
            "tolerance": -0.1,
        },
    )
    assert resp.status_code == 422


def test_endpoint_rejects_bad_reference_index():
    frames, exposures, _ = make_scene()
    resp = client.post(
        "/merges",
        json={
            "frames": [b64_png(f) for f in frames],
            "exposures": exposures,
            "reference_index": 9,
            "tolerance": 0.03,
        },
    )
    assert resp.status_code == 422


def test_endpoint_rejects_size_mismatch():
    frames, exposures, _ = make_scene()
    small, _, _ = make_scene(size=96)
    frames[2] = small[0]
    resp = _post(frames, exposures, 1, 0.03)
    assert resp.status_code == 422
    assert "不一致" in resp.json()["detail"]


def test_endpoint_rejects_8bit_png():
    frames, exposures, _ = make_scene()
    import base64
    import cv2

    eight = (frames[0] >> 8).astype(np.uint8)
    ok, buf = cv2.imencode(".png", cv2.cvtColor(eight, cv2.COLOR_RGB2BGR))
    payload = base64.b64encode(buf.tobytes()).decode()
    body = {
        "frames": [b64_png(f) for f in frames[:-1]] + [payload],
        "exposures": exposures,
        "reference_index": 1,
        "tolerance": 0.03,
    }
    resp = client.post("/merges", json=body)
    assert resp.status_code == 422
    assert "16 位" in resp.json()["detail"]


def test_endpoint_rejects_garbage_base64():
    frames, exposures, _ = make_scene()
    body = {
        "frames": [b64_png(f) for f in frames[:-1]] + ["@@@not-base64@@@"],
        "exposures": exposures,
        "reference_index": 1,
        "tolerance": 0.03,
    }
    resp = client.post("/merges", json=body)
    assert resp.status_code == 422


def test_endpoint_registration_failure_returns_422():
    rng = np.random.default_rng(7)
    frames = [
        np.clip(rng.uniform(0.05, 0.3, (128, 128, 3)) * exp * 65535, 1, 65534).astype(
            np.uint16
        )
        for exp in (0.25, 1.0, 4.0)
    ]
    resp = _post(frames, [0.25, 1.0, 4.0], 1, 0.03)
    assert resp.status_code == 422
    assert "配准" in resp.json()["detail"]


def test_registration_at_shift_boundary():
    """每轴 ±8 为允许范围的边界，配准应能恢复；±8 之外不可达。"""
    frames, exposures, _ = make_scene(
        shifts=((8, 8), (0, 0), (-8, -8)), moving=False
    )
    shifts = register_frames(frames, exposures, 1)
    assert shifts == [(-8, -8), (0, 0), (8, 8)]
    assert all(abs(dy) <= 8 and abs(dx) <= 8 for dy, dx in shifts)


def test_endpoint_accepts_nine_frames():
    frames3, exposures3, _ = make_scene()
    frames = frames3 * 3
    exposures = exposures3 * 3
    resp = client.post(
        "/merges",
        json={
            "frames": [b64_png(f) for f in frames],
            "exposures": exposures,
            "reference_index": 4,
            "tolerance": 0.03,
        },
    )
    assert resp.status_code == 200, resp.text
    assert len(resp.json()["shifts"]) == 9


def test_endpoint_rejects_ten_frames():
    frames3, exposures3, _ = make_scene()
    frames = frames3 * 4  # 12 > 9
    exposures = exposures3 * 4
    resp = client.post(
        "/merges",
        json={
            "frames": [b64_png(f) for f in frames],
            "exposures": exposures,
            "reference_index": 0,
            "tolerance": 0.03,
        },
    )
    assert resp.status_code == 422


def test_health():
    assert client.get("/health").json() == {"status": "ok"}
