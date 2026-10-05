"""图像的 Base64 / PNG / TIFF 编解码。"""
from __future__ import annotations

import base64
import binascii

import cv2
import numpy as np

from . import config as C

PNG_SIGNATURE = b"\x89PNG\r\n\x1a\n"


def decode_png16_base64(payload: str) -> np.ndarray:
    """解码 Base64 16 位线性 RGB PNG，返回 HxWx3 uint16（RGB 顺序）。"""
    try:
        raw = base64.b64decode(payload, validate=True)
    except (binascii.Error, ValueError) as exc:
        raise ValueError("图像必须是合法的 Base64 编码 PNG") from exc

    if not raw.startswith(PNG_SIGNATURE):
        raise ValueError("图像必须是 PNG 格式")

    buf = np.frombuffer(raw, dtype=np.uint8)
    image = cv2.imdecode(buf, cv2.IMREAD_UNCHANGED)
    if image is None:
        raise ValueError("PNG 无法解码")
    if image.dtype != np.uint16:
        raise ValueError(f"PNG 必须为 16 位图像，实际位深为 {image.dtype}")
    if image.ndim != 3 or image.shape[2] != 3:
        raise ValueError("PNG 必须为 3 通道 RGB 图像")

    # OpenCV 按 BGR 解码，转换为 RGB 返回。
    return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)


def encode_png_base64(mask: np.ndarray) -> str:
    """将单通道 uint8 掩膜编码为 Base64 PNG。"""
    ok, buf = cv2.imencode(".png", mask)
    if not ok:
        raise RuntimeError("掩膜 PNG 编码失败")
    return base64.b64encode(buf.tobytes()).decode("ascii")


def encode_tiff_float32_base64(radiance: np.ndarray) -> str:
    """将 float32 RGB 辐亮度（未知像素为 NaN）编码为 Base64 TIFF。"""
    # OpenCV 按 BGR 写入 TIFF。
    bgr = cv2.cvtColor(radiance, cv2.COLOR_RGB2BGR)
    ok, buf = cv2.imencode(".tif", bgr)
    if not ok:
        raise RuntimeError("float32 TIFF 编码失败")
    return base64.b64encode(buf.tobytes()).decode("ascii")
