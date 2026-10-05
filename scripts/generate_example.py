"""生成 README 使用的真实请求/响应示例（合成 3 帧 16 位线性 RGB PNG）。

运行后写出：
  examples/request.json    可直接 POST 给 /merges 的完整请求体
  examples/response.json   运行该服务得到的实际响应
  examples/summary.txt     响应的人类可读摘要（尺寸/位移/掩膜像素数）

用法：先启动服务（docker compose up 或 uvicorn app.main:app），再执行：
    python scripts/generate_example.py
"""
from __future__ import annotations

import base64
import json
import pathlib

import cv2
import numpy as np

import urllib.request

import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

from tests.test_pipeline import make_scene  # noqa: E402

ROOT = pathlib.Path(__file__).resolve().parent.parent
EX = ROOT / "examples"
EX.mkdir(exist_ok=True)

import urllib.request

API = "http://127.0.0.1:8000/merges"


def main() -> None:
    frames, exposures, _info = make_scene(
        size=128,
        exposures=(0.25, 1.0, 4.0),
        shifts=((3, -4), (0, 0), (-5, 2)),
        reference_index=1,
    )

    encoded = []
    for f in frames:
        ok, buf = cv2.imencode(".png", cv2.cvtColor(f, cv2.COLOR_RGB2BGR))
        assert ok
        encoded.append(base64.b64encode(buf.tobytes()).decode("ascii"))

    request = {
        "frames": encoded,
        "exposures": exposures,
        "reference_index": 1,
        "tolerance": 0.03,
    }
    (EX / "request.json").write_text(json.dumps(request, ensure_ascii=False))

    req = urllib.request.Request(
        API,
        data=json.dumps(request).encode(),
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(req, timeout=60) as resp:
        response = json.loads(resp.read().decode())

    (EX / "response.json").write_text(json.dumps(response, ensure_ascii=False))

    rad = cv2.imdecode(
        np.frombuffer(base64.b64decode(response["radiance_tiff"]), np.uint8),
        cv2.IMREAD_UNCHANGED,
    )
    motion = cv2.imdecode(
        np.frombuffer(base64.b64decode(response["motion_mask_png"]), np.uint8),
        cv2.IMREAD_UNCHANGED,
    )
    unknown = cv2.imdecode(
        np.frombuffer(base64.b64decode(response["unknown_mask_png"]), np.uint8),
        cv2.IMREAD_UNCHANGED,
    )
    valid_rad = rad[~np.isnan(rad).any(axis=2)]
    summary = (
        f"radiance TIFF: shape={rad.shape}, dtype={rad.dtype}, "
        f"min={valid_rad.min():.4f}, max={valid_rad.max():.4f}\n"
        f"motion pixels:  {(motion > 0).sum()}\n"
        f"unknown pixels: {(unknown > 0).sum()}\n"
        f"shifts: {response['shifts']}\n"
        f"base64 lengths: tiff={len(response['radiance_tiff'])}, "
        f"motion_png={len(response['motion_mask_png'])}, "
        f"unknown_png={len(response['unknown_mask_png'])}\n"
    )
    (EX / "summary.txt").write_text(summary)
    print(summary)


if __name__ == "__main__":
    main()
