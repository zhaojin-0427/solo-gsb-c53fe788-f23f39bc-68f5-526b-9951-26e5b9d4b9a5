"""FastAPI 入口：POST /merges 包围曝光合成。"""
from __future__ import annotations

import logging

from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse

from .fusion import merge_frames
from .image_io import (
    decode_png16_base64,
    encode_png_base64,
    encode_tiff_float32_base64,
)
from .models import MergeRequest, MergeResponse
from .registration import RegistrationError, register_frames

logger = logging.getLogger("exposure_merge")

app = FastAPI(
    title="包围曝光合成 API",
    description="依据未剪切静态背景配准多曝光 16 位线性 PNG，输出 float32 辐亮度 TIFF。",
    version="1.0.0",
)


@app.exception_handler(RequestValidationError)
async def _validation_exception_handler(
    request: Request, exc: RequestValidationError
) -> JSONResponse:
    # 默认处理器在输入含 Infinity/NaN 等非标准 JSON 数值时，回显会再次触发
    # 序列化异常而变成 500；这里只返回安全可序列化的精简错误信息。
    safe_errors = []
    for err in exc.errors():
        safe_errors.append(
            {
                "type": str(err.get("type", "value_error")),
                "loc": [str(x) for x in err.get("loc", [])],
                "msg": str(err.get("msg", "")),
            }
        )
    return JSONResponse(status_code=422, content={"detail": safe_errors})


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.post("/merges", response_model=MergeResponse)
def post_merges(req: MergeRequest) -> MergeResponse:
    try:
        req.validate_cross_fields()

        # 解码并校验所有帧：必须是同尺寸 16 位 RGB PNG。
        frames = []
        shape = None
        for i, payload in enumerate(req.frames):
            image = decode_png16_base64(payload)
            if shape is None:
                shape = image.shape
            elif image.shape != shape:
                raise ValueError(
                    f"第 {i} 帧尺寸 {image.shape[:2]} 与首帧 {shape[:2]} 不一致"
                )
            frames.append(image)

        # 依据未剪切的静态背景配准（不允许移动前景牵引位移）。
        shifts = register_frames(frames, req.exposures, req.reference_index)

        # 运动检测 + 辐亮度融合。
        radiance, motion_mask, unknown_mask = merge_frames(
            frames, req.exposures, shifts, req.reference_index, req.tolerance
        )

        return MergeResponse(
            radiance_tiff=encode_tiff_float32_base64(radiance),
            motion_mask_png=encode_png_base64(motion_mask),
            unknown_mask_png=encode_png_base64(unknown_mask),
            shifts=[{"dy": int(dy), "dx": int(dx)} for dy, dx in shifts],
        )
    except RegistrationError as exc:
        logger.info("背景配准失败: %s", exc)
        return JSONResponse(
            status_code=422,
            content={"detail": f"背景无法配准: {exc}"},
        )
    except ValueError as exc:
        logger.info("输入非法: %s", exc)
        return JSONResponse(status_code=422, content={"detail": str(exc)})
