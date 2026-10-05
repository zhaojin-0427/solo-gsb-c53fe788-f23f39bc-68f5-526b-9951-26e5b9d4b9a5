"""/merges 的请求/响应模型与字段校验。"""
from __future__ import annotations

import math

from pydantic import BaseModel, Field, field_validator

from . import config as C


class MergeRequest(BaseModel):
    frames: list[str] = Field(
        ...,
        description="3~9 张同尺寸 16 位线性 RGB PNG 的 Base64（与 exposures 顺序一致）",
    )
    exposures: list[float] = Field(
        ..., description="各帧有限、正的曝光秒数（如 0.01、0.1、1.0）"
    )
    reference_index: int = Field(..., description="参考帧序号，0 起始")
    tolerance: float = Field(..., description="有限、非负的辐亮度容差")

    @field_validator("frames")
    @classmethod
    def _validate_frames(cls, v: list[str]) -> list[str]:
        if not (C.MIN_FRAMES <= len(v) <= C.MAX_FRAMES):
            raise ValueError(f"帧数必须在 {C.MIN_FRAMES}~{C.MAX_FRAMES} 之间")
        return v

    @field_validator("exposures")
    @classmethod
    def _validate_exposures(cls, v: list[float]) -> list[float]:
        if not all(isinstance(x, (int, float)) and math.isfinite(x) and x > 0 for x in v):
            raise ValueError("曝光秒数必须为有限正数")
        return [float(x) for x in v]

    @field_validator("tolerance")
    @classmethod
    def _validate_tolerance(cls, v: float) -> float:
        if not math.isfinite(v) or v < 0:
            raise ValueError("tolerance 必须为有限非负数")
        return float(v)

    @field_validator("reference_index")
    @classmethod
    def _validate_reference_index(cls, v: int) -> int:
        if v < 0:
            raise ValueError("reference_index 必须是非负的帧序号（0 起始）")
        return v

    def validate_cross_fields(self) -> None:
        if len(self.exposures) != len(self.frames):
            raise ValueError("exposures 数量必须与 frames 数量相同")
        if not (0 <= self.reference_index < len(self.frames)):
            raise ValueError("reference_index 必须是合法的帧序号（0 起始）")
        if not math.isfinite(self.tolerance) or self.tolerance < 0:
            raise ValueError("tolerance 必须为有限非负数")


class MergeResponse(BaseModel):
    radiance_tiff: str = Field(..., description="float32 RGB TIFF（Base64），未知像素为 NaN")
    motion_mask_png: str = Field(..., description="运动掩膜 8 位 PNG（Base64），运动像素=255")
    unknown_mask_png: str = Field(..., description="未知掩膜 8 位 PNG（Base64），未知像素=255")
    shifts: list[dict[str, int]] = Field(
        ...,
        description=(
            "每帧相对参考帧的整数平移 {dy, dx}，顺序与输入一致；参考帧为 {0,0}。"
            "含义：注册后参考坐标 (y,x) 的内容取自源帧 (y+dy, x+dx)，每轴绝对值不超过 8"
        ),
    )
