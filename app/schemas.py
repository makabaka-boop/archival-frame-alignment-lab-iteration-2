"""Pydantic 请求/响应模型。

Pydantic 的类型校验（非数组、元素非字符串等）自动产出 422；域校验
（长度、重复次数、ASCII 范围）在 :mod:`app.validation` 中实现，
于路由内调用以复用同一份错误消息。
"""

from __future__ import annotations

from datetime import datetime

from pydantic import BaseModel, Field, field_validator


class JobCreate(BaseModel):
    left: list[str] = Field(description="左扫描机指纹数组（1..20000 项）")
    right: list[str] = Field(description="右扫描机指纹数组（1..20000 项）")


class AnchorIn(BaseModel):
    anchors: list[tuple[int, int]] = Field(
        default_factory=list,
        description="锚点索引对集合；空数组（或缺省）表示清空锚点、恢复全局最优",
    )
    base_version: int | None = Field(
        default=None,
        description=(
            "本次修改所基于的作业版本（取自最近一次 GET/PUT 响应的 version）；"
            "与当前版本不一致时返回 409 且状态不变，缺省时不做该检查"
        ),
    )

    @field_validator("anchors", mode="before")
    @classmethod
    def _none_becomes_empty(cls, value: object) -> object:
        # 允许 {"anchors": null}，等价于清空。
        return [] if value is None else value


class Pair(BaseModel):
    left_index: int
    right_index: int


class JobOut(BaseModel):
    id: str
    left: list[str]
    right: list[str]
    anchors: list[Pair]
    result: list[Pair]
    length: int
    version: int
    updated_at: datetime
