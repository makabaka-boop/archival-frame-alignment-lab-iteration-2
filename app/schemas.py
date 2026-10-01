"""Pydantic 请求/响应模型。

Pydantic 的类型校验（非数组、元素非字符串等）自动产出 422；域校验
（长度、重复次数、ASCII 范围）在 :mod:`app.validation` 中实现，
于路由内调用以复用同一份错误消息。

索引对在接口层同样使用 ``[left_index, right_index]`` 整数二元组的规范身份，
与算法层、存储层（:mod:`app.pairs`）完全一致。
"""

from __future__ import annotations

from datetime import datetime

from pydantic import BaseModel, Field, field_validator

from .pairs import IndexPair, coerce_pairs


class JobCreate(BaseModel):
    left: list[str] = Field(description="左扫描机指纹数组（1..20000 项）")
    right: list[str] = Field(description="右扫描机指纹数组（1..20000 项）")


class _PairSetIn(BaseModel):
    """替换类约束请求的公共字段：乐观版本。"""

    base_version: int | None = Field(
        default=None,
        description=(
            "本次修改所基于的作业版本（取自最近一次 GET/PUT 响应的 version）；"
            "与当前版本不一致时返回 409 且状态不变，缺省时不做该检查"
        ),
    )


class AnchorIn(_PairSetIn):
    anchors: list[IndexPair] = Field(
        default_factory=list,
        description="锚点索引对集合；空数组（或缺省）表示清空锚点、恢复全局最优",
    )

    @field_validator("anchors", mode="before")
    @classmethod
    def _coerce(cls, value: object) -> object:
        # 允许 {"anchors": null}，等价于清空；其余经共享身份的严格形状校验
        # （拒绝布尔等伪整数），与算法层/存储层看到的是同一类对象。
        return [] if value is None else coerce_pairs("锚点", value)


class ForbiddenIn(_PairSetIn):
    forbidden_pairs: list[IndexPair] = Field(
        default_factory=list,
        description=(
            "禁止对应对（人工确认的同指纹误配）集合；空数组（或缺省）表示"
            "不禁止任何对应。元素必须是索引合法且两侧指纹相等的索引对；"
            "与当前锚点相交时整次请求被拒绝（422）"
        ),
    )

    @field_validator("forbidden_pairs", mode="before")
    @classmethod
    def _coerce(cls, value: object) -> object:
        return [] if value is None else coerce_pairs("禁止对应对", value)


class Pair(BaseModel):
    left_index: int
    right_index: int


class JobOut(BaseModel):
    id: str
    left: list[str]
    right: list[str]
    anchors: list[Pair]
    forbidden_pairs: list[Pair]
    result: list[Pair]
    length: int
    version: int
    updated_at: datetime
