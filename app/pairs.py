"""规范化的索引对身份：接口、算法与存储层共享同一份表示与校验。

两侧扫描机上的一对索引在任何层都规范为 ``(left_index, right_index)``
整数二元组：

* 接口层（Pydantic 模型）用 :data:`IndexPair` 接收请求体；
* 算法层（:mod:`app.lcs`）直接消费规范化后的元组；
* 存储层（PostgreSQL JSON 列）读回时经 :func:`rows_to_pairs` 转回同一身份。

集合语义统一：重复元素去重、按 ``(左索引, 右索引)`` 升序排列后作为规范形态。
锚点与禁止对的具体合法性（范围、指纹、相容性）在 :mod:`app.lcs` 中校验。
"""

from __future__ import annotations

from typing import Iterable, Sequence, TypeAlias

IndexPair: TypeAlias = tuple[int, int]


class ConstraintError(ValueError):
    """对应约束（锚点或禁止对应对）不合法。消息可直接返回给调用方。"""


def coerce_pairs(kind: str, raw: Iterable[object]) -> list[IndexPair]:
    """把任意输入元素强转为合法形状的整数对；形状非法抛 :class:`ConstraintError`。

    ``kind`` 是约束的中文名（如 "锚点"/"禁止对应对"），仅用于错误消息。
    这里只负责形状与类型（拒绝布尔等伪整数）；范围、指纹相等与集合相容性
    由算法层结合具体作业判定。
    """
    # 字符串/字节虽可迭代，但绝不是“索引对的集合”；先挡住，避免按字符枚举。
    if not isinstance(raw, (list, tuple, set)):
        raise ConstraintError(f"{kind}必须是 [左索引, 右索引] 整数对的数组")
    pairs: list[IndexPair] = []
    for pos, item in enumerate(raw):
        if (
            not isinstance(item, (list, tuple))
            or len(item) != 2
            or not all(isinstance(v, int) and not isinstance(v, bool) for v in item)
        ):
            raise ConstraintError(f"第 {pos} 个{kind}不是 [左索引, 右索引] 整数对")
        pairs.append((int(item[0]), int(item[1])))
    return pairs


def normalize_pairs(raw: Sequence[object], kind: str) -> list[IndexPair]:
    """强转后去重并按两侧索引升序排列——跨层统一的规范集合形态。"""
    return sorted(set(coerce_pairs(kind, raw)))


def rows_to_pairs(raw: object) -> list[IndexPair]:
    """把数据库 JSON 列读回的值转回规范索引对（元素均为合法整数）。

    存储层写入的值来自算法层规范化结果；这里仍做一次形状强转并排序，
    保证任何路径读到的都是同一份规范身份。
    """
    if raw is None:
        return []
    return normalize_pairs(raw, kind="索引对")
