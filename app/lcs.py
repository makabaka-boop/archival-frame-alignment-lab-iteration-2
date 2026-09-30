"""最长对应（LCS）求解。

目标：在左右两条指纹数组之间，求零基索引对序列，使两侧索引都严格递增且
指纹相等；先最大化对数，并列时取索引对序列字典序最小者。

输入约束（由 API 层保证）：

* 每侧 1..20000 项；
* 指纹为 1..32 个 ASCII 可打印字符（32..126），按字节区分大小写；
* 每个指纹在每侧至多出现 4 次。

设 K 为相等指纹的索引对总数（K <= 80000）。算法把 LCS 归约为二维点列上的
最长链（LIS），用 Fenwick 树求后缀最大值，整体 O(K log K)；锚点把索引平面
切成互不相交的矩形段，逐段独立求解，复杂度不变。
"""

from __future__ import annotations

from bisect import bisect_left, bisect_right
from collections import defaultdict
from typing import Sequence


class AnchorError(ValueError):
    """锚点不合法。消息可直接返回给调用方。"""


def validate_anchors(
    left: Sequence[str],
    right: Sequence[str],
    anchors: Sequence[tuple[int, int]],
) -> list[tuple[int, int]]:
    """校验锚点集合并返回按左索引（也是右索引）升序排列的列表。

    合法条件与结果条件一致：索引落在各自数组范围内、两侧指纹相等、
    左右索引各自严格递增（即不得共用索引或交叉）。
    """
    n_left, n_right = len(left), len(right)
    normalized: list[tuple[int, int]] = []
    for pos, anchor in enumerate(anchors):
        if (
            not isinstance(anchor, (list, tuple))
            or len(anchor) != 2
            or not all(isinstance(v, int) and not isinstance(v, bool) for v in anchor)
        ):
            raise AnchorError(f"第 {pos} 个锚点不是 [左索引, 右索引] 整数对")
        i, j = int(anchor[0]), int(anchor[1])
        if not 0 <= i < n_left:
            raise AnchorError(f"第 {pos} 个锚点左索引 {i} 超出范围 [0, {n_left - 1}]")
        if not 0 <= j < n_right:
            raise AnchorError(f"第 {pos} 个锚点右索引 {j} 超出范围 [0, {n_right - 1}]")
        if left[i] != right[j]:
            raise AnchorError(f"第 {pos} 个锚点两侧指纹不相等：({i!r}, {j!r})")
        normalized.append((i, j))

    ordered = sorted(normalized)
    prev_i = prev_j = -1
    for i, j in ordered:
        # 左索引相等会被严格递增挡住；右索引相等或交叉同样被挡住。
        if i <= prev_i or j <= prev_j:
            raise AnchorError(
                f"锚点集合必须在两侧索引上都严格递增，违规点：({i}, {j})"
            )
        prev_i, prev_j = i, j
    return ordered


class _FenwickMax:
    """前缀最大值 Fenwick 树，下标 1..size。"""

    __slots__ = ("size", "tree")

    def __init__(self, size: int) -> None:
        self.size = size
        self.tree = [0] * (size + 1)

    def update(self, idx: int, value: int) -> None:
        size, tree = self.size, self.tree
        while idx <= size:
            if value > tree[idx]:
                tree[idx] = value
            idx += idx & -idx

    def query(self, idx: int) -> int:
        """[1..idx] 上的最大值。"""
        tree, best = self.tree, 0
        while idx > 0:
            if tree[idx] > best:
                best = tree[idx]
            idx -= idx & -idx
        return best


def _solve_segment(
    left: Sequence[str],
    right: Sequence[str],
    positions_right: dict[str, list[int]],
    lo_i: int,
    hi_i: int,
    lo_j: int,
    hi_j: int,
) -> list[tuple[int, int]]:
    """求矩形 lo_i<=i<hi_i、lo_j<=j<hi_j 内的字典序最小最长链。

    矩形边界来自锚点，保证段间索引不相交。
    """
    if lo_i >= hi_i or lo_j >= hi_j:
        return []

    # 收集段内的相等指纹对，按 (i, j) 升序；顺便压缩 j 坐标。
    pairs: list[tuple[int, int]] = []
    js_used: list[int] = []
    for i in range(lo_i, hi_i):
        spots = positions_right.get(left[i])
        if not spots:
            continue
        start = bisect_left(spots, lo_j)
        stop = bisect_right(spots, hi_j - 1, start)
        for k in range(start, stop):
            j = spots[k]
            pairs.append((i, j))
            js_used.append(j)

    if not pairs:
        return []

    js_sorted = sorted(set(js_used))
    j_to_idx = {j: k for k, j in enumerate(js_sorted)}
    m = len(js_sorted)

    # 按下 i 降序处理；反向映射后，Fenwick 前缀即“j 严格更大”的后缀最大值。
    # 同 i 一组必须先全部查询、再全部写入，避免同 i 的点互相连接。
    buckets: dict[int, list[tuple[int, int]]] = defaultdict(list)
    bit = _FenwickMax(m)
    p = len(pairs) - 1
    best = 0
    while p >= 0:
        group_end = p
        group_i = pairs[p][0]
        group_vals: list[int] = []
        while p >= 0 and pairs[p][0] == group_i:
            i, j = pairs[p]
            rev = m - j_to_idx[j]  # 1..m；j 越大，rev 越小
            value = 1 + bit.query(rev - 1)
            group_vals.append(value)
            if value > best:
                best = value
            p -= 1
        # 该 i 组统一落盘并写树。组内 j 降序写入，桶内之后整体反转即为升序。
        for offset in range(group_end - p):
            idx = group_end - offset
            value = group_vals[offset]
            i, j = pairs[idx]
            buckets[value].append((i, j))
            bit.update(m - j_to_idx[j], value)

    if best == 0:
        return []

    for value in buckets:
        buckets[value].reverse()  # 变为 (i, j) 升序

    # 贪心重建：第 rem 步在 dp==rem 的点里取第一个“晚于当前游标”的点，
    # 即剩余可行解中首对字典序最小者。桶已按 (i,j) 升序：先用 i 二分，
    # 再在该后缀里线性找首个 j 更大的点。
    result: list[tuple[int, int]] = []
    ci, cj = lo_i - 1, lo_j - 1
    for rem in range(best, 0, -1):
        bucket = buckets[rem]
        k = bisect_right(bucket, (ci, hi_j))  # 第一个 i > ci 的位置
        while k < len(bucket) and bucket[k][1] <= cj:
            k += 1
        if k >= len(bucket):  # 理论上不可达：best 是真实链长
            raise RuntimeError("内部错误：最长链重建失败")
        ci, cj = bucket[k]
        result.append((ci, cj))
    return result


def solve(
    left: Sequence[str],
    right: Sequence[str],
    anchors: Sequence[tuple[int, int]] = (),
) -> list[tuple[int, int]]:
    """求包含全部锚点的最长对应；先最长，再字典序最小。

    调用前可用 :func:`validate_anchors` 校验锚点。锚点沿两条轴把平面切成
    互不相交的矩形段，逐段取字典序最小最优解，再以锚点连接。
    """
    ordered = validate_anchors(left, right, anchors) if anchors else []

    positions_right: dict[str, list[int]] = defaultdict(list)
    for j, fp in enumerate(right):
        positions_right[fp].append(j)

    result: list[tuple[int, int]] = []
    prev_i, prev_j = -1, -1
    for i, j in ordered:
        result.extend(
            _solve_segment(left, right, positions_right, prev_i + 1, i, prev_j + 1, j)
        )
        result.append((i, j))
        prev_i, prev_j = i, j
    result.extend(
        _solve_segment(
            left,
            right,
            positions_right,
            prev_i + 1,
            len(left),
            prev_j + 1,
            len(right),
        )
    )
    return result
