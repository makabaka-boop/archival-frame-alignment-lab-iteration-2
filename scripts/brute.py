"""小规模穷举参考实现，仅用于与生产算法交叉核对。"""

from __future__ import annotations

from typing import Sequence


def _brute_segment(left, right, lo_i, hi_i, lo_j, hi_j):
    # dp[(i,j)] = 该点之后（不含）的最大链长；从后往前枚举。
    matches = [
        (i, j)
        for i in range(lo_i, hi_i)
        for j in range(lo_j, hi_j)
        if left[i] == right[j]
    ]
    matches.sort()  # 升序；(i,j) 全序
    dp = {pair: 1 for pair in matches}
    best_by_suffix: dict[tuple[int, int], int] = {}
    # 从后向前：dp[p] = 1 + max{dp[q] : q>p, q.i>p.i, q.j>p.j}
    for idx in range(len(matches) - 1, -1, -1):
        p = matches[idx]
        best = 1
        for q in matches[idx + 1 :]:
            if q[0] > p[0] and q[1] > p[1]:
                cand = dp[q] + 1
                if cand > best:
                    best = cand
        dp[p] = best
    if not matches:
        return []
    total = max(dp.values())
    # 字典序最小重建：每步取满足可行的最小 (i,j)。
    result = []
    ci, cj = lo_i - 1, lo_j - 1
    remaining = total
    while remaining:
        candidates = [
            p
            for p in matches
            if p[0] > ci
            and p[1] > cj
            and dp[p] == remaining
        ]
        pick = min(candidates)
        result.append(pick)
        ci, cj = pick
        remaining -= 1
    return result


def brute(left: Sequence[str], right: Sequence[str], anchors=()) -> list[tuple[int, int]]:
    ordered = sorted(tuple(a) for a in anchors)
    result: list[tuple[int, int]] = []
    prev_i, prev_j = -1, -1
    for i, j in ordered:
        result.extend(
            _brute_segment(left, right, prev_i + 1, i, prev_j + 1, j)
        )
        result.append((i, j))
        prev_i, prev_j = i, j
    result.extend(
        _brute_segment(left, right, prev_i + 1, len(left), prev_j + 1, len(right))
    )
    return result
