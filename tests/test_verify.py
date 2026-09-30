"""验收用例：协议、最长性、字典序裁决、锚点、性能与持久化。"""

from __future__ import annotations

import random
import sys
import time
from collections import Counter
from pathlib import Path

import httpx
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from brute import brute  # noqa: E402

from conftest import assert_chain_valid, create_job, set_anchors  # noqa: E402


def _pairs(body: dict) -> list[tuple[int, int]]:
    return [(p["left_index"], p["right_index"]) for p in body["result"]]


def _anchor_pairs(body: dict) -> set[tuple[int, int]]:
    return {(p["left_index"], p["right_index"]) for p in body["anchors"]}


# ---------------------------------------------------------------- 基础协议


def test_health(client: httpx.Client) -> None:
    assert client.get("/health").json() == {"status": "ok"}


def test_create_and_get_roundtrip(client: httpx.Client) -> None:
    left = ["a", "b", "c"]
    right = ["b", "c", "d"]
    body = create_job(client, left, right)
    assert body["length"] == 2
    pairs = assert_chain_valid(body, left, right)
    assert pairs == [(1, 0), (2, 1)]  # b,c 的唯一对齐
    # 查询同一任务，结果一致（可重启续作的前提：已落库）
    got = client.get(f"/api/jobs/{body['id']}").json()
    assert got == body


def test_get_missing_job_404(client: httpx.Client) -> None:
    resp = client.get("/api/jobs/00000000-0000-0000-0000-000000000000")
    assert resp.status_code == 404
    resp = set_anchors(client, "00000000-0000-0000-0000-000000000000", [])
    assert resp.status_code == 404


def test_no_match_returns_empty(client: httpx.Client) -> None:
    body = create_job(client, ["a"], ["b"])
    assert body["result"] == []
    assert body["length"] == 0
    assert body["anchors"] == []


def test_fingerprints_are_byte_case_sensitive_no_normalization(
    client: httpx.Client,
) -> None:
    # A 与 a 不同；带空格、~、32 字符等可打印字符按字节比较，不做规范化。
    fp32 = "x" * 32
    left = ["A", "a", " x", "~", fp32]
    right = ["a", "A", " x", "y", fp32]
    body = create_job(client, left, right)
    pairs = assert_chain_valid(body, left, right)
    # A(0)->A(1) 与 a(1)->a(0) 交叉只能二选一，字典序从 (0,1) 起；
    # 随后空格前缀、32 字符长指纹按字节精确相等匹配。
    assert pairs == [(0, 1), (2, 2), (4, 4)]


# ---------------------------------------------------------------- 重复指纹


def test_repeated_fingerprints_tie_break_lexicographic(client: httpx.Client) -> None:
    # 每侧同一指纹重复 4 次：任何完美对齐都有 4 对，字典序最小为前对角。
    left = ["v"] * 4
    right = ["v"] * 4
    body = create_job(client, left, right)
    assert _pairs(body) == [(0, 0), (1, 1), (2, 2), (3, 3)]


def test_repeated_with_offsets_lexicographic(client: httpx.Client) -> None:
    left = ["a", "a", "b"]
    right = ["a", "a", "a", "b"]
    body = create_job(client, left, right)
    pairs = assert_chain_valid(body, left, right)
    # 最长 3；字典序最小：从 (0,0) 起，之后 a->(1,1)，再 b->(2,3)
    assert pairs == [(0, 0), (1, 1), (2, 3)]


# ---------------------------------------------------------------- 锚点


def test_invalid_anchors_422_and_state_unchanged(client: httpx.Client) -> None:
    left = ["a", "b", "a", "c"]
    right = ["b", "a", "c", "a"]
    body = create_job(client, left, right)
    jid = body["id"]
    original = client.get(f"/api/jobs/{jid}").json()

    bad_anchors = [
        [[4, 0]],  # 左索引越界
        [[0, 9]],  # 右索引越界
        [[-1, 0]],  # 负索引
        [[0, 0]],  # a != b 指纹不等
        [[1, 2]],  # b != c
        [[1, 0], [0, 1]],  # 两侧都不递增（交叉）
        [[0, 1], [2, 1]],  # 右索引重复
        [[0, 1], [0, 3]],  # 左索引重复
        [["x", 1]],  # 类型错误
        [[0]],  # 形状错误
    ]
    for bad in bad_anchors:
        resp = set_anchors(client, jid, bad)
        assert resp.status_code == 422, f"应对 {bad} 返回 422，实际 {resp.status_code}: {resp.text}"

    # 全部拒绝后，状态与结果必须与初始完全一致
    after = client.get(f"/api/jobs/{jid}").json()
    assert after["anchors"] == original["anchors"]
    assert after["result"] == original["result"]


def test_anchors_replace_and_clear_to_global_optimum(client: httpx.Client) -> None:
    left = ["a", "b", "a", "b", "c"]
    right = ["b", "a", "b", "a", "c"]
    body = create_job(client, left, right)
    jid = body["id"]
    global_pairs = _pairs(body)
    assert len(global_pairs) == 4  # a,b,a,c 可对齐

    # 合法锚点：强制第二个 a 对齐到右侧第二个 a（索引 3）
    resp = set_anchors(client, jid, [[2, 3]])
    assert resp.status_code == 200, resp.text
    anchored = resp.json()
    assert _anchor_pairs(anchored) == {(2, 3)}
    pairs = assert_chain_valid(anchored, left, right)
    assert (2, 3) in pairs

    # 用另一个锚点集合替换旧集合：旧锚点不得残留，结果随之改变
    # a(0)->a(3) 后只能再接 c，(2,3) 必然消失
    resp2 = set_anchors(client, jid, [[0, 3]])
    assert resp2.status_code == 200
    replaced = resp2.json()
    assert _anchor_pairs(replaced) == {(0, 3)}
    pairs2 = assert_chain_valid(replaced, left, right)
    assert pairs2 == [(0, 3), (4, 4)]
    assert (2, 3) not in pairs2

    # 清空锚点 -> 回到全局最长且字典序最小解
    resp3 = set_anchors(client, jid, [])
    assert resp3.status_code == 200
    cleared = resp3.json()
    assert cleared["anchors"] == []
    assert _pairs(cleared) == global_pairs


def test_first_and_last_anchors(client: httpx.Client) -> None:
    left = ["a", "x", "b", "y"]
    right = ["y", "a", "x", "b"]
    body = create_job(client, left, right)
    jid = body["id"]
    # 首锚点 (0,1)=a、尾锚点 (3,0)=y，把可行解夹在中间
    resp = set_anchors(client, jid, [[0, 1], [3, 0]])
    # (3,0) j=0 小于前锚点 j=1 -> 非法交叉
    assert resp.status_code == 422

    resp = set_anchors(client, jid, [[0, 1], [2, 3]])  # 首 a 与尾 b
    assert resp.status_code == 200, resp.text
    pairs = assert_chain_valid(resp.json(), left, right)
    assert pairs[0] == (0, 1)
    assert pairs[-1] == (2, 3)


def test_anchor_can_force_shorter_optimum(client: httpx.Client) -> None:
    # 受约束的较短最优解：全局 LCS 长度 3；合法锚点 x(0,3) 不属于任何
    # 最长链，强制包含后最优仅 1 对。
    # left:  0:x 1:a 2:b 3:c      right: 0:a 1:b 2:c 3:x
    left = ["x", "a", "b", "c"]
    right = ["a", "b", "c", "x"]
    body = create_job(client, left, right)
    assert body["length"] == 3
    jid = body["id"]

    resp = set_anchors(client, jid, [[0, 3]])  # x == x，锚点本身合法
    assert resp.status_code == 200, resp.text
    anchored = resp.json()
    pairs = assert_chain_valid(anchored, left, right)
    assert pairs == [(0, 3)]  # 锚点把唯一可行对固定，全局最优被排除
    assert anchored["length"] == 1


# ---------------------------------------------------------------- 穷举交叉核对


def _bounded_side(alphabet: str, n: int, seed: int) -> list[str]:
    rng = random.Random(seed)
    while True:
        side = [rng.choice(alphabet) for _ in range(n)]
        if all(v <= 4 for v in Counter(side).values()):
            return side


def test_small_cases_exhaustive_crosscheck(client: httpx.Client) -> None:
    """小规模随机用例：生产 API 结果必须与 O(n^2) 穷举完全一致（含字典序）。"""
    rng = random.Random(20260920)
    checked = 0
    for case in range(200):
        n_left = rng.randint(1, 8)
        n_right = rng.randint(1, 8)
        alphabet = "abcde"
        left = _bounded_side(alphabet, n_left, case * 2 + 1)
        right = _bounded_side(alphabet, n_right, case * 2 + 2)
        body = create_job(client, left, right)
        pairs = assert_chain_valid(body, left, right)
        expected = brute(left, right)
        assert pairs == expected, f"无锚点不一致: {left} / {right}: {pairs} != {expected}"

        # 从穷举最优解中随机抽取合法锚点链子集，再核锚点结果
        jid = body["id"]
        if expected:
            k = rng.randint(1, min(3, len(expected)))
            anchors = [list(p) for p in sorted(rng.sample(expected, k))]
            anchors.sort()
            resp = set_anchors(client, jid, anchors)
            assert resp.status_code == 200, resp.text
            anchored_body = resp.json()
            anchored_pairs = assert_chain_valid(anchored_body, left, right)
            expected_anchored = brute(left, right, anchors)
            assert anchored_pairs == expected_anchored, (
                f"锚点不一致: {left} / {right} / {anchors}: "
                f"{anchored_pairs} != {expected_anchored}"
            )
            assert set(map(tuple, anchors)).issubset(set(anchored_pairs))

            # 清空后恢复全局解
            cleared = set_anchors(client, jid, []).json()
            assert _pairs(cleared) == expected
        checked += 1
    assert checked == 200


def test_lexicographic_tiebreak_bruteforce(client: httpx.Client) -> None:
    """专门制造并列最长：每个值重复多次，核对字典序最小裁决。"""
    cases = [
        (["a", "a"], ["a", "a"]),
        (["a", "a", "b"], ["b", "a", "a"]),
        (["z", "z", "z"], ["z", "z", "z"]),
        (["a", "b", "a", "b"], ["b", "a", "b", "a"]),
    ]
    for left, right in cases:
        body = create_job(client, left, right)
        assert _pairs(body) == brute(left, right), (left, right)


# ---------------------------------------------------------------- 输入域 422


@pytest.mark.parametrize(
    "payload",
    [
        {"left": [], "right": ["a"]},  # 长度 0
        {"left": ["a"], "right": []},
        {"left": ["a"] * 20001, "right": ["a"]},  # 超长
        {"left": ["a", "a", "a", "a", "a"], "right": ["a"]},  # 同值 5 次
        {"left": ["a"], "right": ["A", "A", "A", "A", "A"]},
        {"left": [""], "right": ["a"]},  # 空指纹
        {"left": ["x" * 33], "right": ["x" * 33]},  # 超长指纹
        {"left": ["café"], "right": ["café"]},  # 非 ASCII
        {"left": ["a\nb"], "right": ["a\nb"]},  # 不可打印字符
        {"left": "ab", "right": ["a"]},  # 非数组
        {"left": [1, 2], "right": [1, 2]},  # 非字符串
    ],
)
def test_invalid_create_payload_422(client: httpx.Client, payload: dict) -> None:
    resp = client.post("/api/jobs", json=payload)
    assert resp.status_code == 422, resp.text


# ---------------------------------------------------------------- 性能：3 秒


def test_max_input_completes_within_3_seconds(client: httpx.Client) -> None:
    # 最大规模：两侧 20000 项，每值恰好 4 次（K=80000 个相等对），结果 20000 对。
    pattern = [f"f{g:04d}" for g in range(5000) for _ in range(4)]
    assert len(pattern) == 20000
    started = time.perf_counter()
    body = create_job(client, pattern, pattern)
    elapsed = time.perf_counter() - started
    assert body["length"] == 20000
    assert _pairs(body) == [(i, i) for i in range(20000)]
    assert elapsed < 3.0, f"最大输入耗时 {elapsed:.3f}s 超过 3 秒"

    # 同任务再设一组首尾锚点重算，仍应在 3 秒内
    started = time.perf_counter()
    resp = set_anchors(client, body["id"], [[0, 0], [19999, 19999]])
    elapsed2 = time.perf_counter() - started
    assert resp.status_code == 200
    assert resp.json()["length"] == 20000
    assert elapsed2 < 3.0, f"锚点重算耗时 {elapsed2:.3f}s 超过 3 秒"


def test_max_input_disjoint_within_3_seconds(client: httpx.Client) -> None:
    left = [f"L{g:05d}" for g in range(20000)]
    right = [f"R{g:05d}" for g in range(20000)]
    started = time.perf_counter()
    body = create_job(client, left, right)
    elapsed = time.perf_counter() - started
    assert body["length"] == 0
    assert body["result"] == []
    assert elapsed < 3.0, f"无匹配最大输入耗时 {elapsed:.3f}s"


# ---------------------------------------------------------------- 持久化


def test_persistence_in_database(client: httpx.Client) -> None:
    """结果与锚点实际写入 PostgreSQL，而非仅保存在进程内存。"""
    psycopg = pytest.importorskip("psycopg")
    left = ["a", "b", "c", "a"]
    right = ["a", "c", "b", "a"]
    body = create_job(client, left, right)
    jid = body["id"]
    set_anchors(client, jid, [[0, 0]])

    import os

    dsn = os.environ["PG_DSN"]
    with psycopg.connect(dsn) as conn, conn.cursor() as cur:
        cur.execute(
            "SELECT left_data, right_data, anchors, result FROM jobs WHERE id = %s",
            (jid,),
        )
        row = cur.fetchone()
    assert row is not None, "任务未写入 PostgreSQL"
    left_col, right_col, anchors_col, result_col = row
    assert list(left_col) == left
    assert list(right_col) == right
    assert [tuple(p) for p in anchors_col] == [(0, 0)]
    assert (0, 0) in [tuple(p) for p in result_col]
