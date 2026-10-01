"""禁止对应对（人工确认的同指纹误配）验收：协议、最长性、字典序、锚点冲突、
两类修改交错、穷举交叉核对与迁移后读取。

面向真实 HTTP 服务与真实 PostgreSQL，不使用任何桩。
"""

from __future__ import annotations

import json
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

from conftest import (  # noqa: E402
    assert_chain_valid,
    create_job,
    forbidden_pairs,
    set_anchors,
    set_forbidden,
)


def _pairs(body: dict) -> list[tuple[int, int]]:
    return [(p["left_index"], p["right_index"]) for p in body["result"]]


def _anchor_set(body: dict) -> set[tuple[int, int]]:
    return {(p["left_index"], p["right_index"]) for p in body["anchors"]}


# ---------------------------------------------------------------- 基础协议


def test_job_representation_includes_forbidden_pairs(client: httpx.Client) -> None:
    body = create_job(client, ["a", "b"], ["a", "b"])
    assert body["forbidden_pairs"] == []
    got = client.get(f"/api/jobs/{body['id']}").json()
    assert got == body
    assert got["forbidden_pairs"] == []


def test_get_and_put_unknown_job_404(client: httpx.Client) -> None:
    jid = "00000000-0000-0000-0000-000000000000"
    resp = client.put(f"/api/jobs/{jid}/forbidden-pairs", json={"forbidden_pairs": []})
    assert resp.status_code == 404


# ---------------------------------------------------------------- 语义：重复指纹


def test_forbidden_reroutes_duplicate_fingerprint_chain(client: httpx.Client) -> None:
    # 同一指纹在两侧各出现 4 次：全局字典序最小解为对角 (0,0)..(3,3)。
    left = ["v"] * 4
    right = ["v"] * 4
    body = create_job(client, left, right)
    assert _pairs(body) == [(0, 0), (1, 1), (2, 2), (3, 3)]
    jid = body["id"]

    # 禁止 (0,0)：最长降为 3，字典序最小链整体顺移 (0,1)(1,2)(2,3)。
    resp = set_forbidden(client, jid, [[0, 0]])
    assert resp.status_code == 200, resp.text
    first = resp.json()
    assert forbidden_pairs(first) == [(0, 0)]
    pairs = assert_chain_valid(first, left, right)
    assert pairs == [(0, 1), (1, 2), (2, 3)]
    assert (0, 0) not in pairs
    assert first["version"] == 1

    # 替换为更大的禁止集合（含交叉、共用索引——禁止对允许无序）。
    resp2 = set_forbidden(
        client, jid, [[0, 0], [0, 1], [1, 0], [2, 2]]
    )
    assert resp2.status_code == 200
    second = resp2.json()
    assert forbidden_pairs(second) == [(0, 0), (0, 1), (1, 0), (2, 2)]
    pairs2 = assert_chain_valid(second, left, right)
    assert set(pairs2).isdisjoint({(0, 0), (0, 1), (1, 0), (2, 2)})
    assert second["version"] == 2

    # 清空禁止集合 -> 恢复全局字典序最小最优。
    cleared = set_forbidden(client, jid, []).json()
    assert cleared["forbidden_pairs"] == []
    assert _pairs(cleared) == [(0, 0), (1, 1), (2, 2), (3, 3)]


def test_forbidden_every_equal_pair_yields_no_solution(client: httpx.Client) -> None:
    # 无解情形：把全部相等指纹对都禁止，结果为空、长度 0。
    left = ["a", "b", "a"]
    right = ["a", "b", "a"]
    body = create_job(client, left, right)
    assert body["length"] == 3
    jid = body["id"]
    all_pairs = [
        [i, j]
        for i in range(len(left))
        for j in range(len(right))
        if left[i] == right[j]
    ]
    resp = set_forbidden(client, jid, all_pairs)
    assert resp.status_code == 200, resp.text
    blocked = resp.json()
    assert blocked["result"] == []
    assert blocked["length"] == 0
    assert_chain_valid(blocked, left, right)

    # 无匹配数组天然无解；禁止集合也必须为空可清空。
    none = create_job(client, ["x"], ["y"])
    assert none["result"] == []
    resp2 = set_forbidden(client, none["id"], [])
    assert resp2.status_code == 200 and resp2.json()["result"] == []


def test_forbidden_set_is_normalized(client: httpx.Client) -> None:
    """规范化身份：重复元素去重、乱序按 (左,右) 升序排列，跨响应/读取一致。"""
    left = ["a", "b", "c"]
    right = ["a", "b", "c"]
    jid = create_job(client, left, right)["id"]
    resp = set_forbidden(client, jid, [[2, 2], [0, 0], [0, 0], [1, 1]])
    assert resp.status_code == 200, resp.text
    body = resp.json()
    assert forbidden_pairs(body) == [(0, 0), (1, 1), (2, 2)]
    # GET 读回（存储层经同一规范化身份转换）逐项一致。
    assert client.get(f"/api/jobs/{jid}").json() == body


# ---------------------------------------------------------------- 422 与状态不变


@pytest.mark.parametrize(
    "bad",
    [
        [[3, 0]],  # 左索引越界
        [[0, 3]],  # 右索引越界
        [[-1, 0]],  # 负索引
        [[0, 1]],  # 指纹不等（a != b）
        [["x", 1]],  # 类型错误
        [[0]],  # 形状错误
        [[0, 1, 2]],
        [[True, 0]],  # 布尔不是整数
    ],
)
def test_invalid_forbidden_422_and_state_unchanged(
    client: httpx.Client, bad: list
) -> None:
    left = ["a", "b", "c"]
    right = ["a", "x", "c"]
    jid = create_job(client, left, right)["id"]
    before = client.get(f"/api/jobs/{jid}").json()
    resp = set_forbidden(client, jid, bad)
    assert resp.status_code == 422, f"应对 {bad} 返回 422，实际 {resp.status_code}"
    assert set(resp.json()) == {"detail"}
    after = client.get(f"/api/jobs/{jid}").json()
    assert after == before, "非法禁止对应对不得留下半次修改"


@pytest.mark.parametrize(
    "payload",
    [
        {"forbidden_pairs": 5},  # 非数组
        {"forbidden_pairs": "ab"},  # 字符串不是集合
        {"forbidden_pairs": {}},  # 对象不是数组
        {"forbidden_pairs": [{}]},  # 元素不是对
        {"forbidden_pairs": [None]},
    ],
)
def test_malformed_forbidden_payload_422(
    client: httpx.Client, payload: dict
) -> None:
    jid = create_job(client, ["a"], ["a"])["id"]
    resp = client.put(f"/api/jobs/{jid}/forbidden-pairs", json=payload)
    assert resp.status_code == 422, resp.text


def test_forbidden_intersecting_anchors_rejected_entirely(
    client: httpx.Client,
) -> None:
    # 锚点与禁止集合相交：无论从哪一端提交都整次拒绝（422），状态不变。
    left = ["a", "b", "a", "c"]
    right = ["b", "a", "c", "a"]
    jid = create_job(client, left, right)["id"]

    # 先放锚点 (2,1)（a==a），再提交包含该点的禁止集合 -> 422。
    ok = set_anchors(client, jid, [[2, 1]])
    assert ok.status_code == 200
    before = client.get(f"/api/jobs/{jid}").json()
    resp = set_forbidden(client, jid, [[2, 1], [3, 2]])  # 与锚点撞，另一点合法
    assert resp.status_code == 422
    after = client.get(f"/api/jobs/{jid}").json()
    assert after == before, "相交拒绝不得部分落库"

    # 反向：清空锚点后先放禁止集合，再提交相交锚点同样 422。
    assert set_anchors(client, jid, []).status_code == 200
    assert set_forbidden(client, jid, [[2, 1]]).status_code == 200
    before2 = client.get(f"/api/jobs/{jid}").json()
    resp2 = set_anchors(client, jid, [[2, 1]])
    assert resp2.status_code == 422
    assert client.get(f"/api/jobs/{jid}").json() == before2


def test_forbidden_persists_while_anchors_change(client: httpx.Client) -> None:
    """替换锚点不触碰禁止集合；替换禁止集合不触碰锚点。"""
    left = ["a", "b", "a", "b", "c"]
    right = ["b", "a", "b", "a", "c"]
    jid = create_job(client, left, right)["id"]
    f = set_forbidden(client, jid, [[0, 1]])
    assert f.status_code == 200 and forbidden_pairs(f.json()) == [(0, 1)]

    a = set_anchors(client, jid, [[2, 3]])
    assert a.status_code == 200
    assert _anchor_set(a.json()) == {(2, 3)}
    assert forbidden_pairs(a.json()) == [(0, 1)], "替换锚点必须保留禁止集合"
    pairs = assert_chain_valid(a.json(), left, right)
    assert (0, 1) not in pairs and (2, 3) in pairs

    # 替换禁止集合不得改动锚点。
    f2 = set_forbidden(client, jid, [])
    assert f2.status_code == 200
    assert forbidden_pairs(f2.json()) == []
    assert _anchor_set(f2.json()) == {(2, 3)}


# ---------------------------------------------------------------- 穷举交叉核对


def _bounded_side(alphabet: str, n: int, seed: int) -> list[str]:
    rng = random.Random(seed)
    while True:
        side = [rng.choice(alphabet) for _ in range(n)]
        if all(v <= 4 for v in Counter(side).values()):
            return side


def test_forbidden_exhaustive_small_arrays(client: httpx.Client) -> None:
    """二元/三元字母表小数组：枚举全部禁止子集，API 结果必须与穷举一致。

    覆盖重复指纹、无解（全部相等对被禁）与字典序裁决。
    """
    rng = random.Random(20261001)
    cases = 0
    for left, right in [
        (["a", "a"], ["a", "a"]),
        (["a", "a", "b"], ["b", "a", "a"]),
        (["a", "b", "a", "b"], ["b", "a", "b", "a"]),
        (["x", "a"], ["a", "x"]),  # 全局唯一匹配也会被禁成空解
    ]:
        matches = [
            [i, j]
            for i in range(len(left))
            for j in range(len(right))
            if left[i] == right[j]
        ]
        jid = create_job(client, left, right)["id"]
        for mask in range(1 << len(matches)):
            forb = [matches[k] for k in range(len(matches)) if mask & (1 << k)]
            resp = set_forbidden(client, jid, forb)
            assert resp.status_code == 200, (left, right, forb, resp.text)
            body = resp.json()
            got = assert_chain_valid(body, left, right)
            expected = brute(left, right, (), forb)
            assert got == expected, (left, right, forb, got, expected)
            assert set(got).isdisjoint(map(tuple, forb))
            # 清空后恢复全局解。
            if mask:
                cleared = set_forbidden(client, jid, []).json()
                assert _pairs(cleared) == brute(left, right)
        cases += 1 << len(matches)
    assert cases > 0


def test_forbidden_random_crosscheck_with_and_without_anchors(
    client: httpx.Client,
) -> None:
    """随机小数组：随机禁止集合（与锚点不相交）下与 O(n^2) 穷举逐项一致。"""
    rng = random.Random(20261002)
    for case in range(120):
        n_left = rng.randint(1, 8)
        n_right = rng.randint(1, 8)
        left = _bounded_side("abcd", n_left, case * 3 + 1)
        right = _bounded_side("abcd", n_right, case * 3 + 2)
        jid = create_job(client, left, right)["id"]
        matches = [
            (i, j)
            for i in range(n_left)
            for j in range(n_right)
            if left[i] == right[j]
        ]
        # 随机一条合法锚点链（约一半用例带锚点）。
        anchors: list[list[int]] = []
        if rng.random() < 0.5:
            pi = pj = -1
            for i, j in sorted(matches):
                if i > pi and j > pj and rng.random() < 0.35:
                    anchors.append([i, j])
                    pi, pj = i, j
            if anchors:
                r = set_anchors(client, jid, anchors)
                assert r.status_code == 200, (left, right, anchors, r.text)
        anchor_set = {tuple(a) for a in anchors}
        forb = [list(p) for p in matches if rng.random() < 0.3 and p not in anchor_set]
        resp = set_forbidden(client, jid, forb)
        assert resp.status_code == 200, (left, right, forb, resp.text)
        body = resp.json()
        got = assert_chain_valid(body, left, right)
        expected = brute(left, right, anchors, forb)
        assert got == expected, (
            f"不一致: {left} / {right} / anchors={anchors} / forb={forb}: "
            f"{got} != {expected}"
        )
        assert anchor_set.issubset(set(got))
        assert set(got).isdisjoint(map(tuple, forb))


# ---------------------------------------------------------------- 性能


def test_constraints_result_version_atomic_in_one_row(client: httpx.Client) -> None:
    """约束、结果、版本落在 jobs 同一行：交错两类修改后逐列核验原子一致。"""
    psycopg = pytest.importorskip("psycopg")
    import os

    left = ["a", "b", "a", "b", "c"]
    right = ["b", "a", "b", "a", "c"]
    jid = create_job(client, left, right)["id"]
    r1 = set_forbidden(client, jid, [[0, 1]])
    assert r1.status_code == 200
    r2 = set_anchors(client, jid, [[2, 3]])
    assert r2.status_code == 200
    expected = client.get(f"/api/jobs/{jid}").json()
    assert expected == r2.json()

    dsn = os.environ["PG_DSN"]
    with psycopg.connect(dsn) as conn, conn.cursor() as cur:
        cur.execute(
            "SELECT anchors, forbidden_pairs, result, version, updated_at "
            "FROM jobs WHERE id = %s",
            (jid,),
        )
        row = cur.fetchone()
    assert row is not None
    anchors_col, forbidden_col, result_col, version_col, updated_col = row
    assert [tuple(p) for p in anchors_col] == [(2, 3)]
    assert [tuple(p) for p in forbidden_col] == [(0, 1)]
    assert [tuple(p) for p in result_col] == [
        (p["left_index"], p["right_index"]) for p in expected["result"]
    ]
    assert version_col == 2
    assert updated_col is not None


def test_forbidden_same_state_replay_returns_first_verdict(
    client: httpx.Client,
) -> None:
    """状态未推进时同键重放禁止请求：返回首次裁决，不再次推进版本。"""
    left = ["a", "b", "c"]
    right = ["c", "b", "a"]
    jid = create_job(client, left, right)["id"]
    key = f"forbidden-replay-{jid}"
    first = set_forbidden(client, jid, [[0, 2]], key=key)
    assert first.status_code == 200 and first.json()["version"] == 1
    replay = set_forbidden(client, jid, [[0, 2]], key=key)
    assert replay.status_code == 200
    assert replay.json() == first.json()
    final = client.get(f"/api/jobs/{jid}").json()
    assert final == first.json()
    assert final["version"] == 1


# ---------------------------------------------------------------- 性能


def test_max_input_forbidden_within_3_seconds(client: httpx.Client) -> None:
    pattern = [f"f{g:04d}" for g in range(5000) for _ in range(4)]
    assert len(pattern) == 20000
    jid = create_job(client, pattern, pattern)["id"]
    started = time.perf_counter()
    resp = set_forbidden(client, jid, [[0, 0]])
    elapsed = time.perf_counter() - started
    assert resp.status_code == 200
    body = resp.json()
    assert (0, 0) not in _pairs(body)
    assert elapsed < 3.0, f"禁止对重算耗时 {elapsed:.3f}s 超过 3 秒"


# ---------------------------------------------------------------- 迁移：旧作业按空禁止集合读取


def test_legacy_job_row_reads_back_with_empty_forbidden_set(
    client: httpx.Client,
) -> None:
    """升级前写入的旧 jobs 行（无 forbidden_pairs 列）：迁移后按空集合读取。

    直接用底层连接在“旧形态”表上插入历史行，再执行与启动时完全相同的
    幂等迁移，随后经 HTTP 验证旧作业可读、结果可查、可在新版本上修改。
    测试结束前恢复列，避免降级状态外溢到同进程的其他用例。
    """
    psycopg = pytest.importorskip("psycopg")
    import os

    from sqlalchemy import create_engine

    from app.db import _migrate_jobs

    left = ["a", "b", "c"]
    right = ["a", "b", "c"]
    dsn = os.environ["PG_DSN"]
    # 用验收环境的 PG_DSN 自建连接（而非引用服务进程的 engine），与其他
    # 落库核验用例保持同一环境依赖。
    migration_engine = create_engine(f"postgresql+psycopg://{dsn.split('://', 1)[1]}")
    legacy_id = "00000000-0000-4000-8000-000000000abc"
    try:
        with psycopg.connect(dsn, autocommit=True) as conn, conn.cursor() as cur:
            cur.execute("ALTER TABLE jobs DROP COLUMN IF EXISTS forbidden_pairs")
            cur.execute("DELETE FROM jobs WHERE id = %s", (legacy_id,))
            # 旧版本服务写出的历史行：JSON 列以文本绑定，需显式 ::json。
            cur.execute(
                "INSERT INTO jobs (id, left_data, right_data, anchors, result, "
                "version, updated_at) VALUES ("
                "%s, %s::json, %s::json, %s::json, %s::json, %s, %s)",
                (
                    legacy_id,
                    json.dumps(left),
                    json.dumps(right),
                    json.dumps([[0, 0]]),
                    json.dumps([[0, 0], [1, 1], [2, 2]]),
                    7,
                    "2026-09-01T00:00:00+00:00",
                ),
            )

        # 与容器启动同一条迁移路径（幂等）：旧作业按空禁止集合读取。
        with migration_engine.begin() as conn:
            _migrate_jobs(conn)

        got = client.get(f"/api/jobs/{legacy_id}").json()
        assert got["forbidden_pairs"] == []
        assert _pairs(got) == [(0, 0), (1, 1), (2, 2)]
        assert {(p["left_index"], p["right_index"]) for p in got["anchors"]} == {(0, 0)}
        assert got["version"] == 7

        # 旧作业可正常接受禁止集合修改（基于版本 7），版本推进、原子一致。
        resp = set_forbidden(client, legacy_id, [[1, 1]], base_version=7)
        assert resp.status_code == 200, resp.text
        body = resp.json()
        assert body["version"] == 8
        assert forbidden_pairs(body) == [(1, 1)]
        assert _anchor_set(body) == {(0, 0)}
        assert client.get(f"/api/jobs/{legacy_id}").json() == body

        # 落库核验：约束、结果、版本同列同行原子一致。
        with psycopg.connect(dsn) as conn, conn.cursor() as cur:
            cur.execute(
                "SELECT anchors, forbidden_pairs, result, version FROM jobs WHERE id = %s",
                (legacy_id,),
            )
            row = cur.fetchone()
        assert row is not None
        anchors_col, forbidden_col, result_col, version_col = row
        assert [tuple(p) for p in anchors_col] == [(0, 0)]
        assert [tuple(p) for p in forbidden_col] == [(1, 1)]
        assert (1, 1) not in [tuple(p) for p in result_col]
        assert version_col == 8
    finally:
        # 无论断言是否失败都恢复迁移后形态，保证其他用例看到的是新 schema。
        with migration_engine.begin() as conn:
            _migrate_jobs(conn)
