"""禁止对应对（误配排除）验收。

两台扫描机的少数同指纹画面经人工核实并非同一帧：这些索引对进入作业的
禁止集合后不得出现在最终最长对应中。本模块核对：

- 禁止集合的校验（422 且状态不变）、规范化（排序去重）与重算语义；
- 锚点与禁止集合相交时两个方向都整次拒绝；
- 两类修改共用版本裁决：交错推进、过期版本拒绝、并发恰好一个成功；
- 幂等键重放不重复推进版本、不覆盖另一类约束的较新状态；
- 小数组穷举交叉核对（重复指纹、无解、锚点+禁止组合）与落库重读一致。
"""

from __future__ import annotations

import random
import sys
import threading
from collections import Counter
from pathlib import Path

import httpx
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from brute import brute  # noqa: E402

from conftest import (  # noqa: E402
    API_BASE_URL,
    assert_chain_valid,
    create_job,
    set_anchors,
    set_forbidden,
)

LEFT = ["a", "b", "a", "b", "c"]
RIGHT = ["b", "a", "b", "a", "c"]


def _pairs(body: dict) -> list[tuple[int, int]]:
    return [(p["left_index"], p["right_index"]) for p in body["result"]]


def _forbidden_pairs(body: dict) -> list[tuple[int, int]]:
    return [(p["left_index"], p["right_index"]) for p in body["forbidden"]]


def _anchor_pairs(body: dict) -> set[tuple[int, int]]:
    return {(p["left_index"], p["right_index"]) for p in body["anchors"]}


def _get(client: httpx.Client, job_id: str) -> dict:
    resp = client.get(f"/api/jobs/{job_id}")
    assert resp.status_code == 200, resp.text
    return resp.json()


def _assert_avoids_forbidden(body: dict) -> None:
    forbidden = set(_forbidden_pairs(body))
    assert not (forbidden & set(_pairs(body))), "结果不得包含任何禁止对应对"


# ---------------------------------------------------------------- 基本语义


def test_create_has_empty_forbidden_set(client: httpx.Client) -> None:
    body = create_job(client, LEFT, RIGHT)
    assert body["forbidden"] == []
    assert _get(client, body["id"]) == body


def test_forbidden_missing_job_404(client: httpx.Client) -> None:
    resp = set_forbidden(client, "00000000-0000-0000-0000-000000000000", [])
    assert resp.status_code == 404


def test_forbidden_excludes_mispaired_duplicate_frames(client: httpx.Client) -> None:
    """重复指纹里的误配对：禁止后结果避开该对，仍先最长再字典序。"""
    left = ["a", "a"]
    right = ["a", "a"]
    body = create_job(client, left, right)
    assert _pairs(body) == [(0, 0), (1, 1)]
    jid = body["id"]

    resp = set_forbidden(client, jid, [[0, 0]])
    assert resp.status_code == 200, resp.text
    fb = resp.json()
    assert fb["version"] == 1
    assert _forbidden_pairs(fb) == [(0, 0)]
    pairs = assert_chain_valid(fb, left, right)
    _assert_avoids_forbidden(fb)
    # 剩余可行对 (0,1)/(1,0)/(1,1) 互不成链，最长为 1，字典序最小为 (0,1)
    assert pairs == [(0, 1)] == brute(left, right, forbidden=[(0, 0)])

    # 清空禁止集合 -> 恢复全局最优
    cleared = set_forbidden(client, jid, [], base_version=1).json()
    assert cleared["forbidden"] == []
    assert _pairs(cleared) == [(0, 0), (1, 1)]


def test_forbidden_set_is_normalized_sorted_and_deduplicated(
    client: httpx.Client,
) -> None:
    """乱序 + 重复元素：接口、算法、存储共享同一规范化索引身份。"""
    body = create_job(client, LEFT, RIGHT)
    jid = body["id"]
    resp = set_forbidden(client, jid, [[2, 1], [0, 1], [2, 1]])
    assert resp.status_code == 200, resp.text
    assert _forbidden_pairs(resp.json()) == [(0, 1), (2, 1)]
    assert _forbidden_pairs(_get(client, jid)) == [(0, 1), (2, 1)]


def test_forbidden_can_eliminate_all_correspondences(client: httpx.Client) -> None:
    """无解：唯一可行对被禁止后结果为空。"""
    body = create_job(client, ["a"], ["a"])
    assert _pairs(body) == [(0, 0)]
    resp = set_forbidden(client, body["id"], [[0, 0]])
    assert resp.status_code == 200, resp.text
    fb = resp.json()
    assert fb["result"] == [] and fb["length"] == 0
    assert _forbidden_pairs(fb) == [(0, 0)]


# ---------------------------------------------------------------- 校验 422


def test_invalid_forbidden_422_and_state_unchanged(client: httpx.Client) -> None:
    left = ["a", "b", "a", "c"]
    right = ["b", "a", "c", "a"]
    body = create_job(client, left, right)
    jid = body["id"]
    original = _get(client, jid)

    bad_payloads = [
        [[4, 0]],  # 左索引越界
        [[0, 9]],  # 右索引越界
        [[-1, 0]],  # 负索引
        [[0, 0]],  # a != b 指纹不等
        [[1, 2]],  # b != c
        [["x", 1]],  # 类型错误
        [[0]],  # 形状错误
    ]
    for bad in bad_payloads:
        resp = set_forbidden(client, jid, bad)
        assert resp.status_code == 422, (
            f"应对 {bad} 返回 422，实际 {resp.status_code}: {resp.text}"
        )
        assert set(resp.json()) == {"detail"}

    after = _get(client, jid)
    assert after == original, "非法禁止集合不得改变任何状态"


# ---------------------------------------------------------------- 锚点冲突


def test_anchor_forbidden_intersection_rejected_in_both_directions(
    client: httpx.Client,
) -> None:
    body = create_job(client, LEFT, RIGHT)
    jid = body["id"]

    anchored = set_anchors(client, jid, [[2, 3]], base_version=0)
    assert anchored.status_code == 200 and anchored.json()["version"] == 1

    # 方向一：新禁止集合与既有锚点相交 -> 整次拒绝，状态不变
    resp = set_forbidden(client, jid, [[0, 1], [2, 3]], base_version=1)
    assert resp.status_code == 422, resp.text
    assert set(resp.json()) == {"detail"}
    assert _get(client, jid) == anchored.json()

    # 不含锚点的禁止集合合法：锚点原样保留，版本推进
    forb = set_forbidden(client, jid, [[0, 1]], base_version=1)
    assert forb.status_code == 200, forb.text
    assert forb.json()["version"] == 2
    assert _forbidden_pairs(forb.json()) == [(0, 1)]
    assert _anchor_pairs(forb.json()) == {(2, 3)}, "替换禁止集合不得改动锚点"

    # 方向二：新锚点集合与既有禁止对应对相交 -> 整次拒绝，状态不变
    resp2 = set_anchors(client, jid, [[0, 1], [2, 3]], base_version=2)
    assert resp2.status_code == 422, resp2.text
    assert _get(client, jid) == forb.json()


# ---------------------------------------------------------------- 交错修改与版本裁决


def test_interleaved_anchor_and_forbidden_modifications_share_version(
    client: httpx.Client,
) -> None:
    body = create_job(client, LEFT, RIGHT)
    jid = body["id"]
    assert body["version"] == 0

    a1 = set_anchors(client, jid, [[2, 3]], base_version=0)
    assert a1.status_code == 200 and a1.json()["version"] == 1

    # 禁止对应对真实改变最优解：全局链 (0,1),(1,2),(2,3),(4,4) 被截短。
    f1 = set_forbidden(client, jid, [[1, 2]], base_version=1)
    assert f1.status_code == 200, f1.text
    f1b = f1.json()
    assert f1b["version"] == 2
    assert _forbidden_pairs(f1b) == [(1, 2)]
    assert _anchor_pairs(f1b) == {(2, 3)}, "替换禁止集合不得改动既有锚点"
    pairs = assert_chain_valid(f1b, LEFT, RIGHT)
    _assert_avoids_forbidden(f1b)
    assert pairs == brute(LEFT, RIGHT, anchors=[(2, 3)], forbidden=[(1, 2)])
    assert pairs == [(0, 1), (2, 3), (4, 4)]

    a2 = set_anchors(client, jid, [[4, 4]], base_version=2)
    assert a2.status_code == 200, a2.text
    a2b = a2.json()
    assert a2b["version"] == 3
    assert _anchor_pairs(a2b) == {(4, 4)}
    assert _forbidden_pairs(a2b) == [(1, 2)], "替换锚点不得改动既有禁止集合"
    assert _pairs(a2b) == brute(LEFT, RIGHT, anchors=[(4, 4)], forbidden=[(1, 2)])

    # 旧版本请求不能覆盖另一类约束：两个方向的过期 base_version 都拒绝。
    stale_anchor = set_anchors(client, jid, [[2, 3]], base_version=1)
    assert stale_anchor.status_code == 409
    stale_forbidden = set_forbidden(client, jid, [], base_version=2)
    assert stale_forbidden.status_code == 409

    final = _get(client, jid)
    assert final == a2b, "过期请求不得改变任何状态"
    assert _get(client, jid) == final


# ---------------------------------------------------------------- 并发：两类修改竞争


def _fire_concurrently(job_id: str, specs: list[dict]) -> list[httpx.Response]:
    """同步屏障让两类修改请求严格同时发出，按 specs 顺序返回响应。"""
    barrier = threading.Barrier(len(specs))
    responses: list[httpx.Response | None] = [None] * len(specs)

    def worker(index: int, spec: dict) -> None:
        with httpx.Client(base_url=API_BASE_URL, timeout=60.0) as client:
            barrier.wait(timeout=10)
            if spec["kind"] == "anchors":
                responses[index] = set_anchors(
                    client,
                    job_id,
                    spec["pairs"],
                    base_version=spec.get("base_version"),
                    key=spec.get("key"),
                )
            else:
                responses[index] = set_forbidden(
                    client,
                    job_id,
                    spec["pairs"],
                    base_version=spec.get("base_version"),
                    key=spec.get("key"),
                )

    threads = [
        threading.Thread(target=worker, args=(index, spec))
        for index, spec in enumerate(specs)
    ]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join(timeout=60)
    assert all(response is not None for response in responses), "并发请求未全部完成"
    return [response for response in responses if response is not None]


def test_concurrent_anchor_and_forbidden_replacement_exactly_one_wins(
    client: httpx.Client,
) -> None:
    """两类修改基于同一版本并发：共用版本裁决，恰好一个成功。"""
    body = create_job(client, LEFT, RIGHT)
    jid = body["id"]

    specs = [
        {"kind": "anchors", "pairs": [[2, 3]], "base_version": 0},
        {"kind": "forbidden", "pairs": [[1, 2]], "base_version": 0},
    ]
    responses = _fire_concurrently(jid, specs)
    assert sorted(resp.status_code for resp in responses) == [200, 409]

    final = _get(client, jid)
    assert final["version"] == 1, "共用版本裁决：版本恰好推进一次"
    anchor_resp, forbidden_resp = responses
    if anchor_resp.status_code == 200:
        assert final == anchor_resp.json()
        assert _anchor_pairs(final) == {(2, 3)}
        assert final["forbidden"] == []
        assert forbidden_resp.status_code == 409
    else:
        assert final == forbidden_resp.json()
        assert _forbidden_pairs(final) == [(1, 2)]
        assert final["anchors"] == []
        assert anchor_resp.status_code == 409
    assert _get(client, jid) == final


# ---------------------------------------------------------------- 幂等重放


def test_forbidden_idempotent_replay_does_not_advance_version(
    client: httpx.Client,
) -> None:
    body = create_job(client, LEFT, RIGHT)
    jid = body["id"]
    key = f"forbidden-replay-{jid}"

    first = set_forbidden(client, jid, [[1, 2]], base_version=0, key=key)
    assert first.status_code == 200 and first.json()["version"] == 1

    replay = set_forbidden(client, jid, [[1, 2]], base_version=0, key=key)
    assert replay.status_code == 200
    assert replay.json() == first.json(), "重放必须返回首次裁决结果"
    assert _get(client, jid)["version"] == 1, "重放不得推进版本"

    # 另一类约束推进版本后，旧请求重放：拒绝且不得覆盖较新状态。
    newer = set_anchors(client, jid, [[2, 3]], base_version=1)
    assert newer.status_code == 200 and newer.json()["version"] == 2
    stale_replay = set_forbidden(client, jid, [[1, 2]], base_version=0, key=key)
    assert stale_replay.status_code == 409

    # 同一键搭配不同请求体同样拒绝。
    other = set_forbidden(client, jid, [[0, 1]], base_version=2, key=key)
    assert other.status_code == 409

    final = _get(client, jid)
    assert final == newer.json()
    assert _forbidden_pairs(final) == [(1, 2)], "禁止集合必须完整保留"
    assert _anchor_pairs(final) == {(2, 3)}


def test_idempotency_key_not_shared_across_constraint_kinds(
    client: httpx.Client,
) -> None:
    """同一幂等键用于另一类约束：指纹不同，拒绝且状态不变。"""
    body = create_job(client, LEFT, RIGHT)
    jid = body["id"]
    key = f"cross-kind-key-{jid}"
    first = set_forbidden(client, jid, [[1, 2]], base_version=0, key=key)
    assert first.status_code == 200

    resp = set_anchors(client, jid, [[2, 3]], base_version=1, key=key)
    assert resp.status_code == 409
    assert _get(client, jid) == first.json()


# ---------------------------------------------------------------- 穷举交叉核对


def _bounded_side(alphabet: str, n: int, seed: int) -> list[str]:
    rng = random.Random(seed)
    while True:
        side = [rng.choice(alphabet) for _ in range(n)]
        if all(v <= 4 for v in Counter(side).values()):
            return side


def test_forbidden_exhaustive_crosscheck(client: httpx.Client) -> None:
    """小数组穷举：重复指纹、无解、锚点+禁止组合，与 O(n^2) 参考实现一致。"""
    rng = random.Random(20261001)
    checked = 0
    for case in range(150):
        n_left = rng.randint(1, 7)
        n_right = rng.randint(1, 7)
        # 小字母表制造大量重复指纹；无匹配用例自然落入“无解”分支。
        left = _bounded_side("abc", n_left, case * 3 + 1)
        right = _bounded_side("abc", n_right, case * 3 + 2)
        body = create_job(client, left, right)
        jid = body["id"]

        all_pairs = [
            (i, j)
            for i in range(n_left)
            for j in range(n_right)
            if left[i] == right[j]
        ]
        forbidden = (
            sorted(rng.sample(all_pairs, rng.randint(0, min(len(all_pairs), 6))))
            if all_pairs
            else []
        )
        resp = set_forbidden(client, jid, [list(p) for p in forbidden])
        assert resp.status_code == 200, resp.text
        fb = resp.json()
        assert _forbidden_pairs(fb) == forbidden
        pairs = assert_chain_valid(fb, left, right)
        _assert_avoids_forbidden(fb)
        expected = brute(left, right, forbidden=forbidden)
        assert pairs == expected, (
            f"禁止集合不一致: {left} / {right} / {forbidden}: {pairs} != {expected}"
        )

        # 从受约束最优解抽锚点（必不与禁止集合相交），核锚点+禁止组合
        if expected:
            anchors = sorted(rng.sample(expected, rng.randint(1, min(2, len(expected)))))
            r2 = set_anchors(client, jid, [list(a) for a in anchors])
            assert r2.status_code == 200, r2.text
            ab = r2.json()
            apairs = assert_chain_valid(ab, left, right)
            _assert_avoids_forbidden(ab)
            assert apairs == brute(left, right, anchors, forbidden), (
                f"锚点+禁止不一致: {left} / {right} / {anchors} / {forbidden}"
            )
            assert set(anchors) <= set(apairs)

            # 锚点与禁止集合相交 -> 整次拒绝，状态不变
            if forbidden:
                clash = set_anchors(client, jid, [list(forbidden[0])])
                assert clash.status_code == 422
                assert _get(client, jid) == ab
        checked += 1
    assert checked == 150


# ---------------------------------------------------------------- 落库与重读


def test_forbidden_constraints_result_and_version_persisted(
    client: httpx.Client,
) -> None:
    """约束、结果、版本原子落库；重新读取（如重启后）逐项一致。"""
    psycopg = pytest.importorskip("psycopg")
    import os

    body = create_job(client, LEFT, RIGHT)
    jid = body["id"]
    resp = set_forbidden(client, jid, [[1, 2]], base_version=0)
    assert resp.status_code == 200
    anchored = set_anchors(client, jid, [[2, 3]], base_version=1)
    assert anchored.status_code == 200
    final = anchored.json()

    dsn = os.environ["PG_DSN"]
    with psycopg.connect(dsn) as conn, conn.cursor() as cur:
        cur.execute(
            "SELECT anchors, forbidden, result, version FROM jobs WHERE id = %s",
            (jid,),
        )
        row = cur.fetchone()
    assert row is not None, "任务未写入 PostgreSQL"
    anchors_col, forbidden_col, result_col, version_col = row
    assert [tuple(p) for p in anchors_col] == [(2, 3)]
    assert [tuple(p) for p in forbidden_col] == [(1, 2)]
    assert [tuple(p) for p in result_col] == _pairs(final)
    assert version_col == final["version"] == 2

    # 重新读取（重启后亦从此持久状态恢复）逐项一致
    again = _get(client, jid)
    assert again == final
    assert _forbidden_pairs(again) == [(1, 2)]


def test_legacy_jobs_read_as_empty_forbidden_set(client: httpx.Client) -> None:
    """旧作业（禁止列引入前创建）按空禁止集合读取：列非空且默认 '[]'。"""
    psycopg = pytest.importorskip("psycopg")
    import os

    dsn = os.environ["PG_DSN"]
    with psycopg.connect(dsn) as conn, conn.cursor() as cur:
        cur.execute(
            "SELECT column_default, is_nullable FROM information_schema.columns "
            "WHERE table_name = 'jobs' AND column_name = 'forbidden'"
        )
        row = cur.fetchone()
    assert row is not None, "jobs 表缺少 forbidden 列（迁移未执行）"
    default, nullable = row
    assert nullable == "NO"
    assert default is not None and "[]" in default

    # 未设置过禁止集合的作业读取为空集合
    body = create_job(client, LEFT, RIGHT)
    assert body["forbidden"] == []
    assert _get(client, body["id"])["forbidden"] == []
