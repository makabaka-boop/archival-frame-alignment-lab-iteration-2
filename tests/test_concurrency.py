"""并发锚点替换验收：同步屏障交错、超时重放、版本裁决与幂等。

面向真实 HTTP 服务与真实 PostgreSQL，不使用任何桩：

- 两名修复师基于同一旧状态并发替换锚点时，恰好一个请求成功（200），
  另一个收到可解释的 409；失败/过期请求不改变锚点、结果、版本或更新时间；
- 成功响应与随后重复 GET 读到的持久状态逐项一致；
- 超时重放（同一 Idempotency-Key + 同一请求体）返回首次裁决结果，
  绝不重复执行，也绝不覆盖其他修复师已完成的较新集合。
"""

from __future__ import annotations

import threading

import httpx
import pytest

from conftest import API_BASE_URL, assert_chain_valid, create_job

# 合法锚点充足的小数组：a/b 多次交错，c 在末尾。
LEFT = ["a", "b", "a", "b", "c"]
RIGHT = ["b", "a", "b", "a", "c"]

# 注意：幂等记录真实落库且 pgdata 卷跨运行保留，因此各用例的 Idempotency-Key
# 都拼入任务 id，保证每次验收运行使用的键全局唯一。


def _put(
    client: httpx.Client,
    job_id: str,
    anchors: list[list[int]],
    base_version: int | None = None,
    key: str | None = None,
) -> httpx.Response:
    payload: dict = {"anchors": anchors}
    if base_version is not None:
        payload["base_version"] = base_version
    headers = {"Idempotency-Key": key} if key is not None else {}
    return client.put(f"/api/jobs/{job_id}/anchors", json=payload, headers=headers)


def _fire_concurrently(job_id: str, specs: list[dict]) -> list[httpx.Response]:
    """用同步屏障让多组请求严格同时发出，按 specs 顺序返回响应。"""
    barrier = threading.Barrier(len(specs))
    responses: list[httpx.Response | None] = [None] * len(specs)

    def worker(index: int, spec: dict) -> None:
        with httpx.Client(base_url=API_BASE_URL, timeout=60.0) as client:
            barrier.wait(timeout=10)
            responses[index] = _put(
                client,
                job_id,
                spec["anchors"],
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


def _get(client: httpx.Client, job_id: str) -> dict:
    resp = client.get(f"/api/jobs/{job_id}")
    assert resp.status_code == 200, resp.text
    return resp.json()


def _anchor_set(body: dict) -> set[tuple[int, int]]:
    return {(p["left_index"], p["right_index"]) for p in body["anchors"]}


def _pairs(body: dict) -> list[tuple[int, int]]:
    return [(p["left_index"], p["right_index"]) for p in body["result"]]


def _assert_conflict_structure(resp: httpx.Response) -> None:
    """失败结构：409 + {"detail": 可解释的原因}。"""
    assert resp.status_code == 409, f"应为 409，实际 {resp.status_code}: {resp.text}"
    body = resp.json()
    assert set(body) == {"detail"}
    assert isinstance(body["detail"], str) and body["detail"]


# ---------------------------------------------------------------- 并发裁决


def test_concurrent_nonempty_replacements_exactly_one_wins(
    client: httpx.Client,
) -> None:
    """同步屏障交错两组非空替换：恰好一个成功，最终状态为胜出者。"""
    body = create_job(client, LEFT, RIGHT)
    jid = body["id"]
    assert body["version"] == 0

    specs = [
        {"anchors": [[2, 3]], "base_version": 0},
        {"anchors": [[0, 3]], "base_version": 0},
    ]
    responses = _fire_concurrently(jid, specs)

    statuses = sorted(resp.status_code for resp in responses)
    assert statuses == [200, 409], f"并发替换应恰好一个成功：{statuses}"

    winner_body: dict | None = None
    expected_anchors: set[tuple[int, int]] | None = None
    for spec, resp in zip(specs, responses):
        if resp.status_code == 200:
            winner_body = resp.json()
            expected_anchors = {tuple(a) for a in spec["anchors"]}
        else:
            _assert_conflict_structure(resp)
    assert winner_body is not None and expected_anchors is not None

    # 最终持久状态 = 胜出请求的集合，版本恰好推进一次，重复 GET 一致。
    first = _get(client, jid)
    second = _get(client, jid)
    assert first == second, "重复 GET 必须一致"
    assert first == winner_body, "成功响应必须与随后读取的持久状态逐项一致"
    assert first["version"] == 1
    assert _anchor_set(first) == expected_anchors
    assert _anchor_set(first).issubset(set(_pairs(first)))
    assert_chain_valid(first, LEFT, RIGHT)


def test_concurrent_clear_and_nonempty_replacement_exactly_one_wins(
    client: httpx.Client,
) -> None:
    """同步屏障交错清空与非空替换：恰好一个成功。"""
    body = create_job(client, LEFT, RIGHT)
    jid = body["id"]
    global_pairs = _pairs(body)
    assert len(global_pairs) == 4

    setup = _put(client, jid, [[2, 3]], base_version=0)
    assert setup.status_code == 200 and setup.json()["version"] == 1

    specs = [
        {"anchors": [], "base_version": 1},  # 清空锚点
        {"anchors": [[0, 3]], "base_version": 1},  # 非空替换
    ]
    responses = _fire_concurrently(jid, specs)
    assert sorted(resp.status_code for resp in responses) == [200, 409]

    clear_resp, set_resp = responses
    final = _get(client, jid)
    assert final["version"] == 2, "并发裁决后版本恰好推进一次"
    if clear_resp.status_code == 200:
        _assert_conflict_structure(set_resp)
        assert final["anchors"] == []
        assert _pairs(final) == global_pairs, "清空后必须恢复全局最优"
        assert final == clear_resp.json()
    else:
        _assert_conflict_structure(clear_resp)
        assert _anchor_set(final) == {(0, 3)}
        assert _pairs(final) == [(0, 3), (4, 4)]
        assert final == set_resp.json()
    assert _get(client, jid) == final


def test_concurrent_duplicate_delivery_applied_exactly_once(
    client: httpx.Client,
) -> None:
    """同一逻辑请求（同幂等键）并发到达两次：只应用一次，双方拿到同一裁决。"""
    body = create_job(client, LEFT, RIGHT)
    jid = body["id"]

    specs = [
        {"anchors": [[2, 3]], "base_version": 0, "key": f"duplicate-delivery-{jid}"},
        {"anchors": [[2, 3]], "base_version": 0, "key": f"duplicate-delivery-{jid}"},
    ]
    responses = _fire_concurrently(jid, specs)
    assert [resp.status_code for resp in responses] == [200, 200]
    assert responses[0].json() == responses[1].json()

    final = _get(client, jid)
    assert final["version"] == 1, "同一逻辑请求只能被应用一次"
    assert final == responses[0].json()
    assert _get(client, jid) == final


# ---------------------------------------------------------------- 超时重放


def test_timeout_replay_while_state_unchanged_returns_first_verdict(
    client: httpx.Client,
) -> None:
    """状态未变时重放：返回首次成功响应，版本与更新时间不再推进。"""
    body = create_job(client, LEFT, RIGHT)
    jid = body["id"]

    first = _put(client, jid, [[2, 3]], base_version=0, key=f"replay-same-state-{jid}")
    assert first.status_code == 200

    # 客户端超时后用同一幂等键、同一请求体重试。
    replay = _put(client, jid, [[2, 3]], base_version=0, key=f"replay-same-state-{jid}")
    assert replay.status_code == 200
    assert replay.json() == first.json(), "重放必须返回首次裁决结果"

    final = _get(client, jid)
    assert final == first.json()
    assert final["version"] == 1, "重放不得推进版本"
    assert _get(client, jid) == final


def test_timeout_replay_after_newer_state_does_not_overwrite(
    client: httpx.Client,
) -> None:
    """重放到达时状态已被其他修复师推进：拒绝且不得覆盖较新集合。"""
    body = create_job(client, LEFT, RIGHT)
    jid = body["id"]

    # 修复师 A 的替换实际已成功，但响应因超时未送达客户端。
    first = _put(client, jid, [[2, 3]], base_version=0, key=f"repairer-a-{jid}")
    assert first.status_code == 200 and first.json()["version"] == 1

    # 修复师 B 基于最新状态完成了新的核实。
    newer = _put(client, jid, [[0, 3]], base_version=1, key=f"repairer-b-{jid}")
    assert newer.status_code == 200 and newer.json()["version"] == 2

    # A 超时重放较早请求：可解释地拒绝，不得覆盖 B 的较新集合。
    replay = _put(client, jid, [[2, 3]], base_version=0, key=f"repairer-a-{jid}")
    _assert_conflict_structure(replay)

    final = _get(client, jid)
    assert final == newer.json()
    assert final["version"] == 2
    assert _anchor_set(final) == {(0, 3)}
    assert _pairs(final) == [(0, 3), (4, 4)]
    assert _get(client, jid) == final


def test_clear_request_replay_does_not_wipe_newer_anchors(
    client: httpx.Client,
) -> None:
    """清空锚点的重试不得抹去其他修复师较新的人工结论。"""
    body = create_job(client, LEFT, RIGHT)
    jid = body["id"]

    assert _put(client, jid, [[2, 3]], base_version=0).status_code == 200  # v1
    cleared = _put(client, jid, [], base_version=1, key=f"clear-request-{jid}")
    assert cleared.status_code == 200 and cleared.json()["anchors"] == []  # v2
    renewed = _put(client, jid, [[0, 3]], base_version=2)
    assert renewed.status_code == 200  # v3

    # 清空请求的超时重放：拒绝，较新锚点集合必须完整保留。
    replay = _put(client, jid, [], base_version=1, key=f"clear-request-{jid}")
    _assert_conflict_structure(replay)

    final = _get(client, jid)
    assert final == renewed.json()
    assert final["version"] == 3
    assert _anchor_set(final) == {(0, 3)}
    assert _get(client, jid) == final


def test_idempotency_key_reused_with_different_payload_409(
    client: httpx.Client,
) -> None:
    """同一幂等键用于不同请求体：拒绝且状态不变。"""
    body = create_job(client, LEFT, RIGHT)
    jid = body["id"]

    first = _put(client, jid, [[2, 3]], base_version=0, key=f"shared-key-{jid}")
    assert first.status_code == 200

    other = _put(client, jid, [[0, 3]], base_version=1, key=f"shared-key-{jid}")
    _assert_conflict_structure(other)

    final = _get(client, jid)
    assert final == first.json()
    assert final["version"] == 1


# ---------------------------------------------------------------- 失败不改状态


def test_stale_base_version_409_and_state_untouched(client: httpx.Client) -> None:
    """基于过期版本的替换：409，锚点、结果、版本、更新时间全部不变。"""
    body = create_job(client, LEFT, RIGHT)
    jid = body["id"]
    assert _put(client, jid, [[2, 3]], base_version=0).status_code == 200
    before = _get(client, jid)

    stale = _put(client, jid, [[0, 3]], base_version=0)  # 基于过期版本 0
    _assert_conflict_structure(stale)

    after = _get(client, jid)
    assert after == before, "过期请求不得改变锚点、结果、版本或更新时间"


def test_invalid_anchors_422_keeps_version_and_timestamp(
    client: httpx.Client,
) -> None:
    """422 校验失败同样不得推进版本或更新时间。"""
    body = create_job(client, LEFT, RIGHT)
    jid = body["id"]
    assert _put(client, jid, [[2, 3]], base_version=0).status_code == 200
    before = _get(client, jid)

    resp = _put(client, jid, [[0, 0]], base_version=1)  # a != b，非法锚点
    assert resp.status_code == 422
    assert set(resp.json()) == {"detail"}

    after = _get(client, jid)
    assert after == before


# ---------------------------------------------------------------- 兼容性


def test_sequential_version_chain_and_legacy_calls(client: httpx.Client) -> None:
    """顺序执行、首次设置、合法清空与不带版本号的旧客户端调用保持兼容。"""
    body = create_job(client, LEFT, RIGHT)
    jid = body["id"]
    assert body["version"] == 0

    r1 = _put(client, jid, [[2, 3]], base_version=0)
    assert r1.status_code == 200 and r1.json()["version"] == 1
    r2 = _put(client, jid, [[0, 3]], base_version=1)
    assert r2.status_code == 200 and r2.json()["version"] == 2
    r3 = _put(client, jid, [], base_version=2)
    assert r3.status_code == 200 and r3.json()["version"] == 3
    assert r3.json()["anchors"] == []
    assert _pairs(r3.json()) == _pairs(body), "合法清空必须恢复全局最优"

    # 旧客户端（不带 base_version / 幂等键）顺序调用行为不变。
    r4 = _put(client, jid, [[4, 4]])
    assert r4.status_code == 200 and r4.json()["version"] == 4
    assert _anchor_set(r4.json()) == {(4, 4)}

    final = _get(client, jid)
    assert final == r4.json()


# ---------------------------------------------------------------- 落库核验


def test_version_and_idempotency_record_persisted_in_database(
    client: httpx.Client,
) -> None:
    """版本与幂等记录真实写入 PostgreSQL，重放不在库中重复推进版本。"""
    psycopg = pytest.importorskip("psycopg")
    import os

    body = create_job(client, LEFT, RIGHT)
    jid = body["id"]
    first = _put(client, jid, [[2, 3]], base_version=0, key=f"db-check-{jid}")
    assert first.status_code == 200
    replay = _put(client, jid, [[2, 3]], base_version=0, key=f"db-check-{jid}")
    assert replay.status_code == 200

    dsn = os.environ["PG_DSN"]
    with psycopg.connect(dsn) as conn, conn.cursor() as cur:
        cur.execute("SELECT version, anchors FROM jobs WHERE id = %s", (jid,))
        row = cur.fetchone()
        assert row is not None, "任务未写入 PostgreSQL"
        assert row[0] == 1, "重放不得在数据库中重复推进版本"
        assert [tuple(p) for p in row[1]] == [(2, 3)]
        cur.execute(
            "SELECT status_code FROM idempotency_records WHERE key = %s",
            (f"db-check-{jid}",),
        )
        assert cur.fetchone() == (200,), "首次成功响应未持久化幂等记录"
