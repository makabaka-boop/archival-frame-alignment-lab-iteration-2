"""验收测试夹具：面向真实 HTTP 服务与真实 PostgreSQL，无桩、无假接口。"""

from __future__ import annotations

import os
import time
from collections.abc import Iterator

import httpx
import pytest

API_BASE_URL = os.environ.get("API_BASE_URL", "http://localhost:8000")
PG_DSN = os.environ.get("PG_DSN", "postgresql://lcs:lcs@localhost:5432/lcs")


@pytest.fixture(scope="session")
def client() -> Iterator[httpx.Client]:
    deadline = time.time() + 30
    last_exc: Exception | None = None
    while time.time() < deadline:
        try:
            with httpx.Client(base_url=API_BASE_URL, timeout=30.0) as probe:
                if probe.get("/health").status_code == 200:
                    break
        except Exception as exc:  # 服务尚未就绪
            last_exc = exc
        time.sleep(0.5)
    else:
        pytest.fail(f"API 在 30 秒内未就绪：{last_exc}")

    with httpx.Client(base_url=API_BASE_URL, timeout=60.0) as c:
        yield c


def create_job(client: httpx.Client, left: list[str], right: list[str]) -> dict:
    resp = client.post("/api/jobs", json={"left": left, "right": right})
    assert resp.status_code == 201, resp.text
    return resp.json()


def set_anchors(client: httpx.Client, job_id: str, anchors: list[list[int]]) -> httpx.Response:
    return client.put(f"/api/jobs/{job_id}/anchors", json={"anchors": anchors})


def assert_chain_valid(body: dict, left: list[str], right: list[str]) -> list[tuple[int, int]]:
    """通用合法断言：严格递增、指纹相等。"""
    pairs = [(p["left_index"], p["right_index"]) for p in body["result"]]
    assert body["length"] == len(pairs)
    prev_i, prev_j = -1, -1
    for i, j in pairs:
        assert i > prev_i and j > prev_j
        assert left[i] == right[j]
        prev_i, prev_j = i, j
    return pairs
