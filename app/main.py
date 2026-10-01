"""FastAPI 纯后端：扫描对应（LCS）任务的创建、查询与约束重算。

协议见 README。所有非法输入（请求体域校验、非法锚点、非法禁止对应对）
统一返回 422，且约束非法时数据库原状态不变；任务不存在返回 404。

作业约束分两类，共用同一份版本裁决：

- 锚点：人工核实、必须出现在结果中的索引对；
- 禁止对应对：人工确认误配、不得出现在结果中的索引对。

任一成功修改（替换锚点或替换禁止集合）都把 ``version`` 加一并重算。携带
``base_version`` 的请求若与当前版本不一致，或落库时版本已被并发请求推进，
都返回 409 且不改变任何状态——并发修改中恰好一个成功，旧版本请求不能覆盖
另一类约束。携带 ``Idempotency-Key`` 请求头的逻辑请求，其首次成功响应被
持久化；超时重放返回首次裁决结果，绝不重复执行、绝不覆盖较新状态。
"""

from __future__ import annotations

import hashlib
import json
import uuid
from collections.abc import Sequence
from datetime import datetime, timezone
from typing import Annotated

from fastapi import Depends, FastAPI, Header, HTTPException, Request
from fastapi.responses import JSONResponse
from sqlalchemy import select, update
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session

from . import lcs
from .db import IdempotencyRecord, Job, get_session, init_db
from .schemas import AnchorIn, ForbiddenIn, JobCreate, JobOut, Pair
from .validation import InvalidInput, validate_side

app = FastAPI(
    title="胶片扫描对应 API",
    version="1.2.0",
    description="两台扫描机指纹数组的最长对应（LCS）求解，支持锚点、禁止对应对与并发裁决。",
)

MAX_IDEMPOTENCY_KEY_LEN = 200


class ConflictError(RuntimeError):
    """请求与当前作业状态冲突（过期版本、并发竞争或幂等键冲突）。"""


@app.on_event("startup")
def _startup() -> None:
    init_db()


@app.exception_handler(InvalidInput)
async def _invalid_input_handler(_request: Request, exc: InvalidInput) -> JSONResponse:
    return JSONResponse(status_code=422, content={"detail": str(exc)})


@app.exception_handler(lcs.ConstraintError)
async def _constraint_error_handler(
    _request: Request, exc: lcs.ConstraintError
) -> JSONResponse:
    # AnchorError / ForbiddenError 均为其子类，统一映射 422。
    return JSONResponse(status_code=422, content={"detail": str(exc)})


@app.exception_handler(ConflictError)
async def _conflict_handler(_request: Request, exc: ConflictError) -> JSONResponse:
    return JSONResponse(status_code=409, content={"detail": str(exc)})


def _as_utc(value: datetime) -> datetime:
    """统一为 UTC 时间，保证响应与落库读回的值逐项一致。"""
    if value.tzinfo is None:
        return value.replace(tzinfo=timezone.utc)
    return value.astimezone(timezone.utc)


def _stored_pairs(value: list | None) -> list[tuple[int, int]]:
    """存储层 JSON 对应对 → 算法层索引对：接口、算法、存储共享同一规范化身份。"""
    return [(int(p[0]), int(p[1])) for p in (value or [])]


def _serialize(job: Job) -> JobOut:
    return JobOut(
        id=job.id,
        left=job.left_data,
        right=job.right_data,
        anchors=[Pair(left_index=i, right_index=j) for i, j in job.anchors],
        forbidden=[
            Pair(left_index=i, right_index=j) for i, j in _stored_pairs(job.forbidden)
        ],
        result=[Pair(left_index=i, right_index=j) for i, j in job.result],
        length=len(job.result),
        version=job.version,
        updated_at=_as_utc(job.updated_at),
    )


def _fingerprint(
    job_id: str,
    kind: str,
    pairs: Sequence[Sequence[int]],
    base_version: int | None,
) -> str:
    """逻辑请求的规范指纹：同一请求的超时重放必然得到同一指纹。

    ``kind`` 区分约束类别（``"anchors"`` / ``"forbidden"``）：同一幂等键
    搭配另一类约束或不同集合会得到不同指纹，从而被裁决为冲突而非误重放。
    """
    canonical = json.dumps(
        {
            "job_id": job_id,
            kind: sorted([[int(p[0]), int(p[1])] for p in pairs]),
            "base_version": base_version,
        },
        ensure_ascii=False,
        sort_keys=True,
    )
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def _replay(
    record: IdempotencyRecord, fingerprint: str, current_version: int
) -> JSONResponse:
    """返回已记录请求的首次裁决；状态已被后续请求推进时给出可解释的 409。"""
    if record.request_hash != fingerprint:
        raise ConflictError("Idempotency-Key 已被不同的请求使用")
    applied_version = record.response_body.get("version")
    if record.status_code == 200 and applied_version != current_version:
        raise ConflictError(
            f"该请求已在版本 {applied_version} 成功应用；当前版本为 "
            f"{current_version}，状态已被后续请求推进，未重复执行"
        )
    return JSONResponse(status_code=record.status_code, content=record.response_body)


def _current_version(session: Session, job_id: str, fallback: int) -> int:
    """读取当前已提交的版本（新语句快照，不受 ORM 身份映射中旧值影响）。"""
    version = session.execute(
        select(Job.version).where(Job.id == job_id)
    ).scalar_one_or_none()
    return fallback if version is None else version


def _get_job_or_404(session: Session, job_id: str) -> Job:
    job = session.get(Job, job_id)
    if job is None:
        raise HTTPException(status_code=404, detail="任务不存在")
    return job


def _check_idempotency_key(idempotency_key: str | None) -> None:
    if idempotency_key is not None and len(idempotency_key) > MAX_IDEMPOTENCY_KEY_LEN:
        raise InvalidInput(
            f"Idempotency-Key 长度不能超过 {MAX_IDEMPOTENCY_KEY_LEN} 字符"
        )


def _find_replay(
    session: Session,
    job_id: str,
    idempotency_key: str | None,
    fingerprint: str,
    fallback_version: int,
) -> JSONResponse | None:
    """已记录同一逻辑请求时返回其首次裁决；否则返回 None。"""
    if idempotency_key is None:
        return None
    record = session.get(IdempotencyRecord, idempotency_key)
    if record is None:
        return None
    # job 可能是并发提交前读到的旧快照；用新语句取当前版本再裁决。
    return _replay(
        record, fingerprint, _current_version(session, job_id, fallback_version)
    )


def _check_base_version(job: Job, base_version: int | None) -> None:
    if base_version is not None and base_version != job.version:
        raise ConflictError(
            f"作业状态已变更：请求基于版本 {base_version}，"
            f"当前版本为 {job.version}；请重新获取任务后重试"
        )


def _reject_clash(
    anchors: Sequence[tuple[int, int]], forbidden: Sequence[tuple[int, int]]
) -> None:
    """锚点与禁止对应对相交时整次拒绝（422），不留半次修改。"""
    clash = sorted(set(anchors) & set(forbidden))
    if clash:
        shown = ", ".join(f"({i}, {j})" for i, j in clash[:3])
        suffix = " 等" if len(clash) > 3 else ""
        raise lcs.ConstraintError(
            f"锚点与禁止对应对相交：{shown}{suffix}共 {len(clash)} 对"
            "既被锚点强制包含又被禁止集合排除；请调整其中一类约束后重试"
        )


def _apply_constraint_change(
    session: Session,
    job: Job,
    *,
    anchors: list[tuple[int, int]],
    forbidden: list[tuple[int, int]],
    idempotency_key: str | None,
    fingerprint: str,
) -> JobOut | JSONResponse:
    """两类约束修改共用的重算与落库裁决。

    约束（锚点 + 禁止对应对）、重算结果、版本与时间戳在同一条条件 UPDATE
    中落库，幂等记录在同一事务提交：要么全部生效，要么整体回滚，失败不留
    半次修改。仅当版本仍等于读取版本时更新生效，并发下恰好一个请求胜出。
    """
    job_id = job.id
    read_version = job.version
    result = lcs.solve(job.left_data, job.right_data, anchors, forbidden)

    new_version = read_version + 1
    now = datetime.now(timezone.utc)

    # 原子裁决：仅当版本仍等于读取时的版本，本次替换才落库。并发下恰好
    # 一个请求胜出；落败请求（rowcount != 1）不得改变约束、结果、版本或
    # 更新时间。并发事务提交前，本语句在行锁上等待，保证落败方能读到
    # 胜出方已提交的幂等记录。
    outcome = session.execute(
        update(Job)
        .where(Job.id == job_id, Job.version == read_version)
        .values(
            anchors=[[i, j] for i, j in anchors],
            forbidden=[[i, j] for i, j in forbidden],
            result=[[i, j] for i, j in result],
            version=new_version,
            updated_at=now,
        )
    )
    if outcome.rowcount != 1:
        session.rollback()
        replay = _find_replay(session, job_id, idempotency_key, fingerprint, read_version)
        if replay is not None:
            # 同一逻辑请求的并发副本已胜出并提交：返回其首次裁决。
            return replay
        raise ConflictError("作业已被并发修改，本次替换未应用；请重新获取任务后重试")

    body = JobOut(
        id=job_id,
        left=job.left_data,
        right=job.right_data,
        anchors=[Pair(left_index=i, right_index=j) for i, j in anchors],
        forbidden=[Pair(left_index=i, right_index=j) for i, j in forbidden],
        result=[Pair(left_index=i, right_index=j) for i, j in result],
        length=len(result),
        version=new_version,
        updated_at=now,
    )
    if idempotency_key is not None:
        # 与状态变更同一事务落库：成功响应即持久化真值，崩溃也不会脱节。
        session.add(
            IdempotencyRecord(
                key=idempotency_key,
                job_id=job_id,
                request_hash=fingerprint,
                status_code=200,
                response_body=body.model_dump(mode="json"),
            )
        )
    try:
        session.commit()
    except IntegrityError:
        # 幂等键被并发请求先一步记录（例如同一键用于其他任务）。
        session.rollback()
        replay = _find_replay(session, job_id, idempotency_key, fingerprint, read_version)
        if replay is not None:
            return replay
        raise ConflictError("幂等键冲突：请更换 Idempotency-Key 后重试") from None
    return body


@app.post("/api/jobs", response_model=JobOut, status_code=201)
def create_job(payload: JobCreate, session: Session = Depends(get_session)) -> JobOut:
    # 域校验失败抛 InvalidInput -> 422，不写库。
    validate_side("left", payload.left)
    validate_side("right", payload.right)
    result = lcs.solve(payload.left, payload.right)
    job = Job(
        id=str(uuid.uuid4()),
        left_data=payload.left,
        right_data=payload.right,
        anchors=[],
        forbidden=[],
        result=result,
    )
    session.add(job)
    session.commit()
    return _serialize(job)


@app.get("/api/jobs/{job_id}", response_model=JobOut)
def get_job(job_id: str, session: Session = Depends(get_session)) -> JobOut:
    return _serialize(_get_job_or_404(session, job_id))


@app.put("/api/jobs/{job_id}/anchors", response_model=JobOut)
def replace_anchors(
    job_id: str,
    payload: AnchorIn,
    idempotency_key: Annotated[str | None, Header()] = None,
    session: Session = Depends(get_session),
) -> JobOut | JSONResponse:
    job = _get_job_or_404(session, job_id)
    _check_idempotency_key(idempotency_key)
    fingerprint = _fingerprint(job_id, "anchors", payload.anchors, payload.base_version)

    # 幂等重放：同一逻辑请求已成功过，直接返回首次裁决，绝不重复执行。
    replay = _find_replay(session, job_id, idempotency_key, fingerprint, job.version)
    if replay is not None:
        return replay

    _check_base_version(job, payload.base_version)

    # 先完整校验（非法抛 AnchorError -> 422），通过后才重算与落库，
    # 保证“非法锚点返回 422，原状态不变”。
    ordered = lcs.validate_anchors(job.left_data, job.right_data, payload.anchors)
    forbidden = _stored_pairs(job.forbidden)
    _reject_clash(ordered, forbidden)
    return _apply_constraint_change(
        session,
        job,
        anchors=ordered,
        forbidden=forbidden,
        idempotency_key=idempotency_key,
        fingerprint=fingerprint,
    )


@app.put("/api/jobs/{job_id}/forbidden", response_model=JobOut)
def replace_forbidden(
    job_id: str,
    payload: ForbiddenIn,
    idempotency_key: Annotated[str | None, Header()] = None,
    session: Session = Depends(get_session),
) -> JobOut | JSONResponse:
    """用新禁止对应对集合替换旧集合并重算；与替换锚点共用版本裁决。"""
    job = _get_job_or_404(session, job_id)
    _check_idempotency_key(idempotency_key)
    fingerprint = _fingerprint(
        job_id, "forbidden", payload.forbidden, payload.base_version
    )

    replay = _find_replay(session, job_id, idempotency_key, fingerprint, job.version)
    if replay is not None:
        return replay

    _check_base_version(job, payload.base_version)

    # 禁止对应对是集合语义：元素合法且指纹相等即可，规范化（排序去重）后落库。
    ordered = lcs.validate_forbidden(job.left_data, job.right_data, payload.forbidden)
    anchors = _stored_pairs(job.anchors)
    _reject_clash(anchors, ordered)
    return _apply_constraint_change(
        session,
        job,
        anchors=anchors,
        forbidden=ordered,
        idempotency_key=idempotency_key,
        fingerprint=fingerprint,
    )


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}
