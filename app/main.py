"""FastAPI 纯后端：扫描对应（LCS）任务的创建、查询与锚点重算。

协议见 README。所有非法输入（请求体域校验、非法锚点）统一返回 422，
且锚点非法时数据库原状态不变；任务不存在返回 404。

并发裁决：每次成功替换锚点都会把 ``version`` 加一。携带 ``base_version``
的请求若与当前版本不一致，或落库时版本已被并发请求推进，都返回 409 且
不改变任何状态——两个并发替换中恰好一个成功。携带 ``Idempotency-Key``
请求头的逻辑请求，其首次成功响应被持久化；超时重放返回首次裁决结果，
绝不重复执行、绝不覆盖较新状态。
"""

from __future__ import annotations

import hashlib
import json
import uuid
from datetime import datetime, timezone
from typing import Annotated

from fastapi import Depends, FastAPI, Header, HTTPException, Request
from fastapi.responses import JSONResponse
from sqlalchemy import select, update
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session

from . import lcs
from .db import IdempotencyRecord, Job, get_session, init_db
from .schemas import AnchorIn, JobCreate, JobOut, Pair
from .validation import InvalidInput, validate_side

app = FastAPI(
    title="胶片扫描对应 API",
    version="1.1.0",
    description="两台扫描机指纹数组的最长对应（LCS）求解，支持锚点重算与并发裁决。",
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


@app.exception_handler(lcs.AnchorError)
async def _anchor_error_handler(_request: Request, exc: lcs.AnchorError) -> JSONResponse:
    return JSONResponse(status_code=422, content={"detail": str(exc)})


@app.exception_handler(ConflictError)
async def _conflict_handler(_request: Request, exc: ConflictError) -> JSONResponse:
    return JSONResponse(status_code=409, content={"detail": str(exc)})


def _as_utc(value: datetime) -> datetime:
    """统一为 UTC 时间，保证响应与落库读回的值逐项一致。"""
    if value.tzinfo is None:
        return value.replace(tzinfo=timezone.utc)
    return value.astimezone(timezone.utc)


def _serialize(job: Job) -> JobOut:
    return JobOut(
        id=job.id,
        left=job.left_data,
        right=job.right_data,
        anchors=[Pair(left_index=i, right_index=j) for i, j in job.anchors],
        result=[Pair(left_index=i, right_index=j) for i, j in job.result],
        length=len(job.result),
        version=job.version,
        updated_at=_as_utc(job.updated_at),
    )


def _fingerprint(job_id: str, payload: AnchorIn) -> str:
    """逻辑请求的规范指纹：同一请求的超时重放必然得到同一指纹。"""
    canonical = json.dumps(
        {
            "job_id": job_id,
            "anchors": sorted([[int(a[0]), int(a[1])] for a in payload.anchors]),
            "base_version": payload.base_version,
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
        result=result,
    )
    session.add(job)
    session.commit()
    return _serialize(job)


@app.get("/api/jobs/{job_id}", response_model=JobOut)
def get_job(job_id: str, session: Session = Depends(get_session)) -> JobOut:
    job = session.get(Job, job_id)
    if job is None:
        raise HTTPException(status_code=404, detail="任务不存在")
    return _serialize(job)


@app.put("/api/jobs/{job_id}/anchors", response_model=JobOut)
def replace_anchors(
    job_id: str,
    payload: AnchorIn,
    idempotency_key: Annotated[str | None, Header()] = None,
    session: Session = Depends(get_session),
) -> JobOut | JSONResponse:
    job = session.get(Job, job_id)
    if job is None:
        raise HTTPException(status_code=404, detail="任务不存在")

    if idempotency_key is not None and len(idempotency_key) > MAX_IDEMPOTENCY_KEY_LEN:
        raise InvalidInput(
            f"Idempotency-Key 长度不能超过 {MAX_IDEMPOTENCY_KEY_LEN} 字符"
        )

    fingerprint = _fingerprint(job_id, payload)

    # 幂等重放：同一逻辑请求已成功过，直接返回首次裁决，绝不重复执行。
    if idempotency_key is not None:
        record = session.get(IdempotencyRecord, idempotency_key)
        if record is not None:
            # job 可能是并发提交前读到的旧快照；用新语句取当前版本再裁决。
            return _replay(
                record, fingerprint, _current_version(session, job_id, job.version)
            )

    read_version = job.version
    if payload.base_version is not None and payload.base_version != read_version:
        raise ConflictError(
            f"作业状态已变更：请求基于版本 {payload.base_version}，"
            f"当前版本为 {read_version}；请重新获取任务后重试"
        )

    # 先完整校验（非法抛 AnchorError -> 422），通过后才重算与落库，
    # 保证“非法锚点返回 422，原状态不变”。
    ordered = lcs.validate_anchors(job.left_data, job.right_data, payload.anchors)
    result = lcs.solve(job.left_data, job.right_data, ordered)

    new_version = read_version + 1
    now = datetime.now(timezone.utc)
    anchors_json = [[i, j] for i, j in ordered]
    result_json = [[i, j] for i, j in result]

    # 原子裁决：仅当版本仍等于读取时的版本，本次替换才落库。并发下恰好
    # 一个请求胜出；落败请求（rowcount != 1）不得改变锚点、结果、版本或
    # 更新时间。并发事务提交前，本语句在行锁上等待，保证落败方能读到
    # 胜出方已提交的幂等记录。
    outcome = session.execute(
        update(Job)
        .where(Job.id == job_id, Job.version == read_version)
        .values(
            anchors=anchors_json,
            result=result_json,
            version=new_version,
            updated_at=now,
        )
    )
    if outcome.rowcount != 1:
        session.rollback()
        if idempotency_key is not None:
            # 同一逻辑请求的并发副本已胜出并提交：返回其首次裁决。
            record = session.get(IdempotencyRecord, idempotency_key)
            if record is not None:
                return _replay(
                    record,
                    fingerprint,
                    _current_version(session, job_id, read_version),
                )
        raise ConflictError("作业已被并发修改，本次替换未应用；请重新获取任务后重试")

    body = JobOut(
        id=job_id,
        left=job.left_data,
        right=job.right_data,
        anchors=[Pair(left_index=i, right_index=j) for i, j in ordered],
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
        if idempotency_key is not None:
            record = session.get(IdempotencyRecord, idempotency_key)
            if record is not None:
                return _replay(
                    record,
                    fingerprint,
                    _current_version(session, job_id, read_version),
                )
        raise ConflictError("幂等键冲突：请更换 Idempotency-Key 后重试") from None
    return body


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}
