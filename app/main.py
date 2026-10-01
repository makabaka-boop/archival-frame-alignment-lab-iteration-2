"""FastAPI 纯后端：扫描对应（LCS）任务的创建、查询与约束重算。

协议见 README。所有非法输入（请求体域校验、非法锚点/禁止对）统一返回 422，
且约束非法时数据库原状态不变；任务不存在返回 404。

两类可替换约束共用同一套版本裁决与幂等语义：

* 锚点集合 ``PUT /api/jobs/{id}/anchors``——结果必须包含全部锚点；
* 禁止对应对集合 ``PUT /api/jobs/{id}/forbidden-pairs``——人工确认的
  同指纹误配点，重算时绝不进入最终最长对应；与锚点相交则整次拒绝。

任一成功修改都把 ``version`` 加一。携带 ``base_version`` 的请求若与当前版本
不一致，或落库时版本已被并发请求（包括另一类约束的修改）推进，都返回 409
且不改变任何状态——两个并发替换中恰好一个成功。携带 ``Idempotency-Key``
请求头的逻辑请求，其首次成功响应被持久化；超时重放返回首次裁决结果，绝不
重复执行、绝不覆盖较新状态。
"""

from __future__ import annotations

import hashlib
import json
import uuid
from datetime import datetime, timezone
from typing import Annotated, Any

from fastapi import Depends, FastAPI, Header, HTTPException, Request
from fastapi.responses import JSONResponse
from sqlalchemy import select, update
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session

from . import lcs
from .db import IdempotencyRecord, Job, get_session, init_db
from .pairs import ConstraintError, IndexPair
from .schemas import AnchorIn, ForbiddenIn, JobCreate, JobOut, Pair
from .validation import InvalidInput, validate_side

app = FastAPI(
    title="胶片扫描对应 API",
    version="1.2.0",
    description=(
        "两台扫描机指纹数组的最长对应（LCS）求解，支持锚点、禁止对应对"
        "重算与并发裁决。"
    ),
)

MAX_IDEMPOTENCY_KEY_LEN = 200

# 约束种类（即各自请求体字段名）：两类替换的裁决、幂等、重算路径完全
# 共用，仅写入的列与返回字段不同。
_KIND_ANCHORS = "anchors"
_KIND_FORBIDDEN = "forbidden_pairs"


class ConflictError(RuntimeError):
    """请求与当前作业状态冲突（过期版本、并发竞争或幂等键冲突）。"""


@app.on_event("startup")
def _startup() -> None:
    init_db()


@app.exception_handler(InvalidInput)
async def _invalid_input_handler(_request: Request, exc: InvalidInput) -> JSONResponse:
    return JSONResponse(status_code=422, content={"detail": str(exc)})


@app.exception_handler(ConstraintError)
async def _constraint_error_handler(
    _request: Request, exc: ConstraintError
) -> JSONResponse:
    return JSONResponse(status_code=422, content={"detail": str(exc)})


@app.exception_handler(ConflictError)
async def _conflict_handler(_request: Request, exc: ConflictError) -> JSONResponse:
    return JSONResponse(status_code=409, content={"detail": str(exc)})


def _as_utc(value: datetime) -> datetime:
    """统一为 UTC 时间，保证响应与落库读回的值逐项一致。"""
    if value.tzinfo is None:
        return value.replace(tzinfo=timezone.utc)
    return value.astimezone(timezone.utc)


def _pair_models(pairs: list[IndexPair]) -> list[Pair]:
    return [Pair(left_index=i, right_index=j) for i, j in pairs]


def _serialize(
    job: Job,
    anchors: list[IndexPair] | None = None,
    forbidden: list[IndexPair] | None = None,
    result: list[IndexPair] | None = None,
    version: int | None = None,
    updated_at: datetime | None = None,
) -> JobOut:
    """组装作业完整表示；可显式传入尚未落库或已重算的新值。

    所有 JSON 列都经存储层的规范化索引身份转换，保证响应与落库读回一致。
    """
    resolved_result = job.result_pairs() if result is None else result
    return JobOut(
        id=job.id,
        left=job.left_data,
        right=job.right_data,
        anchors=_pair_models(job.anchor_pairs() if anchors is None else anchors),
        forbidden_pairs=_pair_models(
            job.forbidden_index_pairs() if forbidden is None else forbidden
        ),
        result=_pair_models(resolved_result),
        length=len(resolved_result),
        version=job.version if version is None else version,
        updated_at=_as_utc(job.updated_at if updated_at is None else updated_at),
    )


def _canonical(pairs: Any) -> list[list[int]]:
    """逻辑请求指纹用的规范形态：排序后的整数对数组。"""
    return sorted([[int(p[0]), int(p[1])] for p in pairs])


def _fingerprint(
    job_id: str, kind: str, pairs: Any, base_version: int | None
) -> str:
    """逻辑请求的规范指纹：同一种类、同一集合、同一基准版本的超时重放必然同指纹。

    以请求体字段名（``anchors`` / ``forbidden_pairs``）作为 JSON 键，两类
    请求天然区分；锚点分支的字节形态与升级前完全一致，旧客户端在服务升级
    后的超时重放仍能命中原幂等记录。
    """
    canonical = json.dumps(
        {
            "job_id": job_id,
            kind: _canonical(pairs),
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


@app.post("/api/jobs", response_model=JobOut, status_code=201)
def create_job(payload: JobCreate, session: Session = Depends(get_session)) -> JobOut:
    # 域校验失败抛 InvalidInput -> 422，不写库。
    validate_side("left", payload.left)
    validate_side("right", payload.right)
    # 新作业没有任何人工约束：空锚点、空禁止集合。
    result = lcs.solve(payload.left, payload.right)
    job = Job(
        id=str(uuid.uuid4()),
        left_data=payload.left,
        right_data=payload.right,
        anchors=[],
        forbidden_pairs=[],
        result=result,
    )
    session.add(job)
    session.commit()
    return _serialize(job, result=result)


@app.get("/api/jobs/{job_id}", response_model=JobOut)
def get_job(job_id: str, session: Session = Depends(get_session)) -> JobOut:
    job = session.get(Job, job_id)
    if job is None:
        raise HTTPException(status_code=404, detail="任务不存在")
    return _serialize(job)


def _replace_constraint_set(
    job_id: str,
    kind: str,
    new_pairs: Any,
    base_version: int | None,
    idempotency_key: str | None,
    session: Session,
) -> JobOut | JSONResponse:
    """两类约束替换共用的裁决、重算与落库路径。

    ``kind`` 为 ``_KIND_ANCHORS`` 或 ``_KIND_FORBIDDEN``；``new_pairs`` 为
    本次提交的规范整数对序列。约束、结果与版本在同一事务原子更新：任何
    校验/冲突失败都不留半次修改。
    """
    job = session.get(Job, job_id)
    if job is None:
        raise HTTPException(status_code=404, detail="任务不存在")

    if idempotency_key is not None and len(idempotency_key) > MAX_IDEMPOTENCY_KEY_LEN:
        raise InvalidInput(
            f"Idempotency-Key 长度不能超过 {MAX_IDEMPOTENCY_KEY_LEN} 字符"
        )

    fingerprint = _fingerprint(job_id, kind, new_pairs, base_version)

    # 幂等重放：同一逻辑请求已成功过，直接返回首次裁决，绝不重复执行。
    if idempotency_key is not None:
        record = session.get(IdempotencyRecord, idempotency_key)
        if record is not None:
            # job 可能是并发提交前读到的旧快照；用新语句取当前版本再裁决。
            return _replay(
                record, fingerprint, _current_version(session, job_id, job.version)
            )

    read_version = job.version
    if base_version is not None and base_version != read_version:
        raise ConflictError(
            f"作业状态已变更：请求基于版本 {base_version}，"
            f"当前版本为 {read_version}；请重新获取任务后重试"
        )

    # 以读快照中的“另一类约束”配合本次新集合重算；当前类约束被整集替换。
    incoming = [tuple(p) for p in new_pairs]
    if kind == _KIND_ANCHORS:
        anchors_raw, forbidden_raw = incoming, job.forbidden_index_pairs()
    else:
        anchors_raw, forbidden_raw = job.anchor_pairs(), incoming

    # 先完整校验并规范化（非法抛 ConstraintError -> 422，含锚点∩禁止集合相交），
    # 通过后才落库，保证“非法请求返回 422，原状态不变”，且持久化的约束与
    # 算法消费的是同一份规范索引身份。
    anchors, forbidden, result = lcs.prepare_and_solve(
        job.left_data, job.right_data, anchors_raw, forbidden_raw
    )

    anchors_json = [[i, j] for i, j in anchors]
    forbidden_json = [[i, j] for i, j in forbidden]
    result_json = [[i, j] for i, j in result]
    new_version = read_version + 1
    now = datetime.now(timezone.utc)

    # 原子裁决：仅当版本仍等于读取时的版本，本次替换才落库。并发（含另一类
    # 约束的并发修改）下恰好一个请求胜出；落败请求（rowcount != 1）不得改变
    # 约束、结果、版本或更新时间。并发事务提交前，本语句在行锁上等待，保证
    # 落败方能读到胜出方已提交的幂等记录。
    outcome = session.execute(
        update(Job)
        .where(Job.id == job_id, Job.version == read_version)
        .values(
            anchors=anchors_json,
            forbidden_pairs=forbidden_json,
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
        anchors=_pair_models(anchors),
        forbidden_pairs=_pair_models(forbidden),
        result=_pair_models(result),
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


@app.put("/api/jobs/{job_id}/anchors", response_model=JobOut)
def replace_anchors(
    job_id: str,
    payload: AnchorIn,
    idempotency_key: Annotated[str | None, Header()] = None,
    session: Session = Depends(get_session),
) -> JobOut | JSONResponse:
    return _replace_constraint_set(
        job_id,
        _KIND_ANCHORS,
        payload.anchors,
        payload.base_version,
        idempotency_key,
        session,
    )


@app.put("/api/jobs/{job_id}/forbidden-pairs", response_model=JobOut)
def replace_forbidden_pairs(
    job_id: str,
    payload: ForbiddenIn,
    idempotency_key: Annotated[str | None, Header()] = None,
    session: Session = Depends(get_session),
) -> JobOut | JSONResponse:
    return _replace_constraint_set(
        job_id,
        _KIND_FORBIDDEN,
        payload.forbidden_pairs,
        payload.base_version,
        idempotency_key,
        session,
    )


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}
