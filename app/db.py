"""PostgreSQL 持久化：扫描数组、锚点、禁止对应对、结果、版本与幂等记录全部落库，重启后可继续工作。"""

from __future__ import annotations

import os
from collections.abc import Iterator
from datetime import datetime, timezone

from sqlalchemy import JSON, DateTime, Integer, String, create_engine, text
from sqlalchemy.orm import (
    DeclarativeBase,
    Mapped,
    mapped_column,
    sessionmaker,
    Session,
)

from .pairs import IndexPair, rows_to_pairs

DATABASE_URL = os.environ.get(
    "DATABASE_URL", "postgresql+psycopg://lcs:lcs@db:5432/lcs"
)


class Base(DeclarativeBase):
    pass


class Job(Base):
    __tablename__ = "jobs"

    id: Mapped[str] = mapped_column(String(36), primary_key=True)
    # PostgreSQL 原生 JSON 列；存为 JSON 字符串数组/索引对数组。
    left_data: Mapped[list] = mapped_column("left_data", JSON, nullable=False)
    right_data: Mapped[list] = mapped_column("right_data", JSON, nullable=False)
    anchors: Mapped[list] = mapped_column(JSON, nullable=False, default=list)
    # 禁止对应对（人工确认的误配点）：替换语义，空数组 = 不禁止任何对。
    forbidden_pairs: Mapped[list] = mapped_column(
        "forbidden_pairs", JSON, nullable=False, default=list
    )
    result: Mapped[list] = mapped_column(JSON, nullable=False, default=list)
    # 乐观并发令牌：创建为 0，每次成功替换锚点或禁止集合 +1。
    # 过期或并发落败的请求据此被唯一裁决，不得改变任何状态。
    version: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        nullable=False,
        default=lambda: datetime.now(timezone.utc),
    )

    def anchor_pairs(self) -> list[IndexPair]:
        """数据库 JSON 值 -> 算法层规范化索引对身份。"""
        return rows_to_pairs(self.anchors)

    def forbidden_index_pairs(self) -> list[IndexPair]:
        return rows_to_pairs(self.forbidden_pairs)

    def result_pairs(self) -> list[IndexPair]:
        return rows_to_pairs(self.result)


class IdempotencyRecord(Base):
    """已成功替换请求的幂等记录：超时重放返回首次裁决，绝不重复执行。"""

    __tablename__ = "idempotency_records"

    key: Mapped[str] = mapped_column(String(200), primary_key=True)
    job_id: Mapped[str] = mapped_column(String(36), nullable=False, index=True)
    # 逻辑请求（任务 + 约束种类 + 索引对集合 + 基准版本）的 SHA-256 指纹。
    request_hash: Mapped[str] = mapped_column(String(64), nullable=False)
    status_code: Mapped[int] = mapped_column(Integer, nullable=False)
    response_body: Mapped[dict] = mapped_column(JSON, nullable=False)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        nullable=False,
        default=lambda: datetime.now(timezone.utc),
    )


engine = create_engine(
    DATABASE_URL,
    pool_pre_ping=True,
    pool_size=5,
    max_overflow=10,
)
SessionLocal = sessionmaker(bind=engine, autoflush=False, expire_on_commit=False)


def _migrate_jobs(conn) -> None:
    """jobs 表轻量迁移（幂等）：旧库就地加列，旧作业按空约束/空禁止集合读取。"""
    # 既有 jobs 表（旧版本库）就地升级；新库下列已存在，语句为空操作。
    conn.execute(
        text(
            "ALTER TABLE jobs ADD COLUMN IF NOT EXISTS version "
            "INTEGER NOT NULL DEFAULT 0"
        )
    )
    conn.execute(
        text("ALTER TABLE jobs ADD COLUMN IF NOT EXISTS updated_at TIMESTAMPTZ")
    )
    conn.execute(
        text("UPDATE jobs SET updated_at = CURRENT_TIMESTAMP WHERE updated_at IS NULL")
    )
    conn.execute(text("ALTER TABLE jobs ALTER COLUMN updated_at SET NOT NULL"))
    conn.execute(
        text("ALTER TABLE jobs ALTER COLUMN updated_at SET DEFAULT CURRENT_TIMESTAMP")
    )
    # 禁止对应对：旧作业按空禁止集合读取（JSON 默认 '[]'）。
    conn.execute(
        text(
            "ALTER TABLE jobs ADD COLUMN IF NOT EXISTS forbidden_pairs "
            "JSON NOT NULL DEFAULT '[]'"
        )
    )
    conn.execute(
        text("UPDATE jobs SET forbidden_pairs = '[]' WHERE forbidden_pairs IS NULL")
    )


def init_db() -> None:
    """建表与迁移（幂等）；容器启动时调用。"""
    Base.metadata.create_all(engine)
    with engine.begin() as conn:
        _migrate_jobs(conn)


def get_session() -> Iterator[Session]:
    session = SessionLocal()
    try:
        yield session
    finally:
        session.close()
