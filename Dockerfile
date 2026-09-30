FROM python:3.12-slim

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /app

# psycopg[binary] 使用预编译 wheel，无需额外构建工具。
COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt \
    && groupadd --system app \
    && useradd --system --gid app --home-dir /app --shell /usr/sbin/nologin app

COPY --chown=app:app . .

USER app

# 由 API_PORT 决定监听端口（compose 注入，默认 8000）。
CMD uvicorn app.main:app --host 0.0.0.0 --port "${API_PORT:-8000}"
