# 胶片扫描对应 API

同卷胶片被两台扫描机分别采集。掉帧与重复画面会让首次按指纹拼接时错序。
本服务计算两台扫描机指纹数组之间的**全局最长对应**（最长公共子序列，LCS）：

- 对应关系是零基索引对 `(left_index, right_index)` 的序列；
- 两侧索引都严格递增，且两侧指纹相等；
- **先最大化对数**；对数并列时，取索引对序列（逐对比较）**字典序最小**者；
- 无任何匹配时返回空数组。

修复师可提交**锚点集合**（已人工核实的帧对应）替换旧集合并触发重算：
结果必须包含全部锚点，再按相同目标（最长、字典序最小）裁决。锚点非法时返回
`422` 且数据库原状态不变；提交空集合即清空锚点、恢复全局最优。

两名修复师可能同时编辑同一作业。作业带有单调递增的 `version`，替换请求可
携带所基于的 `base_version` 与 `Idempotency-Key` 请求头：并发替换中**恰好
一个**请求成功，过期或落败的请求收到 `409` 且不改变任何状态；超时重放返回
首次裁决结果，绝不重复执行、绝不覆盖其他修复师的较新核实结论。

纯后端 API：Python 3.12 + FastAPI + PostgreSQL。无任何桩实现或假接口。

## 输入域

| 项 | 约束 |
| --- | --- |
| 数组长度 | 每侧 1..20000 项 |
| 指纹 | 1..32 个 ASCII 可打印字符（`0x20`..`0x7E`，空格至 `~`） |
| 大小写 | 按字节精确比较，区分大小写，不做任何规范化 |
| 重复 | 每个指纹值在每侧至多出现 4 次 |

不满足约束的请求返回 `422`。

## 用 Docker Compose 启动

```bash
docker compose up --build
```

服务通过环境变量 `API_PORT` 暴露（未设置时默认 `8000`）：

```bash
API_PORT=9000 docker compose up --build
# API: http://localhost:9000
```

栈包含三个服务：

- `db`：PostgreSQL 16，扫描数组、锚点、结果全部落库（命名卷 `pgdata`），
  容器重启后任务可继续查询与重算；
- `api`：FastAPI 应用，容器内**非 root 用户**运行；
- `verify`：pytest 验收服务，对运行中的真实 API 与真实 PostgreSQL 执行
  `tests/`，包含小规模穷举交叉核对、重复指纹、首尾锚点、受约束的较短最优解、
  清空锚点恢复全局最优、最大输入 3 秒性能、落库核验，以及同步屏障交错的
  并发替换裁决、超时重放幂等验收。

查看验收结果：

```bash
docker compose up --build verify   # 或直接 docker compose up --build 后看 verify 日志
```

## HTTP 协议

### `POST /api/jobs` — 创建任务并计算全局最长对应

请求体：

```json
{ "left": ["A", "a", "b"], "right": ["a", "b", "A"] }
```

`201` 响应：

```json
{
  "id": "7e3..uuid",
  "left": ["A", "a", "b"],
  "right": ["a", "b", "A"],
  "anchors": [],
  "result": [
    { "left_index": 0, "right_index": 2 },
    { "left_index": 2, "right_index": 1 }
  ],
  "length": 2,
  "version": 0,
  "updated_at": "2026-09-23T12:00:00.000000+00:00"
}
```

> `A(0)→A(2)` 与 `a(1)→a(0)` 交叉不能同取；两条长度 2 的最优解中，
> 首对字典序更小的 `(0,2)` 胜出。

### `GET /api/jobs/{id}` — 查询任务（含当前锚点与结果）

不存在返回 `404`。响应始终包含当前 `version` 与 `updated_at`（UTC），
供后续替换请求携带 `base_version`。

### `PUT /api/jobs/{id}/anchors` — 用新锚点集合替换旧集合并重算

请求体（空数组 = 清空锚点；`base_version` 可选）：

```json
{ "anchors": [[2, 1], [4, 3]], "base_version": 2 }
```

返回与 `GET` 相同的完整任务表示（含新 `version` 与 `updated_at`），其中
`anchors` 为规范化（按两侧索引升序）后的集合，`result` 包含全部锚点。

锚点合法条件（与结果条件一致）：

1. 每个锚点是 `[left_index, right_index]` 整数对，索引在各自数组范围内；
2. 该位置两侧指纹相等；
3. 全部锚点在左索引、右索引上都严格递增（不得共用索引或交叉）。

任一不满足返回 `422`（响应体 `{"detail": "原因"}`），**锚点与结果均保持
调用前状态**。锚点可以排除掉全局最优解，使含锚点的最优比对数更短。

### 并发裁决与幂等

- `version` 创建时为 `0`，每次**成功**替换锚点加一；`updated_at` 记录最近
  一次成功修改（创建或替换）的 UTC 时间。任何 `404/422/409` 都不改变
  锚点、结果、`version` 或 `updated_at`。
- `base_version` 表示本次编辑所基于的作业版本（通常取自最近一次 `GET`）。
  与当前版本不一致时返回 `409`（`{"detail": "原因"}`）并拒绝执行；
  缺省时不做该检查，顺序调用的旧客户端行为不变。
- 即使请求读到的版本与 `base_version` 一致，落库仍通过
  `UPDATE ... WHERE version = <读取版本>` 做原子裁决：两名修复师基于同一
  旧状态并发替换时，数据库行锁串行化提交，**恰好一个**请求成功，另一个
  返回 `409`，先前已确认的锚点与重算结果不会被静默覆盖。
- `Idempotency-Key` 请求头标识同一逻辑请求。首次成功响应与状态变更在同一
  事务内持久化；超时或断线后用**同一键、同一请求体**重放，直接返回首次
  裁决结果，不重复执行、不推进版本。若首次应用后状态已被其他请求推进，
  重放返回 `409` 并说明已应用的版本；同一键搭配不同请求体同样返回 `409`。
- 成功响应与随后读取到的持久状态逐项一致；重复 `GET` 结果相同。

### `GET /health`

返回 `{"status":"ok"}`。

## 算法

普通的 O(n·m) 动态规划无法处理 20000×20000。本实现把问题归约为二维点列上的
最长链（LIS）：

1. 收集所有指纹相等的索引对（因每个指纹每侧至多 4 次，总数 K ≤ 80000）；
2. 用 Fenwick 树按左索引分组、在右索引上维护后缀最大值求每点链长，
   整体 O(K log K)；
3. 重建阶段按链长分桶，每步取“晚于当前游标”的字典序首个点，得到
   最长解中唯一的字典序最小序列；
4. 锚点沿两条轴把索引平面切成互不相交的矩形段，逐段独立求解后以锚点连接。

最大规模实测远低于 3 秒预算。

## 本地开发（无 Docker）

```bash
python3.12 -m venv .venv && . .venv/bin/activate
pip install -r requirements.txt
export DATABASE_URL=postgresql+psycopg://lcs:lcs@localhost:5432/lcs
uvicorn app.main:app --port 8000
pip install -r requirements-test.txt
API_BASE_URL=http://localhost:8000 PG_DSN=postgresql://lcs:lcs@localhost:5432/lcs pytest -v
```

`scripts/brute.py` 是 O(n²) 穷举参考实现，仅供验收交叉核对，不参与服务运行。
