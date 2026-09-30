"""输入域校验：纯函数，不依赖 Web 框架，方便单测直接调用。"""

from __future__ import annotations

from collections import Counter
from typing import Sequence

MIN_LEN = 1
MAX_LEN = 20_000
MIN_FP_LEN = 1
MAX_FP_LEN = 32
MAX_REPEAT = 4


class InvalidInput(ValueError):
    """请求体不满足输入域约束。"""


def validate_side(name: str, side: Sequence[str]) -> None:
    if not isinstance(side, list):
        raise InvalidInput(f"{name} 必须是 JSON 字符串数组")
    n = len(side)
    if not MIN_LEN <= n <= MAX_LEN:
        raise InvalidInput(
            f"{name} 长度必须在 [{MIN_LEN}, {MAX_LEN}] 之间，实际为 {n}"
        )
    for pos, fp in enumerate(side):
        if not isinstance(fp, str):
            raise InvalidInput(f"{name}[{pos}] 不是字符串")
        try:
            raw = fp.encode("ascii")
        except UnicodeEncodeError:
            raise InvalidInput(
                f"{name}[{pos}] 含非 ASCII 字符，指纹仅限 ASCII 可打印字符"
            ) from None
        if not MIN_FP_LEN <= len(raw) <= MAX_FP_LEN:
            raise InvalidInput(
                f"{name}[{pos}] 字节长度 {len(raw)} 不在 [{MIN_FP_LEN}, {MAX_FP_LEN}] 内"
            )
        # 可打印 ASCII：0x20..0x7E（空格至 ~），禁止换行等控制字符。
        if any(not (0x20 <= b <= 0x7E) for b in raw):
            raise InvalidInput(f"{name}[{pos}] 含非可打印字符")
    counts = Counter(side)
    duplicated = [fp for fp, cnt in counts.items() if cnt > MAX_REPEAT]
    if duplicated:
        raise InvalidInput(
            f"{name} 中每个指纹至多出现 {MAX_REPEAT} 次，"
            f"超限值例如 {duplicated[0]!r} 出现 {counts[duplicated[0]]} 次"
        )
