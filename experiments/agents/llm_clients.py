"""统一 LLM 客户端：DeepSeek-V3 / GPT-4o / Anthropic（可选）。

Phase 1 prompt-only 阶段用。Phase 2 co-training 直接走 transformers + PEFT，不走这层。

设计：
- 同一接口 caller(system, payload) -> dict（已 json.loads 的 message）
- 本地文件缓存：相同 (system_hash, payload_hash) 命中 → 不重复调 API（省钱）
- 指数退避重试（network / rate limit error）
- JSON 解析容错：去 ```json ``` markdown 包裹、自动 truncate 尾部 garbage
- API key 通过 .env 读取（.env 已 gitignored）

实现状态：DeepSeek ✅；GPT-4o + Claude 待用户提供 key 后实现
"""
from __future__ import annotations

import hashlib
import json
import os
import re
import time
from functools import wraps
from pathlib import Path
from typing import Any, Callable

# ---------------------------------------------------------------------------
# .env 加载（轻量实现，避免再引一个 python-dotenv 依赖）
# ---------------------------------------------------------------------------

_ENV_PATHS = (
    Path(__file__).parent.parent.parent / ".env",     # paper&experiment/.env
    Path(__file__).parent.parent / ".env",            # experiments/.env (备用)
)


def _load_env() -> None:
    for env_path in _ENV_PATHS:
        if not env_path.exists():
            continue
        for line in env_path.read_text().splitlines():
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            if "=" not in line:
                continue
            key, _, value = line.partition("=")
            key = key.strip()
            value = value.strip().strip('"').strip("'")
            # 已存在的环境变量优先（shell export 覆盖 .env）
            os.environ.setdefault(key, value)


_load_env()


# ---------------------------------------------------------------------------
# 缓存层
# ---------------------------------------------------------------------------

CACHE_DIR = Path(__file__).parent / ".cache"


def _cache_key(system: str, payload: dict[str, Any]) -> str:
    blob = system + "\n---\n" + json.dumps(payload, sort_keys=True, ensure_ascii=False)
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()[:24]


def cached(fn: Callable[[str, dict[str, Any]], dict[str, Any]]) -> Callable[[str, dict[str, Any]], dict[str, Any]]:
    """本地 JSON 缓存装饰器。设 LLM_NO_CACHE=1 关闭。"""

    @wraps(fn)
    def wrapper(system: str, payload: dict[str, Any]) -> dict[str, Any]:
        if os.environ.get("LLM_NO_CACHE") == "1":
            return fn(system, payload)
        CACHE_DIR.mkdir(parents=True, exist_ok=True)
        cache_path = CACHE_DIR / f"{fn.__name__}_{_cache_key(system, payload)}.json"
        if cache_path.exists():
            return json.loads(cache_path.read_text())
        result = fn(system, payload)
        cache_path.write_text(json.dumps(result, ensure_ascii=False, indent=2))
        return result

    return wrapper


# ---------------------------------------------------------------------------
# JSON 解析容错
# ---------------------------------------------------------------------------


_JSON_FENCE = re.compile(r"```(?:json)?\s*(\{.*?\})\s*```", re.DOTALL)
_BARE_OBJECT = re.compile(r"(\{.*\})", re.DOTALL)


def _extract_json_object(text: str) -> dict[str, Any]:
    """从 LLM 响应里抠出第一个 JSON 对象。处理：
    1. ```json {...} ``` 围栏
    2. 自由文本前后缀 + 一个对象
    3. 已经是干净 JSON
    """
    text = text.strip()
    # 优先匹配围栏
    match = _JSON_FENCE.search(text)
    if match:
        return json.loads(match.group(1))
    # 直接 parse
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass
    # 抠第一个 {...}
    match = _BARE_OBJECT.search(text)
    if match:
        return json.loads(match.group(1))
    raise ValueError(f"无法解析 JSON：{text[:200]!r}…")


def _abstain(reason: str) -> dict[str, Any]:
    """协议层兜底：调用失败时返回 abstain 消息。"""
    return {
        "message_type": "abstain",
        "agent": "unknown",
        "round": 0,
        "rationale": reason,
    }


# ---------------------------------------------------------------------------
# 重试装饰器
# ---------------------------------------------------------------------------


def retry_with_backoff(max_attempts: int = 5, base_delay: float = 1.0) -> Callable:
    """指数退避重试装饰器。"""

    def decorator(fn: Callable[..., dict[str, Any]]) -> Callable[..., dict[str, Any]]:
        @wraps(fn)
        def wrapper(*args: Any, **kwargs: Any) -> dict[str, Any]:
            last_err: Exception | None = None
            for attempt in range(max_attempts):
                try:
                    return fn(*args, **kwargs)
                except Exception as err:  # noqa: BLE001
                    last_err = err
                    if attempt == max_attempts - 1:
                        break
                    delay = base_delay * (2**attempt)
                    print(f"  ⚠️  {fn.__name__} 失败 ({type(err).__name__}: {err})，{delay:.1f}s 后重试")
                    time.sleep(delay)
            return _abstain(f"{fn.__name__} 重试 {max_attempts} 次失败：{last_err}")
        return wrapper
    return decorator


# ---------------------------------------------------------------------------
# DeepSeek-V3
# ---------------------------------------------------------------------------


def _format_messages_for_chat(system: str, payload: dict[str, Any]) -> list[dict[str, str]]:
    """三 agent persona system + payload 拼成 chat messages。"""
    user_content = (
        "Region 与可见特征数据（JSON）：\n"
        f"```json\n{json.dumps(payload, ensure_ascii=False, indent=2)}\n```\n"
        "\n请按 persona 规则输出 typed JSON message，只输出 JSON 不要前后缀。"
    )
    return [
        {"role": "system", "content": system},
        {"role": "user", "content": user_content},
    ]


@cached
@retry_with_backoff(max_attempts=4, base_delay=1.5)
def call_deepseek(system: str, payload: dict[str, Any]) -> dict[str, Any]:
    """DeepSeek-V3 chat completion → JSON dict.

    需要 DEEPSEEK_API_KEY 环境变量（.env 加载）。
    """
    try:
        from openai import OpenAI
    except ImportError as exc:
        raise RuntimeError("缺 openai 包：pip install openai>=1.30") from exc

    api_key = os.environ.get("DEEPSEEK_API_KEY")
    if not api_key:
        raise RuntimeError("DEEPSEEK_API_KEY 未设置；检查 paper&experiment/.env")

    base_url = os.environ.get("DEEPSEEK_BASE_URL", "https://api.deepseek.com")
    model = os.environ.get("DEEPSEEK_MODEL", "deepseek-chat")

    client = OpenAI(api_key=api_key, base_url=base_url)
    response = client.chat.completions.create(
        model=model,
        messages=_format_messages_for_chat(system, payload),
        response_format={"type": "json_object"},
        temperature=0.3,  # 协作判定不需要太创意
        max_tokens=1024,
    )
    raw = response.choices[0].message.content or ""
    return _extract_json_object(raw)


# ---------------------------------------------------------------------------
# OpenAI / GPT-4o
# ---------------------------------------------------------------------------


@cached
@retry_with_backoff(max_attempts=4, base_delay=1.5)
def call_gpt4o(system: str, payload: dict[str, Any]) -> dict[str, Any]:
    try:
        from openai import OpenAI
    except ImportError as exc:
        raise RuntimeError("缺 openai 包：pip install openai>=1.30") from exc

    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError("OPENAI_API_KEY 未设置；Phase 1 cross-backbone 实验前用户需提供")

    model = os.environ.get("OPENAI_MODEL", "gpt-4o")
    client = OpenAI(api_key=api_key)
    response = client.chat.completions.create(
        model=model,
        messages=_format_messages_for_chat(system, payload),
        response_format={"type": "json_object"},
        temperature=0.3,
        max_tokens=1024,
    )
    raw = response.choices[0].message.content or ""
    return _extract_json_object(raw)


# ---------------------------------------------------------------------------
# Anthropic Claude（可选第三家）
# ---------------------------------------------------------------------------


@cached
@retry_with_backoff(max_attempts=4, base_delay=1.5)
def call_claude(system: str, payload: dict[str, Any]) -> dict[str, Any]:
    try:
        import anthropic
    except ImportError as exc:
        raise RuntimeError("缺 anthropic 包：pip install anthropic>=0.30") from exc

    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        raise RuntimeError("ANTHROPIC_API_KEY 未设置；可选实验用")

    model = os.environ.get("ANTHROPIC_MODEL", "claude-sonnet-4-6")
    client = anthropic.Anthropic(api_key=api_key)
    user_content = (
        "Region 与可见特征数据（JSON）：\n"
        f"```json\n{json.dumps(payload, ensure_ascii=False, indent=2)}\n```\n"
        "请按 persona 规则输出 typed JSON message，只输出 JSON 不要前后缀。"
    )
    response = client.messages.create(
        model=model,
        max_tokens=1024,
        system=system,
        messages=[{"role": "user", "content": user_content}],
        temperature=0.3,
    )
    raw = "".join(block.text for block in response.content if hasattr(block, "text"))
    return _extract_json_object(raw)


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------

REGISTRY: dict[str, Callable[[str, dict[str, Any]], dict[str, Any]]] = {
    "deepseek-v3": call_deepseek,
    "deepseek": call_deepseek,
    "gpt-4o": call_gpt4o,
    "gpt": call_gpt4o,
    "claude": call_claude,
}


def get_caller(name: str) -> Callable[[str, dict[str, Any]], dict[str, Any]]:
    if name not in REGISTRY:
        raise ValueError(f"unknown LLM: {name}; available: {sorted(set(REGISTRY))}")
    return REGISTRY[name]


__all__ = ["get_caller", "call_deepseek", "call_gpt4o", "call_claude", "CACHE_DIR"]
