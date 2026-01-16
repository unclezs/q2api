import os
import json
import traceback
import uuid
import time
import asyncio
import importlib.util
import random
import secrets
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, Optional, List, Any, AsyncGenerator, Tuple

from fastapi import FastAPI, Depends, HTTPException, Header, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, StreamingResponse, HTMLResponse, FileResponse
from pydantic import BaseModel
from dotenv import load_dotenv
import httpx
import tiktoken

from db import init_db, close_db, row_to_dict

# ------------------------------------------------------------------------------
# Tokenizer
# ------------------------------------------------------------------------------

try:
    # cl100k_base is used by gpt-4, gpt-3.5-turbo, text-embedding-ada-002
    ENCODING = tiktoken.get_encoding("cl100k_base")
except Exception:
    ENCODING = None

def count_tokens(text: str, apply_multiplier: bool = False) -> int:
    """Counts tokens with tiktoken."""
    if not text or not ENCODING:
        return 0
    token_count = len(ENCODING.encode(text))
    if apply_multiplier:
        token_count = int(token_count * TOKEN_COUNT_MULTIPLIER)
    return token_count

# ------------------------------------------------------------------------------
# Bootstrap
# ------------------------------------------------------------------------------

BASE_DIR = Path(__file__).resolve().parent

load_dotenv(BASE_DIR / ".env")

app = FastAPI(title="v2 OpenAI-compatible Server (Amazon Q Backend)")

# CORS for simple testing in browser
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ------------------------------------------------------------------------------
# Dynamic import of replicate.py to avoid package __init__ needs
# ------------------------------------------------------------------------------

def _load_replicate_module():
    mod_path = BASE_DIR / "replicate.py"
    spec = importlib.util.spec_from_file_location("v2_replicate", str(mod_path))
    module = importlib.util.module_from_spec(spec)  # type: ignore[arg-type]
    assert spec is not None and spec.loader is not None
    spec.loader.exec_module(module)  # type: ignore[attr-defined]
    return module

_replicate = _load_replicate_module()
send_chat_request = _replicate.send_chat_request

# ------------------------------------------------------------------------------
# Dynamic import of Claude modules
# ------------------------------------------------------------------------------

def _load_claude_modules():
    # claude_types
    spec_types = importlib.util.spec_from_file_location("v2_claude_types", str(BASE_DIR / "claude_types.py"))
    mod_types = importlib.util.module_from_spec(spec_types)
    spec_types.loader.exec_module(mod_types)
    
    # claude_converter
    spec_conv = importlib.util.spec_from_file_location("v2_claude_converter", str(BASE_DIR / "claude_converter.py"))
    mod_conv = importlib.util.module_from_spec(spec_conv)
    # We need to inject claude_types into converter's namespace if it uses relative imports or expects them
    # But since we used relative import in claude_converter.py (.claude_types), we need to be careful.
    # Actually, since we are loading dynamically, relative imports might fail if not in sys.modules correctly.
    # Let's patch sys.modules temporarily or just rely on file location.
    # A simpler way for this single-file script style is to just load them.
    # However, claude_converter does `from .claude_types import ...`
    # To make that work, we should probably just use standard import if v2 is a package,
    # but v2 is just a folder.
    # Let's assume the user runs this with v2 in pythonpath or we just fix imports in the files.
    # But I wrote `from .claude_types` in the file.
    # Let's try to load it. If it fails, we might need to adjust.
    # Actually, for simplicity in this `app.py` dynamic loading context,
    # it is better if `claude_converter.py` used absolute import or we mock the package.
    # BUT, let's try to just load them and see.
    # To avoid relative import issues, I will inject the module into sys.modules
    import sys
    sys.modules["v2.claude_types"] = mod_types
    
    spec_conv.loader.exec_module(mod_conv)
    
    # claude_stream
    spec_stream = importlib.util.spec_from_file_location("v2_claude_stream", str(BASE_DIR / "claude_stream.py"))
    mod_stream = importlib.util.module_from_spec(spec_stream)
    spec_stream.loader.exec_module(mod_stream)
    
    return mod_types, mod_conv, mod_stream

try:
    _claude_types, _claude_converter, _claude_stream = _load_claude_modules()
    ClaudeRequest = _claude_types.ClaudeRequest
    convert_claude_to_amazonq_request = _claude_converter.convert_claude_to_amazonq_request
    map_model_name = _claude_converter.map_model_name
    ClaudeStreamHandler = _claude_stream.ClaudeStreamHandler
except Exception as e:
    print(f"Failed to load Claude modules: {e}")
    traceback.print_exc()
    # Define dummy classes to avoid NameError on startup if loading fails
    class ClaudeRequest(BaseModel):
        pass
    convert_claude_to_amazonq_request = None
    map_model_name = lambda m: m  # Pass through if module fails to load
    ClaudeStreamHandler = None

# ------------------------------------------------------------------------------
# Global HTTP Client
# ------------------------------------------------------------------------------

GLOBAL_CLIENT: Optional[httpx.AsyncClient] = None

def _get_proxies() -> Optional[Dict[str, str]]:
    proxy = os.getenv("HTTP_PROXY", "").strip()
    if proxy:
        return {"http": proxy, "https": proxy}
    return None

async def _init_global_client():
    global GLOBAL_CLIENT
    proxies = _get_proxies()
    mounts = None
    if proxies:
        proxy_url = proxies.get("https") or proxies.get("http")
        if proxy_url:
            mounts = {
                "https://": httpx.AsyncHTTPTransport(proxy=proxy_url),
                "http://": httpx.AsyncHTTPTransport(proxy=proxy_url),
            }
    # Increased limits for high concurrency with streaming
    # max_connections: 总连接数上限
    # max_keepalive_connections: 保持活跃的连接数
    # keepalive_expiry: 连接保持时间
    limits = httpx.Limits(
        max_keepalive_connections=60,
        max_connections=60,  # 提高到500以支持更高并发
        keepalive_expiry=30.0  # 30秒后释放空闲连接
    )
    # 为流式响应设置更长的超时
    timeout = httpx.Timeout(
        connect=30.0,  # 连接超时（包括 TLS 握手）
        read=300.0,    # 读取超时(流式响应需要更长时间)
        write=30.0,    # 写入超时
        pool=10.0      # 从连接池获取连接的超时时间
    )
    GLOBAL_CLIENT = httpx.AsyncClient(mounts=mounts, timeout=timeout, limits=limits)

async def _close_global_client():
    global GLOBAL_CLIENT
    if GLOBAL_CLIENT:
        await GLOBAL_CLIENT.aclose()
        GLOBAL_CLIENT = None

# ------------------------------------------------------------------------------
# Database helpers
# ------------------------------------------------------------------------------

# Database backend instance (initialized on startup)
_db = None

async def _ensure_db():
    """Initialize database backend."""
    global _db
    _db = await init_db()

def _row_to_dict(r: Dict[str, Any]) -> Dict[str, Any]:
    """Convert database row to dict with JSON parsing."""
    return row_to_dict(r)

# _ensure_db() will be called in startup event

# ------------------------------------------------------------------------------
# Background token refresh thread
# ------------------------------------------------------------------------------

async def _refresh_stale_tokens():
    while True:
        try:
            await asyncio.sleep(300)  # 5 minutes
            if _db is None:
                print("[Error] Database not initialized, skipping token refresh cycle.")
                continue
            now = time.time()
            
            if LAZY_ACCOUNT_POOL_ENABLED:
                limit = LAZY_ACCOUNT_POOL_SIZE + LAZY_ACCOUNT_POOL_REFRESH_OFFSET
                order_direction = "DESC" if LAZY_ACCOUNT_POOL_ORDER_DESC else "ASC"
                query = f"SELECT id, last_refresh_time FROM accounts WHERE enabled=1 ORDER BY {LAZY_ACCOUNT_POOL_ORDER_BY} {order_direction} LIMIT {limit}"
                rows = await _db.fetchall(query)
            else:
                rows = await _db.fetchall("SELECT id, last_refresh_time FROM accounts WHERE enabled=1")

            for row in rows:
                acc_id, last_refresh = row['id'], row['last_refresh_time']
                should_refresh = False
                if not last_refresh or last_refresh == "never":
                    should_refresh = True
                else:
                    try:
                        last_time = time.mktime(time.strptime(last_refresh, "%Y-%m-%dT%H:%M:%S"))
                        if now - last_time > 1500:  # 25 minutes
                            should_refresh = True
                    except Exception:
                        # Malformed or unparsable timestamp; force refresh
                        should_refresh = True

                if should_refresh:
                    try:
                        await refresh_access_token_in_db(acc_id)
                    except Exception:
                        traceback.print_exc()
                        # Ignore per-account refresh failure; timestamp/status are recorded inside
                        pass
        except Exception:
            traceback.print_exc()
            pass

# ------------------------------------------------------------------------------
# Env and API Key authorization (keys are independent of AWS accounts)
# ------------------------------------------------------------------------------
def _parse_allowed_keys_env() -> List[str]:
    """
    OPENAI_KEYS is a comma-separated whitelist of API keys for authorization only.
    Example: OPENAI_KEYS="key1,key2,key3"
    - When the list is non-empty, incoming Authorization: Bearer {key} must be one of them.
    - When empty or unset, authorization is effectively disabled (dev mode).
    """
    s = os.getenv("OPENAI_KEYS", "") or ""
    keys: List[str] = []
    for k in [x.strip() for x in s.split(",") if x.strip()]:
        keys.append(k)
    return keys

ALLOWED_API_KEYS: List[str] = _parse_allowed_keys_env()
MAX_ERROR_COUNT: int = int(os.getenv("MAX_ERROR_COUNT", "100"))
MAX_RETRY_COUNT: int = int(os.getenv("MAX_RETRY_COUNT", "3"))
TOKEN_COUNT_MULTIPLIER: float = float(os.getenv("TOKEN_COUNT_MULTIPLIER", "1.0"))

# Lazy Account Pool settings
LAZY_ACCOUNT_POOL_ENABLED: bool = os.getenv("LAZY_ACCOUNT_POOL_ENABLED", "false").lower() in ("true", "1", "yes")
LAZY_ACCOUNT_POOL_SIZE: int = int(os.getenv("LAZY_ACCOUNT_POOL_SIZE", "20"))
LAZY_ACCOUNT_POOL_REFRESH_OFFSET: int = int(os.getenv("LAZY_ACCOUNT_POOL_REFRESH_OFFSET", "10"))
LAZY_ACCOUNT_POOL_ORDER_BY: str = os.getenv("LAZY_ACCOUNT_POOL_ORDER_BY", "created_at")
LAZY_ACCOUNT_POOL_ORDER_DESC: bool = os.getenv("LAZY_ACCOUNT_POOL_ORDER_DESC", "false").lower() in ("true", "1", "yes")

# Validate LAZY_ACCOUNT_POOL_ORDER_BY to prevent SQL injection
if LAZY_ACCOUNT_POOL_ORDER_BY not in ["created_at", "id", "success_count"]:
    LAZY_ACCOUNT_POOL_ORDER_BY = "created_at"

def _is_console_enabled() -> bool:
    """检查是否启用管理控制台"""
    console_env = os.getenv("ENABLE_CONSOLE", "true").strip().lower()
    return console_env not in ("false", "0", "no", "disabled")

CONSOLE_ENABLED: bool = _is_console_enabled()

# Admin authentication configuration
ADMIN_PASSWORD: str = os.getenv("ADMIN_PASSWORD", "admin")
LOGIN_MAX_ATTEMPTS: int = 5
LOGIN_LOCKOUT_SECONDS: int = 3600  # 1 hour
_login_failures: Dict[str, Dict] = {}  # {ip: {"count": int, "locked_until": float}}

def _extract_bearer(token_header: Optional[str]) -> Optional[str]:
    if not token_header:
        return None
    if token_header.startswith("Bearer "):
        return token_header.split(" ", 1)[1].strip()
    return token_header.strip()

async def _list_enabled_accounts(limit: Optional[int] = None) -> List[Dict[str, Any]]:
    if LAZY_ACCOUNT_POOL_ENABLED:
        order_direction = "DESC" if LAZY_ACCOUNT_POOL_ORDER_DESC else "ASC"
        query = f"SELECT * FROM accounts WHERE enabled=1 ORDER BY {LAZY_ACCOUNT_POOL_ORDER_BY} {order_direction}"
        if limit:
            query += f" LIMIT {limit}"
        rows = await _db.fetchall(query)
    else:
        query = "SELECT * FROM accounts WHERE enabled=1 ORDER BY created_at DESC"
        if limit:
            query += f" LIMIT {limit}"
        rows = await _db.fetchall(query)
    return [_row_to_dict(r) for r in rows]

async def _list_disabled_accounts() -> List[Dict[str, Any]]:
    rows = await _db.fetchall("SELECT * FROM accounts WHERE enabled=0 ORDER BY created_at DESC")
    return [_row_to_dict(r) for r in rows]

async def verify_account(account: Dict[str, Any]) -> Tuple[bool, Optional[str]]:
    """验证账号可用性"""
    try:
        account = await refresh_access_token_in_db(account['id'])
        test_request = {
            "conversationState": {
                "currentMessage": {"userInputMessage": {"content": "hello"}},
                "chatTriggerType": "MANUAL"
            }
        }
        _, _, tracker, event_gen = await send_chat_request(
            access_token=account['accessToken'],
            messages=[],
            stream=True,
            raw_payload=test_request
        )
        if event_gen:
            async for _ in event_gen:
                break
        return True, None
    except Exception as e:
        if "AccessDenied" in str(e) or "403" in str(e):
            return False, "AccessDenied"
        return False, None

async def resolve_account_for_key(bearer_key: Optional[str], exclude_ids: Optional[List[str]] = None) -> Dict[str, Any]:
    """
    Authorize request by OPENAI_KEYS (if configured), then select an AWS account.
    Selection strategy: weighted random based on error rate (lower error rate = higher probability).
    """
    # Authorization: allow admin password to bypass OPENAI_KEYS check (for console testing)
    is_admin = bearer_key and bearer_key == ADMIN_PASSWORD
    if ALLOWED_API_KEYS and not is_admin:
        if not bearer_key or bearer_key not in ALLOWED_API_KEYS:
            raise HTTPException(status_code=401, detail="Invalid or missing API key")

    # Get candidate accounts
    if LAZY_ACCOUNT_POOL_ENABLED:
        candidates = await _list_enabled_accounts(limit=LAZY_ACCOUNT_POOL_SIZE)
    else:
        candidates = await _list_enabled_accounts()

    # Exclude specified accounts (for retry)
    if exclude_ids:
        candidates = [acc for acc in candidates if acc.get("id") not in exclude_ids]

    if not candidates:
        raise HTTPException(status_code=401, detail="No enabled account available")

    # Weighted random selection: lower error rate = higher weight
    def get_weight(acc):
        total = acc.get("success_count", 0) + acc.get("error_count", 0)
        if total == 0:
            return 0.5  # 新账号中等权重
        error_rate = acc.get("error_count", 0) / total
        return max(0.1, 1 - error_rate)  # 最低0.1，保证都有机会

    weights = [get_weight(acc) for acc in candidates]
    return random.choices(candidates, weights=weights, k=1)[0]

# ------------------------------------------------------------------------------
# Pydantic Schemas
# ------------------------------------------------------------------------------

class AccountCreate(BaseModel):
    label: Optional[str] = None
    clientId: str
    clientSecret: str
    refreshToken: Optional[str] = None
    accessToken: Optional[str] = None
    other: Optional[Dict[str, Any]] = None
    enabled: Optional[bool] = True

class BatchAccountCreate(BaseModel):
    accounts: List[AccountCreate]

class AccountUpdate(BaseModel):
    label: Optional[str] = None
    clientId: Optional[str] = None
    clientSecret: Optional[str] = None
    refreshToken: Optional[str] = None
    accessToken: Optional[str] = None
    other: Optional[Dict[str, Any]] = None
    enabled: Optional[bool] = None

class ChatMessage(BaseModel):
    role: str
    content: Any
    tool_calls: Optional[List[Dict[str, Any]]] = None
    tool_call_id: Optional[str] = None

class ChatCompletionRequest(BaseModel):
    model: Optional[str] = None
    messages: List[ChatMessage]
    stream: Optional[bool] = False
    max_tokens: Optional[int] = None
    temperature: Optional[float] = None
    top_p: Optional[float] = None
    tools: Optional[List[Dict[str, Any]]] = None
    tool_choice: Optional[Any] = None
    user: Optional[str] = None

# ------------------------------------------------------------------------------
# Token refresh (OIDC)
# ------------------------------------------------------------------------------

OIDC_BASE = "https://oidc.us-east-1.amazonaws.com"
TOKEN_URL = f"{OIDC_BASE}/token"

def _oidc_headers() -> Dict[str, str]:
    return {
        "content-type": "application/json",
        "user-agent": "aws-sdk-rust/1.3.9 os/windows lang/rust/1.87.0",
        "x-amz-user-agent": "aws-sdk-rust/1.3.9 ua/2.1 api/ssooidc/1.88.0 os/windows lang/rust/1.87.0 m/E app/AmazonQ-For-CLI",
        "amz-sdk-request": "attempt=1; max=3",
        "amz-sdk-invocation-id": str(uuid.uuid4()),
    }

async def refresh_access_token_in_db(account_id: str) -> Dict[str, Any]:
    row = await _db.fetchone("SELECT * FROM accounts WHERE id=?", (account_id,))
    if not row:
        raise HTTPException(status_code=404, detail="Account not found")
    acc = _row_to_dict(row)

    if not acc.get("clientId") or not acc.get("clientSecret") or not acc.get("refreshToken"):
        raise HTTPException(status_code=400, detail="Account missing clientId/clientSecret/refreshToken for refresh")

    payload = {
        "grantType": "refresh_token",
        "clientId": acc["clientId"],
        "clientSecret": acc["clientSecret"],
        "refreshToken": acc["refreshToken"],
    }

    try:
        # Use global client if available, else fallback (though global should be ready)
        client = GLOBAL_CLIENT
        if not client:
            # Fallback for safety
            async with httpx.AsyncClient(timeout=60.0) as temp_client:
                r = await temp_client.post(TOKEN_URL, headers=_oidc_headers(), json=payload)
                r.raise_for_status()
                data = r.json()
        else:
            r = await client.post(TOKEN_URL, headers=_oidc_headers(), json=payload)
            r.raise_for_status()
            data = r.json()

        new_access = data.get("accessToken")
        new_refresh = data.get("refreshToken", acc.get("refreshToken"))
        now = time.strftime("%Y-%m-%dT%H:%M:%S", time.gmtime())
        status = "success"
    except httpx.HTTPError as e:
        now = time.strftime("%Y-%m-%dT%H:%M:%S", time.gmtime())
        status = "failed"
        await _db.execute(
            """
            UPDATE accounts
            SET last_refresh_time=?, last_refresh_status=?, updated_at=?
            WHERE id=?
            """,
            (now, status, now, account_id),
        )
        # 记录刷新失败次数
        await _update_stats(account_id, False)
        raise HTTPException(status_code=502, detail=f"Token refresh failed: {str(e)}")
    except Exception as e:
        # Ensure last_refresh_time is recorded even on unexpected errors
        now = time.strftime("%Y-%m-%dT%H:%M:%S", time.gmtime())
        status = "failed"
        await _db.execute(
            """
            UPDATE accounts
            SET last_refresh_time=?, last_refresh_status=?, updated_at=?
            WHERE id=?
            """,
            (now, status, now, account_id),
        )
        # 记录刷新失败次数
        await _update_stats(account_id, False)
        raise

    await _db.execute(
        """
        UPDATE accounts
        SET accessToken=?, refreshToken=?, last_refresh_time=?, last_refresh_status=?, updated_at=?
        WHERE id=?
        """,
        (new_access, new_refresh, now, status, now, account_id),
    )

    row2 = await _db.fetchone("SELECT * FROM accounts WHERE id=?", (account_id,))
    return _row_to_dict(row2)

async def get_account(account_id: str) -> Dict[str, Any]:
    row = await _db.fetchone("SELECT * FROM accounts WHERE id=?", (account_id,))
    if not row:
        raise HTTPException(status_code=404, detail="Account not found")
    return _row_to_dict(row)

async def _update_stats(account_id: str, success: bool) -> None:
    if success:
        await _db.execute("UPDATE accounts SET success_count=success_count+1, error_count=0, updated_at=? WHERE id=?",
                    (time.strftime("%Y-%m-%dT%H:%M:%S", time.gmtime()), account_id))
    else:
        row = await _db.fetchone("SELECT error_count FROM accounts WHERE id=?", (account_id,))
        if row:
            new_count = (row['error_count'] or 0) + 1
            if new_count >= MAX_ERROR_COUNT:
                await _db.execute("UPDATE accounts SET error_count=?, enabled=0, updated_at=? WHERE id=?",
                           (new_count, time.strftime("%Y-%m-%dT%H:%M:%S", time.gmtime()), account_id))
            else:
                await _db.execute("UPDATE accounts SET error_count=?, updated_at=? WHERE id=?",
                           (new_count, time.strftime("%Y-%m-%dT%H:%M:%S", time.gmtime()), account_id))

# ------------------------------------------------------------------------------
# Dependencies
# ------------------------------------------------------------------------------

async def require_account(
    authorization: Optional[str] = Header(default=None),
    x_api_key: Optional[str] = Header(default=None)
) -> Dict[str, Any]:
    key = _extract_bearer(authorization) if authorization else x_api_key
    return await resolve_account_for_key(key)

def verify_admin_password(authorization: Optional[str] = Header(None)) -> bool:
    """Verify admin password for console access"""
    if not authorization or not authorization.startswith("Bearer "):
        raise HTTPException(
            status_code=401,
            detail={"error": "Unauthorized access", "code": "UNAUTHORIZED"}
        )

    password = authorization[7:]  # Remove "Bearer " prefix

    if password != ADMIN_PASSWORD:
        raise HTTPException(
            status_code=401,
            detail={"error": "Invalid password", "code": "INVALID_PASSWORD"}
        )

    return True

# ------------------------------------------------------------------------------
# OpenAI-compatible Chat endpoint
# ------------------------------------------------------------------------------

def _openai_non_streaming_response(
    text: str,
    model: Optional[str],
    prompt_tokens: int = 0,
    completion_tokens: int = 0,
) -> Dict[str, Any]:
    created = int(time.time())
    return {
        "id": f"chatcmpl-{uuid.uuid4()}",
        "object": "chat.completion",
        "created": created,
        "model": model or "unknown",
        "choices": [
            {
                "index": 0,
                "message": {
                    "role": "assistant",
                    "content": text,
                },
                "finish_reason": "stop",
            }
        ],
        "usage": {
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "total_tokens": prompt_tokens + completion_tokens,
        },
    }

def _sse_format(obj: Dict[str, Any]) -> str:
    return f"data: {json.dumps(obj, ensure_ascii=False)}\n\n"


def _trim_context_by_tokens(messages: List, trim_tokens: int = 10000) -> List:
    """
    从消息历史中截断指定数量的 tokens。
    优先从旧消息开始截断，保留最近的系统提示和用户消息。
    """
    if trim_tokens <= 0 or not messages or not ENCODING:
        return messages

    # 计算当前总 tokens
    def count_msg_tokens(msg):
        if isinstance(msg, str):
            return len(ENCODING.encode(msg))
        elif isinstance(msg, dict):
            content = msg.get("content", "")
            if isinstance(content, str):
                return len(ENCODING.encode(content))
            elif isinstance(content, list):
                tokens = 0
                for item in content:
                    if isinstance(item, dict) and item.get("type") == "text":
                        tokens += len(ENCODING.encode(item.get("text", "")))
                return tokens
        return 0

    # 优先截断 history 中的旧消息
    # aq_request["conversationState"]["history"] 是需要处理的主要对象
    total_trimmed = 0
    remaining_to_trim = trim_tokens

    # 从开头开始删除历史消息，直到达到目标截断量
    while remaining_to_trim > 0 and len(messages) > 1:
        # 跳过保留的消息数量（如最后几条重要的对话）
        keep_count = 2  # 保留最后2条消息
        if len(messages) <= keep_count:
            break

        # 删除最旧的消息
        removed = messages.pop(0)
        removed_tokens = count_msg_tokens(removed)
        remaining_to_trim -= removed_tokens
        total_trimmed += removed_tokens

    import logging
    logging.getLogger(__name__).info(f"Trimmed ~{total_trimmed} tokens from context, {len(messages)} messages remain")

    return messages


@app.post("/v1/messages")
async def claude_messages(
    req: ClaudeRequest,
    authorization: Optional[str] = Header(default=None),
    x_api_key: Optional[str] = Header(default=None),
    x_conversation_id: Optional[str] = Header(default=None, alias="x-conversation-id")
):
    """
    Claude-compatible messages endpoint with retry support.
    Includes automatic context truncation on "Input is too long" errors.
    """
    # Extract bearer key for authorization
    bearer_key = _extract_bearer(authorization) if authorization else x_api_key

    # 1. Convert request (do this once, before retry loop)
    try:
        aq_request = convert_claude_to_amazonq_request(req, conversation_id=None)
    except Exception as e:
        traceback.print_exc()
        raise HTTPException(status_code=400, detail=f"Request conversion failed: {str(e)}")

    # Post-process history to fix message ordering (prevents infinite loops)
    from message_processor import process_claude_history_for_amazonq
    conversation_state = aq_request.get("conversationState", {})
    history = conversation_state.get("history", [])
    if history:
        processed_history = process_claude_history_for_amazonq(history)
        aq_request["conversationState"]["history"] = processed_history

    # Remove duplicate tail userInputMessage that matches currentMessage content
    conversation_state = aq_request.get("conversationState", {})
    current_msg = conversation_state.get("currentMessage", {}).get("userInputMessage", {})
    current_content = (current_msg.get("content") or "").strip()
    history = conversation_state.get("history", [])

    if history and current_content:
        last = history[-1]
        if "userInputMessage" in last:
            last_content = (last["userInputMessage"].get("content") or "").strip()
            if last_content and last_content == current_content:
                history = history[:-1]
                aq_request["conversationState"]["history"] = history
                import logging
                logging.getLogger(__name__).info("Removed duplicate tail userInputMessage to prevent repeated response")

    conversation_state = aq_request.get("conversationState", {})
    conversation_id = conversation_state.get("conversationId")
    response_headers: Dict[str, str] = {}
    if conversation_id:
        response_headers["x-conversation-id"] = conversation_id

    # Calculate input tokens (do this once)
    text_to_count = ""
    if req.system:
        if isinstance(req.system, str):
            text_to_count += req.system
        elif isinstance(req.system, list):
            for item in req.system:
                if isinstance(item, dict) and item.get("type") == "text":
                    text_to_count += item.get("text", "")
    for msg in req.messages:
        if isinstance(msg.content, str):
            text_to_count += msg.content
        elif isinstance(msg.content, list):
            for item in msg.content:
                if isinstance(item, dict) and item.get("type") == "text":
                    text_to_count += item.get("text", "")
    input_tokens = count_tokens(text_to_count, apply_multiplier=True)

    # Retry loop with exponential backoff and context truncation
    tried_account_ids: List[str] = []
    last_error: Optional[Exception] = None
    max_attempts = MAX_RETRY_COUNT + 1  # +1 for initial attempt
    trim_retry_count = 0  # 跟踪上下文截断重试次数
    max_trim_retries = 3  # 最多截断重试3次
    trim_tokens_per_retry = 10000  # 每次截断10k tokens

    for attempt in range(max_attempts):
        event_iter = None
        account = None
        try:
            # Get account (excluding previously failed ones)
            # If no accounts available, reset the exclusion list and try again
            try:
                account = await resolve_account_for_key(bearer_key, exclude_ids=tried_account_ids if tried_account_ids else None)
            except HTTPException as he:
                if "No enabled account available" in str(he.detail) and tried_account_ids:
                    # All accounts tried, reset and try again
                    tried_account_ids = []
                    account = await resolve_account_for_key(bearer_key, exclude_ids=None)
                else:
                    raise
            tried_account_ids.append(account["id"])

            access = account.get("accessToken")
            if not access:
                refreshed = await refresh_access_token_in_db(account["id"])
                access = refreshed.get("accessToken")

            # Send request
            _, _, tracker, event_iter = await send_chat_request(
                access_token=access,
                messages=[],
                model=map_model_name(req.model),
                stream=True,
                client=GLOBAL_CLIENT,
                raw_payload=aq_request
            )

            if not event_iter:
                raise Exception("No event stream returned")

            # Try to get the first event to ensure the connection is valid
            first_event = None
            try:
                first_event = await event_iter.__anext__()
            except StopAsyncIteration:
                raise Exception("Empty response from upstream")

            # Success! Create handler and return response
            handler = ClaudeStreamHandler(model=req.model, input_tokens=input_tokens, conversation_id=conversation_id)

            # Capture account for closure
            current_account = account
            current_tracker = tracker

            async def event_generator():
                try:
                    if first_event:
                        event_type, payload = first_event
                        async for sse in handler.handle_event(event_type, payload):
                            yield sse
                    async for event_type, payload in event_iter:
                        async for sse in handler.handle_event(event_type, payload):
                            yield sse
                    async for sse in handler.finish():
                        yield sse
                    await _update_stats(current_account["id"], True)
                except GeneratorExit:
                    await _update_stats(current_account["id"], current_tracker.has_content if current_tracker else False)
                except Exception:
                    await _update_stats(current_account["id"], False)
                    raise

            if req.stream:
                return StreamingResponse(
                    event_generator(),
                    media_type="text/event-stream",
                    headers=response_headers or None
                )
            else:
                # Non-streaming: accumulate response
                content_blocks = []
                usage = {"input_tokens": 0, "output_tokens": 0}
                stop_reason = None
                final_content = []

                async for sse_chunk in event_generator():
                    data_str = None
                    for line in sse_chunk.strip().split('\n'):
                        if line.startswith("data:"):
                            data_str = line[6:].strip()
                            break
                    if not data_str or data_str == "[DONE]":
                        continue
                    try:
                        data = json.loads(data_str)
                        dtype = data.get("type")
                        if dtype == "content_block_start":
                            idx = data.get("index", 0)
                            while len(final_content) <= idx:
                                final_content.append(None)
                            final_content[idx] = data.get("content_block")
                        elif dtype == "content_block_delta":
                            idx = data.get("index", 0)
                            delta = data.get("delta", {})
                            if final_content[idx]:
                                if delta.get("type") == "text_delta":
                                    final_content[idx]["text"] += delta.get("text", "")
                                elif delta.get("type") == "thinking_delta":
                                    final_content[idx].setdefault("thinking", "")
                                    final_content[idx]["thinking"] += delta.get("thinking", "")
                                elif delta.get("type") == "input_json_delta":
                                    if "partial_json" not in final_content[idx]:
                                        final_content[idx]["partial_json"] = ""
                                    final_content[idx]["partial_json"] += delta.get("partial_json", "")
                        elif dtype == "content_block_stop":
                            idx = data.get("index", 0)
                            if final_content[idx] and final_content[idx].get("type") == "tool_use":
                                if "partial_json" in final_content[idx]:
                                    try:
                                        final_content[idx]["input"] = json.loads(final_content[idx]["partial_json"])
                                    except json.JSONDecodeError:
                                        final_content[idx]["input"] = {"error": "invalid json", "partial": final_content[idx]["partial_json"]}
                                    del final_content[idx]["partial_json"]
                        elif dtype == "message_delta":
                            usage = data.get("usage", usage)
                            stop_reason = data.get("delta", {}).get("stop_reason")
                    except (json.JSONDecodeError, Exception):
                        pass

                final_content_cleaned = [c for c in final_content if c is not None]
                for c in final_content_cleaned:
                    c.pop("partial_json", None)

                response_body = {
                    "id": f"msg_{uuid.uuid4()}",
                    "type": "message",
                    "role": "assistant",
                    "model": req.model,
                    "content": final_content_cleaned,
                    "stop_reason": stop_reason,
                    "stop_sequence": None,
                    "usage": usage
                }
                if conversation_id:
                    response_body["conversation_id"] = conversation_id
                    response_body["conversationId"] = conversation_id
                return JSONResponse(content=response_body, headers=response_headers or None)

        except HTTPException:
            # Don't retry on HTTP exceptions (auth errors, etc.)
            if account:
                await _update_stats(account["id"], False)
            raise
        except Exception as e:
            # Close event_iter if exists
            if event_iter and hasattr(event_iter, "aclose"):
                try:
                    await event_iter.aclose()
                except Exception:
                    pass
            if account:
                await _update_stats(account["id"], False)

            # Check for context too long errors - apply automatic truncation
            error_str = str(e)
            if "CONTENT_LENGTH_EXCEEDS_THRESHOLD" in error_str or "Input is too long" in error_str:
                if trim_retry_count < max_trim_retries:
                    # 自动截断上下文并重试
                    trim_retry_count += 1
                    import logging
                    logging.getLogger(__name__).warning(
                        f"Context too long (trim retry {trim_retry_count}/{max_trim_retries}), "
                        f"truncating {trim_tokens_per_retry} tokens and retrying..."
                    )
                    # 重置账户重试计数，因为这是上下文问题不是账户问题
                    tried_account_ids = []
                    # 截断 aq_request 中的 history
                    history = aq_request.get("conversationState", {}).get("history", [])
                    if history:
                        aq_request["conversationState"]["history"] = _trim_context_by_tokens(history, trim_tokens_per_retry)
                    continue
                else:
                    # 超过最大截断重试次数
                    raise HTTPException(
                        status_code=400,
                        detail=f"Input is too long even after {max_trim_retries} truncation attempts. "
                               f"Please significantly reduce the context length."
                    )

            last_error = e

            # Log retry attempt and apply exponential backoff
            if attempt < max_attempts - 1:
                import logging
                import asyncio
                # Exponential backoff: 3s, 6s, 12s, 24s... max 30s
                wait_time = min(3 * (2 ** attempt), 30)
                logging.getLogger(__name__).warning(f"Request failed (attempt {attempt + 1}/{max_attempts}), retrying in {wait_time}s with different account: {str(e)}")
                await asyncio.sleep(wait_time)
                continue
            else:
                # All retries exhausted
                raise HTTPException(status_code=502, detail=f"All {max_attempts} attempts failed. Last error: {str(last_error)}")

@app.post("/v1/messages/count_tokens")
async def count_tokens_endpoint(req: ClaudeRequest):
    """
    Count tokens in a message without sending it.
    Compatible with Claude API's /v1/messages/count_tokens endpoint.
    Uses tiktoken for local token counting.
    """
    text_to_count = ""
    
    # Count system prompt tokens
    if req.system:
        if isinstance(req.system, str):
            text_to_count += req.system
        elif isinstance(req.system, list):
            for item in req.system:
                if isinstance(item, dict) and item.get("type") == "text":
                    text_to_count += item.get("text", "")
    
    # Count message tokens
    for msg in req.messages:
        if isinstance(msg.content, str):
            text_to_count += msg.content
        elif isinstance(msg.content, list):
            for item in msg.content:
                if isinstance(item, dict) and item.get("type") == "text":
                    text_to_count += item.get("text", "")
    
    # Count tool definition tokens if present
    if req.tools:
        text_to_count += json.dumps([tool.model_dump() if hasattr(tool, 'model_dump') else tool for tool in req.tools], ensure_ascii=False)
    
    input_tokens = count_tokens(text_to_count, apply_multiplier=True)
    
    return {"input_tokens": input_tokens}

@app.post("/v1/chat/completions")
async def chat_completions(
    req: ChatCompletionRequest,
    authorization: Optional[str] = Header(default=None),
    x_api_key: Optional[str] = Header(default=None)
):
    """
    OpenAI-compatible chat endpoint.
    将 OpenAI 格式转换为 Claude 格式，通过现有的 Claude 处理流程处理请求。
    """
    # 动态导入转换模块
    from openai_converter import convert_openai_to_claude, apply_model_mapping, MODEL_MAPPING, MAX_TOKENS_MAPPING
    from openai_stream import OpenAIStreamHandler, convert_claude_response_to_openai

    # 提取 API Key
    bearer_key = _extract_bearer(authorization) if authorization else x_api_key

    # 构建 OpenAI 请求数据
    openai_req = {
        "model": req.model or "",
        "messages": [m.model_dump() for m in req.messages],
        "stream": bool(req.stream),
    }
    if req.max_tokens is not None:
        openai_req["max_tokens"] = req.max_tokens
    if req.temperature is not None:
        openai_req["temperature"] = req.temperature
    if req.top_p is not None:
        openai_req["top_p"] = req.top_p
    if req.tools:
        openai_req["tools"] = req.tools
    if req.tool_choice is not None:
        openai_req["tool_choice"] = req.tool_choice
    if req.user:
        openai_req["user"] = req.user

    # 应用模型映射
    original_model = openai_req["model"]
    openai_req["model"] = apply_model_mapping(original_model)
    if openai_req["model"] != original_model:
        import logging
        logging.getLogger(__name__).info(f"Model mapped: {original_model} -> {openai_req['model']}")

    # 转换 OpenAI 请求为 Claude 格式
    try:
        claude_req_dict = convert_openai_to_claude(
            openai_req,
            max_tokens_mapping=MAX_TOKENS_MAPPING,
            api_key=bearer_key or ""
        )
    except Exception as e:
        traceback.print_exc()
        raise HTTPException(status_code=400, detail=f"Request conversion failed: {str(e)}")

    # 构建 ClaudeRequest 对象
    claude_req = ClaudeRequest(
        model=claude_req_dict.get("model", ""),
        max_tokens=claude_req_dict.get("max_tokens", 8192),
        messages=[_claude_types.ClaudeMessage(**m) for m in claude_req_dict.get("messages", [])],
        system=claude_req_dict.get("system"),
        temperature=claude_req_dict.get("temperature"),
        top_p=claude_req_dict.get("top_p"),
        stream=bool(req.stream),
        tools=[_claude_types.ClaudeTool(**t) for t in claude_req_dict.get("tools", [])] if claude_req_dict.get("tools") else None,
        tool_choice=claude_req_dict.get("tool_choice"),
        metadata=claude_req_dict.get("metadata")
    )

    # 转换请求为 Amazon Q 格式
    try:
        aq_request = convert_claude_to_amazonq_request(claude_req, conversation_id=None)
    except Exception as e:
        traceback.print_exc()
        raise HTTPException(status_code=400, detail=f"Claude to AmazonQ conversion failed: {str(e)}")

    # Post-process history to fix message ordering
    from message_processor import process_claude_history_for_amazonq
    conversation_state = aq_request.get("conversationState", {})
    history = conversation_state.get("history", [])
    if history:
        processed_history = process_claude_history_for_amazonq(history)
        aq_request["conversationState"]["history"] = processed_history

    # 计算输入 tokens
    text_to_count = ""
    if claude_req.system:
        if isinstance(claude_req.system, str):
            text_to_count += claude_req.system
        elif isinstance(claude_req.system, list):
            for item in claude_req.system:
                if isinstance(item, dict) and item.get("type") == "text":
                    text_to_count += item.get("text", "")
    for msg in claude_req.messages:
        if isinstance(msg.content, str):
            text_to_count += msg.content
        elif isinstance(msg.content, list):
            for item in msg.content:
                if isinstance(item, dict) and item.get("type") == "text":
                    text_to_count += item.get("text", "")
    input_tokens = count_tokens(text_to_count, apply_multiplier=True)

    # 重试循环
    tried_account_ids: List[str] = []
    last_error: Optional[Exception] = None
    max_attempts = MAX_RETRY_COUNT + 1
    trim_retry_count = 0  # 跟踪上下文截断重试次数
    max_trim_retries = 3  # 最多截断重试3次
    trim_tokens_per_retry = 10000  # 每次截断10k tokens

    for attempt in range(max_attempts):
        event_iter = None
        account = None
        try:
            # 获取账户
            try:
                account = await resolve_account_for_key(bearer_key, exclude_ids=tried_account_ids if tried_account_ids else None)
            except HTTPException as he:
                if "No enabled account available" in str(he.detail) and tried_account_ids:
                    tried_account_ids = []
                    account = await resolve_account_for_key(bearer_key, exclude_ids=None)
                else:
                    raise
            tried_account_ids.append(account["id"])

            access = account.get("accessToken")
            if not access:
                refreshed = await refresh_access_token_in_db(account["id"])
                access = refreshed.get("accessToken")

            # 发送请求
            _, _, tracker, event_iter = await send_chat_request(
                access_token=access,
                messages=[],
                model=map_model_name(claude_req.model),
                stream=True,
                client=GLOBAL_CLIENT,
                raw_payload=aq_request
            )

            if not event_iter:
                raise Exception("No event stream returned")

            # 尝试获取第一个事件以确保连接有效
            first_event = None
            try:
                first_event = await event_iter.__anext__()
            except StopAsyncIteration:
                raise Exception("Empty response from upstream")

            # 成功！创建处理器并返回响应
            current_account = account
            current_tracker = tracker

            if req.stream:
                # 流式响应 - 使用 OpenAI 流式处理器
                openai_handler = OpenAIStreamHandler(model=req.model or claude_req.model)

                async def event_generator():
                    try:
                        # 确保先发送 role
                        if not openai_handler.sent_role:
                            openai_handler.sent_role = True
                            yield openai_handler._format_sse(openai_handler._create_chunk({
                                "role": "assistant",
                                "content": ""
                            }))

                        if first_event:
                            event_type, payload = first_event
                            async for sse in openai_handler.handle_event(event_type, payload):
                                yield sse
                        async for event_type, payload in event_iter:
                            async for sse in openai_handler.handle_event(event_type, payload):
                                yield sse
                        async for sse in openai_handler.finish():
                            yield sse
                        await _update_stats(current_account["id"], True)
                    except GeneratorExit:
                        await _update_stats(current_account["id"], current_tracker.has_content if current_tracker else False)
                    except Exception as e:
                        import traceback
                        traceback.print_exc()
                        await _update_stats(current_account["id"], False)
                        raise

                return StreamingResponse(
                    event_generator(),
                    media_type="text/event-stream"
                )
            else:
                # 非流式响应 - 累积所有事件然后转换
                content_blocks = []
                usage = {"input_tokens": 0, "output_tokens": 0}
                stop_reason = None
                message_id = None
                role = "assistant"

                # 创建一个简单的 Claude 流处理器来累积响应
                handler = ClaudeStreamHandler(model=claude_req.model, input_tokens=input_tokens)

                async def accumulate_response():
                    nonlocal content_blocks, usage, stop_reason, message_id, role
                    try:
                        if first_event:
                            event_type, payload = first_event
                            async for _ in handler.handle_event(event_type, payload):
                                pass
                        async for event_type, payload in event_iter:
                            if event_type == "message_start":
                                msg = payload.get("message", {})
                                message_id = msg.get("id")
                                role = msg.get("role", "assistant")
                            async for _ in handler.handle_event(event_type, payload):
                                pass
                        async for _ in handler.finish():
                            pass
                    except Exception:
                        pass

                await accumulate_response()

                # 从 handler 构建 Claude 响应
                claude_response = {
                    "id": message_id or f"msg_{uuid.uuid4()}",
                    "type": "message",
                    "role": role,
                    "model": claude_req.model,
                    "content": handler.get_content_blocks() if hasattr(handler, 'get_content_blocks') else [],
                    "stop_reason": handler.stop_reason if hasattr(handler, 'stop_reason') else "end_turn",
                    "usage": {
                        "input_tokens": handler.usage.get("input_tokens", input_tokens) if hasattr(handler, 'usage') else input_tokens,
                        "output_tokens": handler.usage.get("output_tokens", 0) if hasattr(handler, 'usage') else 0
                    }
                }

                # 如果 handler 没有 get_content_blocks 方法，从 response_buffer 构建
                if not hasattr(handler, 'get_content_blocks'):
                    text_content = handler.response_buffer if hasattr(handler, 'response_buffer') else ""
                    claude_response["content"] = [{"type": "text", "text": text_content}] if text_content else []

                # 转换为 OpenAI 格式
                openai_response = convert_claude_response_to_openai(claude_response, model=req.model or claude_req.model)
                await _update_stats(current_account["id"], True)
                return JSONResponse(content=openai_response)

        except HTTPException:
            if account:
                await _update_stats(account["id"], False)
            raise
        except Exception as e:
            if event_iter and hasattr(event_iter, "aclose"):
                try:
                    await event_iter.aclose()
                except Exception:
                    pass
            if account:
                await _update_stats(account["id"], False)

            # 检查不可重试的错误
            error_str = str(e)
            if "CONTENT_LENGTH_EXCEEDS_THRESHOLD" in error_str or "Input is too long" in error_str:
                if trim_retry_count < max_trim_retries:
                    # 自动截断上下文并重试
                    trim_retry_count += 1
                    import logging
                    logging.getLogger(__name__).warning(
                        f"Context too long (trim retry {trim_retry_count}/{max_trim_retries}), "
                        f"truncating {trim_tokens_per_retry} tokens and retrying..."
                    )
                    # 重置账户重试计数，因为这是上下文问题不是账户问题
                    tried_account_ids = []
                    # 截断 aq_request 中的 history
                    history = aq_request.get("conversationState", {}).get("history", [])
                    if history:
                        aq_request["conversationState"]["history"] = _trim_context_by_tokens(history, trim_tokens_per_retry)
                    continue
                else:
                    # 超过最大截断重试次数
                    raise HTTPException(
                        status_code=400,
                        detail=f"Input is too long even after {max_trim_retries} truncation attempts. "
                               f"Please significantly reduce the context length."
                    )

            last_error = e
            traceback.print_exc()  # 打印详细堆栈

            if attempt < max_attempts - 1:
                import logging
                wait_time = min(3 * (2 ** attempt), 30)
                logging.getLogger(__name__).warning(f"Request failed (attempt {attempt + 1}/{max_attempts}), retrying in {wait_time}s: {str(e)}")
                await asyncio.sleep(wait_time)
                continue
            else:
                raise HTTPException(status_code=502, detail=f"All {max_attempts} attempts failed. Last error: {str(last_error)}")

# ------------------------------------------------------------------------------
# OpenAI Models API
# ------------------------------------------------------------------------------

AVAILABLE_MODELS = [
    {
        "id": "claude-opus-4-5-20251101",
        "object": "model",
        "created": 1730419200,
        "owned_by": "anthropic",
        "context_window": 200000,
        "max_output_tokens": 16384,
    },
    {
        "id": "claude-sonnet-4-5-20251101",
        "object": "model",
        "created": 1730419200,
        "owned_by": "anthropic",
        "context_window": 200000,
        "max_output_tokens": 8192,
    },
    {
        "id": "claude-3-5-sonnet-20241022",
        "object": "model",
        "created": 1729555200,
        "owned_by": "anthropic",
        "context_window": 200000,
        "max_output_tokens": 8192,
    },
    {
        "id": "claude-3-5-haiku-20241022",
        "object": "model",
        "created": 1729555200,
        "owned_by": "anthropic",
        "context_window": 200000,
        "max_output_tokens": 4096,
    },
    {
        "id": "claude-3-opus-20240229",
        "object": "model",
        "created": 1709164800,
        "owned_by": "anthropic",
        "context_window": 200000,
        "max_output_tokens": 4096,
    },
    {
        "id": "claude-3-sonnet-20240229",
        "object": "model",
        "created": 1709164800,
        "owned_by": "anthropic",
        "context_window": 200000,
        "max_output_tokens": 4096,
    },
    {
        "id": "claude-3-haiku-20240307",
        "object": "model",
        "created": 1709769600,
        "owned_by": "anthropic",
        "context_window": 200000,
        "max_output_tokens": 4096,
    },
]

@app.get("/v1/models")
async def list_models():
    """
    OpenAI-compatible models list endpoint.
    Returns available Claude models with context window information.
    Also includes mapped models from OPENAI_MODEL_MAPPING.
    """
    from openai_converter import parse_model_mapping

    # 每次请求时重新解析环境变量，确保获取最新配置
    model_mapping = parse_model_mapping(os.getenv("OPENAI_MODEL_MAPPING", ""))

    # 构建模型 ID 到模型信息的映射
    model_info_map = {m["id"]: m for m in AVAILABLE_MODELS}

    # 收集所有模型
    all_models = list(AVAILABLE_MODELS)

    # 添加映射的模型（克隆目标模型的信息，但使用映射的名称）
    for source_model, target_model in model_mapping.items():
        if source_model not in model_info_map:
            # 找到目标模型的信息
            target_info = model_info_map.get(target_model)
            if target_info:
                # 克隆目标模型信息，但使用源模型名称
                mapped_model = {
                    "id": source_model,
                    "object": "model",
                    "created": target_info.get("created", int(time.time())),
                    "owned_by": target_info.get("owned_by", "anthropic"),
                    "context_window": target_info.get("context_window", 200000),
                    "max_output_tokens": target_info.get("max_output_tokens", 8192),
                }
                all_models.append(mapped_model)
            else:
                # 目标模型不在列表中，使用默认值
                mapped_model = {
                    "id": source_model,
                    "object": "model",
                    "created": int(time.time()),
                    "owned_by": "anthropic",
                    "context_window": 200000,
                    "max_output_tokens": 8192,
                }
                all_models.append(mapped_model)

    return {
        "object": "list",
        "data": all_models
    }

@app.get("/v1/models/{model_id}")
async def get_model(model_id: str):
    """
    OpenAI-compatible model detail endpoint.
    """
    from openai_converter import parse_model_mapping

    # 每次请求时重新解析环境变量
    model_mapping = parse_model_mapping(os.getenv("OPENAI_MODEL_MAPPING", ""))

    # 先查找原始模型
    for model in AVAILABLE_MODELS:
        if model["id"] == model_id:
            return model

    # 查找映射模型
    if model_id in model_mapping:
        target_model = model_mapping[model_id]
        for model in AVAILABLE_MODELS:
            if model["id"] == target_model:
                return {
                    "id": model_id,
                    "object": "model",
                    "created": model.get("created", int(time.time())),
                    "owned_by": model.get("owned_by", "anthropic"),
                    "context_window": model.get("context_window", 200000),
                    "max_output_tokens": model.get("max_output_tokens", 8192),
                }
        # 目标模型不在列表中，使用默认值
        return {
            "id": model_id,
            "object": "model",
            "created": int(time.time()),
            "owned_by": "anthropic",
            "context_window": 200000,
            "max_output_tokens": 8192,
        }

    raise HTTPException(status_code=404, detail=f"Model '{model_id}' not found")

# ------------------------------------------------------------------------------
# Device Authorization (URL Login, 5-minute timeout)
# ------------------------------------------------------------------------------

# Dynamic import of auth_flow.py (device-code login helpers)
def _load_auth_flow_module():
    mod_path = BASE_DIR / "auth_flow.py"
    spec = importlib.util.spec_from_file_location("v2_auth_flow", str(mod_path))
    module = importlib.util.module_from_spec(spec)  # type: ignore[arg-type]
    assert spec is not None and spec.loader is not None
    spec.loader.exec_module(module)  # type: ignore[attr-defined]
    return module

_auth_flow = _load_auth_flow_module()
register_client_min = _auth_flow.register_client_min
device_authorize = _auth_flow.device_authorize
poll_token_device_code = _auth_flow.poll_token_device_code

# In-memory auth sessions (ephemeral)
AUTH_SESSIONS: Dict[str, Dict[str, Any]] = {}

class AuthStartBody(BaseModel):
    label: Optional[str] = None
    enabled: Optional[bool] = True

class AdminLoginRequest(BaseModel):
    password: str

class AdminLoginResponse(BaseModel):
    success: bool
    message: str

async def _create_account_from_tokens(
    client_id: str,
    client_secret: str,
    access_token: str,
    refresh_token: Optional[str],
    label: Optional[str],
    enabled: bool,
) -> Dict[str, Any]:
    now = time.strftime("%Y-%m-%dT%H:%M:%S", time.gmtime())
    acc_id = str(uuid.uuid4())
    await _db.execute(
        """
        INSERT INTO accounts (id, label, clientId, clientSecret, refreshToken, accessToken, other, last_refresh_time, last_refresh_status, created_at, updated_at, enabled)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (
            acc_id,
            label,
            client_id,
            client_secret,
            refresh_token,
            access_token,
            None,
            now,
            "success",
            now,
            now,
            1 if enabled else 0,
        ),
    )
    row = await _db.fetchone("SELECT * FROM accounts WHERE id=?", (acc_id,))
    return _row_to_dict(row)

# 管理控制台相关端点 - 仅在启用时注册
if CONSOLE_ENABLED:
    # ------------------------------------------------------------------------------
    # Admin Authentication Endpoints
    # ------------------------------------------------------------------------------

    @app.post("/api/login", response_model=AdminLoginResponse)
    async def admin_login(request: AdminLoginRequest, req: Request) -> AdminLoginResponse:
        """Admin login endpoint - password only, with rate limiting"""
        client_ip = req.client.host if req.client else "unknown"
        now = time.time()

        # Check if locked
        if client_ip in _login_failures:
            info = _login_failures[client_ip]
            if info.get("locked_until", 0) > now:
                remaining = int(info["locked_until"] - now)
                return AdminLoginResponse(
                    success=False,
                    message=f"账号已锁定，请 {remaining // 60} 分钟后重试"
                )

        if request.password == ADMIN_PASSWORD:
            # Clear failures on success
            _login_failures.pop(client_ip, None)
            return AdminLoginResponse(
                success=True,
                message="Login successful"
            )
        else:
            # Track failure
            if client_ip not in _login_failures:
                _login_failures[client_ip] = {"count": 0, "locked_until": 0}
            _login_failures[client_ip]["count"] += 1
            count = _login_failures[client_ip]["count"]

            if count >= LOGIN_MAX_ATTEMPTS:
                _login_failures[client_ip]["locked_until"] = now + LOGIN_LOCKOUT_SECONDS
                return AdminLoginResponse(
                    success=False,
                    message=f"密码错误次数过多，账号已锁定1小时"
                )

            remaining = LOGIN_MAX_ATTEMPTS - count
            return AdminLoginResponse(
                success=False,
                message=f"密码错误，还剩 {remaining} 次尝试机会"
            )

    @app.get("/login", response_class=FileResponse)
    def login_page():
        """Serve the login page"""
        path = BASE_DIR / "frontend" / "login.html"
        if not path.exists():
            raise HTTPException(status_code=404, detail="frontend/login.html not found")
        return FileResponse(str(path))

    # ------------------------------------------------------------------------------
    # Device Authorization Endpoints
    # ------------------------------------------------------------------------------

    @app.post("/v2/auth/start")
    async def auth_start(body: AuthStartBody, _: bool = Depends(verify_admin_password)):
        """
        Start device authorization and return verification URL for user login.
        Session lifetime capped at 5 minutes on claim.
        """
        try:
            cid, csec = await register_client_min()
            dev = await device_authorize(cid, csec)
        except httpx.HTTPError as e:
            raise HTTPException(status_code=502, detail=f"OIDC error: {str(e)}")

        auth_id = str(uuid.uuid4())
        sess = {
            "clientId": cid,
            "clientSecret": csec,
            "deviceCode": dev.get("deviceCode"),
            "interval": int(dev.get("interval", 1)),
            "expiresIn": int(dev.get("expiresIn", 600)),
            "verificationUriComplete": dev.get("verificationUriComplete"),
            "userCode": dev.get("userCode"),
            "startTime": int(time.time()),
            "label": body.label,
            "enabled": True if body.enabled is None else bool(body.enabled),
            "status": "pending",
            "error": None,
            "accountId": None,
        }
        AUTH_SESSIONS[auth_id] = sess
        return {
            "authId": auth_id,
            "verificationUriComplete": sess["verificationUriComplete"],
            "userCode": sess["userCode"],
            "expiresIn": sess["expiresIn"],
            "interval": sess["interval"],
        }

    @app.get("/v2/auth/status/{auth_id}")
    async def auth_status(auth_id: str, _: bool = Depends(verify_admin_password)):
        sess = AUTH_SESSIONS.get(auth_id)
        if not sess:
            raise HTTPException(status_code=404, detail="Auth session not found")
        now_ts = int(time.time())
        deadline = sess["startTime"] + min(int(sess.get("expiresIn", 600)), 300)
        remaining = max(0, deadline - now_ts)
        return {
            "status": sess.get("status"),
            "remaining": remaining,
            "error": sess.get("error"),
            "accountId": sess.get("accountId"),
        }

    @app.post("/v2/auth/claim/{auth_id}")
    async def auth_claim(auth_id: str, _: bool = Depends(verify_admin_password)):
        """
        Block up to 5 minutes to exchange the device code for tokens after user completed login.
        On success, creates an enabled account and returns it.
        """
        sess = AUTH_SESSIONS.get(auth_id)
        if not sess:
            raise HTTPException(status_code=404, detail="Auth session not found")
        if sess.get("status") in ("completed", "timeout", "error"):
            return {
                "status": sess["status"],
                "accountId": sess.get("accountId"),
                "error": sess.get("error"),
            }
        try:
            toks = await poll_token_device_code(
                sess["clientId"],
                sess["clientSecret"],
                sess["deviceCode"],
                sess["interval"],
                sess["expiresIn"],
                max_timeout_sec=300,  # 5 minutes
            )
            access_token = toks.get("accessToken")
            refresh_token = toks.get("refreshToken")
            if not access_token:
                raise HTTPException(status_code=502, detail="No accessToken returned from OIDC")

            acc = await _create_account_from_tokens(
                sess["clientId"],
                sess["clientSecret"],
                access_token,
                refresh_token,
                sess.get("label"),
                sess.get("enabled", True),
            )
            sess["status"] = "completed"
            sess["accountId"] = acc["id"]
            return {
                "status": "completed",
                "account": acc,
            }
        except TimeoutError:
            sess["status"] = "timeout"
            raise HTTPException(status_code=408, detail="Authorization timeout (5 minutes)")
        except httpx.HTTPError as e:
            sess["status"] = "error"
            sess["error"] = str(e)
            raise HTTPException(status_code=502, detail=f"OIDC error: {str(e)}")

    # ------------------------------------------------------------------------------
    # Accounts Management API
    # ------------------------------------------------------------------------------

    @app.post("/v2/accounts")
    async def create_account(body: AccountCreate, _: bool = Depends(verify_admin_password)):
        now = time.strftime("%Y-%m-%dT%H:%M:%S", time.gmtime())
        acc_id = str(uuid.uuid4())
        other_str = json.dumps(body.other, ensure_ascii=False) if body.other is not None else None
        enabled_val = 1 if (body.enabled is None or body.enabled) else 0
        await _db.execute(
            """
            INSERT INTO accounts (id, label, clientId, clientSecret, refreshToken, accessToken, other, last_refresh_time, last_refresh_status, created_at, updated_at, enabled)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                acc_id,
                body.label,
                body.clientId,
                body.clientSecret,
                body.refreshToken,
                body.accessToken,
                other_str,
                None,
                "never",
                now,
                now,
                enabled_val,
            ),
        )
        row = await _db.fetchone("SELECT * FROM accounts WHERE id=?", (acc_id,))
        return _row_to_dict(row)


    async def _verify_and_enable_accounts(account_ids: List[str]):
        """后台异步验证并启用账号"""
        for acc_id in account_ids:
            try:
                # 必须先获取完整的账号信息
                account = await get_account(acc_id)
                verify_success, fail_reason = await verify_account(account)
                now = time.strftime("%Y-%m-%dT%H:%M:%S", time.gmtime())

                if verify_success:
                    await _db.execute("UPDATE accounts SET enabled=1, updated_at=? WHERE id=?", (now, acc_id))
                elif fail_reason:
                    other_dict = account.get("other", {}) or {}
                    other_dict['failedReason'] = fail_reason
                    await _db.execute("UPDATE accounts SET other=?, updated_at=? WHERE id=?", (json.dumps(other_dict, ensure_ascii=False), now, acc_id))
            except Exception as e:
                print(f"Error verifying account {acc_id}: {e}")
                traceback.print_exc()

    @app.post("/v2/accounts/feed")
    async def create_accounts_feed(request: BatchAccountCreate, _: bool = Depends(verify_admin_password)):
        """
        统一的投喂接口，接收账号列表，立即存入并后台异步验证。
        """
        now = time.strftime("%Y-%m-%dT%H:%M:%S", time.gmtime())
        new_account_ids = []

        for i, account_data in enumerate(request.accounts):
            acc_id = str(uuid.uuid4())
            other_dict = account_data.other or {}
            other_dict['source'] = 'feed'
            other_str = json.dumps(other_dict, ensure_ascii=False)

            await _db.execute(
                """
                INSERT INTO accounts (id, label, clientId, clientSecret, refreshToken, accessToken, other, last_refresh_time, last_refresh_status, created_at, updated_at, enabled)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    acc_id,
                    account_data.label or f"批量账号 {i+1}",
                    account_data.clientId,
                    account_data.clientSecret,
                    account_data.refreshToken,
                    account_data.accessToken,
                    other_str,
                    None,
                    "never",
                    now,
                    now,
                    0,  # 初始为禁用状态
                ),
            )
            new_account_ids.append(acc_id)

        # 启动后台任务进行验证，不阻塞当前请求
        if new_account_ids:
            asyncio.create_task(_verify_and_enable_accounts(new_account_ids))

        return {
            "status": "processing",
            "message": f"{len(new_account_ids)} accounts received and are being verified in the background.",
            "account_ids": new_account_ids
        }

    @app.get("/v2/accounts")
    async def list_accounts(_: bool = Depends(verify_admin_password), enabled: Optional[bool] = None, sort_by: str = "created_at", sort_order: str = "desc"):
        query = "SELECT * FROM accounts"
        params = []
        if enabled is not None:
            query += " WHERE enabled=?"
            params.append(1 if enabled else 0)
        sort_field = "created_at" if sort_by not in ["created_at", "success_count"] else sort_by
        order = "DESC" if sort_order.lower() == "desc" else "ASC"
        query += f" ORDER BY {sort_field} {order}"
        rows = await _db.fetchall(query, tuple(params) if params else ())
        accounts = [_row_to_dict(r) for r in rows]
        return {"accounts": accounts, "count": len(accounts)}

    # ------------------------------------------------------------------------------
    # Usage Query
    # ------------------------------------------------------------------------------

    USAGE_LIMITS_URL = "https://q.{region}.amazonaws.com/getUsageLimits"

    async def _get_account_usage(account: Dict[str, Any], client: httpx.AsyncClient) -> Dict[str, Any]:
        """获取单个账号的使用量"""
        result = {"account_id": account["id"], "label": account.get("label", ""), "success": False, "usage": None, "error": None}

        access_token = account.get("accessToken")
        if not access_token:
            result["error"] = "无访问令牌"
            return result

        region = "us-east-1"
        url = USAGE_LIMITS_URL.format(region=region)
        params = {"isEmailRequired": "true", "origin": "AI_EDITOR", "resourceType": "AGENTIC_REQUEST"}
        headers = {
            "Authorization": f"Bearer {access_token}",
            "amz-sdk-invocation-id": str(uuid.uuid4()),
            "amz-sdk-request": "attempt=1; max=1",
        }

        try:
            resp = await client.get(url, params=params, headers=headers, timeout=30.0)
            if resp.status_code == 200:
                result["success"] = True
                result["usage"] = resp.json()
            else:
                result["error"] = f"HTTP {resp.status_code}"
        except Exception as e:
            result["error"] = str(e)

        return result

    @app.get("/v2/accounts/usage")
    async def get_all_accounts_usage(_: bool = Depends(verify_admin_password)):
        """查询所有启用账号的使用量"""
        rows = await _db.fetchall("SELECT * FROM accounts WHERE enabled=1 ORDER BY created_at DESC")
        accounts = [_row_to_dict(r) for r in rows]

        if not accounts:
            return {"results": [], "total": 0, "success_count": 0}

        results = []
        async with httpx.AsyncClient() as client:
            tasks = [_get_account_usage(acc, client) for acc in accounts]
            results = await asyncio.gather(*tasks)

        success_count = sum(1 for r in results if r["success"])
        return {"results": results, "total": len(results), "success_count": success_count}

    @app.get("/v2/accounts/{account_id}/usage")
    async def get_single_account_usage(account_id: str, _: bool = Depends(verify_admin_password)):
        """查询单个账号的使用量"""
        account = await get_account(account_id)
        async with httpx.AsyncClient() as client:
            result = await _get_account_usage(account, client)
        return result

    @app.get("/v2/accounts/{account_id}")
    async def get_account_detail(account_id: str, _: bool = Depends(verify_admin_password)):
        return await get_account(account_id)

    @app.delete("/v2/accounts/{account_id}")
    async def delete_account(account_id: str, _: bool = Depends(verify_admin_password)):
        rowcount = await _db.execute("DELETE FROM accounts WHERE id=?", (account_id,))
        if rowcount == 0:
            raise HTTPException(status_code=404, detail="Account not found")
        return {"deleted": account_id}

    @app.patch("/v2/accounts/{account_id}")
    async def update_account(account_id: str, body: AccountUpdate, _: bool = Depends(verify_admin_password)):
        now = time.strftime("%Y-%m-%dT%H:%M:%S", time.gmtime())
        fields = []
        values: List[Any] = []

        if body.label is not None:
            fields.append("label=?"); values.append(body.label)
        if body.clientId is not None:
            fields.append("clientId=?"); values.append(body.clientId)
        if body.clientSecret is not None:
            fields.append("clientSecret=?"); values.append(body.clientSecret)
        if body.refreshToken is not None:
            fields.append("refreshToken=?"); values.append(body.refreshToken)
        if body.accessToken is not None:
            fields.append("accessToken=?"); values.append(body.accessToken)
        if body.other is not None:
            fields.append("other=?"); values.append(json.dumps(body.other, ensure_ascii=False))
        if body.enabled is not None:
            fields.append("enabled=?"); values.append(1 if body.enabled else 0)

        if not fields:
            return await get_account(account_id)

        fields.append("updated_at=?"); values.append(now)
        values.append(account_id)

        rowcount = await _db.execute(f"UPDATE accounts SET {', '.join(fields)} WHERE id=?", tuple(values))
        if rowcount == 0:
            raise HTTPException(status_code=404, detail="Account not found")
        row = await _db.fetchone("SELECT * FROM accounts WHERE id=?", (account_id,))
        return _row_to_dict(row)

    @app.post("/v2/accounts/{account_id}/refresh")
    async def manual_refresh(account_id: str, _: bool = Depends(verify_admin_password)):
        return await refresh_access_token_in_db(account_id)

    # ------------------------------------------------------------------------------
    # Simple Frontend (minimal dev test page; full UI in v2/frontend/index.html)
    # ------------------------------------------------------------------------------

    # Frontend inline HTML removed; serving ./frontend/index.html instead (see route below)
    # Note: This route is NOT protected - the HTML file is served freely,
    # but the frontend JavaScript checks authentication and redirects to /login if needed.
    # All API endpoints remain protected.

    @app.get("/", response_class=FileResponse)
    def index():
        path = BASE_DIR / "frontend" / "index.html"
        if not path.exists():
            raise HTTPException(status_code=404, detail="frontend/index.html not found")
        return FileResponse(str(path))

# ------------------------------------------------------------------------------
# Health
# ------------------------------------------------------------------------------

@app.get("/healthz")
async def health():
    return {"status": "ok"}

# ------------------------------------------------------------------------------
# Startup / Shutdown Events
# ------------------------------------------------------------------------------

# async def _verify_disabled_accounts_loop():
#     """后台验证禁用账号任务"""
#     while True:
#         try:
#             await asyncio.sleep(1800)
#             async with _conn() as conn:
#                 accounts = await _list_disabled_accounts(conn)
#                 if accounts:
#                     for account in accounts:
#                         other = account.get('other')
#                         if other:
#                             try:
#                                 other_dict = json.loads(other) if isinstance(other, str) else other
#                                 if other_dict.get('failedReason') == 'AccessDenied':
#                                     continue
#                             except:
#                                 pass
#                         try:
#                             verify_success, fail_reason = await verify_account(account)
#                             now = time.strftime("%Y-%m-%dT%H:%M:%S", time.gmtime())
#                             if verify_success:
#                                 await conn.execute("UPDATE accounts SET enabled=1, updated_at=? WHERE id=?", (now, account['id']))
#                             elif fail_reason:
#                                 other_dict = {}
#                                 if account.get('other'):
#                                     try:
#                                         other_dict = json.loads(account['other']) if isinstance(account['other'], str) else account['other']
#                                     except:
#                                         pass
#                                 other_dict['failedReason'] = fail_reason
#                                 await conn.execute("UPDATE accounts SET other=?, updated_at=? WHERE id=?", (json.dumps(other_dict, ensure_ascii=False), now, account['id']))
#                             await conn.commit()
#                         except Exception:
#                             pass
#         except Exception:
#             pass

@app.on_event("startup")
async def startup_event():
    """Initialize database and start background tasks on startup."""
    await _init_global_client()
    await _ensure_db()
    asyncio.create_task(_refresh_stale_tokens())
    # asyncio.create_task(_verify_disabled_accounts_loop())

@app.on_event("shutdown")
async def shutdown_event():
    await _close_global_client()
    await close_db()
