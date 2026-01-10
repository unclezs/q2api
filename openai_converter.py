"""
OpenAI 格式到 Claude 格式的转换器

将 OpenAI Chat Completions API 格式转换为 Anthropic Claude Messages API 格式
参考：openai-claude-proxy/converter.go
"""

import os
import time
import hashlib
import json
import logging
from typing import Dict, List, Any, Optional, Union
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)


def generate_stable_user_id(api_key: str, client_user: str = "") -> str:
    """
    基于 API Key 生成稳定的 user_id
    同一个 API Key 始终生成相同的 user_id，确保缓存可以命中
    支持 SESSION_TTL_MINUTES 环境变量控制 session 轮换周期（默认 60 分钟）
    """
    # 确定种子：优先使用客户端传的 user，否则使用 API Key
    seed = api_key
    if client_user:
        seed = f"{api_key}_{client_user}"

    # 获取 session TTL 配置（分钟），默认 60 分钟
    session_ttl_minutes = 60
    ttl_str = os.getenv("SESSION_TTL_MINUTES", "")
    if ttl_str:
        try:
            ttl = int(ttl_str)
            if ttl > 0:
                session_ttl_minutes = ttl
        except ValueError:
            pass

    # 计算当前时间窗口（基于 TTL 分钟数）
    time_window = int(time.time()) // (session_ttl_minutes * 60)

    # 生成稳定的用户 hash（不随时间变化，保持用户身份一致）
    user_hash = hashlib.sha256(seed.encode()).hexdigest()

    # 生成会话 UUID（加入时间窗口，实现周期性轮换）
    session_seed = f"{seed}_session_{time_window}"
    session_hash = hashlib.sha256(session_seed.encode()).hexdigest()
    session_uuid = f"{session_hash[:8]}-{session_hash[8:12]}-{session_hash[12:16]}-{session_hash[16:20]}-{session_hash[20:32]}"

    user_id = f"user_{user_hash}_account__session_{session_uuid}"

    logger.info(f"[INFO] Session TTL: {session_ttl_minutes} minutes, TimeWindow: {time_window}, UserID: {user_id[:40]}...{user_id[-20:]}")

    return user_id


def get_default_max_tokens(model: str, max_tokens_mapping: Optional[Dict[str, int]] = None) -> int:
    """根据模型名称返回默认的 max_tokens"""
    # 1. 首先检查用户配置的 mapping
    if max_tokens_mapping:
        if model in max_tokens_mapping:
            return max_tokens_mapping[model]

    # 2. 然后检查环境变量
    max_tokens_env = os.getenv("OPENAI_MAX_TOKENS", "")
    if max_tokens_env:
        try:
            tokens = int(max_tokens_env)
            if tokens > 0:
                return tokens
        except ValueError:
            pass

    # 3. 最后根据模型名称选择合适的默认值
    model_lower = model.lower()
    if "opus-4" in model_lower:
        return 16384  # Claude Opus 4.x 支持更大的输出
    elif "opus" in model_lower:
        return 8192  # Claude 3 Opus
    elif "sonnet" in model_lower:
        return 8192  # Claude 3.5 Sonnet
    elif "haiku" in model_lower:
        return 4096  # Claude Haiku (较小模型)
    else:
        return 8192  # 默认使用 8192


def is_string_content(content: Any) -> bool:
    """检查 content 是否为字符串"""
    return isinstance(content, str)


def get_string_content(content: Any) -> str:
    """获取字符串内容"""
    if isinstance(content, str):
        return content
    return ""


@dataclass
class OpenAIMessage:
    role: str
    content: Any
    tool_calls: List[Dict[str, Any]] = field(default_factory=list)
    tool_call_id: str = ""


@dataclass
class OpenAIRequest:
    model: str
    messages: List[Dict[str, Any]]
    max_tokens: int = 0
    temperature: float = 0.0
    top_p: float = 0.0
    stream: bool = False
    tools: List[Dict[str, Any]] = field(default_factory=list)
    tool_choice: Any = None
    user: str = ""


def convert_openai_to_claude(
    req: Dict[str, Any],
    max_tokens_mapping: Optional[Dict[str, int]] = None,
    api_key: str = ""
) -> Dict[str, Any]:
    """
    将 OpenAI 请求格式转换为 Claude 请求格式

    完全参考 openai-claude-proxy/converter.go 的 ConvertOpenAIToAnthropic 函数
    """
    model = req.get("model", "")
    messages = req.get("messages", [])
    max_tokens = req.get("max_tokens", 0)
    temperature = req.get("temperature")
    top_p = req.get("top_p")
    stream = req.get("stream", False)
    tools = req.get("tools", [])
    tool_choice = req.get("tool_choice")
    user = req.get("user", "")

    # 转换工具定义
    claude_tools = []
    for tool in tools:
        if tool.get("type") == "function":
            func = tool.get("function", {})
            params = func.get("parameters", {})
            claude_tool = {
                "name": func.get("name", ""),
                "description": func.get("description", ""),
                "input_schema": {}
            }

            if params.get("type"):
                claude_tool["input_schema"]["type"] = params["type"]
            claude_tool["input_schema"]["properties"] = params.get("properties", {})
            claude_tool["input_schema"]["required"] = params.get("required", [])

            # 复制其他字段
            for key, val in params.items():
                if key not in ("type", "properties", "required"):
                    claude_tool["input_schema"][key] = val

            claude_tools.append(claude_tool)

    # 构建基础请求
    claude_req = {
        "model": model,
        "max_tokens": max_tokens,
        "stream": stream,
    }

    if temperature is not None:
        claude_req["temperature"] = temperature
    if top_p is not None:
        claude_req["top_p"] = top_p
    if claude_tools:
        claude_req["tools"] = claude_tools
    if tool_choice:
        claude_req["tool_choice"] = convert_tool_choice(tool_choice)

    # 生成稳定的 metadata.user_id（基于 API Key）
    claude_req["metadata"] = {
        "user_id": generate_stable_user_id(api_key, user)
    }
    logger.info(f"[INFO] Generated stable user_id: {claude_req['metadata']['user_id'][:30]}...{claude_req['metadata']['user_id'][-20:]}")

    if claude_req["max_tokens"] == 0:
        claude_req["max_tokens"] = get_default_max_tokens(model, max_tokens_mapping)

    # 格式化消息：合并连续相同角色的消息
    format_messages = []
    last_message = {"role": "tool", "content": None}

    for message in messages:
        msg = dict(message)  # 复制消息
        if not msg.get("role"):
            msg["role"] = "user"

        # 合并连续相同角色的消息（tool 除外）
        if last_message["role"] == msg["role"] and last_message["role"] != "tool":
            if is_string_content(last_message.get("content")) and is_string_content(msg.get("content")):
                # 合并文本内容
                combined = f"{get_string_content(last_message.get('content'))} {get_string_content(msg.get('content'))}"
                msg["content"] = combined.strip()
                # 删除上一条消息
                if format_messages:
                    format_messages.pop()

        # 如果 content 是 None，设置为占位符
        if msg.get("content") is None:
            msg["content"] = "..."

        format_messages.append(msg)
        last_message = msg

    # 转换消息
    claude_messages = []
    system_messages = []
    is_first_message = True

    for message in format_messages:
        role = message.get("role", "")
        content = message.get("content")
        tool_calls = message.get("tool_calls", [])
        tool_call_id = message.get("tool_call_id", "")

        # 提取 system 消息
        if role == "system":
            if is_string_content(content):
                system_messages.append({
                    "type": "text",
                    "text": get_string_content(content)
                })
            elif isinstance(content, list):
                for item in content:
                    if isinstance(item, dict) and item.get("type") == "text":
                        text = item.get("text", "")
                        if text:
                            system_messages.append({
                                "type": "text",
                                "text": text
                            })
            continue

        # 确保第一条消息是 user
        if is_first_message:
            is_first_message = False
            if role != "user":
                logger.info("[INFO] First message is not user, adding placeholder user message")
                claude_messages.append({
                    "role": "user",
                    "content": [{"type": "text", "text": "..."}]
                })

        claude_msg = {"role": role}

        # 处理 tool 结果
        if role == "tool" and tool_call_id:
            tool_result = {
                "type": "tool_result",
                "tool_use_id": tool_call_id,
                "content": content
            }

            # 尝试合并到上一条 user 消息
            if claude_messages and claude_messages[-1]["role"] == "user":
                last_msg = claude_messages[-1]

                # 确保 content 是数组格式
                if isinstance(last_msg.get("content"), str):
                    last_msg["content"] = [{"type": "text", "text": last_msg["content"]}]

                if isinstance(last_msg.get("content"), list):
                    last_msg["content"].append(tool_result)
                    logger.info("[INFO] Merged tool_result into previous user message")
                    continue
            else:
                # 创建新的 user 消息
                claude_msg["role"] = "user"
                claude_msg["content"] = [tool_result]

        elif is_string_content(content) and not tool_calls:
            # 纯文本消息
            claude_msg["content"] = get_string_content(content)
        else:
            # 复杂内容或有 tool_calls
            claude_contents = []

            # 转换 content
            if isinstance(content, list):
                for item in content:
                    if not isinstance(item, dict):
                        continue

                    content_type = item.get("type", "")

                    if content_type == "text":
                        text = item.get("text", "")
                        if not text:
                            logger.debug("[DEBUG] Skipping empty text block")
                            continue
                        claude_contents.append({
                            "type": "text",
                            "text": text
                        })
                    elif content_type == "image_url":
                        image_url = item.get("image_url", {})
                        url = image_url.get("url", "")
                        if url:
                            # 检查是否是 base64 数据
                            if url.startswith("data:"):
                                # 解析 data URL: data:image/png;base64,xxxxx
                                try:
                                    header, data = url.split(",", 1)
                                    media_type = header.split(":")[1].split(";")[0]
                                    claude_contents.append({
                                        "type": "image",
                                        "source": {
                                            "type": "base64",
                                            "media_type": media_type,
                                            "data": data
                                        }
                                    })
                                except Exception:
                                    claude_contents.append({
                                        "type": "image",
                                        "source": {
                                            "type": "url",
                                            "url": url
                                        }
                                    })
                            else:
                                claude_contents.append({
                                    "type": "image",
                                    "source": {
                                        "type": "url",
                                        "url": url
                                    }
                                })

            # 添加 tool_calls（不能跳过，否则后续的 tool_result 会找不到对应的 tool_use）
            if tool_calls:
                for tool_call in tool_calls:
                    input_data = {}

                    arguments = tool_call.get("function", {}).get("arguments", "")
                    if arguments and arguments != "{}":
                        try:
                            input_data = json.loads(arguments)
                        except json.JSONDecodeError as e:
                            logger.error(f"[ERROR] Failed to parse tool call arguments: ID={tool_call.get('id')}, Name={tool_call.get('function', {}).get('name')}, Error={e}")

                    claude_contents.append({
                        "type": "tool_use",
                        "id": tool_call.get("id", ""),
                        "name": tool_call.get("function", {}).get("name", ""),
                        "input": input_data
                    })
                    logger.debug(f"[DEBUG] Converted tool_call: ID={tool_call.get('id')}, Name={tool_call.get('function', {}).get('name')}, InputLen={len(input_data)}")

            if claude_contents:
                claude_msg["content"] = claude_contents
            else:
                # 如果没有任何内容，跳过这条消息
                logger.warning("[WARN] Skipping empty message after tool_call filtering")
                continue

        claude_messages.append(claude_msg)

    # 添加 system 消息并设置 cache_control
    if system_messages:
        system_messages[-1]["cache_control"] = {
            "type": "ephemeral"
        }
        logger.info("[INFO] Added cache_control to system (ephemeral)")
        claude_req["system"] = system_messages

    # 在倒数第2条 assistant 消息添加 cache_control
    if len(claude_messages) >= 2:
        second_last = claude_messages[-2]
        if second_last["role"] == "assistant":
            add_cache_control_to_message(second_last)
            logger.info("[INFO] Added cache_control to second-to-last assistant message (ephemeral)")

    claude_req["messages"] = claude_messages
    return claude_req


def add_cache_control_to_message(msg: Dict[str, Any]) -> None:
    """为消息添加 cache_control"""
    content = msg.get("content")

    if isinstance(content, list) and content:
        content[-1]["cache_control"] = {"type": "ephemeral"}
    elif isinstance(content, str) and content:
        msg["content"] = [{
            "type": "text",
            "text": content,
            "cache_control": {"type": "ephemeral"}
        }]


def convert_tool_choice(choice: Any) -> Any:
    """转换 tool_choice 格式"""
    if choice is None:
        return None

    if isinstance(choice, str):
        if choice in ("auto", "required", "none", "any"):
            return {"type": choice}
    elif isinstance(choice, dict):
        return choice

    return None


def convert_stop_reason(reason: str) -> str:
    """转换停止原因"""
    mapping = {
        "end_turn": "stop",
        "max_tokens": "length",
        "stop_sequence": "stop",
        "tool_use": "tool_calls"
    }
    return mapping.get(reason, reason)


def parse_model_mapping(mapping_str: str) -> Dict[str, str]:
    """
    解析模型映射配置
    格式: "gpt-4:claude-opus-4-5-20251101,gpt-3.5-turbo:claude-3-5-haiku-20241022"
    """
    if not mapping_str:
        return {}

    result = {}
    for pair in mapping_str.split(","):
        pair = pair.strip()
        if ":" in pair:
            key, value = pair.split(":", 1)
            result[key.strip()] = value.strip()

    return result


def parse_max_tokens_mapping(mapping_str: str) -> Dict[str, int]:
    """
    解析 max_tokens 映射配置
    格式: "claude-opus-4-5-20251101:16384,claude-3-5-sonnet:8192"
    """
    if not mapping_str:
        return {}

    result = {}
    for pair in mapping_str.split(","):
        pair = pair.strip()
        if ":" in pair:
            key, value = pair.split(":", 1)
            try:
                result[key.strip()] = int(value.strip())
            except ValueError:
                pass

    return result


# 加载配置
MODEL_MAPPING = parse_model_mapping(os.getenv("OPENAI_MODEL_MAPPING", ""))
MAX_TOKENS_MAPPING = parse_max_tokens_mapping(os.getenv("OPENAI_MAX_TOKENS_MAPPING", ""))


def apply_model_mapping(model: str) -> str:
    """应用模型映射"""
    return MODEL_MAPPING.get(model, model)
