"""
OpenAI 流式响应处理器

将 Claude Messages API 的流式响应格式转换为 OpenAI Chat Completions API 格式
参考：openai-claude-proxy/proxy.go
"""

import json
import time
import uuid
import os
import logging
from typing import AsyncGenerator, Dict, Any, Optional, List

logger = logging.getLogger(__name__)

# 调试开关
DEBUG_OPENAI_STREAM = os.getenv("DEBUG_OPENAI_STREAM", "").lower() in ("true", "1", "yes")

def debug_log(msg: str):
    """打印调试日志（受环境变量控制）"""
    if DEBUG_OPENAI_STREAM:
        print(msg)


class OpenAIStreamHandler:
    """
    将 Claude SSE 响应转换为 OpenAI SSE 响应格式
    """

    def __init__(self, model: str = "unknown", stream_id: str = None):
        self.model = model
        self.stream_id = stream_id or f"chatcmpl-{uuid.uuid4()}"
        self.created = int(time.time())
        self.message_id = None
        self.usage = None
        self.tool_index = -1  # 当前工具索引，-1 表示还没有工具
        self.current_tool_id = None
        self.current_tool_name = None
        self.sent_role = False
        self.tool_calls_in_progress = {}  # toolUseId -> {"index": int, "name": str, "arguments": str}

    def _format_sse(self, data: Dict[str, Any]) -> str:
        """格式化 SSE 数据"""
        return f"data: {json.dumps(data, ensure_ascii=False)}\n\n"

    def _create_chunk(
        self,
        delta: Dict[str, Any],
        finish_reason: Optional[str] = None,
        usage: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """创建 OpenAI 格式的 chunk"""
        chunk = {
            "id": self.stream_id,
            "object": "chat.completion.chunk",
            "created": self.created,
            "model": self.model,
            "choices": [{
                "index": 0,
                "delta": delta,
                "finish_reason": finish_reason
            }]
        }
        if usage:
            chunk["usage"] = usage
        return chunk

    async def handle_event(self, event_type: str, payload: Dict[str, Any]) -> AsyncGenerator[str, None]:
        """
        处理事件并转换为 OpenAI 格式
        支持 Amazon Q 和 Claude API 两种事件格式

        Args:
            event_type: 事件类型
            payload: 事件数据

        Yields:
            SSE 格式的字符串
        """
        # 调试日志
        debug_log(f"[OpenAI Stream] Event: {event_type}")
        debug_log(f"[OpenAI Stream] Payload: {json.dumps(payload, ensure_ascii=False, default=str)[:500]}")

        # ==================== Amazon Q 事件格式 ====================
        if event_type == "initial-response":
            # Amazon Q 初始响应
            if not self.sent_role:
                self.sent_role = True
                yield self._format_sse(self._create_chunk({
                    "role": "assistant",
                    "content": ""
                }))

        elif event_type == "assistantResponseEvent":
            # Amazon Q 助手响应内容
            content = payload.get("content", "")
            if content:
                # print(f"[OpenAI Stream] assistantResponseEvent content: {content[:50]}...")
                yield self._format_sse(self._create_chunk({
                    "content": content
                }))

        elif event_type == "toolUseEvent":
            # Amazon Q 工具调用事件
            tool_use_id = payload.get("toolUseId", "")
            tool_name = payload.get("name", "")
            tool_input = payload.get("input", "")
            is_stop = payload.get("stop", False)

            if is_stop:
                # 工具调用结束
                # print(f"[OpenAI Stream] Tool call finished: {tool_use_id}")
                pass
            elif tool_input:
                # 工具参数增量
                if tool_use_id not in self.tool_calls_in_progress:
                    # 新工具调用开始
                    self.tool_index += 1
                    self.tool_calls_in_progress[tool_use_id] = {
                        "index": self.tool_index,
                        "name": tool_name,
                        "arguments": ""
                    }
                    # 发送工具调用开始
                    yield self._format_sse(self._create_chunk({
                        "tool_calls": [{
                            "index": self.tool_index,
                            "id": tool_use_id,
                            "type": "function",
                            "function": {
                                "name": tool_name,
                                "arguments": ""
                            }
                        }]
                    }))

                # 发送参数增量
                tool_info = self.tool_calls_in_progress[tool_use_id]
                tool_info["arguments"] += tool_input
                yield self._format_sse(self._create_chunk({
                    "tool_calls": [{
                        "index": tool_info["index"],
                        "function": {
                            "arguments": tool_input
                        }
                    }]
                }))
            elif tool_name and not tool_input and not is_stop:
                # 工具调用开始（只有 name 和 toolUseId）
                if tool_use_id not in self.tool_calls_in_progress:
                    self.tool_index += 1
                    self.tool_calls_in_progress[tool_use_id] = {
                        "index": self.tool_index,
                        "name": tool_name,
                        "arguments": ""
                    }
                    # 发送工具调用开始
                    yield self._format_sse(self._create_chunk({
                        "tool_calls": [{
                            "index": self.tool_index,
                            "id": tool_use_id,
                            "type": "function",
                            "function": {
                                "name": tool_name,
                                "arguments": ""
                            }
                        }]
                    }))

        elif event_type == "meteringEvent":
            # Amazon Q 使用量统计
            # print(f"[OpenAI Stream] meteringEvent payload: {payload}")
            usage_val = payload.get("usage", 0)
            # usage 可能是数字或字典
            if isinstance(usage_val, dict):
                input_tokens = usage_val.get("inputTokens", 0)
                output_tokens = usage_val.get("outputTokens", 0)
            else:
                # usage 是数字，表示输出 tokens
                input_tokens = 0
                output_tokens = int(usage_val) if usage_val else 0

            self.usage = {
                "input_tokens": input_tokens,
                "output_tokens": output_tokens,
            }
            # 发送最终块
            usage_info = {
                "prompt_tokens": input_tokens,
                "completion_tokens": output_tokens,
                "total_tokens": input_tokens + output_tokens,
            }
            # 如果有工具调用，finish_reason 应该是 "tool_calls"
            finish_reason = "tool_calls" if self.tool_calls_in_progress else "stop"
            yield self._format_sse(self._create_chunk(
                {},
                finish_reason=finish_reason,
                usage=usage_info
            ))

        elif event_type == "assistantResponseEnd":
            # Amazon Q 响应结束
            # 如果还没发送 finish，在这里发送
            if not self.tool_calls_in_progress:
                yield self._format_sse(self._create_chunk(
                    {},
                    finish_reason="stop"
                ))

        # ==================== Claude API 事件格式 ====================
        elif event_type == "message_start":
            # 消息开始，提取 message ID 和初始 usage
            message = payload.get("message", {})
            self.message_id = message.get("id", "")
            usage_data = message.get("usage", {})
            self.usage = {
                "input_tokens": usage_data.get("input_tokens", 0),
                "output_tokens": usage_data.get("output_tokens", 0),
                "cache_creation_input_tokens": usage_data.get("cache_creation_input_tokens", 0),
                "cache_read_input_tokens": usage_data.get("cache_read_input_tokens", 0)
            }
            logger.info(f"Stream started - Message ID: {self.message_id}")
            logger.info(f"Initial usage: input={self.usage['input_tokens']}, cache_creation={self.usage['cache_creation_input_tokens']}, cache_read={self.usage['cache_read_input_tokens']}")

            # 发送初始块（带 role）
            if not self.sent_role:
                self.sent_role = True
                yield self._format_sse(self._create_chunk({
                    "role": "assistant",
                    "content": ""
                }))

        elif event_type == "content_block_start":
            # 内容块开始
            content_block = payload.get("content_block", {})
            block_type = content_block.get("type", "")

            if block_type == "tool_use":
                # 工具调用开始
                tool_id = content_block.get("id", "")
                tool_name = content_block.get("name", "")
                self.current_tool_id = tool_id
                logger.info(f"Tool use started - ID: {tool_id}, Name: {tool_name}, Index: {self.tool_index}")

                # 发送工具调用开始事件
                yield self._format_sse(self._create_chunk({
                    "tool_calls": [{
                        "index": self.tool_index,
                        "id": tool_id,
                        "type": "function",
                        "function": {
                            "name": tool_name,
                            "arguments": ""
                        }
                    }]
                }))

        elif event_type == "content_block_delta":
            # 内容块增量
            delta = payload.get("delta", {})
            delta_type = delta.get("type", "")

            if delta_type == "text_delta":
                # 文本增量
                text = delta.get("text", "")
                if text:
                    yield self._format_sse(self._create_chunk({
                        "content": text
                    }))

            elif delta_type == "input_json_delta":
                # 工具参数增量
                partial_json = delta.get("partial_json", "")
                if partial_json:
                    yield self._format_sse(self._create_chunk({
                        "tool_calls": [{
                            "index": self.tool_index,
                            "function": {
                                "arguments": partial_json
                            }
                        }]
                    }))

            elif delta_type == "thinking_delta":
                # 思考增量 - 可选择是否转发
                # 目前跳过，因为 OpenAI 格式不支持 thinking
                pass

        elif event_type == "content_block_stop":
            # 内容块结束
            block_index = payload.get("index", 0)
            logger.debug(f"Content block {block_index} stopped")
            self.tool_index += 1

        elif event_type == "message_delta":
            # 消息增量（包含停止原因）
            delta = payload.get("delta", {})
            usage_data = payload.get("usage", {})

            if usage_data:
                self.usage["output_tokens"] = usage_data.get("output_tokens", self.usage.get("output_tokens", 0))

            stop_reason = delta.get("stop_reason")
            if stop_reason:
                logger.info(f"Stream ended - Stop reason: {stop_reason}")

                # 构建 usage 信息
                usage_info = None
                if self.usage:
                    usage_info = {
                        "prompt_tokens": self.usage.get("input_tokens", 0),
                        "completion_tokens": self.usage.get("output_tokens", 0),
                        "total_tokens": self.usage.get("input_tokens", 0) + self.usage.get("output_tokens", 0),
                        "prompt_tokens_details": {
                            "cached_tokens": self.usage.get("cache_read_input_tokens", 0),
                            "audio_tokens": 0
                        },
                        "completion_tokens_details": {
                            "reasoning_tokens": 0,
                            "audio_tokens": 0,
                            "accepted_prediction_tokens": 0,
                            "rejected_prediction_tokens": 0
                        }
                    }

                # 发送最终块
                finish_reason = self._convert_stop_reason(stop_reason)
                yield self._format_sse(self._create_chunk(
                    {},
                    finish_reason=finish_reason,
                    usage=usage_info
                ))

        elif event_type == "message_stop":
            # 消息完成
            pass

        elif event_type == "error":
            # 错误事件
            error = payload.get("error", {})
            error_type = error.get("type", "unknown")
            error_message = error.get("message", "Unknown error")
            logger.error(f"Stream error: {error_type} - {error_message}")

    async def finish(self) -> AsyncGenerator[str, None]:
        """完成流式处理"""
        yield "data: [DONE]\n\n"

    def _convert_stop_reason(self, reason: str) -> str:
        """转换停止原因为 OpenAI 格式"""
        mapping = {
            "end_turn": "stop",
            "max_tokens": "length",
            "stop_sequence": "stop",
            "tool_use": "tool_calls"
        }
        return mapping.get(reason, reason)


def convert_claude_response_to_openai(claude_resp: Dict[str, Any], model: str = "unknown") -> Dict[str, Any]:
    """
    将 Claude 非流式响应转换为 OpenAI 格式

    Args:
        claude_resp: Claude API 响应
        model: 模型名称

    Returns:
        OpenAI 格式的响应
    """
    # 提取文本和工具调用
    text_parts = []
    tool_calls = []

    for content in claude_resp.get("content", []):
        content_type = content.get("type", "")

        if content_type == "text":
            text = content.get("text", "")
            if text:
                text_parts.append(text)

        elif content_type == "tool_use":
            tool_id = content.get("id", "")
            tool_name = content.get("name", "")
            tool_input = content.get("input", {})

            tool_calls.append({
                "id": tool_id,
                "type": "function",
                "function": {
                    "name": tool_name,
                    "arguments": json.dumps(tool_input, ensure_ascii=False)
                }
            })

    # 构建响应
    usage = claude_resp.get("usage", {})
    stop_reason = claude_resp.get("stop_reason", "")

    # 确定 finish_reason
    if tool_calls:
        finish_reason = "tool_calls"
    else:
        finish_reason = {
            "end_turn": "stop",
            "max_tokens": "length",
            "stop_sequence": "stop",
            "tool_use": "tool_calls"
        }.get(stop_reason, stop_reason)

    response = {
        "id": claude_resp.get("id", f"chatcmpl-{uuid.uuid4()}"),
        "object": "chat.completion",
        "created": int(time.time()),
        "model": model,
        "choices": [{
            "index": 0,
            "message": {
                "role": "assistant",
                "content": "".join(text_parts) if text_parts else None
            },
            "finish_reason": finish_reason
        }],
        "usage": {
            "prompt_tokens": usage.get("input_tokens", 0),
            "completion_tokens": usage.get("output_tokens", 0),
            "total_tokens": usage.get("input_tokens", 0) + usage.get("output_tokens", 0),
            "prompt_tokens_details": {
                "cached_tokens": usage.get("cache_read_input_tokens", 0),
                "audio_tokens": 0
            },
            "completion_tokens_details": {
                "reasoning_tokens": 0,
                "audio_tokens": 0,
                "accepted_prediction_tokens": 0,
                "rejected_prediction_tokens": 0
            }
        },
        "service_tier": "default"
    }

    # 添加工具调用
    if tool_calls:
        response["choices"][0]["message"]["tool_calls"] = tool_calls
        # 当有 tool_calls 时，content 可以为 None
        if not text_parts:
            response["choices"][0]["message"]["content"] = None

    return response
