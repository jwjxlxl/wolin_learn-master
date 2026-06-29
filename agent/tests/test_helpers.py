"""
Agent 开发范式教学模块 — 工具函数测试

测试 utils/helpers.py 中的共享辅助函数。
"""

import sys
import os
import pytest

# 确保能导入 agent/ 目录下的模块
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from helpers import (
    simulate_search,
    parse_react_response,
    format_conversation_history,
)


class TestSimulateSearch:
    """测试模拟搜索功能"""

    def test_known_query_oscar(self):
        """已知查询：2016年奥斯卡最佳男主角"""
        result = simulate_search("2016年奥斯卡最佳男主角")
        assert "莱昂纳多" in result or "迪卡普里奥" in result

    def test_known_query_python_father(self):
        """已知查询：Python之父"""
        result = simulate_search("Python之父")
        assert "吉多" in result or "Guido" in result

    def test_unknown_query(self):
        """未知查询应返回提示"""
        result = simulate_search("这个关键词不存在xyz123")
        assert "未找到" in result

    def test_case_insensitive(self):
        """搜索应忽略大小写"""
        result1 = simulate_search("2016年奥斯卡最佳男主角")
        result2 = simulate_search("2016年奥斯卡最佳男主角")
        assert result1 == result2


class TestParseReactResponse:
    """测试 ReAct 响应解析功能"""

    def test_parse_action(self):
        """解析带 Action 的响应"""
        text = (
            "Thought: 我需要查询天气信息\n"
            "Action: get_weather\n"
            "Action Input: 杭州"
        )
        result = parse_react_response(text)
        assert result["action"] == "get_weather"
        assert result["action_input"] == "杭州"
        assert "天气" in result["thought"]
        assert result["final_answer"] is None

    def test_parse_final_answer(self):
        """解析带 Final Answer 的响应"""
        text = (
            "Thought: 我已经有了答案\n"
            "Final Answer: 杭州今天晴，15°C"
        )
        result = parse_react_response(text)
        assert result["final_answer"] == "杭州今天晴，15°C"
        assert result["action"] is None

    def test_parse_plain_text(self):
        """解析纯文本响应（无结构化标记）"""
        text = "今天的天气很好。"
        result = parse_react_response(text)
        assert result["action"] is None
        assert result["final_answer"] is None
        assert result["thought"] == text.strip()


class TestFormatConversationHistory:
    """测试对话历史格式化功能"""

    def test_basic_format(self):
        """基本格式化"""
        messages = [
            ("产品经理", "我觉得应该加AI功能"),
            ("工程师", "但成本太高了"),
        ]
        result = format_conversation_history(messages)
        assert "【产品经理】" in result
        assert "【工程师】" in result
        assert "加AI功能" in result

    def test_empty_messages(self):
        """空消息列表"""
        result = format_conversation_history([])
        assert result == ""

    def test_single_message(self):
        """单条消息"""
        messages = [("用户", "你好")]
        result = format_conversation_history(messages)
        assert result == "【用户】你好"
