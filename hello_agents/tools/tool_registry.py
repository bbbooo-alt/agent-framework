"""工具注册表 - 管理Agent可用的工具"""
from typing import Dict, List, Optional, Any


class Tool:
    """工具基类"""

    def __init__(self, name: str, description: str):
        self.name = name
        self.description = description

    def run(self, params: Dict[str, Any]) -> str:
        """执行工具"""
        raise NotImplementedError


class CalculatorTool(Tool):
    """计算器工具"""

    def __init__(self):
        super().__init__(
            name="calculator",
            description="执行数学计算，如加减乘除、括号运算等"
        )

    def run(self, params: Dict[str, Any]) -> str:
        """执行计算"""
        expression = params.get('expression') or params.get('input') or str(params)
        try:
            # 安全计算：只允许基本数学运算
            allowed_chars = set('0123456789+-*/(). ')
            if not all(c in allowed_chars for c in expression):
                return "错误：表达式包含非法字符"

            result = eval(expression)
            return str(result)
        except Exception as e:
            return f"计算错误: {str(e)}"


class ToolRegistry:
    """工具注册表"""

    def __init__(self):
        self._tools: Dict[str, Tool] = {}

    def register_tool(self, tool: Tool) -> None:
        """注册工具"""
        self._tools[tool.name] = tool

    def unregister(self, tool_name: str) -> None:
        """注销工具"""
        if tool_name in self._tools:
            del self._tools[tool_name]

    def get_tool(self, tool_name: str) -> Optional[Tool]:
        """获取工具"""
        return self._tools.get(tool_name)

    def execute_tool(self, tool_name: str, params: Any) -> str:
        """执行工具"""
        tool = self.get_tool(tool_name)
        if not tool:
            return f"错误：未找到工具 '{tool_name}'"

        if isinstance(params, dict):
            return tool.run(params)
        else:
            return tool.run({'input': str(params)})

    def get_tools_description(self) -> str:
        """获取所有工具描述"""
        if not self._tools:
            return "暂无可用工具"

        descriptions = []
        for name, tool in self._tools.items():
            descriptions.append(f"- {name}: {tool.description}")
        return "\n".join(descriptions)

    def list_tools(self) -> List[str]:
        """列出所有工具名称"""
        return list(self._tools.keys())
