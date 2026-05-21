"""Agent基类"""
from abc import ABC, abstractmethod
from typing import Optional, Any, List

from ..core.message import Message
from ..core.llm_client import HelloAgentsLLM
from ..core.config import Config


class Agent(ABC):
    """Agent抽象基类
    
    定义智能体的通用行为和属性，所有具体智能体实现必须继承此类。
    通过 ABC 模块强制子类实现 run 方法，保证统一的执行入口。
    """

    def __init__(
        self,
        name: str,
        llm: HelloAgentsLLM,
        system_prompt: Optional[str] = None,
        config: Optional[Config] = None
    ):
        """
        初始化 Agent

        Args:
            name: Agent 名称
            llm: LLM 客户端实例
            system_prompt: 系统提示词
            config: 配置对象，默认使用 Config()
        """
        self.name = name
        self.llm = llm
        self.system_prompt = system_prompt
        self.config = config or Config()
        self._history: List[Message] = []

    @abstractmethod
    def run(self, input_text: str, **kwargs) -> str:
        """
        运行 Agent

        Args:
            input_text: 用户输入文本
            **kwargs: 其他参数

        Returns:
            Agent 的响应文本
        """
        pass

    def add_message(self, message: Message):
        """添加消息到历史记录"""
        self._history.append(message)

    def clear_history(self):
        """清空历史记录"""
        self._history.clear()

    def get_history(self) -> List[Message]:
        """获取历史记录副本"""
        return self._history.copy()

    def __str__(self) -> str:
        return f"Agent(name={self.name}, provider={self.llm.provider})"
