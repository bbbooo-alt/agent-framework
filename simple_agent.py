"""SimpleAgent - 基础对话Agent实现"""
from typing import Optional, Any, List, Dict
from agent import Agent
from message import Message
from llm_client import HelloAgentsLLM
from config import Config


class SimpleAgent(Agent):
    """
    简单的对话Agent实现
    
    提供基础的对话功能，支持系统提示词和历史记录管理。
    可作为更复杂Agent的基类。
    """

    def __init__(
        self,
        name: str,
        llm: HelloAgentsLLM,
        system_prompt: Optional[str] = None,
        config: Optional[Config] = None
    ):
        """
        初始化 SimpleAgent

        Args:
            name: Agent名称
            llm: LLM客户端实例
            system_prompt: 系统提示词
            config: 配置对象
        """
        super().__init__(name, llm, system_prompt, config)
        print(f"✅ {name} 初始化完成")

    def run(self, input_text: str, **kwargs) -> str:
        """
        运行Agent - 实现简单对话逻辑

        Args:
            input_text: 用户输入文本
            **kwargs: 其他参数（如temperature等）

        Returns:
            Agent的响应文本
        """
        print(f"🤖 {self.name} 正在处理: {input_text}")

        # 构建消息列表
        messages = self._build_messages(input_text)

        # 调用LLM
        response = self.llm.think(messages, **kwargs)

        # 更新历史记录
        self.add_message(Message(content=input_text, role="user"))
        self.add_message(Message(content=response, role="assistant"))

        print(f"✅ {self.name} 响应完成")
        return response

    def _build_messages(self, input_text: str) -> List[Dict[str, str]]:
        """
        构建消息列表

        Args:
            input_text: 当前用户输入

        Returns:
            格式化的消息列表
        """
        messages = []

        # 添加系统提示词
        if self.system_prompt:
            messages.append({"role": "system", "content": self.system_prompt})

        # 添加历史记录
        for msg in self._history:
            messages.append(msg.to_dict())

        # 添加当前用户输入
        messages.append({"role": "user", "content": input_text})

        return messages

    def invoke(self, messages: List[Dict[str, str]], **kwargs) -> str:
        """
        直接调用LLM（供子类使用）

        Args:
            messages: 消息列表
            **kwargs: 其他参数

        Returns:
            LLM响应
        """
        return self.llm.think(messages, **kwargs)
