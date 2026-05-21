"""
Agent 基类测试文件
"""

import sys
import os

# 添加父目录到路径
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from agent import Agent
from message import Message
from my_llm import MyLLM
from config import Config


class SimpleAgent(Agent):
    """简单的 Agent 实现，用于测试"""

    def run(self, input_text: str, **kwargs) -> str:
        """运行 Agent：简单的对话"""
        # 添加用户消息到历史
        user_msg = Message(content=input_text, role="user")
        self.add_message(user_msg)

        # 构建消息列表
        messages = []
        if self.system_prompt:
            messages.append(Message(content=self.system_prompt, role="system"))
        messages.extend(self.get_history())

        # 调用 LLM
        response = self.llm.think([m.to_dict() for m in messages])

        # 添加助手消息到历史
        if response:
            assistant_msg = Message(content=response, role="assistant")
            self.add_message(assistant_msg)

        return response


def test_agent_creation():
    """测试创建 Agent"""
    print("=" * 60)
    print("📌 测试1: 创建 Agent")
    print("=" * 60)

    try:
        llm = MyLLM(provider="ollama")
        agent = SimpleAgent(
            name="TestAgent",
            llm=llm,
            system_prompt="你是一个 helpful assistant",
            config=Config()
        )
        print(f"✅ 创建成功: {agent}")
        return True
    except Exception as e:
        print(f"❌ {e}")
        return False


def test_agent_run():
    """测试运行 Agent"""
    print("\n" + "=" * 60)
    print("📌 测试2: 运行 Agent")
    print("=" * 60)

    try:
        llm = MyLLM(provider="ollama")
        agent = SimpleAgent(
            name="TestAgent",
            llm=llm,
            system_prompt="你是一个 helpful assistant"
        )

        response = agent.run("你好")
        print(f"✅ Agent 响应: {response[:50]}...")
        return True
    except Exception as e:
        print(f"❌ {e}")
        return False


def test_agent_history():
    """测试历史记录管理"""
    print("\n" + "=" * 60)
    print("📌 测试3: 历史记录管理")
    print("=" * 60)

    try:
        llm = MyLLM(provider="ollama")
        agent = SimpleAgent(name="TestAgent", llm=llm)

        # 运行两次对话
        agent.run("你好")
        agent.run("今天天气怎么样")

        # 查看历史
        history = agent.get_history()
        print(f"✅ 历史记录条数: {len(history)}")
        for msg in history:
            print(f"   {msg}")

        # 清空历史
        agent.clear_history()
        print(f"✅ 清空后历史条数: {len(agent.get_history())}")

        return True
    except Exception as e:
        print(f"❌ {e}")
        return False


def test_abstract_class():
    """测试抽象类不能被实例化"""
    print("\n" + "=" * 60)
    print("📌 测试4: 抽象类不能直接实例化")
    print("=" * 60)

    try:
        llm = MyLLM(provider="ollama")
        # 尝试直接实例化 Agent 基类应该失败
        agent = Agent(name="Test", llm=llm)
        print("❌ 应该抛出异常但没有")
        return False
    except TypeError as e:
        print(f"✅ 正确抛出异常: {e}")
        return True
    except Exception as e:
        print(f"❌ 其他异常: {e}")
        return False


if __name__ == '__main__':
    print("\n" + "🧪 " * 30)
    print("Agent 基类测试套件")
    print("🧪 " * 30 + "\n")

    results = []
    results.append(("创建 Agent", test_agent_creation()))
    results.append(("运行 Agent", test_agent_run()))
    results.append(("历史记录管理", test_agent_history()))
    results.append(("抽象类保护", test_abstract_class()))

    print("\n" + "=" * 60)
    print("📊 测试结果汇总")
    print("=" * 60)
    for name, passed in results:
        status = "✅ 通过" if passed else "❌ 失败"
        print(f"  {name:<20} {status}")
