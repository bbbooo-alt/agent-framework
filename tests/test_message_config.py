"""
Message 类和 Config 类测试文件
"""

import sys
import os

# 添加父目录到路径
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from hello_agents import Message, Config, MyLLM


def test_message():
    """测试 Message 类"""
    print("=" * 60)
    print("📌 测试 Message 类")
    print("=" * 60)

    # 创建用户消息
    user_msg = Message(content="你好", role="user")
    print(f"✅ 创建消息: {user_msg}")
    print(f"   to_dict(): {user_msg.to_dict()}")

    # 创建助手消息
    assistant_msg = Message(content="你好！有什么可以帮助你？", role="assistant")
    print(f"✅ 创建消息: {assistant_msg}")
    print(f"   to_dict(): {assistant_msg.to_dict()}")

    # 创建系统消息
    system_msg = Message(content="你是一个 helpful assistant", role="system")
    print(f"✅ 创建消息: {system_msg}")
    print(f"   to_dict(): {system_msg.to_dict()}")

    return True


def test_config():
    """测试 Config 类"""
    print("\n" + "=" * 60)
    print("📌 测试 Config 类")
    print("=" * 60)

    # 使用默认配置
    config = Config()
    print(f"✅ 默认配置: {config.to_dict()}")

    # 从环境变量创建配置
    config_from_env = Config.from_env()
    print(f"✅ 环境变量配置: {config_from_env.to_dict()}")

    return True


def test_message_with_llm():
    """测试使用 Message 对象调用 LLM"""
    print("\n" + "=" * 60)
    print("📌 测试 Message 对象与 LLM 集成")
    print("=" * 60)

    try:
        # 创建消息列表
        messages = [
            Message(content="你是一个 helpful assistant", role="system"),
            Message(content="你好，请介绍一下自己", role="user")
        ]

        print("✅ 创建消息列表:")
        for msg in messages:
            print(f"   {msg}")

        # 使用 Message 对象调用 LLM
        llm = MyLLM(provider="ollama")
        response = llm.think(messages)

        print(f"\n💬 回复: {response}")
        return True
    except Exception as e:
        print(f"❌ {e}")
        return False


if __name__ == '__main__':
    print("\n" + "🧪 " * 30)
    print("Message & Config 类测试套件")
    print("🧪 " * 30 + "\n")

    results = []
    results.append(("Message 类", test_message()))
    results.append(("Config 类", test_config()))
    results.append(("Message + LLM", test_message_with_llm()))

    print("\n" + "=" * 60)
    print("📊 测试结果汇总")
    print("=" * 60)
    for name, passed in results:
        status = "✅ 通过" if passed else "❌ 失败"
        print(f"  {name:<20} {status}")
