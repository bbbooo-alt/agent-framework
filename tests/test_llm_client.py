"""
LLM 客户端测试文件
测试自动检测机制和各服务商配置
"""

import sys
import os

# 添加父目录到路径
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from hello_agents import HelloAgentsLLM, MyLLM


def test_auto_detection():
    """测试自动检测机制"""
    print("=" * 60)
    print("📌 测试1: 自动检测服务商")
    print("=" * 60)
    try:
        llm = HelloAgentsLLM()  # 不指定 provider
        print(f"✅ 自动检测结果: {llm.get_provider_info()}")
        return True
    except Exception as e:
        print(f"❌ {e}")
        return False


def test_manual_provider():
    """测试手动指定服务商"""
    print("\n" + "=" * 60)
    print("📌 测试2: 手动指定 LongCat")
    print("=" * 60)
    try:
        llm = HelloAgentsLLM(provider="longcat")
        print(f"✅ 配置信息: {llm.get_provider_info()}")
        return True
    except Exception as e:
        print(f"❌ {e}")
        return False


def test_ollama():
    """测试本地 Ollama"""
    print("\n" + "=" * 60)
    print("📌 测试3: 本地 Ollama")
    print("=" * 60)
    try:
        llm = HelloAgentsLLM(provider="ollama")
        print(f"✅ 配置信息: {llm.get_provider_info()}")
        messages = [{"role": "user", "content": "你好"}]
        response = llm.think(messages)
        return response is not None
    except Exception as e:
        print(f"❌ {e}")
        return False


def test_my_llm():
    """测试 MyLLM 子类"""
    print("\n" + "=" * 60)
    print("📌 测试4: MyLLM 子类")
    print("=" * 60)
    try:
        llm = MyLLM(provider="ollama")
        print(f"✅ 配置信息: {llm.get_provider_info()}")
        return True
    except Exception as e:
        print(f"❌ {e}")
        return False


if __name__ == '__main__':
    print("\n" + "🧪 " * 30)
    print("LLM 客户端测试套件")
    print("🧪 " * 30 + "\n")

    results = []
    results.append(("自动检测", test_auto_detection()))
    results.append(("手动指定", test_manual_provider()))
    results.append(("Ollama", test_ollama()))
    results.append(("MyLLM", test_my_llm()))

    print("\n" + "=" * 60)
    print("📊 测试结果汇总")
    print("=" * 60)
    for name, passed in results:
        status = "✅ 通过" if passed else "❌ 失败"
        print(f"  {name:<15} {status}")
