"""
Ollama 本地模型测试文件
测试 Qwen 和 Llama 3 模型
"""

import sys
import os

# 添加父目录到路径
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from my_llm import MyLLM


def test_qwen():
    """测试 Qwen 模型（默认）"""
    print("=" * 50)
    print("🧪 测试 Qwen 0.5B 模型")
    print("=" * 50)

    llm = MyLLM(provider="ollama")  # 默认使用 qwen:0.5b
    print(f"模型信息: {llm.get_provider_info()}")

    messages = [{"role": "user", "content": "你好，请介绍一下自己"}]
    response = llm.think(messages)
    print(f"\n💬 回复: {response}")
    return response is not None


def test_llama3():
    """测试 Llama 3 模型"""
    print("\n" + "=" * 50)
    print("🧪 测试 Llama 3.2 1B 模型")
    print("=" * 50)

    llm = MyLLM(provider="ollama", model="llama3.2:1b")
    print(f"模型信息: {llm.get_provider_info()}")

    messages = [{"role": "user", "content": "Hello, introduce yourself"}]
    response = llm.think(messages)
    print(f"\n💬 回复: {response}")
    return response is not None


def list_models():
    """列出可用的 Ollama 模型"""
    print("\n" + "=" * 50)
    print("📋 可用的 Ollama 模型")
    print("=" * 50)

    models = [
        ("qwen:0.5b", "通义千问 0.5B（默认）"),
        ("llama3.2:1b", "Llama 3.2 1B"),
        ("llama3.2:3b", "Llama 3.2 3B"),
        ("gemma2:2b", "Gemma 2 2B"),
        ("phi3:mini", "Phi-3 Mini"),
    ]

    for model, desc in models:
        print(f"  • {model:<15} - {desc}")

    print("\n💡 使用方式:")
    print('  llm = MyLLM(provider="ollama", model="模型名称")')


if __name__ == '__main__':
    import os
    from dotenv import load_dotenv
    load_dotenv()

    # 显示可用模型
    list_models()

    # 测试 Qwen
    try:
        test_qwen()
    except Exception as e:
        print(f"❌ Qwen 测试失败: {e}")

    # 测试 Llama 3
    try:
        test_llama3()
    except Exception as e:
        print(f"❌ Llama 3 测试失败: {e}")
