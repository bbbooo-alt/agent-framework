"""
MyLLM 使用示例
展示如何通过 provider 参数快速切换不同厂商
"""

import os
from dotenv import load_dotenv
from my_llm import MyLLM

# 加载环境变量
load_dotenv()


def demo_switch_provider():
    """演示：切换不同厂商"""
    print("=" * 60)
    print("🎯 演示：一行代码切换厂商")
    print("=" * 60)

    # 只需要修改 provider 参数即可切换
    providers = ["longcat", "openai", "modelscope", "aliyun", "deepseek"]

    for provider in providers:
        print(f"\n📡 尝试使用 {provider}...")
        try:
            llm = MyLLM(provider=provider)
            info = llm.get_provider_info()
            print(f"   ✅ 成功 | 模型: {info['model']}")
            print(f"   🌐 地址: {info['base_url']}")
        except ValueError as e:
            print(f"   ⚠️  跳过 | {e}")


def demo_override_model():
    """演示：覆盖默认模型"""
    print("\n" + "=" * 60)
    print("🎯 演示：覆盖默认模型")
    print("=" * 60)

    try:
        # 使用 LongCat，但指定其他模型
        llm = MyLLM(provider="longcat", model="LongCat-Pro-Chat")
        info = llm.get_provider_info()
        print(f"✅ 使用模型: {info['model']}")
    except ValueError as e:
        print(f"⚠️ {e}")


def demo_chat():
    """演示：实际对话"""
    print("\n" + "=" * 60)
    print("🎯 演示：实际对话")
    print("=" * 60)

    messages = [
        {"role": "system", "content": "你是一个乐于助人的AI助手。"},
        {"role": "user", "content": "用一句话介绍自己"}
    ]

    # 使用 LongCat 进行对话
    try:
        print("\n🤖 LongCat 回复:")
        llm = MyLLM(provider="longcat")
        response = llm.think(messages)
        if response:
            print(f"\n💬 完整回复: {response}")
    except Exception as e:
        print(f"❌ 错误: {e}")


def demo_add_new_provider():
    """演示：如何添加新厂商"""
    print("\n" + "=" * 60)
    print("🎯 演示：添加新厂商（扩展）")
    print("=" * 60)

    # 在 my_llm.py 的 PROVIDERS 字典中添加即可：
    code_example = '''
    # 在 MyLLM 类的 PROVIDERS 字典中添加：
    PROVIDERS = {
        # ... 现有厂商 ...
        "new_provider": {
            "base_url": "https://api.new-provider.com/v1",
            "model": "default-model-name",
            "env_key": "NEW_PROVIDER_API_KEY"
        }
    }
    '''
    print(code_example)
    print("✅ 添加后可直接使用: MyLLM(provider='new_provider')")


if __name__ == '__main__':
    # 运行所有演示
    demo_switch_provider()
    demo_override_model()
    demo_chat()
    demo_add_new_provider()

    print("\n" + "=" * 60)
    print("📚 使用总结")
    print("=" * 60)
    print("""
1. 切换厂商: MyLLM(provider="longcat")  # 或 openai/modelscope/aliyun/deepseek
2. 覆盖模型: MyLLM(provider="longcat", model="其他模型")
3. 自定义配置: MyLLM(apiKey="xxx", baseUrl="xxx", model="xxx")
4. 添加厂商: 修改 MyLLM.PROVIDERS 字典
    """)
