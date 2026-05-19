from llm_client import HelloAgentsLLM


class MyLLM(HelloAgentsLLM):
    """
    自定义LLM客户端，继承自HelloAgentsLLM。
    完全复用父类的 PROVIDERS 配置，无需重复定义。

    如果需要添加新的厂商，可以在这里扩展 PROVIDERS 字典：

    Example:
        class MyLLM(HelloAgentsLLM):
            # 先复制父类的配置
            PROVIDERS = HelloAgentsLLM.PROVIDERS.copy()
            # 添加新厂商
            PROVIDERS["new_provider"] = {
                "base_url": "https://api.new-provider.com/v1",
                "model": "default-model",
                "env_key": "NEW_PROVIDER_API_KEY"
            }
    """
    pass


# --- 使用示例 ---
if __name__ == '__main__':
    import os
    from dotenv import load_dotenv
    load_dotenv()

    # 示例1: 使用 LongCat（默认）
    print("=" * 50)
    print("示例1: 使用 LongCat")
    print("=" * 50)
    try:
        llm = MyLLM(provider="longcat")
        print(f"厂商信息: {llm.get_provider_info()}")
    except ValueError as e:
        print(f"⚠️ {e}")

    # 示例2: 使用 ModelScope
    print("\n" + "=" * 50)
    print("示例2: 使用 ModelScope")
    print("=" * 50)
    try:
        llm = MyLLM(provider="modelscope")
        print(f"厂商信息: {llm.get_provider_info()}")
    except ValueError as e:
        print(f"⚠️ {e}")

    # 示例3: 使用 OpenAI
    print("\n" + "=" * 50)
    print("示例3: 使用 OpenAI")
    print("=" * 50)
    try:
        llm = MyLLM(provider="openai")
        print(f"厂商信息: {llm.get_provider_info()}")
    except ValueError as e:
        print(f"⚠️ {e}")

    # 示例4: 覆盖默认模型
    print("\n" + "=" * 50)
    print("示例4: 覆盖默认模型")
    print("=" * 50)
    try:
        llm = MyLLM(provider="longcat", model="LongCat-Pro-Chat")
        print(f"厂商信息: {llm.get_provider_info()}")
    except ValueError as e:
        print(f"⚠️ {e}")
