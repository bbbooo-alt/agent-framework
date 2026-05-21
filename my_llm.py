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
