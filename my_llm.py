import os
from typing import Optional
from llm_client import HelloAgentsLLM


class MyLLM(HelloAgentsLLM):
    """
    自定义LLM客户端，继承自HelloAgentsLLM。
    通过provider参数支持多厂商快速切换，无需修改父类代码。
    """

    # 厂商配置字典：封装各厂商的默认配置
    PROVIDERS = {
        "longcat": {
            "base_url": "https://api.longcat.chat/openai",
            "model": "LongCat-Flash-Chat",
            "env_key": "LONGCAT_API_KEY"
        },
        "openai": {
            "base_url": "https://api.openai.com/v1",
            "model": "gpt-3.5-turbo",
            "env_key": "OPENAI_API_KEY"
        },
        "modelscope": {
            "base_url": "https://api-inference.modelscope.cn/v1/",
            "model": "Qwen/Qwen2.5-VL-72B-Instruct",
            "env_key": "MODELSCOPE_API_KEY"
        },
        "aliyun": {
            "base_url": "https://dashscope.aliyuncs.com/compatible-mode/v1",
            "model": "qwen-turbo",
            "env_key": "DASHSCOPE_API_KEY"
        },
        "deepseek": {
            "base_url": "https://api.deepseek.com",
            "model": "deepseek-chat",
            "env_key": "DEEPSEEK_API_KEY"
        }
    }

    def __init__(
        self,
        model: Optional[str] = None,
        apiKey: Optional[str] = None,
        baseUrl: Optional[str] = None,
        provider: Optional[str] = "longcat",
        timeout: Optional[int] = None,
        **kwargs
    ):
        """
        初始化LLM客户端。

        Args:
            model: 模型名称，如果不传则使用厂商默认模型
            apiKey: API密钥，如果不传则从环境变量读取
            baseUrl: 基础URL，如果不传则使用厂商默认地址
            provider: 厂商名称，支持 longcat/openai/modelscope/aliyun/deepseek
            timeout: 超时时间（秒）
        """
        # 检查是否为已知的厂商
        if provider and provider in self.PROVIDERS:
            print(f"🔄 正在使用 {provider} 厂商配置")
            config = self.PROVIDERS[provider]

            # 按优先级获取参数：传入参数 > 环境变量 > 厂商默认
            apiKey = apiKey or os.getenv(config["env_key"])
            baseUrl = baseUrl or config["base_url"]
            model = model or os.getenv("LLM_MODEL_ID") or config["model"]

            # 验证API密钥
            if not apiKey:
                raise ValueError(
                    f"{provider} API key not found. "
                    f"Please set {config['env_key']} environment variable."
                )

            # 保存厂商信息
            self.provider = provider

        else:
            # 未知厂商或自定义配置，使用父类逻辑
            print(f"⚙️ 使用自定义配置 (provider: {provider})")
            self.provider = provider or "custom"

        # 调用父类初始化
        super().__init__(model=model, apiKey=apiKey, baseUrl=baseUrl, timeout=timeout)

    def get_provider_info(self) -> dict:
        """获取当前厂商信息"""
        return {
            "provider": self.provider,
            "model": self.model,
            "base_url": self.client.base_url if hasattr(self, 'client') else None
        }


# --- 使用示例 ---
if __name__ == '__main__':
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

    # 示例4: 自定义配置（不使用预设厂商）
    print("\n" + "=" * 50)
    print("示例4: 自定义配置")
    print("=" * 50)
    try:
        llm = MyLLM(
            provider="custom",
            apiKey=os.getenv("CUSTOM_API_KEY"),
            baseUrl="https://custom-api.example.com/v1",
            model="custom-model"
        )
        print(f"厂商信息: {llm.get_provider_info()}")
    except ValueError as e:
        print(f"⚠️ {e}")

    # 示例5: 覆盖默认模型
    print("\n" + "=" * 50)
    print("示例5: 覆盖默认模型")
    print("=" * 50)
    try:
        llm = MyLLM(provider="longcat", model="LongCat-Pro-Chat")
        print(f"厂商信息: {llm.get_provider_info()}")
    except ValueError as e:
        print(f"⚠️ {e}")
