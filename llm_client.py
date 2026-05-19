import os
from openai import OpenAI
from dotenv import load_dotenv
from typing import List, Dict, Optional

# 加载 .env 文件中的环境变量
load_dotenv()


class HelloAgentsLLM:
    """
    为本书 "Hello Agents" 定制的LLM客户端。
    它用于调用任何兼容OpenAI接口的服务，并默认使用流式响应。
    """

    # 厂商配置字典：封装各厂商的默认配置（模型、地址、环境变量名）
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
        model: str = None,
        apiKey: str = None,
        baseUrl: str = None,
        provider: str = "longcat",
        timeout: int = None
    ):
        """
        初始化客户端。

        Args:
            model: 模型名称，如果不传则使用厂商默认模型
            apiKey: API密钥，如果不传则从环境变量读取
            baseUrl: 基础URL，如果不传则使用厂商默认地址
            provider: 厂商名称，支持 longcat/openai/modelscope/aliyun/deepseek
            timeout: 超时时间（秒）
        """
        # 检查是否为已知的厂商
        if provider and provider in self.PROVIDERS:
            config = self.PROVIDERS[provider]

            # 按优先级获取参数：传入参数 > 环境变量 > 厂商默认
            apiKey = apiKey or os.getenv(config["env_key"])
            baseUrl = baseUrl or config["base_url"]
            model = model or config["model"]

            # 保存厂商信息
            self.provider = provider
        else:
            # 未知厂商或自定义配置
            self.provider = provider or "custom"
            # 尝试从环境变量获取（兼容旧版本）
            apiKey = apiKey or os.getenv("LLM_API_KEY")
            baseUrl = baseUrl or os.getenv("LLM_BASE_URL")
            model = model or os.getenv("LLM_MODEL_ID")

        timeout = timeout or int(os.getenv("LLM_TIMEOUT", 60))

        # 验证必要参数
        if not all([model, apiKey, baseUrl]):
            raise ValueError(
                f"模型ID、API密钥和服务地址必须被提供。\n"
                f"当前provider: {self.provider}\n"
                f"请检查：1) 传入参数 2) 环境变量配置"
            )

        self.model = model
        self.client = OpenAI(api_key=apiKey, base_url=baseUrl, timeout=timeout)

    def think(self, messages: List[Dict[str, str]], temperature: float = 0) -> str:
        """
        调用大语言模型进行思考，并返回其响应。
        """
        print(f"🧠 正在调用 {self.model} 模型...")
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                temperature=temperature,
                stream=True,
            )

            # 处理流式响应
            print("✅ 大语言模型响应成功:")
            collected_content = []
            for chunk in response:
                if not chunk.choices:
                    continue
                content = chunk.choices[0].delta.content or ""
                print(content, end="", flush=True)
                collected_content.append(content)
            print()  # 在流式输出结束后换行
            return "".join(collected_content)

        except Exception as e:
            print(f"❌ 调用LLM API时发生错误: {e}")
            return None

    def get_provider_info(self) -> dict:
        """获取当前厂商信息"""
        return {
            "provider": self.provider,
            "model": self.model,
            "base_url": self.client.base_url if hasattr(self, 'client') else None
        }


# --- 客户端使用示例 ---
if __name__ == '__main__':
    try:
        # 使用 provider 参数快速切换厂商
        llmClient = HelloAgentsLLM(provider="longcat")

        print(f"当前配置: {llmClient.get_provider_info()}")

        exampleMessages = [
            {"role": "system", "content": "You are a helpful assistant that writes Python code."},
            {"role": "user", "content": "写一个快速排序算法"}
        ]

        print("\n--- 调用LLM ---")
        responseText = llmClient.think(exampleMessages)
        if responseText:
            print("\n\n--- 完整模型响应 ---")
            print(responseText)

    except ValueError as e:
        print(e)
