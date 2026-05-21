import os
from openai import OpenAI
from dotenv import load_dotenv
from typing import List, Dict, Optional, Union

# 加载 .env 文件中的环境变量
load_dotenv()


class HelloAgentsLLM:
    """
    为本书 "Hello Agents" 定制的LLM客户端。
    它用于调用任何兼容OpenAI接口的服务，并默认使用流式响应。
    支持自动检测服务商配置。
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
        },
        "zhipu": {
            "base_url": "https://open.bigmodel.cn/api/paas/v4",
            "model": "glm-4",
            "env_key": "ZHIPU_API_KEY"
        },
        "ollama": {
            "base_url": "http://localhost:11434/v1",
            "model": "qwen:0.5b",
            "env_key": "OLLAMA_API_KEY"
        },
        "vllm": {
            "base_url": "http://localhost:8000/v1",
            "model": "Qwen/Qwen1.5-0.5B-Chat",
            "env_key": "VLLM_API_KEY"
        }
    }

    # 域名匹配规则：用于自动检测服务商
    DOMAIN_PATTERNS = {
        "modelscope": ["api-inference.modelscope.cn"],
        "openai": ["api.openai.com"],
        "aliyun": ["dashscope.aliyuncs.com"],
        "deepseek": ["api.deepseek.com"],
        "zhipu": ["open.bigmodel.cn"],
        "longcat": ["api.longcat.chat"]
    }

    # 端口匹配规则：用于自动检测本地服务
    PORT_PATTERNS = {
        "ollama": [":11434"],
        "vllm": [":8000", ":8001"]
    }

    @classmethod
    def auto_detect_provider(cls) -> Optional[str]:
        """
        自动检测服务商配置
        
        优先级：
        1. 检查特定服务商的环境变量
        2. 根据 LLM_BASE_URL 进行域名匹配
        3. 根据 LLM_BASE_URL 进行端口匹配
        
        Returns:
            检测到的服务商名称，如果无法检测则返回 None
        """
        # 优先级1：检查特定服务商的环境变量
        for provider, config in cls.PROVIDERS.items():
            env_key = config.get("env_key")
            if env_key and os.getenv(env_key):
                print(f"🔍 自动检测：发现环境变量 {env_key}，使用服务商 '{provider}'")
                return provider
        
        # 优先级2：根据 base_url 进行判断
        base_url = os.getenv("LLM_BASE_URL")
        if base_url:
            # 域名匹配
            for provider, domains in cls.DOMAIN_PATTERNS.items():
                for domain in domains:
                    if domain in base_url:
                        print(f"🔍 自动检测：根据域名匹配到服务商 '{provider}'")
                        return provider
            
            # 端口匹配
            for provider, ports in cls.PORT_PATTERNS.items():
                for port in ports:
                    if port in base_url:
                        print(f"🔍 自动检测：根据端口 {port} 匹配到本地服务 '{provider}'")
                        return provider
        
        return None

    def __init__(
        self,
        model: str = None,
        apiKey: str = None,
        baseUrl: str = None,
        provider: str = None,
        timeout: int = None
    ):
        """
        初始化客户端。

        Args:
            model: 模型名称，如果不传则使用厂商默认模型
            apiKey: API密钥，如果不传则从环境变量读取
            baseUrl: 基础URL，如果不传则使用厂商默认地址
            provider: 厂商名称，如果不传则自动检测
            timeout: 超时时间（秒）
        """
        # 如果没有指定 provider，尝试自动检测
        if provider is None:
            detected_provider = self.auto_detect_provider()
            if detected_provider:
                provider = detected_provider
            else:
                # 默认使用 longcat
                provider = "longcat"
                print(f"⚠️ 无法自动检测服务商，使用默认: {provider}")
        
        # 检查是否为已知的厂商
        if provider and provider in self.PROVIDERS:
            config = self.PROVIDERS[provider]

            # 按优先级获取参数：传入参数 > 环境变量 > 厂商默认
            apiKey = apiKey or os.getenv(config["env_key"])
            baseUrl = baseUrl or os.getenv("LLM_BASE_URL") or config["base_url"]
            model = model or os.getenv("LLM_MODEL_ID") or config["model"]

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

    def think(
        self,
        messages: Union[List[Dict[str, str]], List["Message"]],
        temperature: float = 0
    ) -> str:
        """
        调用大语言模型进行思考，并返回其响应。

        Args:
            messages: 消息列表，可以是字典列表或 Message 对象列表
            temperature: 温度参数，控制生成文本的随机性
        
        Returns:
            模型的响应文本
        """
        # 转换 Message 对象为字典格式
        formatted_messages = []
        for msg in messages:
            if hasattr(msg, 'to_dict'):
                # Message 对象
                formatted_messages.append(msg.to_dict())
            else:
                # 字典格式
                formatted_messages.append(msg)
        
        print(f"🧠 正在调用 {self.model} 模型...")
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=formatted_messages,
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
