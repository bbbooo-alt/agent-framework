"""测试 SimpleAgent 和 MySimpleAgent"""
import sys
import os

# 添加父目录到路径
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dotenv import load_dotenv
from my_llm import MyLLM
from tool_registry import ToolRegistry, CalculatorTool
from my_simple_agent import MySimpleAgent

# 加载环境变量
load_dotenv()


def test_basic_conversation():
    """测试1:基础对话Agent（无工具）"""
    print("=" * 60)
    print("=== 测试1:基础对话 ===")
    print("=" * 60)

    llm = MyLLM(provider="ollama")
    basic_agent = MySimpleAgent(
        name="基础助手",
        llm=llm,
        system_prompt="你是一个友好的AI助手，请用简洁明了的方式回答问题。"
    )

    response = basic_agent.run("你好，请介绍一下自己")
    print(f"\n💬 基础对话响应: {response[:100]}...")
    return True


def test_tool_enhanced():
    """测试2:带工具的Agent"""
    print("\n" + "=" * 60)
    print("=== 测试2:工具增强对话 ===")
    print("=" * 60)

    llm = MyLLM(provider="ollama")
    tool_registry = ToolRegistry()
    calculator = CalculatorTool()
    tool_registry.register_tool(calculator)

    enhanced_agent = MySimpleAgent(
        name="增强助手",
        llm=llm,
        system_prompt="你是一个智能助手，可以使用工具来帮助用户。",
        tool_registry=tool_registry,
        enable_tool_calling=True
    )

    # 测试工具计算
    result = tool_registry.execute_tool("calculator", "15 * 8 + 32")
    print(f"🔧 工具计算结果: {result}")

    response = enhanced_agent.run("请帮我计算 15 * 8 + 32")
    print(f"\n💬 工具增强响应: {response[:100]}...")
    return True


def test_stream_response():
    """测试3:流式响应"""
    print("\n" + "=" * 60)
    print("=== 测试3:流式响应 ===")
    print("=" * 60)

    llm = MyLLM(provider="ollama")
    basic_agent = MySimpleAgent(
        name="流式助手",
        llm=llm,
        system_prompt="你是一个AI助手。"
    )

    print("🌊 流式响应: ", end="")
    for chunk in basic_agent.stream_run("你好"):
        pass  # 内容已在stream_run中实时打印

    return True


def test_dynamic_tools():
    """测试4:动态添加工具"""
    print("\n" + "=" * 60)
    print("=== 测试4:动态工具管理 ===")
    print("=" * 60)

    llm = MyLLM(provider="ollama")
    basic_agent = MySimpleAgent(
        name="动态助手",
        llm=llm,
        system_prompt="你是一个AI助手。"
    )

    print(f"🔧 添加工具前: {basic_agent.has_tools()}")

    # 创建工具注册表并添加工具
    tool_registry = ToolRegistry()
    calculator = CalculatorTool()
    tool_registry.register_tool(calculator)

    # 手动设置工具注册表
    basic_agent.tool_registry = tool_registry
    basic_agent.enable_tool_calling = True

    print(f"🔧 添加工具后: {basic_agent.has_tools()}")
    print(f"🔧 可用工具: {basic_agent.list_tools()}")

    # 查看对话历史
    print(f"\n📜 对话历史: {len(basic_agent.get_history())} 条消息")

    return True


if __name__ == '__main__':
    print("\n" + "🧪 " * 30)
    print("SimpleAgent & MySimpleAgent 测试套件")
    print("🧪 " * 30 + "\n")

    results = []
    results.append(("基础对话", test_basic_conversation()))
    results.append(("工具增强", test_tool_enhanced()))
    results.append(("流式响应", test_stream_response()))
    results.append(("动态工具管理", test_dynamic_tools()))

    print("\n" + "=" * 60)
    print("📊 测试结果汇总")
    print("=" * 60)
    for name, passed in results:
        status = "✅ 通过" if passed else "❌ 失败"
        print(f"  {name:<20} {status}")
