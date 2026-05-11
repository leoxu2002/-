"""
Multi-Tool Agent - 多工具协作 Agent

支持多种工具的 Agent，可以同时使用：
- RAG 知识检索
- Web 搜索
- 代码执行
- 数据分析

适用于需要多源信息整合的复杂任务。
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional

from src.agent.react_agent import ReActAgent, Tool, AgentResult


@dataclass
class ToolResult:
    """工具执行结果"""
    tool_name: str
    success: bool
    content: str
    metadata: Dict[str, Any] = None


class MultiToolAgent(ReActAgent):
    """多工具协作 Agent
    
    在 RAG 基础上扩展更多工具能力：
    
    1. query_knowledge_hub - 知识库检索
    2. web_search - 网络搜索（可选）
    3. code_interpreter - 代码执行（可选）
    4. data_analyzer - 数据分析（可选）
    
    使用场景：
    - 用户问题需要结合内部知识和外部信息
    - 需要执行计算或数据处理
    - 复杂的多步骤推理任务
    
    示例：
        User: "公司最新的产品文档中提到了哪些技术栈？和业界主流方案对比如何？"
        
        Agent 执行流程：
        1. Thought: 需要先检索内部文档
        2. Action: query_knowledge_hub("产品文档 技术栈")
        3. Observation: [内部文档内容...]
        4. Thought: 需要搜索业界主流方案
        5. Action: web_search("主流技术栈对比 2024")
        6. Observation: [网络搜索结果...]
        7. Thought: 整合信息，生成对比分析
        8. Final Answer: [对比分析结果]
    """
    
    def __init__(
        self,
        llm_client: Any,
        enable_web_search: bool = False,
        enable_code_interpreter: bool = False,
        rag_tools: Optional[List[Tool]] = None,
    ):
        tools = rag_tools or self._get_rag_tools()
        
        if enable_web_search:
            tools.extend(self._get_web_search_tools())
        
        if enable_code_interpreter:
            tools.extend(self._get_code_tools())
        
        super().__init__(
            llm_client=llm_client,
            tools=tools,
            max_iterations=8,
            verbose=True,
        )
    
    def _get_rag_tools(self) -> List[Tool]:
        """获取 RAG 工具"""
        from src.mcp_server.tools.query_knowledge_hub import query_knowledge_hub_handler
        from src.mcp_server.tools.list_collections import list_collections_handler
        from src.mcp_server.tools.get_document_summary import get_document_summary_handler
        
        return [
            Tool(
                name="query_knowledge_hub",
                description="搜索知识库，返回相关文档。用于回答基于内部文档的问题。",
                parameters={"query": "string", "top_k": "int"},
                handler=query_knowledge_hub_handler,
            ),
            Tool(
                name="list_collections",
                description="列出所有知识库集合。用于了解可用的数据源。",
                parameters={},
                handler=list_collections_handler,
            ),
            Tool(
                name="get_document_summary",
                description="获取指定文档的摘要。用于快速了解文档内容。",
                parameters={"doc_id": "string"},
                handler=get_document_summary_handler,
            ),
        ]
    
    def _get_web_search_tools(self) -> List[Tool]:
        """获取网络搜索工具（需要配置）"""
        async def web_search_handler(query: str, num_results: int = 5) -> str:
            return f"[Web Search] 搜索 '{query}' 的结果（需要配置搜索 API）"
        
        return [
            Tool(
                name="web_search",
                description="搜索互联网信息。用于获取最新资讯或外部知识。",
                parameters={"query": "string", "num_results": "int"},
                handler=web_search_handler,
            ),
        ]
    
    def _get_code_tools(self) -> List[Tool]:
        """获取代码执行工具（需要沙箱环境）"""
        async def code_interpreter_handler(code: str) -> str:
            return f"[Code Interpreter] 执行代码（需要沙箱环境）：{code[:100]}..."
        
        return [
            Tool(
                name="code_interpreter",
                description="执行 Python 代码。用于计算、数据处理等。",
                parameters={"code": "string"},
                handler=code_interpreter_handler,
            ),
        ]


class ConversationalAgent:
    """多轮对话 Agent
    
    支持上下文记忆的多轮对话 Agent。
    
    特性：
    - 记住对话历史
    - 理解代词引用（"它"、"这个"等）
    - 支持追问和澄清
    
    示例对话：
        User: "如何配置 MCP Server？"
        Agent: [检索并回答配置步骤...]
        
        User: "它支持哪些传输方式？"
        Agent: [理解"它"指 MCP Server，继续回答...]
        
        User: "能给我一个具体例子吗？"
        Agent: [理解上下文，给出示例...]
    """
    
    def __init__(self, base_agent: ReActAgent, max_history: int = 10):
        self.base_agent = base_agent
        self.max_history = max_history
        self.conversation_history: List[Dict[str, str]] = []
    
    def _build_context_prompt(self) -> str:
        """构建上下文提示"""
        if not self.conversation_history:
            return ""
        
        context = "\n\n[对话历史]\n"
        for msg in self.conversation_history[-self.max_history:]:
            role = "用户" if msg["role"] == "user" else "助手"
            context += f"{role}: {msg['content']}\n"
        
        return context
    
    async def chat(self, message: str) -> AgentResult:
        """处理用户消息"""
        self.conversation_history.append({"role": "user", "content": message})
        
        context_prompt = self._build_context_prompt()
        enhanced_message = f"{context_prompt}\n\n[当前问题]\n{message}" if context_prompt else message
        
        result = await self.base_agent.run(enhanced_message)
        
        self.conversation_history.append({"role": "assistant", "content": result.answer})
        
        return result
    
    def clear_history(self):
        """清空对话历史"""
        self.conversation_history = []


async def demo_agent():
    """Agent 使用示例"""
    from src.core.settings import load_settings
    from src.libs.llm.llm_factory import LLMFactory
    
    settings = load_settings()
    llm = LLMFactory.create(settings)
    
    agent = KnowledgeAgent(llm)
    
    result = await agent.run("如何配置 MCP Server？")
    
    print(f"Question: {result.question}")
    print(f"Answer: {result.answer}")
    print(f"Steps: {len(result.steps)}")
    
    return result
