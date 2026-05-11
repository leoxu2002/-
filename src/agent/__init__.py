"""
Agent Module - AI Agent 实现

本模块将 RAG MCP Server 扩展为完整的 Agent 系统。

架构：
┌─────────────────────────────────────────┐
│              Agent Layer                 │
│  ┌─────────────┐  ┌─────────────────┐   │
│  │ ReActAgent  │  │ MultiToolAgent  │   │
│  │ (基础推理)   │  │ (多工具协作)     │   │
│  └──────┬──────┘  └────────┬────────┘   │
│         │                  │            │
│  ┌──────┴──────────────────┴──────┐     │
│  │      ConversationalAgent       │     │
│  │        (多轮对话)               │     │
│  └────────────────────────────────┘     │
└─────────────────────────────────────────┘
                    │
                    ▼
┌─────────────────────────────────────────┐
│           Tool Layer (MCP)               │
│  query_knowledge_hub / list_collections  │
│  get_document_summary / web_search ...   │
└─────────────────────────────────────────┘

使用方式：
    from src.agent import KnowledgeAgent, MultiToolAgent
    
    # 基础知识检索 Agent
    agent = KnowledgeAgent(llm_client)
    result = await agent.run("如何配置 MCP Server？")
    
    # 多工具 Agent
    agent = MultiToolAgent(llm_client, enable_web_search=True)
    result = await agent.run("对比内部文档和业界主流方案")
"""

from src.agent.react_agent import (
    AgentResult,
    AgentState,
    AgentStep,
    KnowledgeAgent,
    ReActAgent,
    Tool,
)
from src.agent.multi_tool_agent import (
    ConversationalAgent,
    MultiToolAgent,
)

__all__ = [
    "ReActAgent",
    "KnowledgeAgent",
    "MultiToolAgent",
    "ConversationalAgent",
    "Tool",
    "AgentResult",
    "AgentState",
    "AgentStep",
]
