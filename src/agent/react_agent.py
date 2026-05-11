"""
Agent Client - ReAct 模式的知识检索 Agent

这个模块展示了如何将 RAG MCP Server 扩展为 Agent 系统。
Agent 采用 ReAct（Reasoning + Acting）模式：
  1. Thought: 分析用户问题，决定下一步行动
  2. Action: 调用工具（RAG 检索）
  3. Observation: 观察工具返回结果
  4. 循环直到得出最终答案

使用方式：
  from src.agent.react_agent import KnowledgeAgent
  
  agent = KnowledgeAgent(llm_client, rag_tools)
  result = await agent.run("如何配置 MCP Server？")
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Dict, List, Optional


class AgentState(Enum):
    """Agent 状态"""
    IDLE = "idle"
    THINKING = "thinking"
    ACTING = "acting"
    OBSERVING = "observing"
    FINISHED = "finished"


@dataclass
class Tool:
    """工具定义"""
    name: str
    description: str
    parameters: Dict[str, Any]
    handler: Callable[..., Any]


@dataclass
class AgentStep:
    """Agent 执行步骤"""
    thought: str
    action: Optional[str] = None
    action_input: Optional[Dict[str, Any]] = None
    observation: Optional[str] = None


@dataclass
class AgentResult:
    """Agent 执行结果"""
    question: str
    answer: str
    steps: List[AgentStep] = field(default_factory=list)
    sources: List[Dict[str, Any]] = field(default_factory=list)
    success: bool = True


class ReActAgent:
    """ReAct 模式 Agent
    
    ReAct = Reasoning + Acting
    每一轮循环包含三个步骤：
    1. Thought: 推理当前状态，决定下一步
    2. Action: 执行工具调用
    3. Observation: 观察结果，更新状态
    
    示例对话：
    User: 如何配置 MCP Server？
    
    Thought 1: 用户想了解 MCP Server 配置，我需要检索相关文档
    Action 1: query_knowledge_hub
    Action Input 1: {"query": "MCP Server 配置", "top_k": 5}
    Observation 1: [检索到的文档内容...]
    
    Thought 2: 检索结果包含配置步骤，我可以直接回答
    Answer: MCP Server 配置步骤如下...
    """
    
    def __init__(
        self,
        llm_client: Any,
        tools: List[Tool],
        max_iterations: int = 5,
        verbose: bool = True,
    ):
        self.llm_client = llm_client
        self.tools = {t.name: t for t in tools}
        self.max_iterations = max_iterations
        self.verbose = verbose
        self.state = AgentState.IDLE
        self.history: List[AgentStep] = []
    
    def _build_system_prompt(self) -> str:
        """构建系统提示词"""
        tool_descriptions = "\n".join([
            f"- {name}: {tool.description}"
            for name, tool in self.tools.items()
        ])
        
        return f"""你是一个知识检索 Agent，可以使用以下工具：

{tool_descriptions}

使用以下格式回答问题：

Thought: 分析问题，决定下一步行动
Action: 工具名称
Action Input: 工具参数（JSON 格式）
Observation: 工具返回结果
... (重复 Thought/Action/Action Input/Observation 直到有足够信息)
Thought: 我现在知道答案了
Final Answer: 最终答案

注意：
1. 每次只能调用一个工具
2. 仔细分析 Observation 后再决定下一步
3. 如果检索结果不足，可以换关键词再次检索
4. 回答时引用来源，增强可信度"""
    
    def _build_context(self, question: str) -> str:
        """构建对话上下文"""
        context = f"Question: {question}\n"
        
        for step in self.history:
            context += f"\nThought: {step.thought}\n"
            if step.action:
                context += f"Action: {step.action}\n"
            if step.action_input:
                context += f"Action Input: {step.action_input}\n"
            if step.observation:
                context += f"Observation: {step.observation}\n"
        
        return context
    
    def _parse_response(self, response: str) -> AgentStep:
        """解析 LLM 响应"""
        step = AgentStep(thought="")
        
        lines = response.strip().split("\n")
        for line in lines:
            line = line.strip()
            if line.startswith("Thought:"):
                step.thought = line[8:].strip()
            elif line.startswith("Action:"):
                step.action = line[7:].strip()
            elif line.startswith("Action Input:"):
                import json
                try:
                    step.action_input = json.loads(line[13:].strip())
                except json.JSONDecodeError:
                    step.action_input = {"raw": line[13:].strip()}
            elif line.startswith("Final Answer:"):
                step.observation = f"FINAL: {line[13:].strip()}"
        
        return step
    
    async def _call_tool(self, tool_name: str, tool_input: Dict[str, Any]) -> str:
        """调用工具"""
        if tool_name not in self.tools:
            return f"Error: Unknown tool '{tool_name}'"
        
        tool = self.tools[tool_name]
        try:
            if asyncio.iscoroutinefunction(tool.handler):
                result = await tool.handler(**tool_input)
            else:
                result = tool.handler(**tool_input)
            
            if hasattr(result, 'content'):
                return str(result.content)
            return str(result)
        except Exception as e:
            return f"Error calling {tool_name}: {str(e)}"
    
    async def run(self, question: str) -> AgentResult:
        """执行 Agent 推理循环"""
        self.state = AgentState.THINKING
        self.history = []
        
        result = AgentResult(question=question, answer="")
        
        for iteration in range(self.max_iterations):
            self.state = AgentState.THINKING
            
            context = self._build_context(question)
            
            response = await self.llm_client.chat(
                messages=[
                    {"role": "system", "content": self._build_system_prompt()},
                    {"role": "user", "content": context + "\nThought:"},
                ]
            )
            
            step = self._parse_response(response)
            self.history.append(step)
            
            if self.verbose:
                print(f"\n[Iteration {iteration + 1}]")
                print(f"Thought: {step.thought}")
            
            if step.observation and step.observation.startswith("FINAL:"):
                result.answer = step.observation[6:].strip()
                result.steps = self.history
                result.success = True
                self.state = AgentState.FINISHED
                return result
            
            if step.action and step.action_input:
                self.state = AgentState.ACTING
                
                if self.verbose:
                    print(f"Action: {step.action}")
                    print(f"Action Input: {step.action_input}")
                
                observation = await self._call_tool(step.action, step.action_input)
                step.observation = observation
                
                if self.verbose:
                    print(f"Observation: {observation[:200]}...")
        
        result.answer = "抱歉，我无法在限定步骤内完成推理。请尝试简化问题。"
        result.steps = self.history
        result.success = False
        self.state = AgentState.FINISHED
        return result


class KnowledgeAgent(ReActAgent):
    """知识检索专用 Agent
    
    预配置了 RAG 相关工具，开箱即用。
    
    使用示例：
        from src.libs.llm.llm_factory import LLMFactory
        from src.mcp_server.tools.query_knowledge_hub import query_knowledge_hub_handler
        
        llm = LLMFactory.create(settings)
        tools = [
            Tool(
                name="query_knowledge_hub",
                description="搜索知识库，返回相关文档",
                parameters={"query": "string", "top_k": "int"},
                handler=query_knowledge_hub_handler,
            ),
            Tool(
                name="list_collections",
                description="列出所有知识库集合",
                parameters={},
                handler=list_collections_handler,
            ),
        ]
        
        agent = KnowledgeAgent(llm, tools)
        result = await agent.run("如何配置 MCP Server？")
    """
    
    def __init__(self, llm_client: Any, rag_tools: Optional[List[Tool]] = None):
        if rag_tools is None:
            rag_tools = self._get_default_tools()
        
        super().__init__(
            llm_client=llm_client,
            tools=rag_tools,
            max_iterations=5,
            verbose=True,
        )
    
    def _get_default_tools(self) -> List[Tool]:
        """获取默认 RAG 工具"""
        from src.mcp_server.tools.query_knowledge_hub import query_knowledge_hub_handler
        from src.mcp_server.tools.list_collections import list_collections_handler
        
        return [
            Tool(
                name="query_knowledge_hub",
                description="搜索知识库，返回与查询相关的文档片段。参数：query(查询内容), top_k(返回数量), collection(集合名称)",
                parameters={
                    "query": {"type": "string", "description": "搜索查询"},
                    "top_k": {"type": "integer", "default": 5},
                    "collection": {"type": "string", "default": "default"},
                },
                handler=query_knowledge_hub_handler,
            ),
            Tool(
                name="list_collections",
                description="列出所有可用的知识库集合",
                parameters={},
                handler=list_collections_handler,
            ),
        ]
