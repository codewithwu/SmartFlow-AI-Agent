from typing import Annotated, TypedDict, Sequence

from langchain_core.messages import BaseMessage, HumanMessage, SystemMessage, AIMessage
from langgraph.graph import StateGraph, END
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode

from app.llm.provider import get_chat_model
from app.agent.tools import get_all_tools
from app.memory.short_term import get_session_history


class AgentState(TypedDict):
    messages: Annotated[Sequence[BaseMessage], add_messages]
    rag_context: str
    iteration_count: int


SYSTEM_PROMPT = """你是 SmartFlow 智能助手，一个强大的 AI Agent。你可以通过思考和调用工具来帮助用户完成各种任务。

你拥有以下工具：
- calculator: 计算数学表达式
- web_search: 搜索网络信息
- weather_query: 查询城市天气
- database_query: 查询业务数据库（销售数据、订单信息）

请根据用户的问题，决定是否需要使用工具。如果需要，请调用合适的工具并基于工具返回的结果回答用户。
如果不需要工具，请直接回答用户的问题。

请用中文回答。"""

RAG_CONTEXT_TEMPLATE = """

{context}

请基于以上知识库内容回答用户的问题。如果知识库内容不足以回答，请结合你自己的知识补充。"""

MAX_ITERATIONS = 10


class ReActAgent:
    """ReAct Agent implemented with LangGraph state graph."""

    def __init__(self):
        self.tools = get_all_tools()
        self.llm = get_chat_model().bind_tools(self.tools)
        self.graph = self._build_graph()

    def _build_graph(self) -> StateGraph:
        workflow = StateGraph(AgentState)

        workflow.add_node("agent", self._agent_node)
        workflow.add_node("tools", ToolNode(self.tools))

        workflow.set_entry_point("agent")
        workflow.add_conditional_edges(
            "agent",
            self._should_continue,
            {"continue": "tools", "end": END},
        )
        workflow.add_edge("tools", "agent")

        return workflow.compile()

    def _agent_node(self, state: AgentState) -> dict:
        """LLM reasoning node: decides whether to call tools or give final answer."""
        messages = list(state["messages"])
        iteration = state.get("iteration_count", 0)

        # Build system prompt with optional RAG context
        sys_prompt = SYSTEM_PROMPT
        rag_ctx = state.get("rag_context", "")
        if rag_ctx:
            sys_prompt += RAG_CONTEXT_TEMPLATE.format(context=rag_ctx)

        # Ensure system message is first
        if not messages or not isinstance(messages[0], SystemMessage):
            messages.insert(0, SystemMessage(content=sys_prompt))

        response = self.llm.invoke(messages)
        return {"messages": [response], "iteration_count": iteration + 1}

    def _should_continue(self, state: AgentState) -> str:
        """Decide whether to continue tool calling or end."""
        if state.get("iteration_count", 0) >= MAX_ITERATIONS:
            return "end"

        last_message = state["messages"][-1]
        if hasattr(last_message, "tool_calls") and last_message.tool_calls:
            return "continue"
        return "end"

    def invoke(
        self, query: str, session_id: str = "default", rag_context: str = ""
    ) -> dict:
        """Run the ReAct agent on a user query.

        Returns dict with 'response', 'intermediate_steps', 'sources'.
        """
        # Load conversation history
        history = get_session_history(session_id)
        messages = list(history.messages) + [HumanMessage(content=query)]

        initial_state: AgentState = {
            "messages": messages,
            "rag_context": rag_context,
            "iteration_count": 0,
        }

        result = self.graph.invoke(initial_state)

        # Extract final response
        ai_messages = [m for m in result["messages"] if isinstance(m, AIMessage)]
        final_response = ai_messages[-1].content if ai_messages else "抱歉，我无法处理这个请求。"

        # Extract intermediate steps (tool calls and results)
        intermediate_steps = []
        all_msgs = result["messages"]
        for i, msg in enumerate(all_msgs):
            if isinstance(msg, AIMessage) and hasattr(msg, "tool_calls") and msg.tool_calls:
                for tc in msg.tool_calls:
                    step = {
                        "tool": tc["name"],
                        "tool_input": str(tc["args"]),
                        "output": "",
                    }
                    # Find matching tool response
                    for next_msg in all_msgs[i + 1 :]:
                        if hasattr(next_msg, "name") and next_msg.name == tc["name"]:
                            step["output"] = next_msg.content
                            break
                    intermediate_steps.append(step)

        # Save to conversation history
        history.add_message(HumanMessage(content=query))
        history.add_message(AIMessage(content=final_response))

        return {
            "response": final_response,
            "intermediate_steps": intermediate_steps,
            "sources": [],
        }
