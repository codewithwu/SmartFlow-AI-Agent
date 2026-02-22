from typing import Annotated, TypedDict, Sequence

from langchain_core.messages import BaseMessage, HumanMessage, SystemMessage, AIMessage
from langchain_core.prompts import ChatPromptTemplate
from langgraph.graph import StateGraph, END
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode
from pydantic import BaseModel, Field

from app.llm.provider import get_chat_model
from app.agent.tools import get_all_tools
from app.memory.short_term import get_session_history


# --- State definition ---

class Plan(BaseModel):
    steps: list[str] = Field(description="Ordered list of steps to complete the task")


class PlanExecuteState(TypedDict):
    messages: Annotated[Sequence[BaseMessage], add_messages]
    plan: list[str]
    current_step: int
    step_results: list[str]
    rag_context: str
    final_response: str


# --- Prompts ---

PLANNER_PROMPT = """你是一个任务规划专家。请根据用户的需求，将其拆解为一系列具体的执行步骤。

要求：
1. 每个步骤应该是具体、可执行的操作
2. 步骤之间应该有合理的顺序
3. 通常 2-5 个步骤即可
4. 最后一步应该是"汇总结果并回复用户"

你有以下工具可以使用：
- calculator: 计算数学表达式
- web_search: 搜索网络信息
- weather_query: 查询城市天气
- database_query: 查询业务数据库（销售数据、订单信息）

请用中文输出步骤列表。"""

EXECUTOR_PROMPT = """你是一个任务执行专家。请根据给定的步骤执行任务。

当前执行的步骤: {current_step}

之前步骤的执行结果:
{previous_results}

请执行当前步骤。如果需要使用工具，请调用合适的工具。"""

SUMMARIZER_PROMPT = """你是 SmartFlow 智能助手。请根据以下任务执行结果，为用户生成一个完整、清晰的回答。

用户原始问题: {query}

执行计划:
{plan}

各步骤执行结果:
{results}

请用中文给出最终回答，要条理清晰、信息完整。"""

MAX_STEPS = 10


class PlanExecuteAgent:
    """Plan-and-Execute Agent: first plans steps, then executes each sequentially."""

    def __init__(self):
        self.tools = get_all_tools()
        self.llm = get_chat_model()
        self.llm_with_tools = self.llm.bind_tools(self.tools)
        self.graph = self._build_graph()

    def _build_graph(self) -> StateGraph:
        workflow = StateGraph(PlanExecuteState)

        workflow.add_node("planner", self._planner_node)
        workflow.add_node("executor", self._executor_node)
        workflow.add_node("executor_tools", ToolNode(self.tools))
        workflow.add_node("summarizer", self._summarizer_node)

        workflow.set_entry_point("planner")
        workflow.add_edge("planner", "executor")
        workflow.add_conditional_edges(
            "executor",
            self._after_executor,
            {"tools": "executor_tools", "next_step": "executor", "summarize": "summarizer"},
        )
        workflow.add_edge("executor_tools", "executor")
        workflow.add_edge("summarizer", END)

        return workflow.compile()

    def _planner_node(self, state: PlanExecuteState) -> dict:
        """Generate a step-by-step plan from the user query."""
        messages = list(state["messages"])
        user_query = ""
        for msg in reversed(messages):
            if isinstance(msg, HumanMessage):
                user_query = msg.content
                break

        prompt = ChatPromptTemplate.from_messages([
            ("system", PLANNER_PROMPT),
            ("human", "请为以下任务制定执行计划:\n{query}"),
        ])

        # Try structured output first, fallback to text parsing
        try:
            planner = prompt | self.llm.with_structured_output(Plan)
            plan = planner.invoke({"query": user_query})
            steps = plan.steps
        except Exception:
            chain = prompt | self.llm
            result = chain.invoke({"query": user_query})
            steps = self._parse_plan_text(result.content)

        if not steps:
            steps = ["直接回答用户的问题"]

        return {"plan": steps, "current_step": 0, "step_results": []}

    def _parse_plan_text(self, text: str) -> list[str]:
        """Parse numbered steps from LLM text output."""
        lines = text.strip().split("\n")
        steps = []
        for line in lines:
            line = line.strip()
            if not line:
                continue
            # Remove leading numbers like "1.", "1)", "- "
            for prefix in [".", ")", "、"]:
                idx = line.find(prefix)
                if idx != -1 and idx < 4 and line[:idx].strip().isdigit():
                    line = line[idx + 1:].strip()
                    break
            if line.startswith("- "):
                line = line[2:].strip()
            if line:
                steps.append(line)
        return steps

    def _executor_node(self, state: PlanExecuteState) -> dict:
        """Execute the current step using LLM with tools."""
        current_idx = state["current_step"]
        plan = state["plan"]

        if current_idx >= len(plan):
            return {}

        current_step_desc = plan[current_idx]
        previous_results = "\n".join(
            f"步骤 {i+1}: {state['plan'][i]}\n结果: {r}"
            for i, r in enumerate(state["step_results"])
        ) or "无"

        # Add RAG context if available
        rag_info = ""
        rag_ctx = state.get("rag_context", "")
        if rag_ctx:
            rag_info = f"\n\n知识库参考:\n{rag_ctx}"

        prompt = EXECUTOR_PROMPT.format(
            current_step=current_step_desc,
            previous_results=previous_results,
        ) + rag_info

        messages = [SystemMessage(content=prompt), HumanMessage(content=f"请执行: {current_step_desc}")]
        response = self.llm_with_tools.invoke(messages)

        return {"messages": [response]}

    def _after_executor(self, state: PlanExecuteState) -> str:
        """Decide what to do after executor runs."""
        last_message = state["messages"][-1]

        # If tool calls are pending, route to tools
        if hasattr(last_message, "tool_calls") and last_message.tool_calls:
            return "tools"

        # Step completed, record result and move to next
        current_idx = state["current_step"]
        plan = state["plan"]
        step_results = list(state["step_results"])

        result_text = last_message.content if isinstance(last_message, AIMessage) else str(last_message)
        step_results.append(result_text)

        next_idx = current_idx + 1
        if next_idx >= len(plan) or next_idx >= MAX_STEPS:
            return "summarize"

        # Update state for next step (via returning to executor)
        state["current_step"] = next_idx
        state["step_results"] = step_results
        return "next_step"

    def _summarizer_node(self, state: PlanExecuteState) -> dict:
        """Summarize all step results into a final response."""
        messages = state["messages"]
        user_query = ""
        for msg in messages:
            if isinstance(msg, HumanMessage):
                user_query = msg.content
                break

        plan_text = "\n".join(f"{i+1}. {s}" for i, s in enumerate(state["plan"]))
        results_text = "\n".join(
            f"步骤 {i+1}: {r}" for i, r in enumerate(state["step_results"])
        )

        # If no step results yet, gather from messages
        if not results_text.strip() or results_text == "":
            ai_msgs = [m.content for m in messages if isinstance(m, AIMessage) and m.content]
            results_text = "\n".join(ai_msgs[-3:]) if ai_msgs else "执行完成"

        prompt = ChatPromptTemplate.from_messages([
            ("system", SUMMARIZER_PROMPT),
            ("human", "请生成最终回答。"),
        ])

        chain = prompt | self.llm
        result = chain.invoke({
            "query": user_query,
            "plan": plan_text,
            "results": results_text,
        })

        return {"final_response": result.content, "messages": [AIMessage(content=result.content)]}

    def invoke(
        self, query: str, session_id: str = "default", rag_context: str = ""
    ) -> dict:
        """Run the Plan-and-Execute agent on a user query."""
        history = get_session_history(session_id)
        messages = list(history.messages) + [HumanMessage(content=query)]

        initial_state: PlanExecuteState = {
            "messages": messages,
            "plan": [],
            "current_step": 0,
            "step_results": [],
            "rag_context": rag_context,
            "final_response": "",
        }

        result = self.graph.invoke(initial_state)

        final_response = result.get("final_response", "")
        if not final_response:
            ai_messages = [m for m in result["messages"] if isinstance(m, AIMessage)]
            final_response = ai_messages[-1].content if ai_messages else "任务执行完成。"

        # Extract intermediate steps
        intermediate_steps = []
        for i, (step, res) in enumerate(zip(result.get("plan", []), result.get("step_results", []))):
            intermediate_steps.append({
                "tool": f"步骤{i+1}",
                "tool_input": step,
                "output": res,
            })

        # Save to conversation history
        history.add_message(HumanMessage(content=query))
        history.add_message(AIMessage(content=final_response))

        return {
            "response": final_response,
            "intermediate_steps": intermediate_steps,
            "sources": [],
            "plan": result.get("plan", []),
        }
