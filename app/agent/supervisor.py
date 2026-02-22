from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.prompts import ChatPromptTemplate

from app.llm.provider import get_chat_model
from app.agent.react_agent import ReActAgent
from app.agent.plan_execute_agent import PlanExecuteAgent
from app.rag.retriever import RAGRetriever
from app.rag.vector_store import VectorStoreManager


CLASSIFIER_PROMPT = """你是一个任务分类专家。请根据用户的输入，判断应该使用哪种处理模式。

分类规则：
- "react": 简单的单步任务，如直接回答问题、单个工具调用（查天气、算数学、单次搜索）
- "plan_execute": 复杂的多步任务，需要拆解执行，如多个工具协同、数据分析对比、多步骤流程

请只输出一个词: react 或 plan_execute"""


class SupervisorAgent:
    """Routes user queries to the appropriate agent based on intent classification.

    Supports three modes:
    - "react": Direct to ReAct agent
    - "plan_execute": Direct to Plan-and-Execute agent
    - "auto": LLM classifies the query and routes automatically
    """

    def __init__(self):
        self._react_agent: ReActAgent | None = None
        self._plan_execute_agent: PlanExecuteAgent | None = None
        self._vector_store: VectorStoreManager | None = None
        self._rag_retriever: RAGRetriever | None = None
        self._llm = None

    @property
    def react_agent(self) -> ReActAgent:
        if self._react_agent is None:
            self._react_agent = ReActAgent()
        return self._react_agent

    @property
    def plan_execute_agent(self) -> PlanExecuteAgent:
        if self._plan_execute_agent is None:
            self._plan_execute_agent = PlanExecuteAgent()
        return self._plan_execute_agent

    @property
    def vector_store(self) -> VectorStoreManager:
        if self._vector_store is None:
            self._vector_store = VectorStoreManager()
        return self._vector_store

    @property
    def rag_retriever(self) -> RAGRetriever:
        if self._rag_retriever is None:
            self._rag_retriever = RAGRetriever(self.vector_store)
        return self._rag_retriever

    @property
    def llm(self):
        if self._llm is None:
            self._llm = get_chat_model()
        return self._llm

    def _classify_query(self, query: str) -> str:
        """Use LLM to classify whether the query needs react or plan_execute."""
        try:
            prompt = ChatPromptTemplate.from_messages([
                ("system", CLASSIFIER_PROMPT),
                ("human", "{query}"),
            ])
            chain = prompt | self.llm
            result = chain.invoke({"query": query})
            classification = result.content.strip().lower()
            if "plan" in classification:
                return "plan_execute"
            return "react"
        except Exception:
            return "react"

    def invoke(
        self,
        query: str,
        session_id: str = "default",
        mode: str = "auto",
        use_rag: bool = False,
        collection_name: str = "default",
    ) -> dict:
        """Process a user query through the appropriate agent.

        Args:
            query: User's natural language input
            session_id: Conversation session ID
            mode: "react", "plan_execute", or "auto"
            use_rag: Whether to retrieve from knowledge base
            collection_name: Which RAG collection to search
        """
        # Retrieve RAG context if enabled
        rag_context = ""
        if use_rag:
            rag_context = self.rag_retriever.retrieve_as_context(
                query, collection_name
            )

        # Determine agent mode
        if mode == "auto":
            agent_mode = self._classify_query(query)
        else:
            agent_mode = mode

        # Route to appropriate agent
        if agent_mode == "plan_execute":
            result = self.plan_execute_agent.invoke(
                query, session_id=session_id, rag_context=rag_context
            )
        else:
            result = self.react_agent.invoke(
                query, session_id=session_id, rag_context=rag_context
            )

        result["agent_mode"] = agent_mode

        # Add RAG sources
        if rag_context:
            result["sources"] = [collection_name]

        return result
