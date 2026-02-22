import uuid
import requests
import streamlit as st

# --- Page config ---
st.set_page_config(
    page_title="SmartFlow AI Agent",
    page_icon="🤖",
    layout="wide",
)

# --- Configuration ---
BACKEND_URL = "http://localhost:8000"


# --- Session state initialization ---
def init_session_state():
    if "session_id" not in st.session_state:
        st.session_state.session_id = str(uuid.uuid4())[:8]
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []
    if "agent_mode" not in st.session_state:
        st.session_state.agent_mode = "auto"


init_session_state()


# --- API helpers ---
def api_chat(message: str, agent_mode: str, use_rag: bool, collection_name: str) -> dict:
    try:
        resp = requests.post(
            f"{BACKEND_URL}/api/chat",
            json={
                "message": message,
                "session_id": st.session_state.session_id,
                "agent_mode": agent_mode,
                "use_rag": use_rag,
                "collection_name": collection_name,
            },
            timeout=120,
        )
        resp.raise_for_status()
        return resp.json()
    except requests.exceptions.ConnectionError:
        return {"response": "无法连接到后端服务，请确保 FastAPI 服务已启动 (默认端口 8000)。", "intermediate_steps": [], "sources": [], "agent_mode": "error"}
    except Exception as e:
        return {"response": f"请求错误: {e}", "intermediate_steps": [], "sources": [], "agent_mode": "error"}


def api_upload_doc(file_bytes: bytes, filename: str, collection_name: str) -> dict:
    try:
        resp = requests.post(
            f"{BACKEND_URL}/api/documents/upload",
            files={"file": (filename, file_bytes)},
            data={"collection_name": collection_name},
            timeout=60,
        )
        resp.raise_for_status()
        return resp.json()
    except Exception as e:
        return {"message": f"上传失败: {e}", "num_chunks": 0, "collection_name": ""}


def api_list_collections() -> list:
    try:
        resp = requests.get(f"{BACKEND_URL}/api/documents/collections", timeout=10)
        resp.raise_for_status()
        return resp.json()
    except Exception:
        return []


def api_delete_collection(name: str) -> bool:
    try:
        resp = requests.delete(f"{BACKEND_URL}/api/documents/collections/{name}", timeout=10)
        return resp.status_code == 200
    except Exception:
        return False


def api_clear_memory(session_id: str) -> str:
    try:
        resp = requests.post(f"{BACKEND_URL}/api/memory/clear", params={"session_id": session_id}, timeout=10)
        resp.raise_for_status()
        return resp.json().get("message", "已清除")
    except Exception as e:
        return f"清除失败: {e}"


def api_health() -> dict:
    try:
        resp = requests.get(f"{BACKEND_URL}/api/health", timeout=5)
        resp.raise_for_status()
        return resp.json()
    except Exception:
        return {"status": "unavailable", "llm_provider": "-", "model": "-"}


# ================================================================
# Sidebar
# ================================================================
with st.sidebar:
    st.title("SmartFlow AI Agent")
    st.caption("智能业务流助手")

    st.divider()

    # Health status
    health = api_health()
    if health["status"] == "ok":
        st.success(f"服务状态: 正常 | {health['llm_provider']} / {health['model']}")
    else:
        st.error("服务状态: 未连接")

    st.divider()

    # Agent mode selector
    st.subheader("Agent 模式")
    mode_options = {"自动 (Auto)": "auto", "ReAct": "react", "Plan-Execute": "plan_execute"}
    selected_mode = st.radio(
        "选择 Agent 执行模式",
        options=list(mode_options.keys()),
        index=0,
        help="Auto: 自动判断任务复杂度选择模式\nReAct: 思考-行动-观察循环\nPlan-Execute: 先规划后执行",
    )
    st.session_state.agent_mode = mode_options[selected_mode]

    st.divider()

    # RAG settings
    st.subheader("知识库设置")
    use_rag = st.toggle("启用知识库 (RAG)", value=False)
    collections = api_list_collections()
    col_names = [c["name"] for c in collections] if collections else ["default"]
    selected_collection = st.selectbox("选择知识库", col_names) if col_names else "default"

    st.divider()

    # Session management
    st.subheader("会话管理")
    st.text(f"Session ID: {st.session_state.session_id}")
    col1, col2 = st.columns(2)
    with col1:
        if st.button("清空对话"):
            st.session_state.chat_history = []
            api_clear_memory(st.session_state.session_id)
            st.rerun()
    with col2:
        if st.button("新建会话"):
            st.session_state.session_id = str(uuid.uuid4())[:8]
            st.session_state.chat_history = []
            st.rerun()

    st.divider()
    st.caption("示例查询:")
    examples = [
        "帮我算一下 (123 + 456) * 2",
        "查一下北京的天气，推荐穿搭",
        "查上个月的销售额",
        "搜索一下人工智能的最新趋势",
    ]
    for ex in examples:
        if st.button(ex, key=f"ex_{ex}", use_container_width=True):
            st.session_state["pending_input"] = ex
            st.rerun()


# ================================================================
# Main Area - Tabs
# ================================================================
tab_chat, tab_kb = st.tabs(["💬 对话", "📚 知识库管理"])

# ======================== Chat Tab ========================
with tab_chat:
    # Display chat history
    for msg in st.session_state.chat_history:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])
            if msg.get("steps"):
                with st.expander("🔍 执行过程", expanded=False):
                    for step in msg["steps"]:
                        st.markdown(f"**工具**: {step.get('tool', '-')}")
                        st.markdown(f"**输入**: {step.get('tool_input', '-')}")
                        st.markdown(f"**输出**: {step.get('output', '-')}")
                        st.divider()
            if msg.get("agent_mode"):
                st.caption(f"Agent 模式: {msg['agent_mode']}")

    # Handle pending input from sidebar example buttons
    pending = st.session_state.pop("pending_input", None)

    # Chat input
    user_input = st.chat_input("输入你的问题...") or pending

    if user_input:
        # Display user message
        st.session_state.chat_history.append({"role": "user", "content": user_input})
        with st.chat_message("user"):
            st.markdown(user_input)

        # Get response
        with st.chat_message("assistant"):
            with st.spinner("思考中..."):
                result = api_chat(
                    user_input,
                    st.session_state.agent_mode,
                    use_rag,
                    selected_collection,
                )

            st.markdown(result["response"])

            steps = result.get("intermediate_steps", [])
            if steps:
                with st.expander("🔍 执行过程", expanded=False):
                    for step in steps:
                        st.markdown(f"**工具**: {step.get('tool', '-')}")
                        st.markdown(f"**输入**: {step.get('tool_input', '-')}")
                        st.markdown(f"**输出**: {step.get('output', '-')}")
                        st.divider()

            agent_mode = result.get("agent_mode", "")
            if agent_mode:
                st.caption(f"Agent 模式: {agent_mode}")

        # Save assistant response to history
        st.session_state.chat_history.append({
            "role": "assistant",
            "content": result["response"],
            "steps": steps,
            "agent_mode": agent_mode,
        })

# ======================== Knowledge Base Tab ========================
with tab_kb:
    st.subheader("上传文档到知识库")
    uploaded_file = st.file_uploader(
        "选择文件 (PDF / TXT / MD)",
        type=["pdf", "txt", "md"],
        help="上传文档后将自动分块并向量化存储",
    )
    kb_collection = st.text_input("知识库名称", value="default")

    if uploaded_file and st.button("上传并索引", type="primary"):
        with st.spinner("处理中..."):
            result = api_upload_doc(
                uploaded_file.getvalue(),
                uploaded_file.name,
                kb_collection,
            )
        if result.get("num_chunks", 0) > 0:
            st.success(f"上传成功! 文件: {uploaded_file.name}, 生成 {result['num_chunks']} 个文档片段")
        else:
            st.error(result.get("message", "上传失败"))

    st.divider()
    st.subheader("已有知识库")
    collections = api_list_collections()
    if collections:
        for col in collections:
            c1, c2, c3 = st.columns([3, 1, 1])
            c1.write(f"**{col['name']}**")
            c2.write(f"{col['count']} 片段")
            if c3.button("删除", key=f"del_{col['name']}"):
                if api_delete_collection(col["name"]):
                    st.success(f"已删除知识库: {col['name']}")
                    st.rerun()
                else:
                    st.error("删除失败")
    else:
        st.info("暂无知识库，请上传文档创建。")
