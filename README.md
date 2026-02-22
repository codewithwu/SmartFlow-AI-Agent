# SmartFlow AI Agent - 智能业务流助手

一个基于大语言模型的智能 Agent 系统，支持自然语言理解、任务规划与执行、工具调用、知识库问答（RAG）、多轮对话记忆及多 Agent 协同。

## 功能特性

- **多种 Agent 模式**：支持 ReAct（思考-行动-观察）和 Plan-and-Execute（先规划后执行）两种模式
- **智能任务路由**：Supervisor 自动判断任务复杂度，选择最优 Agent 模式
- **工具调用**：内置计算器、网络搜索、天气查询、数据库查询四种工具
- **知识库问答（RAG）**：支持上传 PDF/TXT 文档，自动分块向量化，检索增强生成
- **对话记忆**：短期对话记忆（滑动窗口）+ 长期语义记忆（ChromaDB）
- **多 Agent 协同**：Planner + Executor 双 Agent 协作完成复杂任务
- **双 LLM 支持**：支持 OpenAI API 和 Ollama 本地模型，通过配置切换
- **Docker 部署**：提供 docker-compose 一键启动

## 技术架构

```
┌─────────────────────────────────────┐
│  Frontend (Streamlit)               │
│  对话界面 / 知识库管理 / Agent选择    │
└──────────────┬──────────────────────┘
               │ HTTP REST API
┌──────────────▼──────────────────────┐
│  API Layer (FastAPI)                │
│  /api/chat  /api/documents  ...     │
└──────────────┬──────────────────────┘
               │
┌──────────────▼──────────────────────┐
│  Supervisor Agent (任务路由)          │
│  ├── ReAct Agent (LangGraph)        │
│  └── Plan-Execute Agent (LangGraph) │
└──────────────┬──────────────────────┘
               │
┌──────────────▼──────────────────────┐
│  Tools / Memory / RAG               │
│  计算器|搜索|天气|数据库              │
│  短期记忆|长期记忆|向量检索           │
└──────────────┬──────────────────────┘
               │
┌──────────────▼──────────────────────┐
│  LLM Provider (OpenAI / Ollama)     │
│  ChromaDB (向量存储)                 │
└─────────────────────────────────────┘
```

## 技术栈

| 组件 | 技术选型 |
|------|---------|
| 后端框架 | FastAPI |
| Agent 框架 | LangChain + LangGraph |
| 大模型 | OpenAI GPT / Ollama (Llama 3 等) |
| 向量数据库 | ChromaDB |
| 前端 | Streamlit |
| 部署 | Docker + docker-compose |

## 项目结构

```
smartflow-ai-agent/
├── app/
│   ├── main.py                    # FastAPI 入口 + API 端点
│   ├── config.py                  # 配置管理 (Pydantic Settings)
│   ├── llm/
│   │   └── provider.py            # LLM 工厂 (OpenAI/Ollama)
│   ├── agent/
│   │   ├── react_agent.py         # ReAct Agent (LangGraph)
│   │   ├── plan_execute_agent.py  # Plan-Execute Agent (LangGraph)
│   │   ├── supervisor.py          # 多 Agent 协调器
│   │   └── tools/                 # 工具集
│   │       ├── calculator.py      # 数学计算器
│   │       ├── web_search.py      # 网络搜索 (模拟)
│   │       ├── weather.py         # 天气查询 (模拟)
│   │       └── database.py        # 数据库查询 (模拟)
│   ├── memory/
│   │   ├── short_term.py          # 短期对话记忆
│   │   └── long_term.py           # 长期语义记忆
│   ├── rag/
│   │   ├── document_processor.py  # 文档加载与分块
│   │   ├── vector_store.py        # ChromaDB 封装
│   │   └── retriever.py           # RAG 检索逻辑
│   └── schemas/
│       └── models.py              # 请求/响应模型
├── frontend/
│   └── streamlit_app.py           # Streamlit UI
├── data/
│   └── sample_docs/               # 示例文档
├── docker/
│   ├── Dockerfile
│   └── docker-compose.yml
├── requirements.txt
├── .env.example
└── README.md
```

## 快速开始

### 方式一：Docker 一键启动（推荐）

1. 克隆项目并配置环境变量：
```bash
git clone <repo-url>
cd smartflow-ai-agent
cp .env.example .env
# 编辑 .env，填入你的 OpenAI API Key（或配置为 Ollama）
```

2. 启动服务：
```bash
cd docker
docker-compose up --build
```

3. 访问：
   - 前端界面：http://localhost:8501
   - API 文档：http://localhost:8000/docs

### 方式二：本地开发启动

1. 安装依赖：
```bash
cd smartflow-ai-agent
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

2. 配置环境变量：
```bash
cp .env.example .env
# 编辑 .env 填入配置
```

3. 启动后端：
```bash
uvicorn app.main:app --reload --port 8000
```

4. 启动前端（新终端窗口）：
```bash
streamlit run frontend/streamlit_app.py
```

5. 访问：
   - 前端界面：http://localhost:8501
   - API 文档：http://localhost:8000/docs

## 配置说明

在 `.env` 文件中配置以下参数：

| 变量 | 说明 | 默认值 |
|------|------|--------|
| `LLM_PROVIDER` | LLM 提供者 (`openai` 或 `ollama`) | `openai` |
| `OPENAI_API_KEY` | OpenAI API 密钥 | - |
| `OPENAI_MODEL` | OpenAI 模型名称 | `gpt-4o-mini` |
| `OLLAMA_BASE_URL` | Ollama 服务地址 | `http://localhost:11434` |
| `OLLAMA_MODEL` | Ollama 模型名称 | `llama3.1` |
| `CHROMA_PERSIST_DIR` | ChromaDB 持久化路径 | `./data/chroma_db` |

### 使用 Ollama 本地模型

1. 安装 Ollama: https://ollama.ai
2. 拉取模型：
```bash
ollama pull llama3.1
ollama pull nomic-embed-text
```
3. 修改 `.env`：
```
LLM_PROVIDER=ollama
OLLAMA_MODEL=llama3.1
```

## API 接口

| 方法 | 路径 | 说明 |
|------|------|------|
| POST | `/api/chat` | 聊天对话（支持选择 Agent 模式） |
| POST | `/api/documents/upload` | 上传文档到知识库 |
| GET | `/api/documents/collections` | 列出所有知识库集合 |
| DELETE | `/api/documents/collections/{name}` | 删除知识库集合 |
| POST | `/api/memory/clear` | 清空会话记忆 |
| GET | `/api/health` | 健康检查 |

### 聊天接口示例

```bash
curl -X POST http://localhost:8000/api/chat \
  -H "Content-Type: application/json" \
  -d '{
    "message": "帮我算一下 (123 + 456) * 2",
    "session_id": "test",
    "agent_mode": "react"
  }'
```

## 示例场景

### 1. 工具调用：天气 + 穿搭推荐
> 用户: "查一下北京的天气，然后根据天气推荐穿搭"
>
> Agent 执行: 调用 weather_query -> 获取北京天气 -> 基于结果推荐穿搭

### 2. 数学计算
> 用户: "帮我算一下 (123 + 456) * 2"
>
> Agent 执行: 调用 calculator -> 返回计算结果

### 3. 业务数据分析
> 用户: "查一下上个月的销售额"
>
> Agent 执行: 调用 database_query -> 返回销售数据

### 4. RAG 知识库问答
> 上传 company_policy.txt 后
>
> 用户: "年假怎么休？"
>
> Agent 执行: 检索知识库 -> 找到年假制度 -> 生成回答

## 项目亮点

1. **架构设计**：模块化分层架构，Agent/Tools/Memory/RAG 完全解耦，易于扩展
2. **LangGraph 状态图**：使用 LangGraph 构建 Agent 状态机，展示对前沿框架的掌握
3. **双 Agent 模式**：ReAct 适合简单任务，Plan-Execute 适合复杂多步任务，Supervisor 智能路由
4. **RAG 完整链路**：文档上传 -> 分块 -> 向量化 -> 检索 -> 增强生成
5. **LLM 灵活切换**：通过配置无缝切换 OpenAI 和 Ollama，兼顾云端和本地部署
6. **工程实践**：类型标注、Pydantic 数据校验、配置管理、Docker 容器化

## License

MIT
