# 🤖 AI Agent — LangGraph × Ollama × Streamlit

A **100% free**, production-ready AI Agent with autonomous tool use — no paid API keys required.

## Stack
| Layer | Technology |
|---|---|
| LLM | `llama3` via **Ollama** (local, free) |
| Agent framework | **LangGraph** `StateGraph` agentic loop |
| Web Search | **DuckDuckGo** (`DuckDuckGoSearchRun`) |
| Math | Custom `@tool` calculator |
| UI | **Streamlit** chat interface |

## Quick Start

### 1. Install Ollama & pull llama3
```bash
# Download Ollama from https://ollama.com/download, then:
ollama pull llama3
ollama serve          # keep this running in a separate terminal
```

### 2. Install Python dependencies
```bash
pip install streamlit langgraph langchain langchain-core langchain-ollama langchain-community duckduckgo-search
```

### 3. Run the app
```bash
streamlit run app.py
```

Visit **http://localhost:8501** in your browser.

## Features
- 🔍 **Web search** — DuckDuckGo, no API key needed
- 🧮 **Math calculator** — evaluates complex expressions safely
- 🔄 **Autonomous loop** — agent decides when to call tools, chains multiple calls
- 💬 **Persistent chat history** — synced via `st.session_state`
- 📊 **Live stats** — query & tool-call counters in sidebar
- 🗑 **Clear chat** button to reset session
