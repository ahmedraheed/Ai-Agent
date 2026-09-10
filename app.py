"""
AI Agent powered by Groq (Cloud) / Ollama (Local) + Streamlit — manual ReAct loop.
100% free — supports ultra-fast cloud inference or local offline models.

Run:
    streamlit run app.py
"""

import os
import re
import math
from dotenv import load_dotenv

load_dotenv()

import streamlit as st
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langchain_ollama import ChatOllama
from langchain_groq import ChatGroq
from langchain_core.tools import tool
try:
    from duckduckgo_search import DDGS
except ImportError:
    try:
        from ddgs import DDGS
    except ImportError:
        DDGS = None

# ─────────────────────────────────────────────
#  Page configuration (must be first Streamlit call)
# ─────────────────────────────────────────────
st.set_page_config(
    page_title="AI Agent — ReAct × Groq & Ollama",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─────────────────────────────────────────────
#  Global CSS — dark glassmorphism theme
# ─────────────────────────────────────────────
st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&family=JetBrains+Mono:wght@400;500&display=swap');

    /* ── Root variables ── */
    :root {
        --bg-primary:   #ffffff;
        --bg-secondary: #f4f6fb;
        --bg-card:      rgba(0,0,0,0.04);
        --border:       rgba(0,0,0,0.10);
        --accent:       #6c63ff;
        --accent-2:     #00b894;
        --accent-glow:  rgba(108,99,255,0.20);
        --text-primary: #1a1d2e;
        --text-muted:   #5a6080;
        --user-bubble:  linear-gradient(135deg, #6c63ff 0%, #4facfe 100%);
        --ai-bubble:    #f0f2ff;
        --tool-bubble:  rgba(0,184,148,0.08);
        --danger:       #e84393;
        --success:      #00b894;
        --warning:      #f39c12;
    }

    /* ── App background ── */
    .stApp, .stApp > div { background: #ffffff !important; }
    html, body { font-family: 'Inter', sans-serif !important; background: #ffffff !important; }


    /* ── Hide Streamlit chrome ── */
    #MainMenu, footer { visibility: hidden; }
    .block-container { padding-top: 1.5rem !important; max-width: 900px; }

    /* ── Sidebar ── */
    [data-testid="stSidebar"] {
        background: #f4f6fb !important;
        border-right: 2px solid #e0e4f0 !important;
    }
    [data-testid="stSidebar"] * { color: #1a1d2e !important; }

    /* ── Sidebar cards ── */
    .status-card {
        background: var(--bg-card);
        border: 1px solid var(--border);
        border-radius: 12px;
        padding: 14px 16px;
        margin-bottom: 12px;
        backdrop-filter: blur(8px);
    }
    .status-card h4 { margin: 0 0 8px; font-size: 0.75rem; letter-spacing: 0.1em;
                      text-transform: uppercase; color: var(--text-muted) !important; }
    .status-row { display: flex; align-items: center; gap: 8px; margin: 4px 0;
                  font-size: 0.88rem; }
    .dot { width: 8px; height: 8px; border-radius: 50%; flex-shrink: 0; }
    .dot-green { background: var(--success); box-shadow: 0 0 8px var(--success); }
    .dot-blue  { background: var(--accent);  box-shadow: 0 0 8px var(--accent);  }
    .dot-yellow{ background: var(--warning); box-shadow: 0 0 8px var(--warning); }

    /* ── Chat header ── */
    .chat-header {
        text-align: center;
        padding: 2rem 0 1.5rem;
        margin-bottom: 0.5rem;
    }
    .chat-header h1 {
        font-size: 2rem; font-weight: 700; margin: 0;
        background: linear-gradient(135deg, #6c63ff, #00d4aa);
        -webkit-background-clip: text; -webkit-text-fill-color: transparent;
    }
    .chat-header p { color: var(--text-muted); font-size: 0.9rem; margin: 6px 0 0; }

    /* ── Message bubble entrance animation ── */
    @keyframes bubbleIn {
        from { opacity: 0; transform: translateY(10px); }
        to   { opacity: 1; transform: translateY(0); }
    }

    /* ── Base bubble ── */
    [data-testid="stChatMessage"] {
        border-radius: 18px !important;
        margin-bottom: 14px !important;
        padding: 12px 16px !important;
        animation: bubbleIn 0.25s ease-out both;
    }

    /* ── User messages ── */
    [data-testid="stChatMessage"]:has([data-testid="chatAvatarIcon-user"]) {
        background: linear-gradient(135deg, rgba(108,99,255,0.30) 0%, rgba(79,172,254,0.20) 100%) !important;
        border: 1px solid rgba(108,99,255,0.50) !important;
        box-shadow: 0 0 20px rgba(108,99,255,0.15) !important;
    }
    [data-testid="stChatMessage"]:has([data-testid="chatAvatarIcon-user"]) p,
    [data-testid="stChatMessage"]:has([data-testid="chatAvatarIcon-user"]) div,
    [data-testid="stChatMessage"]:has([data-testid="chatAvatarIcon-user"]) span {
        color: #ffffff !important;
        font-size: 0.97rem !important;
        line-height: 1.6 !important;
    }

    /* ── AI messages — light card on white bg ── */
    [data-testid="stChatMessage"]:has([data-testid="chatAvatarIcon-assistant"]) {
        background: #f0f2ff !important;
        border: 1px solid #d4d8f7 !important;
        box-shadow: 0 2px 12px rgba(108,99,255,0.08) !important;
    }
    [data-testid="stChatMessage"]:has([data-testid="chatAvatarIcon-assistant"]) p,
    [data-testid="stChatMessage"]:has([data-testid="chatAvatarIcon-assistant"]) span,
    [data-testid="stChatMessage"]:has([data-testid="chatAvatarIcon-assistant"]) li,
    [data-testid="stChatMessage"]:has([data-testid="chatAvatarIcon-assistant"]) strong,
    [data-testid="stChatMessage"]:has([data-testid="chatAvatarIcon-assistant"]) em,
    [data-testid="stChatMessage"]:has([data-testid="chatAvatarIcon-assistant"]) label,
    [data-testid="stChatMessage"]:has([data-testid="chatAvatarIcon-assistant"]) .stMarkdown {
        color: #1a1d2e !important;
        font-size: 1rem !important;
        line-height: 1.75 !important;
    }

    /* ── Input bar ── */
    [data-testid="stChatInput"] textarea {
        background: rgba(255,255,255,0.05) !important;
        border: 1px solid var(--border) !important;
        border-radius: 12px !important;
        color: var(--text-primary) !important;
        font-family: 'Inter', sans-serif !important;
        font-size: 0.95rem !important;
        transition: border-color 0.2s;
    }
    [data-testid="stChatInput"] textarea:focus {
        border-color: var(--accent) !important;
        box-shadow: 0 0 0 3px var(--accent-glow) !important;
    }

    /* ── Tool call pills ── */
    .tool-badge {
        display: inline-block;
        background: rgba(0,212,170,0.12);
        border: 1px solid rgba(0,212,170,0.3);
        color: var(--accent-2);
        border-radius: 20px;
        padding: 3px 10px;
        font-size: 0.78rem;
        font-family: 'JetBrains Mono', monospace;
        margin-right: 6px;
        margin-bottom: 4px;
    }

    /* ── Spinner ── */
    [data-testid="stSpinner"] > div { color: var(--accent) !important; }

    /* ── Buttons ── */
    .stButton > button {
        background: linear-gradient(135deg, var(--danger), #ff8a80) !important;
        color: white !important;
        border: none !important;
        border-radius: 10px !important;
        font-weight: 600 !important;
        font-size: 0.88rem !important;
        padding: 0.5rem 1.2rem !important;
        transition: opacity 0.2s, transform 0.1s !important;
        width: 100%;
    }
    .stButton > button:hover { opacity: 0.85 !important; transform: translateY(-1px) !important; }

    /* ── Expander (tool output) ── */
    [data-testid="stExpander"] {
        background: var(--tool-bubble) !important;
        border: 1px solid rgba(0,212,170,0.2) !important;
        border-radius: 10px !important;
    }
    [data-testid="stExpander"] summary { color: var(--accent-2) !important; font-size: 0.85rem !important; }

    /* ── Stat metrics ── */
    [data-testid="stMetric"] {
        background: var(--bg-card);
        border: 1px solid var(--border);
        border-radius: 10px;
        padding: 10px 14px !important;
    }
    [data-testid="stMetricLabel"] { color: var(--text-muted) !important; font-size: 0.75rem !important; }
    [data-testid="stMetricValue"] { color: var(--text-primary) !important; font-size: 1.3rem !important; }

    /* ── Divider ── */
    hr { border-color: var(--border) !important; }

    /* ── Scrollbar ── */
    ::-webkit-scrollbar { width: 5px; }
    ::-webkit-scrollbar-track { background: transparent; }
    ::-webkit-scrollbar-thumb { background: rgba(255,255,255,0.12); border-radius: 3px; }
    ::-webkit-scrollbar-thumb:hover { background: var(--accent); }
    </style>
    """,
    unsafe_allow_html=True,
)



# ─────────────────────────────────────────────
#  Tools
# ─────────────────────────────────────────────

@tool
def calculator(expression: str) -> str:
    """
    Evaluate a mathematical expression safely.
    Supports: +, -, *, /, **, %, sqrt, sin, cos, tan, log, pi, e, abs, round.

    Args:
        expression: A math expression string, e.g. "sqrt(144) + 2**8"

    Returns:
        The numeric result as a string.
    """
    safe_env = {
        "__builtins__": {},
        "abs": abs,
        "round": round,
        "sqrt": math.sqrt,
        "sin": math.sin,
        "cos": math.cos,
        "tan": math.tan,
        "log": math.log,
        "log10": math.log10,
        "log2": math.log2,
        "exp": math.exp,
        "ceil": math.ceil,
        "floor": math.floor,
        "pi": math.pi,
        "e": math.e,
        "inf": math.inf,
        "pow": pow,
        "factorial": math.factorial,
    }
    # Strip potentially dangerous tokens
    forbidden = re.compile(r"(import|exec|eval|open|__|\bos\b|\bsys\b)", re.I)
    if forbidden.search(expression):
        return "Error: forbidden tokens in expression."
    try:
        result = eval(expression, safe_env)  # noqa: S307
        return str(result)
    except Exception as exc:
        return f"Error evaluating expression: {exc}"


@tool
def duckduckgo_search(query: str) -> str:
    """
    Search the web using DuckDuckGo for up-to-date information.

    Args:
        query: The search query string.

    Returns:
        A string with the top search results including titles, URLs, and snippets.
    """
    if DDGS is None:
        return "Search error: duckduckgo-search package is not available."
    try:
        with DDGS() as ddgs:
            results = list(ddgs.text(query, max_results=5))
        if not results:
            return "No results found for your query."
        formatted = []
        for i, r in enumerate(results, 1):
            title = r.get("title", "No title")
            href  = r.get("href", "")
            body  = r.get("body", "No snippet")
            formatted.append(f"{i}. **{title}**\n   {href}\n   {body}")
        return "\n\n".join(formatted)
    except Exception as exc:
        return f"Search error: {exc}"


TOOLS = [duckduckgo_search, calculator]


# ─────────────────────────────────────────────
#  Build LangGraph (cached per session)
# ─────────────────────────────────────────────

# ── ReAct system prompt (plain text — no native tool-call API needed) ──
REACT_SYSTEM_PROMPT = """You are a helpful AI assistant with access to two tools:

1. duckduckgo_search  — search the web for current information
2. calculator         — evaluate math expressions (supports +, -, *, /, **, sqrt, sin, cos, tan, log, pi, e, abs, round, factorial)

When you need a tool, respond EXACTLY in this format (nothing else on those lines):
Thought: <your reasoning>
Action: <tool name>
Action Input: <tool input>

When you have the final answer, respond EXACTLY:
Thought: <your reasoning>
Final Answer: <your answer>

If no tool is needed, go straight to Final Answer.
Do NOT include markdown fences or extra text around the Action lines."""


@st.cache_resource(show_spinner=False)
def build_groq_llm(model_name: str, api_key: str):
    """Return a cached ChatGroq instance."""
    return ChatGroq(
        model=model_name,
        groq_api_key=api_key,
        temperature=0.2,
        max_tokens=800,
    )


@st.cache_resource(show_spinner=False)
def build_ollama_llm(model_name: str):
    """Return a cached ChatOllama instance."""
    return ChatOllama(model=model_name, temperature=0.3)


# ─────────────────────────────────────────────
#  Session state initialisation
# ─────────────────────────────────────────────
if "messages" not in st.session_state:
    st.session_state.messages = []          # list[dict]: role + content + optional meta
if "total_queries" not in st.session_state:
    st.session_state.total_queries = 0
if "tool_calls_made" not in st.session_state:
    st.session_state.tool_calls_made = 0
if "provider" not in st.session_state:
    st.session_state.provider = "Groq Cloud"
if "groq_model" not in st.session_state:
    st.session_state.groq_model = "openai/gpt-oss-120b"
if "ollama_model" not in st.session_state:
    st.session_state.ollama_model = "qwen2.5:3b"


# ─────────────────────────────────────────────
#  Sidebar
# ─────────────────────────────────────────────
with st.sidebar:
    st.markdown(
        """
        <div style="text-align:center; padding: 0.8rem 0 1rem;">
            <div style="font-size:2.8rem;">🤖</div>
            <div style="font-size:1.15rem; font-weight:700; color:#1a1d2e;">AI Agent</div>
            <div style="font-size:0.78rem; color:#5a6080; margin-top:2px;">ReAct Loop · Groq & Ollama Dual-Engine</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown("---")

    # ── Engine Selector ──
    st.markdown("##### ⚙️ Engine Settings")
    provider_choice = st.radio(
        "Choose AI Engine:",
        ["⚡ Groq Cloud (Ultra Fast)", "💻 Ollama (Local Offline)"],
        index=0,
        help="Groq runs on ultra-fast cloud LPUs. Ollama runs offline on your machine.",
    )
    is_groq = "Groq" in provider_choice
    st.session_state.provider = "Groq Cloud" if is_groq else "Ollama Local"

    if is_groq:
        groq_models = [
            "openai/gpt-oss-120b",
            "openai/gpt-oss-20b",
            "qwen/qwen3.6-27b",
        ]
        selected_model = st.selectbox(
            "Groq Model:",
            groq_models,
            index=0,
            help="openai/gpt-oss-120b is a massive 120B model with high accuracy.",
        )
        st.session_state.groq_model = selected_model

        default_key = os.environ.get("GROQ_API_KEY", "")
        # Also check st.secrets if running on Streamlit Cloud
        if not default_key:
            try:
                default_key = st.secrets.get("GROQ_API_KEY", "")
            except Exception:
                pass

        api_key_input = st.text_input(
            "Groq API Key:",
            value=default_key,
            type="password",
            help="100% Free API key from console.groq.com",
        )
        st.session_state.groq_api_key = api_key_input
        if not api_key_input:
            st.warning("⚠️ Enter a Groq API Key to proceed.")
            st.caption("🔑 [Get a free Groq API Key](https://console.groq.com/keys)")
    else:
        ollama_model_input = st.text_input(
            "Ollama Model Name:",
            value=st.session_state.get("ollama_model", "qwen2.5:3b"),
            help="Model pulled locally in Ollama",
        )
        st.session_state.ollama_model = ollama_model_input
        st.caption("💡 Run `ollama serve` in terminal.")

    st.markdown("---")

    # ── Status indicators ──
    active_engine_name = "Groq Cloud" if is_groq else "Ollama Local"
    active_model_name = st.session_state.groq_model if is_groq else st.session_state.ollama_model

    st.markdown(
        f"""
        <div class="status-card">
            <h4>⚡ System Status</h4>
            <div class="status-row"><div class="dot dot-green"></div> Engine: <b>{active_engine_name}</b></div>
            <div class="status-row"><div class="dot dot-blue"></div> Model: <code>{active_model_name}</code></div>
            <div class="status-row"><div class="dot dot-green"></div> DuckDuckGo search</div>
            <div class="status-row"><div class="dot dot-blue"></div> Math calculator</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # ── Stats ──
    col1, col2 = st.columns(2)
    col1.metric("Queries", st.session_state.total_queries)
    col2.metric("Tool Calls", st.session_state.tool_calls_made)

    st.markdown("<br>", unsafe_allow_html=True)

    # ── Available tools info ──
    st.markdown(
        """
        <div class="status-card">
            <h4>🛠 Available Tools</h4>
            <div class="status-row"><span class="tool-badge">🔍 web_search</span></div>
            <div class="status-row"><span class="tool-badge">🧮 calculator</span></div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown("<br>", unsafe_allow_html=True)

    # ── Tip box ──
    st.markdown(
        """
        <div class="status-card" style="border-color:rgba(108,99,255,0.3);">
            <h4>💡 Try asking…</h4>
            <div style="font-size:0.82rem; color:#5a6080; line-height:1.6;">
                • "What is 2 to the power of 32?"<br>
                • "Search for latest news on AI agents"<br>
                • "What is sqrt(7921) + 45 * 12?"<br>
                • "Who won the 2024 ICC T20 World Cup?"
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown("<br>", unsafe_allow_html=True)

    if st.button("🗑  Clear Chat History", key="clear_btn"):
        st.session_state.messages = []
        st.session_state.total_queries = 0
        st.session_state.tool_calls_made = 0
        st.rerun()


# ─────────────────────────────────────────────
#  Main chat area — header
# ─────────────────────────────────────────────
active_engine_name = "Groq Cloud" if st.session_state.provider == "Groq Cloud" else "Ollama Local"
active_model_name = st.session_state.groq_model if st.session_state.provider == "Groq Cloud" else st.session_state.ollama_model

st.markdown(
    f"""
    <div class="chat-header">
        <h1>🤖 AI Agent</h1>
        <p>Powered by ReAct Agent · {active_engine_name} ({active_model_name}) · DuckDuckGo · 100% Free</p>
    </div>
    """,
    unsafe_allow_html=True,
)

# Welcome message when chat is empty
if not st.session_state.messages:
    st.markdown(
        f"""
        <div style="
            background: linear-gradient(135deg, rgba(108,99,255,0.06), rgba(0,212,170,0.04));
            border: 1px solid rgba(108,99,255,0.18);
            border-radius: 14px;
            padding: 1.5rem 1.8rem;
            margin-bottom: 1.5rem;
            text-align: center;
        ">
            <div style="font-size:2rem; margin-bottom:0.6rem;">👋</div>
            <div style="font-size:1rem; font-weight:600; color:#1a1d2e; margin-bottom:0.4rem;">
                Welcome! Ask me anything.
            </div>
            <div style="font-size:0.85rem; color:#5a6080;">
                I can search the web or calculate math expressions autonomously.<br>
                Currently active: <strong style="color:#6c63ff;">{active_engine_name}</strong> (<code style="background:rgba(0,0,0,0.05); padding:1px 6px; border-radius:4px;">{active_model_name}</code>).
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


# ─────────────────────────────────────────────
#  Helper — render stored messages
# ─────────────────────────────────────────────
def render_messages():
    for msg in st.session_state.messages:
        role = msg["role"]

        if role == "user":
            with st.chat_message("user", avatar="👤"):
                st.markdown(msg.get("content", ""))

        elif role == "assistant":
            with st.chat_message("assistant", avatar="🤖"):
                st.markdown(msg.get("content", ""))

        elif role == "tool_call":
            with st.expander(f"🛠 Tool called: `{msg.get('tool_name', 'tool')}`", expanded=False):
                if msg.get("tool_input"):
                    st.markdown(f"**Input:** `{msg['tool_input']}`")
                if msg.get("tool_output"):
                    st.markdown(f"**Output:**\n```\n{msg['tool_output']}\n```")


render_messages()


# ─────────────────────────────────────────────
#  Helper — manual ReAct agent loop
# ─────────────────────────────────────────────
def run_agent(user_text: str) -> None:
    """Manual ReAct loop: parse Action/Action Input from LLM text, run tools, loop."""
    # Obtain LLM according to provider
    if st.session_state.provider == "Groq Cloud":
        api_key = st.session_state.get("groq_api_key", "").strip() or os.environ.get("GROQ_API_KEY", "")
        if not api_key:
            try:
                api_key = st.secrets.get("GROQ_API_KEY", "")
            except Exception:
                pass
        if not api_key:
            st.session_state.messages.append({
                "role": "assistant",
                "content": "⚠️ **Groq API Key missing.** Please enter your free key in the sidebar or set `GROQ_API_KEY` in `.env` / Streamlit Secrets.\n\nGet your free key here: [console.groq.com/keys](https://console.groq.com/keys)",
            })
            return
        llm = build_groq_llm(st.session_state.groq_model, api_key)
    else:
        llm = build_ollama_llm(st.session_state.ollama_model)

    # Regex patterns
    action_re  = re.compile(r"Action:\s*([\w]+).*?\nAction Input:\s*(.+?)(?=(?:\s*(?:Thought:|Action:|Final Answer:|Observation:))|$)", re.DOTALL | re.IGNORECASE)
    final_re   = re.compile(r"Final Answer:\s*(.+)", re.DOTALL | re.IGNORECASE)

    tool_map = {
        "duckduckgo_search": duckduckgo_search,
        "calculator": calculator,
    }

    # Build message list from history + new user message
    lc_messages = [SystemMessage(content=REACT_SYSTEM_PROMPT)]
    for msg in st.session_state.messages:
        if msg["role"] == "user":
            lc_messages.append(HumanMessage(content=msg["content"]))
        elif msg["role"] == "assistant":
            lc_messages.append(AIMessage(content=msg["content"]))
    lc_messages.append(HumanMessage(content=user_text))

    for _ in range(6):  # max iterations
        response = llm.invoke(lc_messages)
        text = response.content
        # Clean think tags if present
        text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL).strip()

        action_match = action_re.search(text)
        final_match  = final_re.search(text)

        # Prefer action if it appears before final answer
        if action_match and (not final_match or action_match.start() < final_match.start()):
            tool_name  = action_match.group(1).strip()
            tool_input = action_match.group(2).strip().strip('" \' `')

            if tool_name in tool_map:
                try:
                    tool_output = tool_map[tool_name].invoke(tool_input)
                except Exception as exc:
                    tool_output = f"Tool error: {exc}"

                st.session_state.tool_calls_made += 1
                st.session_state.messages.append({
                    "role": "tool_call",
                    "tool_name": tool_name,
                    "tool_input": tool_input,
                    "tool_output": tool_output,
                })
                lc_messages.append(AIMessage(content=text))
                lc_messages.append(HumanMessage(content=f"Observation: {tool_output}"))
            else:
                lc_messages.append(AIMessage(content=text))
                lc_messages.append(HumanMessage(content=f"Observation: Tool '{tool_name}' not found. Available: duckduckgo_search, calculator"))

        elif final_match:
            answer = final_match.group(1).strip()
            st.session_state.messages.append({"role": "assistant", "content": answer})
            return

        else:
            # No structured output — use raw response as answer
            st.session_state.messages.append({"role": "assistant", "content": text})
            return

    # Reached max iterations
    st.session_state.messages.append({
        "role": "assistant",
        "content": "I reached the maximum reasoning steps. Please try rephrasing your question.",
    })


# ─────────────────────────────────────────────
#  Chat input — main loop
# ─────────────────────────────────────────────
if prompt := st.chat_input("Ask me anything — I can search the web or do math…", key="chat_input"):
    # 1. Append user message immediately
    st.session_state.messages.append({"role": "user", "content": prompt})
    st.session_state.total_queries += 1

    # 2. Re-render so user message appears before spinner
    with st.chat_message("user", avatar="👤"):
        st.markdown(prompt)

    # 3. Run agent with spinner
    with st.chat_message("assistant", avatar="🤖"):
        with st.spinner("🔄 Agent thinking…"):
            try:
                run_agent(prompt)
            except Exception as exc:
                if st.session_state.get("provider") == "Groq Cloud":
                    error_msg = (
                        f"⚠️ **Groq API Error:** `{exc}`\n\n"
                        "Please verify your **Groq API Key** in the sidebar or check if your free rate limit was exceeded."
                    )
                else:
                    error_msg = (
                        f"⚠️ **Ollama Error:** `{exc}`\n\n"
                        "Make sure **Ollama** is running (`ollama serve`) and "
                        f"the model is pulled (`ollama pull {st.session_state.get('ollama_model', 'qwen2.5:3b')}`)."
                    )
                st.session_state.messages.append({
                    "role": "assistant",
                    "content": error_msg,
                })

    # 4. Rerun to re-render full conversation cleanly
    st.rerun()

