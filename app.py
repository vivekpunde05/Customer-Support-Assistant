"""
Streamlit UI for RAG Customer Support Bot
Professional chat interface with workflow visualization
"""

import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

import streamlit as st
from modules.llm_client import check_setup
from modules.graph_workflow import create_agent
from config import VECTORSTORE_DIR


# ─── Page Configuration ──────────────────────────────────────────
st.set_page_config(
    page_title="TechCorp Support Bot",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)


# ─── Custom CSS ───────────────────────────────────────────────────
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 0.5rem;
    }
    .sub-header {
        font-size: 1.1rem;
        color: #666;
        text-align: center;
        margin-bottom: 2rem;
    }
    .chat-message {
        padding: 1rem;
        border-radius: 0.5rem;
        margin-bottom: 0.5rem;
    }
    .user-message {
        background-color: #e3f2fd;
        border-left: 4px solid #2196f3;
    }
    .bot-message {
        background-color: #f5f5f5;
        border-left: 4px solid #4caf50;
    }
    .bot-message.rag {
        border-left-color: #ff9800;
    }
    .bot-message.direct {
        border-left-color: #9c27b0;
    }
    .bot-message.hitl {
        border-left-color: #f44336;
    }
    .trace-badge {
        display: inline-block;
        padding: 0.2rem 0.5rem;
        border-radius: 0.3rem;
        font-size: 0.75rem;
        margin-top: 0.5rem;
    }
    .trace-rag {
        background-color: #fff3e0;
        color: #e65100;
    }
    .trace-direct {
        background-color: #f3e5f5;
        color: #7b1fa2;
    }
    .trace-hitl {
        background-color: #ffebee;
        color: #c62828;
    }
    .status-indicator {
        display: inline-block;
        width: 10px;
        height: 10px;
        border-radius: 50%;
        margin-right: 0.5rem;
    }
    .status-online {
        background-color: #4caf50;
    }
    .status-offline {
        background-color: #f44336;
    }
    .workflow-box {
        background-color: #fafafa;
        border: 1px solid #e0e0e0;
        border-radius: 0.5rem;
        padding: 1rem;
        margin: 0.5rem 0;
    }
    .workflow-step {
        display: inline-block;
        padding: 0.3rem 0.8rem;
        margin: 0.2rem;
        border-radius: 0.3rem;
        font-size: 0.85rem;
        font-weight: 500;
    }
    .step-input {
        background-color: #e3f2fd;
        color: #1565c0;
    }
    .step-intent {
        background-color: #fff3e0;
        color: #e65100;
    }
    .step-router {
        background-color: #fce4ec;
        color: #c2185b;
    }
    .step-rag {
        background-color: #e8f5e9;
        color: #2e7d32;
    }
    .step-direct {
        background-color: #f3e5f5;
        color: #7b1fa2;
    }
    .step-hitl {
        background-color: #ffebee;
        color: #c62828;
    }
    .step-output {
        background-color: #e0f2f1;
        color: #00695c;
    }
    .ticket-box {
        background-color: #fff3e0;
        border: 1px solid #ff9800;
        border-radius: 0.5rem;
        padding: 1rem;
        margin: 0.5rem 0;
    }
</style>
""", unsafe_allow_html=True)


# ─── Initialize Session State ──────────────────────────────────────
if "messages" not in st.session_state:
    st.session_state.messages = []

if "agent" not in st.session_state:
    st.session_state.agent = None

if "thread_id" not in st.session_state:
    import uuid
    st.session_state.thread_id = f"streamlit-{uuid.uuid4().hex[:8]}"

if "setup_checked" not in st.session_state:
    st.session_state.setup_checked = False


# ─── Sidebar ─────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## 🤖 TechCorp Support Bot")
    st.markdown("---")
    
    # Setup Status
    if not st.session_state.setup_checked:
        with st.spinner("Checking setup..."):
            status = check_setup()
            st.session_state.setup_status = status
            st.session_state.setup_checked = True
    
    status = st.session_state.get("setup_status", {})
    
    st.markdown("### 🔌 System Status")
    
    if status.get("groq", {}).get("available", False):
        st.markdown('<span class="status-indicator status-online"></span> Groq API: **Connected**', unsafe_allow_html=True)
    else:
        st.markdown('<span class="status-indicator status-offline"></span> Groq API: **Disconnected**', unsafe_allow_html=True)
        st.error("⚠️ Add your Groq API key to `.env` file")
    
    # Vector Store Status
    vs_path = VECTORSTORE_DIR / "support_kb"
    if vs_path.exists():
        st.markdown('<span class="status-indicator status-online"></span> Knowledge Base: **Ready**', unsafe_allow_html=True)
    else:
        st.markdown('<span class="status-indicator status-offline"></span> Knowledge Base: **Not Built**', unsafe_allow_html=True)
        st.warning("Run `python main.py --build` first")
    
    st.markdown("---")
    st.markdown("### 📊 Conversation Stats")
    st.markdown(f"**Messages:** {len(st.session_state.messages)}")
    st.markdown(f"**Session ID:** `{st.session_state.thread_id}`")
    
    st.markdown("---")
    st.markdown("### 🛠️ Actions")
    
    if st.button("🗑️ Clear Chat", use_container_width=True):
        st.session_state.messages = []
        st.rerun()
    
    if st.button("📋 View Demo Queries", use_container_width=True):
        st.session_state.show_demo = True
    
    st.markdown("---")
    st.markdown("### 📚 About")
    st.markdown("""
    **RAG Customer Support Bot**
    
    • PDF ingestion with ChromaDB
    • Intent-based routing
    • Human-in-the-Loop escalation
    • Groq LLM (free tier)
    • LangGraph workflow
    """)


# ─── Main Content ──────────────────────────────────────────────────
st.markdown('<div class="main-header">TechCorp Customer Support</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-header">AI-Powered Support Assistant • Ask me anything about our policies</div>', unsafe_allow_html=True)

# Check if ready
status = st.session_state.get("setup_status", {})
vs_ready = (VECTORSTORE_DIR / "support_kb").exists()

if not status.get("groq", {}).get("available", False) or not vs_ready:
    st.error("### ⚠️ System Not Ready")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
        **1. Setup API Key**
        ```bash
        # Create .env file
        echo GROQ_API_KEY=gsk_your_key > .env
        ```
        Get free key at [console.groq.com](https://console.groq.com/keys)
        """)
    
    with col2:
        st.markdown("""
        **2. Build Knowledge Base**
        ```bash
        python main.py --build
        ```
        This creates the vector store from your PDF.
        """)
    
    st.stop()


# ─── Initialize Agent ──────────────────────────────────────────────
if st.session_state.agent is None:
    with st.spinner("Initializing AI agent..."):
        st.session_state.agent = create_agent()


# ─── Demo Queries Section ─────────────────────────────────────────
if st.session_state.get("show_demo", False):
    st.markdown("### 🎬 Demo Queries")
    
    demo_queries = [
        ("📚 RAG", "What is the return policy for electronics?", "Document-based answer"),
        ("📚 RAG", "How much does express shipping cost?", "Pricing from tables"),
        ("💬 Direct", "Hello! How are you today?", "Friendly greeting"),
        ("💬 Direct", "Thank you for your help!", "Polite response"),
        ("🚨 HITL", "I want a refund and I'm contacting my lawyer!", "Escalation trigger"),
        ("🚨 HITL", "My account was hacked! Unauthorized charges!", "Security escalation"),
        ("💬 Direct", "What's the weather like today?", "Out of scope"),
        ("📚 RAG", "How do I enable two-factor authentication?", "FAQ retrieval"),
    ]
    
    cols = st.columns(2)
    for i, (route, query, desc) in enumerate(demo_queries):
        with cols[i % 2]:
            with st.container():
                st.markdown(f"**{route}** • {desc}")
                if st.button(f"▶️ {query[:50]}...", key=f"demo_{i}", use_container_width=True):
                    st.session_state.demo_query = query
                    st.session_state.show_demo = False
                    st.rerun()
    
    if st.button("❌ Close Demo Panel"):
        st.session_state.show_demo = False
        st.rerun()
    
    st.markdown("---")


# ─── Chat Interface ───────────────────────────────────────────────
st.markdown("### 💬 Chat")

# Display chat history
for msg in st.session_state.messages:
    role = msg["role"]
    content = msg["content"]
    mode = msg.get("mode", "unknown")
    trace = msg.get("trace", [])
    ticket = msg.get("ticket_id", "")
    
    if role == "user":
        st.markdown(f"""
        <div class="chat-message user-message">
            <strong>👤 You</strong><br>{content}
        </div>
        """, unsafe_allow_html=True)
    else:
        # Determine style based on route
        css_class = f"bot-message {mode}"
        trace_class = f"trace-{mode}"
        route_label = mode.upper()
        
        trace_html = " → ".join([
            f'<span class="workflow-step step-{step}">{step}</span>' 
            for step in trace
        ])
        
        ticket_html = ""
        if ticket:
            ticket_html = f"""
            <div class="ticket-box">
                🎫 <strong>Escalation Ticket:</strong> #{ticket}<br>
                <small>A human agent will follow up within 2 hours.</small>
            </div>
            """
        
        st.markdown(f"""
        <div class="chat-message {css_class}">
            <strong>🤖 TechCorp Support</strong><br>
            {content}
            {ticket_html}
            <div class="trace-badge {trace_class}">
                Route: {route_label} | Trace: {trace_html}
            </div>
        </div>
        """, unsafe_allow_html=True)


# ─── Input Area ────────────────────────────────────────────────────
st.markdown("---")

# Demo query auto-fill
demo_query = st.session_state.pop("demo_query", "")

col_input, col_button = st.columns([6, 1])

with col_input:
    user_input = st.text_input(
        "Type your question...",
        value=demo_query,
        placeholder="e.g., What is your return policy? How do I reset my password?",
        key="chat_input",
        label_visibility="collapsed"
    )

with col_button:
    send_clicked = st.button("📤 Send", use_container_width=True, type="primary")

# Handle send
if send_clicked and user_input.strip():
    # Add user message
    st.session_state.messages.append({
        "role": "user",
        "content": user_input.strip()
    })
    
    # Process with agent
    with st.spinner("Thinking..."):
        try:
            state = {
                "query": user_input.strip(),
                "messages": [],
                "documents": [],
                "answer": "",
                "ticket_id": "",
                "requires_human": False,
                "workflow_trace": []
            }
            
            config = {"configurable": {"thread_id": st.session_state.thread_id}}
            result = st.session_state.agent.invoke(state, config=config)
            
            # Add bot message
            st.session_state.messages.append({
                "role": "assistant",
                "content": result["answer"],
                "mode": result.get("intent", "unknown"),
                "trace": result.get("workflow_trace", []),
                "ticket_id": result.get("ticket_id", "")
            })
            
        except Exception as e:
            st.session_state.messages.append({
                "role": "assistant",
                "content": f"❌ Error: {str(e)}",
                "mode": "error",
                "trace": ["error"],
                "ticket_id": ""
            })
    
    st.rerun()


# ─── Quick Suggestions ─────────────────────────────────────────────
if len(st.session_state.messages) == 0:
    st.markdown("### 💡 Try Asking")
    
    suggestions = [
        "What is your return policy?",
        "How do I track my order?",
        "What are the Pro plan features?",
        "How do I reset my password?",
    ]
    
    cols = st.columns(4)
    for i, suggestion in enumerate(suggestions):
        with cols[i]:
            if st.button(suggestion, key=f"suggest_{i}", use_container_width=True):
                st.session_state.demo_query = suggestion
                st.rerun()


# ─── Footer ────────────────────────────────────────────────────────
st.markdown("---")
st.markdown("""
<div style="text-align: center; color: #999; font-size: 0.85rem;">
    Powered by <strong>LangGraph</strong> • <strong>Groq</strong> • <strong>ChromaDB</strong> • <strong>Streamlit</strong><br>
    RAG Customer Support Bot • Free Cloud LLM • No OpenAI Required
</div>
""", unsafe_allow_html=True)