#!/usr/bin/env python3
"""
RAG Customer Support Bot - Main Entry Point

Usage:
    # 1. Set up environment
    export GROQ_API_KEY="gsk-your-key-here"
    
    # 2. Prepare your PDF in data/sample_knowledge_base.pdf
    
    # 3. Build the knowledge base (run once)
    python main.py --build
    
    # 4. Start interactive chat
    python main.py --chat
    
    # 5. Run demo queries
    python main.py --demo
"""

import argparse
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

from config import PDF_PATH, VECTORSTORE_DIR
from modules.pdf_processor import PDFProcessor
from modules.vector_store import VectorStoreManager
from modules.llm_client import check_setup
from modules.graph_workflow import create_agent


def setup_wizard():
    """Interactive setup for first-time users."""
    print("\n" + "="*60)
    print("🔧 RAG BOT SETUP WIZARD")
    print("="*60)
    
    status = check_setup()
    
    if status["groq"]["available"]:
        print("\n✅ Groq is ready!")
        return True
    
    print("\n❌ No LLM provider configured.")
    print("\n" + "-"*60)
    print("QUICK SETUP (Choose ONE):")
    print("-"*60)
    
    print("\nOption 1: GROQ (Recommended - fastest, most reliable)")
    print("  1. Go to: https://console.groq.com/keys")
    print("  2. Sign up with Google/GitHub (30 seconds, no credit card)")
    print("  3. Click 'Create API Key'")
    print("  4. Copy the key (starts with 'gsk_')")
    print("  5. Run: echo 'GROQ_API_KEY=your_key' > .env")
    
    return False


def build_knowledge_base():
    """
    Build the vector store from PDF documents.
    Run this once before chatting.
    """
    print("\n" + "="*60)
    print("🔨 BUILDING KNOWLEDGE BASE")
    print("="*60)
    
    if not PDF_PATH.exists():
        print(f"❌ PDF not found: {PDF_PATH}")
        print("Creating sample knowledge base...")
        
        # Create sample text file as fallback
        sample_text = """TechCorp Customer Support Handbook
        
1. Returns: 30-day return policy with receipt. Electronics unopened unless defective.
2. Warranty: 1-year limited warranty. Extended up to 3 years.
3. Password Reset: Click 'Forgot Password' on login page. Email arrives in 2 minutes.
4. Billing: Monthly auto-renewal. Annual saves 20%. Cancel at Account > Billing.
5. Shipping: Standard $5.99 (5-7 days), Express $12.99 (2-3 days), Overnight $24.99. Free over $75.
6. Support: Basic (free, 48h), Pro ($29/mo, 4h), Enterprise (custom, 1h).
7. Privacy: GDPR/CCPA compliant. Encrypted data. Export/delete via Privacy Dashboard.
8. Contact: support@techcorp.com, 1-800-TECH-HELP, Mon-Fri 9AM-6PM EST.
"""
        PDF_PATH.parent.mkdir(parents=True, exist_ok=True)
        with open(PDF_PATH.with_suffix(".txt"), "w") as f:
            f.write(sample_text)
        print(f"✅ Created sample text: {PDF_PATH.with_suffix('.txt')}")
        
        # Use text file instead
        from langchain_core.documents import Document
        processor = PDFProcessor()
        doc = Document(page_content=sample_text, metadata={"source": "sample.txt"})
        chunks = processor.split_documents([doc])
    else:
        processor = PDFProcessor()
        chunks = processor.process(PDF_PATH)
    
    vs_manager = VectorStoreManager(collection_name="support_kb")
    vs_manager.create_vectorstore(chunks)
    
    print("\n✅ Knowledge base ready!")
    return True


def chat_loop():
    """Interactive chat session."""
    print("\n" + "="*60)
    print("🤖 RAG CUSTOMER SUPPORT BOT")
    print("="*60)
    
    # Verify setup
    status = check_setup()
    if not status["groq"]["available"]:
        setup_wizard()
        return
    
    # Check vector store
    vs_path = VECTORSTORE_DIR / "support_kb"
    if not vs_path.exists():
        print("⚠️  Run: python main.py --build")
        return
    
    # Create agent
    agent = create_agent()
    thread_id = f"session-{__import__('uuid').uuid4().hex[:8]}"
    conversation = []
    
    print("\nType your question or 'quit' to exit.")
    print("-"*60)
    
    while True:
        try:
            user_input = input("\n👤 You: ").strip()
            
            if user_input.lower() in ["quit", "exit", "bye"]:
                print("👋 Goodbye!")
                break
            if not user_input:
                continue
            
            # Run graph
            state = {
                "query": user_input,
                "messages": conversation,
                "documents": [],
                "answer": "",
                "ticket_id": "",
                "requires_human": False,
                "workflow_trace": []
            }
            
            config = {"configurable": {"thread_id": thread_id}}
            result = agent.invoke(state, config=config)
            
            conversation = result["messages"]
            
            print(f"\n🤖 Bot: {result['answer']}")
            if result.get("ticket_id"):
                print(f"   🎫 Ticket: {result['ticket_id']}")
            print(f"   📊 Trace: {' → '.join(result['workflow_trace'])}")
            
        except KeyboardInterrupt:
            print("\n👋 Goodbye!")
            break
        except Exception as e:
            print(f"❌ Error: {e}")


def run_demo():
    """Run a set of demo queries to showcase all routing paths."""
    print("\n" + "="*60)
    print("🎬 RUNNING DEMO QUERIES")
    print("="*60)
    
    status = check_setup()
    if not status["groq"]["available"]:
        setup_wizard()
        return
    
    vs_path = VECTORSTORE_DIR / "support_kb"
    if not vs_path.exists():
        print("⚠️  Run: python main.py --build")
        return
    
    agent = create_agent()
    
    demo_queries = [
        ("RAG: Return policy", "What is the return policy for electronics?"),
        ("RAG: Shipping", "How much does express shipping cost?"),
        ("DIRECT: Greeting", "Hello! How are you today?"),
        ("DIRECT: Thanks", "Thank you for your help!"),
        ("HITL: Legal threat", "I want a refund and I'm contacting my lawyer!"),
        ("HITL: Account hacked", "My account was hacked! Unauthorized charges!"),
        ("DIRECT: Out of scope", "What's the weather like today?"),
        ("RAG: Warranty", "How long is the warranty coverage?"),
    ]
    
    for label, query in demo_queries:
        print(f"\n{'─'*60}")
        print(f"[{label}] {query}")
        print('─'*60)
        
        state = {
            "query": query,
            "messages": [],
            "documents": [],
            "answer": "",
            "ticket_id": "",
            "requires_human": False,
            "workflow_trace": []
        }
        
        config = {"configurable": {"thread_id": f"demo-{label.split(':')[0]}"}}
        
        try:
            result = agent.invoke(state, config=config)
            print(f"\n✅ Response:\n{result['answer'][:300]}...")
            print(f"\n   Trace: {' → '.join(result['workflow_trace'])}")
            if result.get("ticket_id"):
                print(f"   Ticket: {result['ticket_id']}")
        except Exception as e:
            print(f"❌ Error: {e}")


def main():
    parser = argparse.ArgumentParser(
        description="RAG Customer Support Bot with LangGraph"
    )
    parser.add_argument("--setup", action="store_true", help="Run setup wizard")
    parser.add_argument("--build", action="store_true", help="Build knowledge base")
    parser.add_argument("--chat", action="store_true", help="Interactive chat")
    parser.add_argument("--demo", action="store_true", help="Run demo queries")
    parser.add_argument("--check", action="store_true", help="Check provider status")
    
    args = parser.parse_args()
    
    if args.check:
        status = check_setup()
        print(f"\nProvider Status:")
        for provider, info in status.items():
            if provider != "recommendation":
                print(f"  {provider}: {'✅' if info.get('available') else '❌'}")
        print(f"\n{status['recommendation']}")
        return
    
    if args.setup:
        setup_wizard()
    elif args.build:
        build_knowledge_base()
    elif args.chat:
        chat_loop()
    elif args.demo:
        run_demo()
    else:
        parser.print_help()
        print("\n💡 First time? Run: python main.py --setup")


if __name__ == "__main__":
    main()