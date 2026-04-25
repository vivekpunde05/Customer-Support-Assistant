"""
LangGraph Workflow Module
Defines the graph-based control flow: Input → Process → Output with routing
"""

from typing import TypedDict, Annotated, Literal
from langgraph.graph import StateGraph, END
from langgraph.graph.message import add_messages
from langgraph.checkpoint.memory import MemorySaver

from modules.intent_classifier import IntentClassifier, IntentType
from modules.vector_store import VectorStoreManager
from modules.rag_engine import RAGEngine
from modules.hitl_manager import HITLManager


# ─── State Definition ──────────────────────────────────────────────
class AgentState(TypedDict):
    """
    Shared state across all nodes in the graph.
    """
    messages: Annotated[list, add_messages]  # Conversation history
    query: str                               # Current user query
    intent: str                              # Classified intent
    confidence: float                        # Classification confidence
    documents: list                          # Retrieved documents
    answer: str                              # Final answer
    ticket_id: str                           # HITL ticket ID (if escalated)
    requires_human: bool                     # Whether human is needed
    workflow_trace: list                     # Execution trace for debugging


# ─── Node Functions ────────────────────────────────────────────────
def input_node(state: AgentState) -> AgentState:
    """
    INPUT NODE: Receives and logs the user query.
    """
    query = state["query"]
    print(f"\n{'='*50}")
    print(f"📥 INPUT NODE")
    print(f"   Query: {query}")
    print(f"{'='*50}")
    
    state["workflow_trace"] = state.get("workflow_trace", []) + ["input"]
    return state


def intent_classifier_node(state: AgentState) -> AgentState:
    """
    PROCESS NODE 1: Classifies intent to determine routing.
    """
    print(f"\n{'='*50}")
    print(f"🧠 INTENT CLASSIFIER NODE")
    print(f"{'='*50}")
    
    classifier = IntentClassifier()
    result = classifier.classify(state["query"])
    
    state["intent"] = result["intent"].value
    state["confidence"] = result["confidence"]
    state["requires_human"] = classifier.should_escalate(result)
    state["workflow_trace"] = state["workflow_trace"] + ["intent_classification"]
    
    print(f"   Intent: {state['intent']}")
    print(f"   Confidence: {state['confidence']:.2f}")
    print(f"   Requires Human: {state['requires_human']}")
    
    return state


def router_node(state: AgentState) -> Literal["rag", "direct", "hitl"]:
    """
    ROUTER NODE: Conditional routing based on intent classification.
    """
    print(f"\n{'='*50}")
    print(f"🚦 ROUTER NODE")
    print(f"{'='*50}")
    
    intent = state["intent"]
    requires_human = state["requires_human"]
    
    if requires_human:
        print("   → Route: HITL (Human-in-the-Loop)")
        return "hitl"
    
    if intent == IntentType.GENERAL_CHAT.value:
        print("   → Route: DIRECT (General Chat)")
        return "direct"
    
    if intent == IntentType.OUT_OF_SCOPE.value:
        print("   → Route: DIRECT (Out of Scope - polite decline)")
        return "direct"
    
    print("   → Route: RAG (Document Query)")
    return "rag"


def rag_node(state: AgentState) -> AgentState:
    """
    PROCESS NODE 2A: RAG Answer Generation.
    Retrieves documents and generates contextual answer.
    """
    print(f"\n{'='*50}")
    print(f"📚 RAG NODE")
    print(f"{'='*50}")
    
    # Load vector store
    vs_manager = VectorStoreManager()
    vs_manager.load_vectorstore()
    
    # Retrieve relevant documents
    documents = vs_manager.similarity_search(state["query"])
    state["documents"] = documents
    
    # Generate answer
    engine = RAGEngine()
    result = engine.generate_rag_answer(state["query"], documents)
    
    state["answer"] = result["answer"]
    state["workflow_trace"] = state["workflow_trace"] + ["rag_generation"]
    
    print(f"   Answer generated ({len(result['answer'])} chars)")
    
    return state


def direct_answer_node(state: AgentState) -> AgentState:
    """
    PROCESS NODE 2B: Direct Answer (no retrieval).
    For general chat and out-of-scope queries.
    """
    print(f"\n{'='*50}")
    print(f"💬 DIRECT ANSWER NODE")
    print(f"{'='*50}")
    
    engine = RAGEngine()
    result = engine.generate_direct_answer(state["query"])
    
    state["answer"] = result["answer"]
    state["documents"] = []
    state["workflow_trace"] = state["workflow_trace"] + ["direct_answer"]
    
    print(f"   Answer generated ({len(result['answer'])} chars)")
    
    return state


def hitl_node(state: AgentState) -> AgentState:
    """
    PROCESS NODE 2C: Human-in-the-Loop Escalation.
    Creates ticket and generates human handoff response.
    """
    print(f"\n{'='*50}")
    print(f"🚨 HITL ESCALATION NODE")
    print(f"{'='*50}")
    
    hitl_manager = HITLManager()
    
    classification = {
        "intent": state["intent"],
        "confidence": state["confidence"],
        "reason": f"Escalated: confidence={state['confidence']:.2f}"
    }
    
    # Create escalation ticket
    ticket = hitl_manager.create_ticket(state["query"], classification)
    state["ticket_id"] = ticket.ticket_id
    
    # Generate response
    result = hitl_manager.generate_escalation_response(ticket)
    state["answer"] = result["answer"]
    state["workflow_trace"] = state["workflow_trace"] + ["hitl_escalation"]
    
    print(f"   Ticket created: {ticket.ticket_id}")
    
    return state


def output_node(state: AgentState) -> AgentState:
    """
    OUTPUT NODE: Finalizes and formats the response.
    """
    print(f"\n{'='*50}")
    print(f"📤 OUTPUT NODE")
    print(f"{'='*50}")
    
    # Add assistant message to conversation history
    state["messages"] = state.get("messages", []) + [
        {"role": "assistant", "content": state["answer"]}
    ]
    
    state["workflow_trace"] = state["workflow_trace"] + ["output"]
    
    print(f"   Mode: {state.get('intent', 'unknown')}")
    if state.get("ticket_id"):
        print(f"   Ticket: {state['ticket_id']}")
    print(f"   Trace: {' → '.join(state['workflow_trace'])}")
    
    return state


# ─── Graph Builder ─────────────────────────────────────────────────
def build_workflow() -> StateGraph:
    """
    Builds the LangGraph workflow with all nodes and edges.
    
    Graph Structure:
        input → intent_classifier → router → [rag | direct | hitl] → output → END
    """
    # Initialize the graph with state schema
    workflow = StateGraph(AgentState)
    
    # ─── Add Nodes ─────────────────────────────────────────────────
    workflow.add_node("input", input_node)
    workflow.add_node("intent_classifier", intent_classifier_node)
    workflow.add_node("router", router_node)  # This is actually the conditional edge
    workflow.add_node("rag", rag_node)
    workflow.add_node("direct_answer", direct_answer_node)
    workflow.add_node("hitl", hitl_node)
    workflow.add_node("output", output_node)
    
    # ─── Add Edges ─────────────────────────────────────────────────
    # Start → Input → Intent Classifier
    workflow.set_entry_point("input")
    workflow.add_edge("input", "intent_classifier")
    
    # Intent Classifier → Router (conditional)
    workflow.add_conditional_edges(
        "intent_classifier",
        router_node,
        {
            "rag": "rag",
            "direct": "direct_answer",
            "hitl": "hitl"
        }
    )
    
    # All processing nodes → Output
    workflow.add_edge("rag", "output")
    workflow.add_edge("direct_answer", "output")
    workflow.add_edge("hitl", "output")
    
    # Output → END
    workflow.add_edge("output", END)
    
    return workflow


def create_agent():
    """
    Creates the compiled agent with memory checkpointing.
    
    Returns:
        Compiled StateGraph ready for invocation
    """
    workflow = build_workflow()
    
    # Add memory for conversation persistence
    memory = MemorySaver()
    
    # Compile the graph
    agent = workflow.compile(checkpointer=memory)
    
    print("✅ Agent compiled successfully!")
    print("   Graph nodes: input → intent_classifier → [rag|direct|hitl] → output")
    
    return agent


# ─── Standalone usage ──────────────────────────────────────────────
if __name__ == "__main__":
    agent = create_agent()
    
    # Test the workflow
    test_state = {
        "query": "How do I reset my password?",
        "messages": []
    }
    
    # Run the agent
    config = {"configurable": {"thread_id": "test-thread-1"}}
    result = agent.invoke(test_state, config=config)
    
    print(f"\n{'='*50}")
    print("FINAL RESULT:")
    print(f"{'='*50}")
    print(f"Answer: {result['answer']}")
    print(f"Trace: {' → '.join(result['workflow_trace'])}")