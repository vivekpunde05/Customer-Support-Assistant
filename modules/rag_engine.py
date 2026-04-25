"""
RAG Engine Module
Handles: Context retrieval + Prompt construction + LLM generation
"""

from typing import List, Dict
from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

from modules.llm_client import get_llm


class RAGEngine:
    """
    Retrieval-Augmented Generation engine.
    
    Combines retrieved document chunks with LLM to generate
    contextually accurate answers.
    """
    
    def __init__(self):
        self.llm = get_llm(temperature=0.0, max_tokens=1024)
        print(f"🤖 RAG engine using {self.llm._provider}")
        
        # RAG prompt template
        self.rag_prompt = ChatPromptTemplate.from_messages([
    ("system", """You are a helpful customer support assistant for TechCorp.
Answer the user's question based ONLY on the provided context from our company policies and regulations.
If the context doesn't contain enough information to answer fully, say so honestly.
Be concise, professional, and helpful.

Context from company policy:
{context}
"""),
    ("human", "{question}")
])        
        # Direct answer prompt (no retrieval needed)
        self.direct_prompt = ChatPromptTemplate.from_messages([
    ("system", """You are a friendly customer support representative for TechCorp.
Respond naturally to the user's message as a real human agent would. Keep responses brief and professional."""),
    ("human", "{question}")
])
    
    def _format_context(self, documents: List[Document]) -> str:
        """
        Format retrieved documents into context string.
        
        Args:
            documents: List of retrieved Document objects
            
        Returns:
            Formatted context string
        """
        context_parts = []
        for i, doc in enumerate(documents, 1):
            source = doc.metadata.get("source", "Unknown")
            page = doc.metadata.get("page", "N/A")
            context_parts.append(
                f"[Document {i}] Source: {source}, Page: {page}\n{doc.page_content}"
            )
        return "\n\n---\n\n".join(context_parts)
    
    def generate_rag_answer(self, query: str, documents: List[Document]) -> Dict:
        """
        Generate answer using retrieved context (RAG mode).
        
        Args:
            query: User query
            documents: Retrieved relevant documents
            
        Returns:
            Dict with answer, sources used, and confidence
        """
        context = self._format_context(documents)
        
        print(f"\n🤖 Generating RAG answer...")
        print(f"   Context length: {len(context)} chars")
        print(f"   Sources: {[d.metadata.get('page', 'N/A') for d in documents]}")
        
        chain = self.rag_prompt | self.llm | StrOutputParser()
        answer = chain.invoke({
            "context": context,
            "question": query
        })
        
        return {
            "answer": answer,
            "sources": [d.metadata for d in documents],
            "mode": "rag",
            "context_used": True
        }
    
    def generate_direct_answer(self, query: str) -> Dict:
        """
        Generate direct answer without retrieval (for general chat).
        
        Args:
            query: User query
            
        Returns:
            Dict with answer and mode
        """
        print(f"\n🤖 Generating direct answer (no retrieval)...")
        
        chain = self.direct_prompt | self.llm | StrOutputParser()
        answer = chain.invoke({"question": query})
        
        return {
            "answer": answer,
            "sources": [],
            "mode": "direct",
            "context_used": False
        }


# ─── Standalone usage ──────────────────────────────────────────────
if __name__ == "__main__":
    from config import PDF_PATH
    from modules.pdf_processor import PDFProcessor
    from modules.vector_store import VectorStoreManager
    
    # Setup
    processor = PDFProcessor()
    chunks = processor.process(PDF_PATH)
    
    vs = VectorStoreManager()
    vs.create_vectorstore(chunks)
    
    engine = RAGEngine()
    
    # Test RAG
    query = "What is the return policy?"
    docs = vs.similarity_search(query, k=2)
    result = engine.generate_rag_answer(query, docs)
    print(f"\nAnswer: {result['answer']}")