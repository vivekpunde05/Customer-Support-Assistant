"""
Vector Store Module
Handles: Initialize embeddings → Store in ChromaDB → Retrieve relevant chunks
"""

from pathlib import Path
from typing import List, Optional

from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_core.documents import Document

from config import VECTORSTORE_DIR, EMBEDDING_MODEL, TOP_K_RETRIEVAL


class VectorStoreManager:
    """
    Manages ChromaDB vector store for document retrieval.
    
    Features:
    - Initialize embedding model (free local HuggingFace)
    - Create/load vector store from documents
    - Similarity search with relevance scores
    """
    
    def __init__(self, collection_name: str = "support_kb"):
        self.collection_name = collection_name
        self.vectorstore_dir = VECTORSTORE_DIR / collection_name
        self.embeddings = self._initialize_embeddings()
        self.vectorstore: Optional[Chroma] = None
    
    def _initialize_embeddings(self):
        """Initialize the free local embedding model."""
        print(f"🤗 Using local HuggingFace embeddings (free, no API key)")
        return HuggingFaceEmbeddings(
            model_name=EMBEDDING_MODEL,
            model_kwargs={"device": "cpu"},
            encode_kwargs={"normalize_embeddings": True}
        )
    
    def create_vectorstore(self, documents: List[Document]) -> Chroma:
        """
        Create a new ChromaDB vector store from documents.
        
        Args:
            documents: List of chunked Document objects
            
        Returns:
            Chroma vector store instance
        """
        print(f"\n💾 Creating vector store in: {self.vectorstore_dir}")
        
        self.vectorstore = Chroma.from_documents(
            documents=documents,
            embedding=self.embeddings,
            persist_directory=str(self.vectorstore_dir),
            collection_name=self.collection_name
        )
        self.vectorstore.persist()
        print(f"✅ Vector store created with {len(documents)} documents")
        return self.vectorstore
    
    def load_vectorstore(self) -> Chroma:
        """
        Load an existing ChromaDB vector store.
        
        Returns:
            Chroma vector store instance
        """
        if not self.vectorstore_dir.exists():
            raise FileNotFoundError(
                f"Vector store not found at {self.vectorstore_dir}. "
                f"Run create_vectorstore() first."
            )
        
        print(f"\n📂 Loading vector store from: {self.vectorstore_dir}")
        self.vectorstore = Chroma(
            persist_directory=str(self.vectorstore_dir),
            embedding_function=self.embeddings,
            collection_name=self.collection_name
        )
        print("✅ Vector store loaded")
        return self.vectorstore
    
    def similarity_search(self, query: str, k: int = TOP_K_RETRIEVAL) -> List[Document]:
        """
        Retrieve top-k most relevant documents for a query.
        
        Args:
            query: User query string
            k: Number of documents to retrieve
            
        Returns:
            List of relevant Document objects
        """
        if self.vectorstore is None:
            raise RuntimeError("Vector store not initialized. "
                             "Call create_vectorstore() or load_vectorstore() first.")
        
        results = self.vectorstore.similarity_search_with_score(query, k=k)
        
        print(f"\n🔍 Retrieved {len(results)} chunks for query: '{query[:60]}...'")
        for i, (doc, score) in enumerate(results):
            print(f"  [{i+1}] Score: {score:.4f} | {doc.page_content[:80]}...")
        
        # Return just the documents (without scores) for the RAG chain
        return [doc for doc, score in results]
    
    def get_retriever(self, k: int = TOP_K_RETRIEVAL):
        """
        Get a LangChain retriever interface for the vector store.
        
        Args:
            k: Number of documents to retrieve
            
        Returns:
            BaseRetriever instance
        """
        if self.vectorstore is None:
            raise RuntimeError("Vector store not initialized.")
        return self.vectorstore.as_retriever(search_kwargs={"k": k})


# ─── Standalone usage ──────────────────────────────────────────────
if __name__ == "__main__":
    from config import PDF_PATH
    from modules.pdf_processor import PDFProcessor
    
    # Process PDF
    processor = PDFProcessor()
    chunks = processor.process(PDF_PATH)
    
    # Create and test vector store
    vs_manager = VectorStoreManager(collection_name="support_kb")
    vs_manager.create_vectorstore(chunks)
    
    # Test retrieval
    test_query = "How do I reset my password?"
    results = vs_manager.similarity_search(test_query, k=2)
    print(f"\nTop result: {results[0].page_content[:200]}...")