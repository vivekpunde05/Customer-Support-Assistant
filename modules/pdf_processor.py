"""
PDF Processing Module
Handles: Load PDF → Split into chunks → Prepare for embedding
"""

from pathlib import Path
from typing import List

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document

from config import CHUNK_SIZE, CHUNK_OVERLAP


class PDFProcessor:
    """
    Processes PDF documents for RAG pipeline.
    
    Pipeline: PDF File → Document Objects → Text Chunks
    """
    
    def __init__(self, chunk_size: int = CHUNK_SIZE, chunk_overlap: int = CHUNK_OVERLAP):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            length_function=len,
            separators=["\n\n", "\n", ". ", " ", ""]
        )
    
    def load_pdf(self, pdf_path: Path) -> List[Document]:
        """
        Load a PDF file into LangChain Document objects.
        
        Args:
            pdf_path: Path to the PDF file
            
        Returns:
            List of Document objects (one per page)
        """
        if not pdf_path.exists():
            raise FileNotFoundError(f"PDF not found: {pdf_path}")
        
        loader = PyPDFLoader(str(pdf_path))
        documents = loader.load()
        print(f"✅ Loaded {len(documents)} pages from {pdf_path.name}")
        return documents
    
    def split_documents(self, documents: List[Document]) -> List[Document]:
        """
        Split documents into smaller chunks for better retrieval.
        
        Args:
            documents: List of Document objects
            
        Returns:
            List of chunked Document objects
        """
        chunks = self.text_splitter.split_documents(documents)
        print(f"✅ Split into {len(chunks)} chunks "
              f"(size={self.chunk_size}, overlap={self.chunk_overlap})")
        return chunks
    
    def process(self, pdf_path: Path) -> List[Document]:
        """
        Complete pipeline: Load PDF → Split into chunks.
        
        Args:
            pdf_path: Path to the PDF file
            
        Returns:
            List of chunked Document objects ready for embedding
        """
        print(f"\n📄 Processing PDF: {pdf_path.name}")
        documents = self.load_pdf(pdf_path)
        chunks = self.split_documents(documents)
        return chunks


# ─── Standalone usage ──────────────────────────────────────────────
if __name__ == "__main__":
    from config import PDF_PATH
    
    processor = PDFProcessor(chunk_size=500, chunk_overlap=50)
    chunks = processor.process(PDF_PATH)
    
    print(f"\n--- Sample Chunk ---")
    print(f"Content: {chunks[0].page_content[:300]}...")
    print(f"Metadata: {chunks[0].metadata}")