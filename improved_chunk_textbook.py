#!/usr/bin/env python3
"""
Improved Textbook Chunking and Embedding
Enhanced version with better chunking strategy and Pinecone integration
"""

import os
import sys
import re
import json
import time
import logging
from pathlib import Path
from typing import List, Dict, Optional, Tuple
from datetime import datetime

import tiktoken
import numpy as np
from pinecone import Pinecone, ServerlessSpec
from openai import OpenAI
import PyPDF2
from dotenv import load_dotenv

# Load environment variables from .env file (for local development)
load_dotenv()

# Add backend to path for integration
sys.path.append(os.path.join(os.path.dirname(__file__), 'backend'))

# Get environment variables
PINECONE_API_KEY = os.environ.get('PINECONE_API_KEY')
OPENAI_API_KEY = os.environ.get('OPENAI_API_KEY')

if not PINECONE_API_KEY or not OPENAI_API_KEY:
    print("❌ Error: Please set PINECONE_API_KEY and OPENAI_API_KEY environment variables")
    print("   You can set them in your deployment platform or create a .env file locally")
    exit(1)

# Configuration
INDEX_NAME = "professorjames-experiment-grading"
NAMESPACE = "textbook"

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('textbook_chunking.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class TextbookChunker:
    """Enhanced textbook chunking and embedding system"""
    
    def __init__(self):
        self.pc = Pinecone(api_key=PINECONE_API_KEY)
        self.openai_client = OpenAI(api_key=OPENAI_API_KEY)
        self.tokenizer = tiktoken.get_encoding("cl100k_base")
        
        # Initialize index
        try:
            self.index = self.pc.Index(INDEX_NAME)
            logger.info(f"Connected to Pinecone index: {INDEX_NAME}")
        except Exception as e:
            logger.error(f"Failed to connect to Pinecone index: {e}")
            raise
        
        # Chunking parameters
        self.max_tokens = 512
        self.overlap_tokens = 50
        self.min_chunk_tokens = 50
        self.max_chunk_tokens = 800
        
        # Batch processing
        self.batch_size = 96
        self.embedding_model = "text-embedding-3-large"  # Updated to latest model
        
        # Statistics
        self.stats = {
            'total_chunks': 0,
            'successful_embeddings': 0,
            'failed_embeddings': 0,
            'total_tokens': 0,
            'processing_time': 0,
            'start_time': None
        }
    
    def count_tokens(self, text: str) -> int:
        """Count tokens using OpenAI's tokenizer"""
        return len(self.tokenizer.encode(text))
    
    def clean_text(self, text: str) -> str:
        """Clean and normalize text"""
        # Remove excessive whitespace
        text = re.sub(r'\s+', ' ', text)
        
        # Remove page numbers and headers
        text = re.sub(r'^\d+\s*$', '', text, flags=re.MULTILINE)
        
        # Remove common PDF artifacts
        text = re.sub(r'[^\w\s\.\,\;\:\!\?\-\(\)\[\]\{\}\"\']', '', text)
        
        return text.strip()
    
    def split_text_semantically(self, text: str) -> List[str]:
        """
        Advanced semantic chunking with improved logic
        """
        text = self.clean_text(text)
        
        # Split into paragraphs
        paragraphs = re.split(r'\n\s*\n', text)
        paragraphs = [p.strip() for p in paragraphs if p.strip()]
        
        # Process large paragraphs
        processed_paragraphs = []
        for para in paragraphs:
            para_tokens = self.count_tokens(para)
            
            if para_tokens > self.max_tokens * 2:
                # Split large paragraphs by sentences
                sentences = re.split(r'(?<=[.!?])\s+', para)
                current_group = ""
                current_tokens = 0
                
                for sentence in sentences:
                    sentence_tokens = self.count_tokens(sentence)
                    
                    if current_tokens + sentence_tokens > self.max_tokens and current_group:
                        processed_paragraphs.append(current_group.strip())
                        current_group = sentence
                        current_tokens = sentence_tokens
                    else:
                        if current_group:
                            current_group += " " + sentence
                        else:
                            current_group = sentence
                        current_tokens += sentence_tokens
                
                if current_group.strip():
                    processed_paragraphs.append(current_group.strip())
            else:
                processed_paragraphs.append(para)
        
        # Create chunks with overlap
        chunks = []
        current_chunk = ""
        current_tokens = 0
        
        for i, paragraph in enumerate(processed_paragraphs):
            para_tokens = self.count_tokens(paragraph)
            
            if current_tokens + para_tokens > self.max_tokens and current_chunk:
                # Save current chunk
                chunks.append(current_chunk.strip())
                
                # Create intelligent overlap
                overlap_text = self.create_intelligent_overlap(current_chunk)
                current_chunk = overlap_text + " " + paragraph
                current_tokens = self.count_tokens(current_chunk)
            else:
                if current_chunk:
                    current_chunk += " " + paragraph
                else:
                    current_chunk = paragraph
                current_tokens += para_tokens
        
        # Add final chunk
        if current_chunk.strip():
            chunks.append(current_chunk.strip())
        
        return self.post_process_chunks(chunks)
    
    def create_intelligent_overlap(self, text: str) -> str:
        """Create intelligent overlap maintaining semantic coherence"""
        if self.count_tokens(text) <= self.overlap_tokens:
            return text
        
        sentences = re.split(r'(?<=[.!?])\s+', text)
        overlap_text = ""
        overlap_count = 0
        
        # Work backwards to maintain context
        for sentence in reversed(sentences):
            sentence_tokens = self.count_tokens(sentence)
            if overlap_count + sentence_tokens <= self.overlap_tokens:
                overlap_text = sentence + " " + overlap_text
                overlap_count += sentence_tokens
            else:
                # Try partial sentence if needed
                words = sentence.split()
                for word in reversed(words):
                    word_tokens = self.count_tokens(word + " ")
                    if overlap_count + word_tokens <= self.overlap_tokens:
                        overlap_text = word + " " + overlap_text
                        overlap_count += word_tokens
                    else:
                        break
                break
        
        return overlap_text.strip()
    
    def post_process_chunks(self, chunks: List[str]) -> List[str]:
        """Post-process chunks for quality control"""
        final_chunks = []
        
        for chunk in chunks:
            tokens = self.count_tokens(chunk)
            
            if tokens >= self.min_chunk_tokens and tokens <= self.max_chunk_tokens:
                final_chunks.append(chunk)
            elif final_chunks and tokens < self.min_chunk_tokens:
                # Merge small chunks with previous
                final_chunks[-1] = final_chunks[-1] + " " + chunk
        
        return final_chunks
    
    def extract_pdf_text(self, pdf_path: str) -> str:
        """Extract text from PDF with error handling"""
        if not os.path.exists(pdf_path):
            raise FileNotFoundError(f"PDF file not found: {pdf_path}")
        
        text = ""
        try:
            with open(pdf_path, "rb") as f:
                reader = PyPDF2.PdfReader(f)
                total_pages = len(reader.pages)
                
                logger.info(f"Processing PDF with {total_pages} pages")
                
                for i, page in enumerate(reader.pages):
                    try:
                        page_text = page.extract_text()
                        if page_text:
                            text += page_text + "\n"
                        
                        if (i + 1) % 10 == 0:
                            logger.info(f"Processed {i + 1}/{total_pages} pages")
                    
                    except Exception as e:
                        logger.warning(f"Error processing page {i + 1}: {e}")
                        continue
                
                logger.info(f"Successfully extracted text from {total_pages} pages")
                
        except Exception as e:
            logger.error(f"Error reading PDF: {e}")
            raise
        
        return text
    
    def generate_embedding(self, text: str, retries: int = 3) -> Optional[List[float]]:
        """Generate embedding with retry logic"""
        for attempt in range(retries):
            try:
                response = self.openai_client.embeddings.create(
                    input=text,
                    model=self.embedding_model
                )
                return response.data[0].embedding
            
            except Exception as e:
                logger.warning(f"Embedding attempt {attempt + 1} failed: {e}")
                if attempt < retries - 1:
                    time.sleep(2 ** attempt)  # Exponential backoff
                else:
                    logger.error(f"Failed to generate embedding after {retries} attempts")
                    return None
    
    def process_textbook(self, pdf_path: str, course_id: int = None, assignment_id: int = None) -> Dict:
        """Main processing function"""
        self.stats['start_time'] = time.time()
        
        logger.info("Starting textbook processing...")
        
        try:
            # Extract text
            logger.info(f"Extracting text from: {pdf_path}")
            text = self.extract_pdf_text(pdf_path)
            
            if not text.strip():
                raise ValueError("No text extracted from PDF")
            
            # Chunk text
            logger.info("Chunking text...")
            chunks = self.split_text_semantically(text)
            self.stats['total_chunks'] = len(chunks)
            
            logger.info(f"Created {len(chunks)} chunks")
            if chunks:
                avg_tokens = sum(self.count_tokens(c) for c in chunks) / len(chunks)
                logger.info(f"Average chunk size: {avg_tokens:.1f} tokens")
            
            # Generate embeddings and prepare records
            logger.info("Generating embeddings...")
            records = []
            
            for i, chunk in enumerate(chunks):
                logger.info(f"Processing chunk {i + 1}/{len(chunks)}")
                
                embedding = self.generate_embedding(chunk)
                if embedding:
                    record = {
                        "id": f"textbook-chunk-{i}-{int(time.time())}",
                        "values": embedding,
                        "metadata": {
                            "text": chunk,
                            "source": "textbook",
                            "tokens": self.count_tokens(chunk),
                            "chunk_index": i,
                            "course_id": course_id or 0,
                            "assignment_id": assignment_id or 0,
                            "processed_at": datetime.utcnow().isoformat(),
                            "content_type": "textbook_chunk"
                        }
                    }
                    records.append(record)
                    self.stats['successful_embeddings'] += 1
                else:
                    self.stats['failed_embeddings'] += 1
                
                # Progress update
                if (i + 1) % 10 == 0:
                    logger.info(f"Processed {i + 1}/{len(chunks)} chunks")
            
            # Upload to Pinecone in batches
            logger.info("Uploading to Pinecone...")
            for i in range(0, len(records), self.batch_size):
                batch = records[i:i + self.batch_size]
                try:
                    self.index.upsert(vectors=batch, namespace=NAMESPACE)
                    logger.info(f"Uploaded batch {i//self.batch_size + 1}/{(len(records) + self.batch_size - 1)//self.batch_size}")
                except Exception as e:
                    logger.error(f"Failed to upload batch {i//self.batch_size + 1}: {e}")
            
            # Calculate final statistics
            self.stats['processing_time'] = time.time() - self.stats['start_time']
            self.stats['total_tokens'] = sum(self.count_tokens(c) for c in chunks)
            
            logger.info("Textbook processing completed successfully!")
            self.print_statistics()
            
            return {
                'success': True,
                'statistics': self.stats,
                'chunks_processed': len(chunks),
                'embeddings_created': len(records)
            }
            
        except Exception as e:
            logger.error(f"Error processing textbook: {e}")
            return {
                'success': False,
                'error': str(e),
                'statistics': self.stats
            }
    
    def print_statistics(self):
        """Print processing statistics"""
        print("\n" + "="*50)
        print("📊 TEXTBOOK PROCESSING STATISTICS")
        print("="*50)
        print(f"📄 Total Chunks: {self.stats['total_chunks']}")
        print(f"✅ Successful Embeddings: {self.stats['successful_embeddings']}")
        print(f"❌ Failed Embeddings: {self.stats['failed_embeddings']}")
        print(f"🔢 Total Tokens: {self.stats['total_tokens']:,}")
        print(f"⏱️  Processing Time: {self.stats['processing_time']:.2f} seconds")
        print(f"🚀 Average Speed: {self.stats['total_chunks']/self.stats['processing_time']:.2f} chunks/second")
        print("="*50)
    
    def get_index_stats(self) -> Dict:
        """Get Pinecone index statistics"""
        try:
            stats = self.index.describe_index_stats()
            return {
                'success': True,
                'total_vector_count': stats.total_vector_count,
                'dimension': stats.dimension,
                'index_fullness': stats.index_fullness,
                'namespaces': stats.namespaces
            }
        except Exception as e:
            logger.error(f"Error getting index stats: {e}")
            return {'success': False, 'error': str(e)}

def main():
    """Main execution function"""
    print("🚀 Enhanced Textbook Chunking and Embedding System")
    print("="*60)
    
    # Configuration
    pdf_path = "BinderTextbooks_first300.pdf"
    course_id = 1  # Optional: associate with specific course
    assignment_id = 1  # Optional: associate with specific assignment
    
    try:
        # Initialize chunker
        chunker = TextbookChunker()
        
        # Check index status
        index_stats = chunker.get_index_stats()
        if index_stats['success']:
            print(f"📊 Index Status:")
            print(f"   Total Vectors: {index_stats['total_vector_count']}")
            print(f"   Dimension: {index_stats['dimension']}")
            print(f"   Index Fullness: {index_stats['index_fullness']:.2%}")
        
        # Process textbook
        result = chunker.process_textbook(pdf_path, course_id, assignment_id)
        
        if result['success']:
            print(f"\n🎉 Successfully processed textbook!")
            print(f"📚 Chunks created: {result['chunks_processed']}")
            print(f"🔗 Embeddings uploaded: {result['embeddings_created']}")
            print(f"\n💡 The textbook content is now available for AI grading context!")
        else:
            print(f"\n❌ Processing failed: {result['error']}")
    
    except Exception as e:
        print(f"\n❌ Error: {e}")
        logger.error(f"Main execution error: {e}")

if __name__ == "__main__":
    main() 