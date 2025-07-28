#!/usr/bin/env python3
"""
Use Existing Index RAG System
Uses the user's existing 'aiprofessors' index with llama-text-embed-v2
"""

import os
import json
import time
from typing import List, Dict, Any
from pinecone import Pinecone
from openai import OpenAI
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Get environment variables
PINECONE_API_KEY = os.environ.get('PINECONE_API_KEY')
OPENAI_API_KEY = os.environ.get('OPENAI_API_KEY')

if not PINECONE_API_KEY or not OPENAI_API_KEY:
    print("❌ Error: Please set PINECONE_API_KEY and OPENAI_API_KEY environment variables")
    exit(1)

class ExistingIndexRAG:
    def __init__(self):
        self.pc = Pinecone(api_key=PINECONE_API_KEY)
        self.openai_client = OpenAI(api_key=OPENAI_API_KEY)
        self.index_name = "aiprofessorgrading"  # Use your existing index
        
    def upload_textbook_to_existing_index(self, pdf_path: str):
        """Upload textbook content to the existing index"""
        try:
            import PyPDF2
            
            # Extract text from PDF
            text = ""
            with open(pdf_path, 'rb') as file:
                pdf_reader = PyPDF2.PdfReader(file)
                for page in pdf_reader.pages:
                    text += page.extract_text() + "\n"
            
            print(f"✅ Extracted {len(text)} characters from PDF")
            
            # Create semantic chunks
            chunks = self.create_semantic_chunks(text)
            print(f"✅ Created {len(chunks)} semantic chunks")
            
            # Upload to existing index
            success = self.upload_chunks_to_index(chunks)
            
            if success:
                print("🎉 Textbook successfully uploaded to your existing index!")
                print("🚀 Your RAG system now uses the llama-text-embed-v2 model!")
            
            return success
            
        except Exception as e:
            print(f"❌ Failed to process textbook: {e}")
            return False
    
    def create_semantic_chunks(self, text: str) -> List[Dict[str, Any]]:
        """Create semantic chunks for the existing index"""
        import re
        
        # Clean text
        text = re.sub(r'\s+', ' ', text).strip()
        
        # Split into paragraphs
        paragraphs = [p.strip() for p in text.split('\n\n') if p.strip()]
        
        chunks = []
        current_chunk = ""
        chunk_id = 0
        
        for paragraph in paragraphs:
            # If paragraph is too long, split by sentences
            if len(paragraph) > 1000:
                sentences = re.split(r'[.!?]+', paragraph)
                sentences = [s.strip() for s in sentences if s.strip()]
                
                for sentence in sentences:
                    if len(current_chunk) + len(sentence) < 800:
                        current_chunk += sentence + ". "
                    else:
                        if current_chunk.strip():
                            chunks.append({
                                'id': f"chunk_{chunk_id}",
                                'text': current_chunk.strip(),
                                'length': len(current_chunk.strip())
                            })
                            chunk_id += 1
                        current_chunk = sentence + ". "
            else:
                # If adding this paragraph would make chunk too long, start new chunk
                if len(current_chunk) + len(paragraph) > 800:
                    if current_chunk.strip():
                        chunks.append({
                            'id': f"chunk_{chunk_id}",
                            'text': current_chunk.strip(),
                            'length': len(current_chunk.strip())
                        })
                        chunk_id += 1
                    current_chunk = paragraph + "\n\n"
                else:
                    current_chunk += paragraph + "\n\n"
        
        # Add the last chunk
        if current_chunk.strip():
            chunks.append({
                'id': f"chunk_{chunk_id}",
                'text': current_chunk.strip(),
                'length': len(current_chunk.strip())
            })
        
        return chunks
    
    def upload_chunks_to_index(self, chunks: List[Dict[str, Any]]):
        """Upload chunks to the existing index"""
        try:
            index = self.pc.Index(self.index_name)
            
            # Prepare data for the existing index
            records = []
            for i, chunk in enumerate(chunks):
                records.append({
                    "id": f"textbook_chunk_{i}",
                    "text": chunk['text']
                })
            
            # Upload in batches
            batch_size = 50  # Reduced batch size
            total_chunks = len(records)
            
            for i in range(0, total_chunks, batch_size):
                batch = records[i:i + batch_size]
                
                # Upsert to existing index
                index.upsert_records(
                    namespace="textbook",
                    records=batch
                )
                
                print(f"📤 Uploaded batch {i//batch_size + 1}/{(total_chunks + batch_size - 1)//batch_size}")
                time.sleep(1)  # Rate limiting
            
            print(f"✅ Successfully uploaded {total_chunks} chunks to existing index!")
            return True
            
        except Exception as e:
            print(f"❌ Failed to upload chunks: {e}")
            return False
    
    def search_with_existing_index(self, query: str, top_k: int = 5) -> List[Dict[str, Any]]:
        """Search using the existing index with hosted embedding model"""
        try:
            index = self.pc.Index(self.index_name)
            
            print(f"🔍 Searching in namespace: textbook")
            print(f"🔍 Query: {query}")
            
            # Search with hosted embedding model
            results = index.search(
                namespace="textbook",
                query={
                    "inputs": {"text": query},
                    "top_k": top_k
                }
            )
            
            print(f"🔍 Search results: {results}")
            
            # Format results
            formatted_results = []
            if results and hasattr(results, 'result') and hasattr(results.result, 'hits'):
                for hit in results.result.hits:
                    formatted_results.append({
                        'score': hit._score,
                        'text': hit.fields.get('text', '') if hit.fields else '',
                        'metadata': {'id': hit._id}
                    })
            
            return formatted_results
            
        except Exception as e:
            print(f"❌ Search failed: {e}")
            return []
    
    def synthesize_comprehensive_answer(self, query: str, search_results: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Synthesize a comprehensive answer from search results"""
        try:
            if not search_results:
                return {
                    "answer": "I couldn't find relevant information to answer your question. Please try rephrasing your query or ask about a different topic.",
                    "confidence": "low",
                    "sources_used": 0,
                    "key_points": [],
                    "limitations": ["No relevant content found"]
                }
            
            # Prepare comprehensive context from all results
            context_parts = []
            for i, result in enumerate(search_results):
                # Get more content from each result
                text = result['text'][:1500]  # Get up to 1500 chars per result
                context_parts.append(f"Source {i+1} (Relevance: {result['score']:.3f}): {text}")
            
            context = "\n\n".join(context_parts)
            
            prompt = f"""
You are an expert educator with deep knowledge of business ethics and philosophy. Synthesize a comprehensive, well-structured answer to the user's question using the provided sources.

USER QUESTION: "{query}"

SOURCES:
{context}

Please provide a comprehensive answer that:
1. Directly and clearly answers the user's question
2. Uses information from the provided sources
3. Is well-organized with clear sections
4. Includes relevant examples and explanations
5. Acknowledges any limitations or gaps in the available information
6. Uses academic language appropriate for the subject matter

Return your answer in JSON format:
{{
    "answer": "your comprehensive, well-structured answer here",
    "confidence": "high/medium/low",
    "sources_used": number_of_sources_used,
    "key_points": ["main point 1", "main point 2", "main point 3"],
    "limitations": ["any limitations of the answer"],
    "quality_score": 0.85
}}
"""

            response = self.openai_client.chat.completions.create(
                model="gpt-4o",
                messages=[
                    {"role": "system", "content": "You are an expert educator who synthesizes information from multiple sources to provide comprehensive, accurate answers."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.2,  # Lower temperature for more consistent answers
                max_tokens=2000  # Allow longer answers
            )
            
            content = response.choices[0].message.content
            try:
                if '{' in content and '}' in content:
                    start = content.find('{')
                    end = content.rfind('}') + 1
                    json_str = content[start:end]
                    return json.loads(json_str)
                else:
                    return {"error": "No valid JSON found in response", "raw_response": content}
            except json.JSONDecodeError as e:
                return {"error": f"JSON parsing failed: {e}", "raw_response": content}
                
        except Exception as e:
            return {"error": f"Synthesis failed: {e}"}
    
    def test_existing_index_rag(self, query: str) -> Dict[str, Any]:
        """Test the RAG system using the existing index"""
        print(f"🔍 Testing with Existing Index: '{query}'")
        
        # Search with existing index
        search_results = self.search_with_existing_index(query, top_k=8)
        
        if not search_results:
            return {"error": "No results found"}
        
        print(f"✅ Found {len(search_results)} results")
        print(f"📊 Best score: {search_results[0]['score']:.3f}")
        print(f"📊 Average score: {sum(r['score'] for r in search_results) / len(search_results):.3f}")
        
        # Synthesize comprehensive answer
        synthesis = self.synthesize_comprehensive_answer(query, search_results)
        
        if 'error' in synthesis:
            return {"error": f"Answer synthesis failed: {synthesis['error']}"}
        
        return {
            "query": query,
            "search_results": search_results,
            "synthesized_answer": synthesis,
            "total_sources": len(search_results),
            "best_score": search_results[0]['score'] if search_results else 0
        }

def main():
    """Main function to set up and test with existing index"""
    rag_system = ExistingIndexRAG()
    
    print("🚀 Setting up RAG with Your Existing Index")
    print("="*50)
    
    # Upload textbook to existing index
    print("📚 Uploading textbook to existing index...")
    if not rag_system.upload_textbook_to_existing_index("BinderTextbooks_first300.pdf"):
        print("❌ Failed to upload textbook")
        return
    
    # Test queries
    test_queries = [
        "What is utilitarianism?",
        "Explain business ethics",
        "What is the difference between honesty and fidelity?",
        "How to grade short answer questions"
    ]
    
    print("\n🧪 Testing RAG with Existing Index")
    print("="*40)
    
    for query in test_queries:
        print(f"\n🔍 Testing: '{query}'")
        
        result = rag_system.test_existing_index_rag(query)
        
        if 'error' in result:
            print(f"❌ Error: {result['error']}")
        else:
            print(f"✅ Found {result['total_sources']} sources")
            print(f"📊 Best score: {result['best_score']:.3f}")
            print(f"🎯 Confidence: {result['synthesized_answer'].get('confidence', 'unknown')}")
            print(f"📝 Answer: {result['synthesized_answer'].get('answer', 'No answer')[:500]}...")
            
            if result['synthesized_answer'].get('key_points'):
                print(f"🔑 Key points: {', '.join(result['synthesized_answer']['key_points'][:3])}")
            
            if result['synthesized_answer'].get('quality_score'):
                print(f"⭐ Quality score: {result['synthesized_answer']['quality_score']:.2f}")
        
        print("-" * 30)

if __name__ == "__main__":
    main() 