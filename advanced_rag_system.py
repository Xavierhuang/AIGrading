#!/usr/bin/env python3
"""
Advanced RAG System
Uses query understanding, multi-step retrieval, and result synthesis
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

class AdvancedRAGSystem:
    def __init__(self):
        self.pc = Pinecone(api_key=PINECONE_API_KEY)
        self.openai_client = OpenAI(api_key=OPENAI_API_KEY)
        self.index_name = "professorjames-experiment-higheraccuracy"
        
    def understand_query(self, query: str) -> Dict[str, Any]:
        """Understand the query and extract key information"""
        try:
            prompt = f"""
You are an expert at analyzing educational queries. Analyze this query and extract key information:

QUERY: "{query}"

Please provide a detailed analysis in JSON format:
{{
    "query_type": "definition/explanation/comparison/how_to/example",
    "main_topic": "the primary subject",
    "subtopics": ["related concepts"],
    "required_depth": "basic/intermediate/advanced",
    "expected_answer_type": "definition/explanation/step_by_step/comparison",
    "key_terms": ["important terms to search for"],
    "context_needed": ["what background information is needed"]
}}

Focus on understanding what the user is really asking for.
"""

            response = self.openai_client.chat.completions.create(
                model="gpt-4o",
                messages=[
                    {"role": "system", "content": "You are an expert at analyzing educational queries."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.2,
                max_tokens=500
            )
            
            content = response.choices[0].message.content
            try:
                if '{' in content and '}' in content:
                    start = content.find('{')
                    end = content.rfind('}') + 1
                    json_str = content[start:end]
                    return json.loads(json_str)
                else:
                    return {"error": "No valid JSON found in response"}
            except json.JSONDecodeError as e:
                return {"error": f"JSON parsing failed: {e}"}
                
        except Exception as e:
            return {"error": f"Query understanding failed: {e}"}
    
    def generate_search_queries(self, query_analysis: Dict[str, Any]) -> List[str]:
        """Generate multiple search queries based on query analysis"""
        try:
            main_topic = query_analysis.get('main_topic', '')
            key_terms = query_analysis.get('key_terms', [])
            query_type = query_analysis.get('query_type', '')
            
            # Create focused search queries
            search_queries = []
            
            # Primary query
            search_queries.append(f'"{main_topic}" definition explanation')
            
            # Add key terms
            for term in key_terms[:3]:  # Limit to 3 key terms
                search_queries.append(f'"{term}" "{main_topic}"')
            
            # Add query type specific searches
            if query_type == 'definition':
                search_queries.append(f'what is "{main_topic}"')
            elif query_type == 'comparison':
                search_queries.append(f'difference between "{main_topic}"')
            elif query_type == 'how_to':
                search_queries.append(f'how to "{main_topic}"')
            
            return search_queries
            
        except Exception as e:
            print(f"Failed to generate search queries: {e}")
            return [query_analysis.get('main_topic', '')]
    
    def generate_embedding(self, text: str) -> List[float]:
        """Generate embedding using OpenAI"""
        try:
            response = self.openai_client.embeddings.create(
                model="text-embedding-3-large",
                input=text,
                encoding_format="float"
            )
            return response.data[0].embedding
        except Exception as e:
            print(f"Failed to generate embedding: {e}")
            return None
    
    def search_multiple_queries(self, search_queries: List[str], namespace: str = "improved_textbook") -> List[Dict[str, Any]]:
        """Search using multiple queries and combine results"""
        all_results = []
        
        try:
            index = self.pc.Index(self.index_name)
            
            for query in search_queries:
                embedding = self.generate_embedding(query)
                if not embedding:
                    continue
                
                results = index.query(
                    vector=embedding,
                    namespace=namespace,
                    top_k=5,
                    include_metadata=True
                )
                
                for match in results.matches:
                    if match.score > 0.3:
                        all_results.append({
                            'score': match.score,
                            'text': match.metadata.get('text', ''),
                            'metadata': match.metadata,
                            'query': query
                        })
            
            # Remove duplicates and sort by score
            unique_results = []
            seen_texts = set()
            
            for result in sorted(all_results, key=lambda x: x['score'], reverse=True):
                # Simple deduplication based on text similarity
                text_key = result['text'][:100]  # First 100 chars as key
                if text_key not in seen_texts:
                    unique_results.append(result)
                    seen_texts.add(text_key)
            
            return unique_results[:10]  # Return top 10 unique results
            
        except Exception as e:
            print(f"Search failed: {e}")
            return []
    
    def synthesize_answer(self, query: str, query_analysis: Dict[str, Any], search_results: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Synthesize a comprehensive answer from search results"""
        try:
            if not search_results:
                return {
                    "answer": "I couldn't find relevant information to answer your question.",
                    "confidence": "low",
                    "sources_used": 0
                }
            
            # Prepare context from search results
            context_parts = []
            for i, result in enumerate(search_results[:5]):  # Use top 5 results
                context_parts.append(f"Source {i+1} (Score: {result['score']:.3f}): {result['text'][:800]}")
            
            context = "\n\n".join(context_parts)
            
            prompt = f"""
You are an expert educator. Synthesize a comprehensive answer to the user's question using the provided sources.

USER QUESTION: "{query}"

QUERY ANALYSIS:
- Type: {query_analysis.get('query_type', 'unknown')}
- Main Topic: {query_analysis.get('main_topic', 'unknown')}
- Expected Answer Type: {query_analysis.get('expected_answer_type', 'unknown')}
- Required Depth: {query_analysis.get('required_depth', 'basic')}

SOURCES:
{context}

Please provide a comprehensive, well-structured answer that:
1. Directly addresses the user's question
2. Uses information from the sources provided
3. Is organized and easy to understand
4. Includes relevant examples or explanations
5. Acknowledges any limitations in the available information

Return your answer in JSON format:
{{
    "answer": "your comprehensive answer here",
    "confidence": "high/medium/low",
    "sources_used": number_of_sources_used,
    "key_points": ["point1", "point2", "point3"],
    "limitations": ["any limitations of the answer"]
}}
"""

            response = self.openai_client.chat.completions.create(
                model="gpt-4o",
                messages=[
                    {"role": "system", "content": "You are an expert educator who synthesizes information from multiple sources to provide comprehensive answers."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.3,
                max_tokens=1500
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
    
    def advanced_search(self, query: str) -> Dict[str, Any]:
        """Perform advanced RAG search with query understanding and synthesis"""
        
        print(f"🔍 Advanced RAG Search: '{query}'")
        
        # Step 1: Understand the query
        print("📝 Understanding query...")
        query_analysis = self.understand_query(query)
        
        if 'error' in query_analysis:
            return {"error": f"Query understanding failed: {query_analysis['error']}"}
        
        print(f"✅ Query type: {query_analysis.get('query_type', 'unknown')}")
        print(f"✅ Main topic: {query_analysis.get('main_topic', 'unknown')}")
        
        # Step 2: Generate search queries
        print("🔍 Generating search queries...")
        search_queries = self.generate_search_queries(query_analysis)
        print(f"✅ Generated {len(search_queries)} search queries")
        
        # Step 3: Search with multiple queries
        print("🔎 Searching with multiple queries...")
        search_results = self.search_multiple_queries(search_queries)
        print(f"✅ Found {len(search_results)} relevant results")
        
        # Step 4: Synthesize answer
        print("🧠 Synthesizing comprehensive answer...")
        synthesis = self.synthesize_answer(query, query_analysis, search_results)
        
        if 'error' in synthesis:
            return {"error": f"Answer synthesis failed: {synthesis['error']}"}
        
        # Return comprehensive result
        return {
            "query": query,
            "query_analysis": query_analysis,
            "search_results": search_results,
            "synthesized_answer": synthesis,
            "total_sources": len(search_results),
            "best_score": search_results[0]['score'] if search_results else 0
        }

def test_advanced_rag():
    """Test the advanced RAG system"""
    
    rag_system = AdvancedRAGSystem()
    
    test_queries = [
        "What is utilitarianism?",
        "Explain business ethics",
        "What is the difference between honesty and fidelity?",
        "How to grade short answer questions"
    ]
    
    print("🧪 Testing Advanced RAG System")
    print("="*50)
    
    for query in test_queries:
        print(f"\n🔍 Testing: '{query}'")
        
        result = rag_system.advanced_search(query)
        
        if 'error' in result:
            print(f"❌ Error: {result['error']}")
        else:
            print(f"✅ Found {result['total_sources']} sources")
            print(f"📊 Best score: {result['best_score']:.3f}")
            print(f"🎯 Confidence: {result['synthesized_answer'].get('confidence', 'unknown')}")
            print(f"📝 Answer: {result['synthesized_answer'].get('answer', 'No answer')[:200]}...")
        
        print("-" * 30)

if __name__ == "__main__":
    test_advanced_rag() 