#!/usr/bin/env python3
"""
Test Grading with Rubric Integration
Verifies that the grading system uses both textbook and rubric content
"""

import os
import json
from pinecone import Pinecone
from openai import OpenAI
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

class GradingTester:
    def __init__(self):
        self.pc = Pinecone(api_key=os.environ.get('PINECONE_API_KEY'))
        self.openai_client = OpenAI(api_key=os.environ.get('OPENAI_API_KEY'))
        self.index_name = "aiprofessors"
    
    def test_search_with_rubric(self, query: str):
        """Test search to see if rubric content is found"""
        try:
            index = self.pc.Index(self.index_name)
            
            print(f"🔍 Testing search: '{query}'")
            
            results = index.search(
                namespace="textbook",
                query={
                    "inputs": {"text": query},
                    "top_k": 5
                }
            )
            
            if results and hasattr(results, 'result') and hasattr(results.result, 'hits'):
                print(f"✅ Found {len(results.result.hits)} results")
                
                for i, hit in enumerate(results.result.hits):
                    text = hit.fields.get('text', '') if hit.fields else ''
                    is_rubric = 'rubric' in hit._id.lower() or 'grading' in text.lower() or 'criteria' in text.lower()
                    
                    print(f"\n📄 Result {i+1}:")
                    print(f"   Score: {hit._score:.3f}")
                    print(f"   ID: {hit._id}")
                    print(f"   Type: {'📋 Rubric' if is_rubric else '📚 Textbook'}")
                    print(f"   Text: {text[:200]}...")
            else:
                print("❌ No results found")
                
        except Exception as e:
            print(f"❌ Search failed: {e}")
    
    def test_grading_with_rubric(self, question: str, student_answer: str):
        """Test grading with rubric integration"""
        try:
            # Search for relevant content
            index = self.pc.Index(self.index_name)
            
            results = index.search(
                namespace="textbook",
                query={
                    "inputs": {"text": question},
                    "top_k": 8
                }
            )
            
            if not results or not hasattr(results, 'result') or not hasattr(results.result, 'hits'):
                print("❌ No search results found")
                return
            
            # Prepare context
            context_parts = []
            rubric_found = False
            
            for i, hit in enumerate(results.result.hits):
                text = hit.fields.get('text', '') if hit.fields else ''
                is_rubric = 'rubric' in hit._id.lower() or 'grading' in text.lower() or 'criteria' in text.lower()
                
                if is_rubric:
                    rubric_found = True
                
                context_parts.append(f"Source {i+1} ({'Rubric' if is_rubric else 'Textbook'}, Score: {hit._score:.3f}): {text[:500]}")
            
            context = "\n\n".join(context_parts)
            
            print(f"📊 Search Results:")
            print(f"   Total sources: {len(context_parts)}")
            print(f"   Rubric sources: {'✅ Found' if rubric_found else '❌ Not found'}")
            
            # Create grading prompt
            prompt = f"""
You are an expert educator grading a student's answer to a business ethics question. Use the provided sources (which include both textbook content and grading rubrics) to evaluate the student's response.

QUESTION: "{question}"

STUDENT ANSWER: "{student_answer}"

SOURCES (Textbook Content + Grading Rubrics):
{context}

Please grade this answer based on:
1. **Accuracy of content** - does it match the textbook material?
2. **Completeness of response** - does it cover key points from the rubric?
3. **Clarity and organization** - is the answer well-structured?
4. **Use of relevant examples or concepts** - does it demonstrate understanding?
5. **Adherence to grading criteria** - does it meet the rubric standards?

IMPORTANT: The sources include both textbook content and grading rubrics. Use the rubric criteria to ensure consistent and fair grading.

Return your evaluation in JSON format:
{{
    "grade": "A/B/C/D/F",
    "score": 85,
    "feedback": "Detailed feedback explaining the grade",
    "strengths": ["strength 1", "strength 2"],
    "weaknesses": ["weakness 1", "weakness 2"],
    "key_points_missing": ["missing point 1", "missing point 2"],
    "key_points_correct": ["correct point 1", "correct point 2"],
    "confidence": "high/medium/low",
    "suggestions": ["suggestion 1", "suggestion 2"],
    "rubric_applied": "Brief note on which rubric criteria were used"
}}

Be fair but rigorous. A grade of A should be for excellent answers, B for good, C for satisfactory, D for poor, and F for failing.
"""

            response = self.openai_client.chat.completions.create(
                model="gpt-4o",
                messages=[
                    {"role": "system", "content": "You are an expert educator who grades student answers using both textbook content and grading rubrics. Ensure consistent and fair grading by applying rubric criteria alongside content accuracy."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.2,
                max_tokens=2000
            )
            
            content = response.choices[0].message.content
            print(f"\n🤖 AI Response:")
            print(content)
            
            # Try to parse JSON
            try:
                if '{' in content and '}' in content:
                    start = content.find('{')
                    end = content.rfind('}') + 1
                    json_str = content[start:end]
                    result = json.loads(json_str)
                    
                    print(f"\n📋 Grading Result:")
                    print(f"   Grade: {result.get('grade', 'N/A')}")
                    print(f"   Score: {result.get('score', 'N/A')}")
                    print(f"   Rubric Applied: {result.get('rubric_applied', 'N/A')}")
                    
                else:
                    print("❌ No valid JSON found in response")
                    
            except json.JSONDecodeError as e:
                print(f"❌ JSON parsing failed: {e}")
                
        except Exception as e:
            print(f"❌ Grading test failed: {e}")

def main():
    """Run grading tests with rubric integration"""
    print("🧪 Testing Grading with Rubric Integration")
    print("=" * 50)
    
    tester = GradingTester()
    
    # Test 1: Search for rubric content
    print("\n🔍 Test 1: Searching for rubric content")
    tester.test_search_with_rubric("grading criteria")
    
    # Test 2: Search for textbook content
    print("\n🔍 Test 2: Searching for textbook content")
    tester.test_search_with_rubric("utilitarianism")
    
    # Test 3: Full grading test
    print("\n🎓 Test 3: Full grading test with rubric")
    tester.test_grading_with_rubric(
        "What is utilitarianism?",
        "Utilitarianism is an ethical theory that focuses on the consequences of actions to determine what is right or wrong."
    )

if __name__ == "__main__":
    main() 