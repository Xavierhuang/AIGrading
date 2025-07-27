#!/usr/bin/env python3
"""
Simple AI Grading Test
Tests the AI grading system with the Ethics Midterm Rubric
"""

import os
import json
from pinecone import Pinecone, ServerlessSpec
from openai import OpenAI
from dotenv import load_dotenv

# Load environment variables from .env file (for local development)
load_dotenv()

# Get environment variables
PINECONE_API_KEY = os.environ.get('PINECONE_API_KEY')
OPENAI_API_KEY = os.environ.get('OPENAI_API_KEY')

if not PINECONE_API_KEY or not OPENAI_API_KEY:
    print("❌ Error: Please set PINECONE_API_KEY and OPENAI_API_KEY environment variables")
    print("   You can set them in your deployment platform or create a .env file locally")
    exit(1)

def generate_embedding(text, openai_client):
    """Generate embedding using OpenAI"""
    try:
        response = openai_client.embeddings.create(
            model="text-embedding-3-large",
            input=text,
            encoding_format="float"
        )
        return response.data[0].embedding
    except Exception as e:
        print(f"❌ Failed to generate embedding: {e}")
        return None

def search_rubric_content(query, index, openai_client, top_k=3):
    """Search for relevant rubric content"""
    try:
        # Generate embedding for the query
        query_embedding = generate_embedding(query, openai_client)
        if not query_embedding:
            return []
        
        # Search in Pinecone
        results = index.query(
            vector=query_embedding,
            namespace='grading_rubric',
            top_k=top_k,
            include_metadata=True
        )
        
        return results.matches
    except Exception as e:
        print(f"❌ Search failed: {e}")
        return []

def generate_ai_grade(submission_text, rubric_context, openai_client):
    """Generate AI grade using the rubric context"""
    
    prompt = f"""
You are an expert ethics professor grading a midterm exam. Use the following grading rubric context to evaluate the student's response.

GRADING RUBRIC CONTEXT:
{rubric_context}

STUDENT SUBMISSION:
{submission_text}

Please provide a comprehensive grade analysis in the following JSON format:

{{
    "overall_score": 85,
    "grade_letter": "B",
    "confidence": 0.85,
    "criterion_analysis": [
        {{
            "criterion": "Short Answer Questions",
            "score": 18,
            "max_score": 20,
            "feedback": "Clear and accurate answers demonstrating good understanding",
            "strengths": ["Accurate definitions", "Good examples"],
            "weaknesses": ["Could be more detailed"],
            "suggestions": ["Provide more specific examples"]
        }}
    ],
    "overall_feedback": "Strong performance with room for improvement in detail",
    "strengths": ["Good theoretical understanding", "Clear writing"],
    "weaknesses": ["Needs more specific examples", "Some arguments could be developed further"],
    "improvement_suggestions": ["Include more concrete examples", "Develop counter-arguments"]
}}

Focus on the specific criteria from the rubric and provide detailed, constructive feedback.
"""

    try:
        response = openai_client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {"role": "system", "content": "You are an expert ethics professor with deep knowledge of ethical theories and grading standards."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.3,
            max_tokens=2000
        )
        
        # Parse JSON response
        content = response.choices[0].message.content
        try:
            # Try to extract JSON from the response
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
        return {"error": f"AI grading failed: {e}"}

def test_simple_grading():
    """Test the complete AI grading system"""
    
    print("🚀 Simple AI Grading Test")
    print("="*50)
    
    # Initialize services
    try:
        pc = Pinecone(api_key=PINECONE_API_KEY)
        openai_client = OpenAI(api_key=OPENAI_API_KEY)
        index = pc.Index("professorjames-experiment-grading")
        print("✅ Connected to Pinecone and OpenAI")
    except Exception as e:
        print(f"❌ Failed to initialize services: {e}")
        return
    
    # Test submission
    test_submission = """
PART 1: SHORT ANSWER QUESTIONS

1. What's the difference between honesty and fidelity?
Honesty is being true to others by giving a correct representation of the world. Fidelity is being true to oneself and one's own oaths or contracts. A failure of honesty would be misrepresenting facts to others, while a failure of fidelity would be not upholding a commitment to oneself or one's agreement.

2. Why might a libertarian disobey zoning laws?
Libertarianism is a subset of Rights Theory focused on freedom maximization. Libertarians believe property is an extension of themselves and would dislike zoning laws as "capricious or exterior restrictions" on their land use. They believe they should be free to do what they wish with their property up to the point of interfering with others' freedom.

3. Cite a quick example from your own life where you used the "veil of ignorance" theory of fairness.
When dividing pizza with friends, I made the decision about how to cut the slices without knowing which piece I would get, ensuring fairness in the process.

4. What is a global ethics, and why is it important for the consequentialist?
For a consequentialist, Global Ethics means considering all happiness and suffering an act will cause as far into the future as possible and for as many people as possible. It's important because consequentialists must consider the full scope of consequences, not just immediate or local effects.

PART 2: ESSAY - CASE 1 (Data and Algorithm Predictions)

I would answer "No" to using these prediction systems. From a Rights Theory perspective, while I might freely choose to enter the system initially, this choice would limit my future freedom. The system would make decisions for me, reducing my sequential freedom to make authentic choices. Even if the predictions make me happy, this represents a loss of autonomy.

From a Utilitarian perspective, while the systems might increase short-term happiness, they could lead to long-term unhappiness by reducing human agency and potentially creating a society where people become dependent on algorithms for major life decisions. The Global Ethics perspective requires considering how this technology affects society as a whole, not just individual happiness.
"""
    
    print("\n📝 Test Submission:")
    print("-" * 30)
    print(test_submission[:200] + "...")
    
    # Search for relevant rubric content
    print("\n🔍 Searching for relevant rubric content...")
    rubric_matches = search_rubric_content(
        "grading criteria for short answer questions and essay structure", 
        index, 
        openai_client
    )
    
    if rubric_matches:
        print(f"✅ Found {len(rubric_matches)} relevant rubric sections")
        rubric_context = "\n\n".join([match.metadata.get('text', '') for match in rubric_matches])
        print(f"📚 Rubric context length: {len(rubric_context)} characters")
    else:
        print("❌ No rubric content found")
        rubric_context = "Use standard academic grading criteria."
    
    # Generate AI grade
    print("\n🎯 Generating AI grade...")
    grade_result = generate_ai_grade(test_submission, rubric_context, openai_client)
    
    # Display results
    print("\n📊 AI Grading Results:")
    print("="*50)
    
    if "error" in grade_result:
        print(f"❌ Error: {grade_result['error']}")
        if "raw_response" in grade_result:
            print(f"📄 Raw Response: {grade_result['raw_response']}")
    else:
        print(f"🎯 Overall Score: {grade_result.get('overall_score', 'N/A')}/100")
        print(f"📝 Grade Letter: {grade_result.get('grade_letter', 'N/A')}")
        print(f"🎯 Confidence: {grade_result.get('confidence', 'N/A')}")
        
        print(f"\n💬 Overall Feedback:")
        print(grade_result.get('overall_feedback', 'No feedback provided'))
        
        print(f"\n✅ Strengths:")
        for strength in grade_result.get('strengths', []):
            print(f"  • {strength}")
        
        print(f"\n⚠️ Weaknesses:")
        for weakness in grade_result.get('weaknesses', []):
            print(f"  • {weakness}")
        
        print(f"\n💡 Improvement Suggestions:")
        for suggestion in grade_result.get('improvement_suggestions', []):
            print(f"  • {suggestion}")
        
        print(f"\n📋 Criterion Analysis:")
        for criterion in grade_result.get('criterion_analysis', []):
            print(f"  • {criterion.get('criterion', 'Unknown')}: {criterion.get('score', 0)}/{criterion.get('max_score', 0)}")
            print(f"    Feedback: {criterion.get('feedback', 'No feedback')}")
    
    print("\n" + "="*50)
    print("🏁 Test completed!")

if __name__ == "__main__":
    test_simple_grading() 