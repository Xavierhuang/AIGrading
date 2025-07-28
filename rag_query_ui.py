#!/usr/bin/env python3
"""
RAG Query UI
Simple web interface for querying the RAG system
"""

import os
import json
from flask import Flask, render_template_string, request, jsonify
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

class RAGQuerySystem:
    def __init__(self):
        self.pc = Pinecone(api_key=PINECONE_API_KEY)
        self.openai_client = OpenAI(api_key=OPENAI_API_KEY)
        self.index_name = "aiprofessors"
        
    def search_with_existing_index(self, query: str, top_k: int = 8, namespace: str = "textbook") -> list:
        """Search using the existing index with hosted embedding model"""
        try:
            index = self.pc.Index(self.index_name)
            
            # Search with hosted embedding model
            results = index.search(
                namespace=namespace,
                query={
                    "inputs": {"text": query},
                    "top_k": top_k
                }
            )
            
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
    
    def query_rag(self, question: str, namespace: str = "textbook") -> dict:
        """Query RAG system for an answer"""
        try:
            # Search for relevant content
            search_results = self.search_with_existing_index(question, top_k=8, namespace=namespace)
            
            if not search_results:
                return {
                    "error": "No relevant content found",
                    "answer": "I couldn't find relevant information to answer your question. Please try rephrasing your query or ask about a different topic."
                }
            
            # Prepare context from search results
            context_parts = []
            for i, result in enumerate(search_results):
                text = result['text'][:800]  # Limit context per result
                context_parts.append(f"Source {i+1} (Relevance: {result['score']:.3f}): {text}")
            
            context = "\n\n".join(context_parts)
            
            # Create answer synthesis prompt
            prompt = f"""
You are an expert educator providing precise, concise answers to business ethics questions. Use the provided textbook sources to give a clear, focused response.

QUESTION: "{question}"

TEXTBOOK SOURCES:
{context}

Please provide a precise and concise answer that:
1. Directly answers the question in 2-3 clear sentences
2. Uses the most relevant information from the sources
3. Avoids unnecessary details and repetition
4. Focuses on the core concept or definition
5. Uses simple, clear language

Guidelines for concise answers:
- Start with a direct definition or answer
- Include only the most essential details
- Use bullet points or numbered lists for multiple concepts
- Keep each sentence focused and clear
- Avoid lengthy explanations unless specifically requested

Return your answer in JSON format:
{{
    "answer": "your precise and concise answer here (2-3 sentences max)",
    "confidence": "high/medium/low",
    "key_points": ["essential point 1", "essential point 2"],
    "sources_used": number_of_sources,
    "quality_score": 0.85,
    "search_stats": {{
        "total_sources": number_of_sources,
        "best_score": highest_relevance_score,
        "average_score": average_relevance_score
    }}
}}
"""

            response = self.openai_client.chat.completions.create(
                model="gpt-4o",
                messages=[
                    {"role": "system", "content": "You are an expert educator who provides precise, concise answers. Focus on clarity and brevity while maintaining accuracy. Give direct answers in 2-3 sentences maximum."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.2,
                max_tokens=1000
            )
            
            content = response.choices[0].message.content
            try:
                if '{' in content and '}' in content:
                    start = content.find('{')
                    end = content.rfind('}') + 1
                    json_str = content[start:end]
                    result = json.loads(json_str)
                    
                    # Add search statistics
                    if search_results:
                        result['search_stats'] = {
                            'total_sources': len(search_results),
                            'best_score': search_results[0]['score'],
                            'average_score': sum(r['score'] for r in search_results) / len(search_results)
                        }
                    
                    return result
                else:
                    return {"error": "No valid JSON found in response", "raw_response": content}
            except json.JSONDecodeError as e:
                return {"error": f"JSON parsing failed: {e}", "raw_response": content}
                
        except Exception as e:
            return {"error": f"Query failed: {e}"}

# Initialize the query system
query_system = RAGQuerySystem()

# Flask app
app = Flask(__name__)

# HTML template
HTML_TEMPLATE = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>RAG Query System</title>
    <style>
        * {
            margin: 0;
            padding: 0;
            box-sizing: border-box;
        }
        
        body {
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            min-height: 100vh;
            padding: 20px;
        }
        
        .container {
            max-width: 1200px;
            margin: 0 auto;
            background: white;
            border-radius: 15px;
            box-shadow: 0 20px 40px rgba(0,0,0,0.1);
            overflow: hidden;
        }
        
        .header {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            padding: 30px;
            text-align: center;
        }
        
        .header h1 {
            font-size: 2.5em;
            margin-bottom: 10px;
        }
        
        .header p {
            font-size: 1.1em;
            opacity: 0.9;
        }
        
        .content {
            padding: 40px;
        }
        
        .form-group {
            margin-bottom: 25px;
        }
        
        label {
            display: block;
            margin-bottom: 8px;
            font-weight: 600;
            color: #333;
        }
        
        input[type="text"], textarea {
            width: 100%;
            padding: 12px;
            border: 2px solid #e1e5e9;
            border-radius: 8px;
            font-size: 16px;
            transition: border-color 0.3s ease;
        }
        
        input[type="text"]:focus, textarea:focus {
            outline: none;
            border-color: #667eea;
        }
        
        textarea {
            resize: vertical;
            min-height: 120px;
        }
        
        .btn {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            padding: 15px 30px;
            border: none;
            border-radius: 8px;
            font-size: 16px;
            font-weight: 600;
            cursor: pointer;
            transition: transform 0.2s ease;
        }
        
        .btn:hover {
            transform: translateY(-2px);
        }
        
        .btn:disabled {
            opacity: 0.6;
            cursor: not-allowed;
            transform: none;
        }
        
        .result {
            margin-top: 30px;
            padding: 25px;
            border-radius: 10px;
            background: #f8f9fa;
            border-left: 5px solid #667eea;
        }
        
        .answer {
            background: white;
            padding: 20px;
            border-radius: 8px;
            margin-bottom: 20px;
            line-height: 1.6;
        }
        
        .answer h3 {
            margin-bottom: 15px;
            color: #333;
        }
        
        .stats {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
            gap: 15px;
            margin-bottom: 20px;
        }
        
        .stat-card {
            background: white;
            padding: 15px;
            border-radius: 8px;
            text-align: center;
            border: 1px solid #e1e5e9;
        }
        
        .stat-card h4 {
            color: #667eea;
            margin-bottom: 5px;
        }
        
        .stat-card .value {
            font-size: 1.5em;
            font-weight: bold;
            color: #333;
        }
        
        .key-points {
            background: white;
            padding: 20px;
            border-radius: 8px;
            margin-bottom: 20px;
        }
        
        .key-points h4 {
            color: #28a745;
            margin-bottom: 15px;
        }
        
        ul {
            margin-left: 20px;
        }
        
        li {
            margin-bottom: 8px;
            line-height: 1.4;
        }
        
        .loading {
            text-align: center;
            padding: 40px;
            color: #666;
        }
        
        .error {
            background: #f8d7da;
            color: #721c24;
            padding: 15px;
            border-radius: 8px;
            border: 1px solid #f5c6cb;
        }
        
        .confidence {
            display: inline-block;
            padding: 5px 15px;
            border-radius: 20px;
            font-weight: 600;
            margin-bottom: 15px;
        }
        
        .confidence.high { background: #d4edda; color: #155724; }
        .confidence.medium { background: #fff3cd; color: #856404; }
        .confidence.low { background: #f8d7da; color: #721c24; }
        
        .sample-queries {
            margin-top: 30px;
            padding: 20px;
            background: #f8f9fa;
            border-radius: 8px;
        }
        
        .sample-queries h3 {
            margin-bottom: 15px;
            color: #333;
        }
        
        .sample-btn {
            background: #6c757d;
            color: white;
            padding: 8px 15px;
            border: none;
            border-radius: 5px;
            margin: 5px;
            cursor: pointer;
            font-size: 14px;
        }
        
        .sample-btn:hover {
            background: #5a6268;
        }
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>🔍 RAG Query System</h1>
            <p>Ask questions about Business Ethics</p>
        </div>
        
        <div class="content">
            <form id="queryForm">
                <div class="form-group">
                    <label for="question">Your Question:</label>
                    <input type="text" id="question" name="question" placeholder="Ask any question about business ethics..." required>
                </div>
                
                <button type="submit" class="btn" id="queryBtn">Get Answer</button>
            </form>
            
            <div id="result" style="display: none;"></div>
            
            <div class="sample-queries">
                <h3>💡 Sample Questions:</h3>
                <h4>📚 Textbook Questions:</h4>
                <button class="sample-btn" onclick="setQuestion('What is utilitarianism?')">What is utilitarianism?</button>
                <button class="sample-btn" onclick="setQuestion('Explain the difference between honesty and fidelity')">Honesty vs Fidelity</button>
                <button class="sample-btn" onclick="setQuestion('What is business ethics?')">What is business ethics?</button>
                <button class="sample-btn" onclick="setQuestion('How does corporate social responsibility work?')">Corporate Social Responsibility</button>
                <button class="sample-btn" onclick="setQuestion('What are the main ethical theories in business?')">Ethical Theories</button>
                
                <h4>📋 Syllabus Questions:</h4>
                <button class="sample-btn" onclick="setQuestion('What is the grading policy?')">Grading Policy</button>
                <button class="sample-btn" onclick="setQuestion('What are the course objectives?')">Course Objectives</button>
                <button class="sample-btn" onclick="setQuestion('What are the required materials?')">Required Materials</button>
                <button class="sample-btn" onclick="setQuestion('What is the attendance policy?')">Attendance Policy</button>
                <button class="sample-btn" onclick="setQuestion('What are the late work policies?')">Late Work Policies</button>
            </div>
        </div>
    </div>
    
    <script>
        function setQuestion(question) {
            document.getElementById('question').value = question;
        }
        
        document.getElementById('queryForm').addEventListener('submit', async function(e) {
            e.preventDefault();
            
            const question = document.getElementById('question').value;
            const queryBtn = document.getElementById('queryBtn');
            const resultDiv = document.getElementById('result');
            
            // Show loading
            queryBtn.disabled = true;
            queryBtn.textContent = 'Searching...';
            resultDiv.style.display = 'block';
            resultDiv.innerHTML = '<div class="loading">🔍 Searching for relevant information...</div>';
            
            try {
                const response = await fetch('/query', {
                    method: 'POST',
                    headers: {
                        'Content-Type': 'application/json',
                    },
                    body: JSON.stringify({
                        question: question
                    })
                });
                
                const result = await response.json();
                
                if (result.error) {
                    resultDiv.innerHTML = `<div class="error">❌ Error: ${result.error}</div>`;
                } else {
                    const answer = result.answer || 'No answer available';
                    const confidence = result.confidence || 'unknown';
                    const qualityScore = result.quality_score || 'N/A';
                    const sourcesUsed = result.sources_used || 'N/A';
                    const keyPoints = result.key_points || [];
                    const searchStats = result.search_stats || {};
                    
                    let html = `
                        <div class="result">
                            <div class="answer">
                                <h3>📝 Answer</h3>
                                <p>${answer}</p>
                            </div>
                            
                            <div class="stats">
                                <div class="stat-card">
                                    <h4>Confidence</h4>
                                    <div class="value confidence ${confidence}">${confidence.toUpperCase()}</div>
                                </div>
                                <div class="stat-card">
                                    <h4>Quality Score</h4>
                                    <div class="value">${qualityScore}</div>
                                </div>
                                <div class="stat-card">
                                    <h4>Sources Used</h4>
                                    <div class="value">${sourcesUsed}</div>
                                </div>
                                <div class="stat-card">
                                    <h4>Best Match Score</h4>
                                    <div class="value">${searchStats.best_score ? searchStats.best_score.toFixed(3) : 'N/A'}</div>
                                </div>
                            </div>
                    `;
                    
                    if (keyPoints.length > 0) {
                        html += `
                            <div class="key-points">
                                <h4>🔑 Key Points</h4>
                                <ul>${keyPoints.map(point => `<li>${point}</li>`).join('')}</ul>
                            </div>
                        `;
                    }
                    
                    html += '</div>';
                    resultDiv.innerHTML = html;
                }
                
            } catch (error) {
                resultDiv.innerHTML = `<div class="error">❌ Error: ${error.message}</div>`;
            } finally {
                queryBtn.disabled = false;
                queryBtn.textContent = 'Get Answer';
            }
        });
    </script>
</body>
</html>
"""

@app.route('/')
def index():
    return render_template_string(HTML_TEMPLATE)

@app.route('/query', methods=['POST'])
def query():
    try:
        data = request.get_json()
        question = data.get('question', '')
        
        if not question:
            return jsonify({"error": "Question is required"})
        
        # Determine namespace based on question content
        syllabus_keywords = [
            'grading', 'policy', 'attendance', 'late', 'makeup', 'exam', 'assignment',
            'syllabus', 'course', 'objective', 'material', 'textbook', 'schedule',
            'office hour', 'contact', 'outline', 'module', 'unit', 'week', 'final',
            'midterm', 'participation', 'discussion', 'board', 'academic', 'integrity'
        ]
        
        question_lower = question.lower()
        is_syllabus_query = any(keyword in question_lower for keyword in syllabus_keywords)
        
        # Always try both namespaces and use the best result
        syllabus_result = query_system.query_rag(question, namespace="syllabus")
        textbook_result = query_system.query_rag(question, namespace="textbook")
        
        # Compare results and use the better one
        syllabus_score = syllabus_result.get('search_stats', {}).get('best_score', 0) if syllabus_result.get('search_stats') else 0
        textbook_score = textbook_result.get('search_stats', {}).get('best_score', 0) if textbook_result.get('search_stats') else 0
        
        if syllabus_score > textbook_score and syllabus_result.get('answer'):
            result = syllabus_result
            result['namespace_used'] = 'syllabus'
            result['score_comparison'] = f'syllabus: {syllabus_score:.3f} vs textbook: {textbook_score:.3f}'
        elif textbook_result.get('answer'):
            result = textbook_result
            result['namespace_used'] = 'textbook'
            result['score_comparison'] = f'textbook: {textbook_score:.3f} vs syllabus: {syllabus_score:.3f}'
        else:
            # Fallback to whichever has any result
            result = syllabus_result if syllabus_result.get('answer') else textbook_result
            result['namespace_used'] = 'syllabus' if syllabus_result.get('answer') else 'textbook'
        
        return jsonify(result)
        
    except Exception as e:
        return jsonify({"error": f"Server error: {str(e)}"})

if __name__ == '__main__':
    print("🚀 Starting RAG Query UI...")
    print("📱 Open your browser to: http://localhost:5003")
    print("🔍 Ready to answer questions!")
    app.run(host='0.0.0.0', port=5003, debug=True) 