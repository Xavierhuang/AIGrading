#!/usr/bin/env python3
"""
RAG Grading UI
Simple web interface for testing the RAG grading system
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

class RAGGradingSystem:
    def __init__(self):
        self.pc = Pinecone(api_key=PINECONE_API_KEY)
        self.openai_client = OpenAI(api_key=OPENAI_API_KEY)
        self.index_name = "aiprofessors"
        
    def search_with_existing_index(self, query: str, top_k: int = 8) -> list:
        """Search using the existing index with hosted embedding model"""
        try:
            index = self.pc.Index(self.index_name)
            
            # Search with hosted embedding model
            results = index.search(
                namespace="textbook",
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
    
    def fetch_top_rubric_chunks(self, top_n: int = 3) -> list:
        """Fetch the top N rubric chunks by ID (rubric_chunk_0, rubric_chunk_1, ...)"""
        try:
            index = self.pc.Index(self.index_name)
            rubric_chunks = []
            for i in range(top_n):
                chunk_id = f"rubric_chunk_{i}"
                res = index.fetch(namespace="textbook", ids=[chunk_id])
                if res and hasattr(res, 'vectors') and chunk_id in res.vectors:
                    text = res.vectors[chunk_id].fields.get('text', '') if hasattr(res.vectors[chunk_id], 'fields') else ''
                    if text:
                        rubric_chunks.append({
                            'id': chunk_id,
                            'text': text
                        })
            return rubric_chunks
        except Exception as e:
            print(f"❌ Failed to fetch rubric chunks: {e}")
            return []
    
    def grade_student_answer(self, question: str, student_answer: str) -> dict:
        """Grade a student answer using RAG"""
        try:
            # Search for relevant textbook content
            search_results = self.search_with_existing_index(question, top_k=8)
            # Fetch rubric chunks
            rubric_chunks = self.fetch_top_rubric_chunks(top_n=3)
            if not search_results and not rubric_chunks:
                return {
                    "error": "No relevant content found",
                    "grade": "F",
                    "score": 0,
                    "feedback": "Could not find relevant information to grade this answer."
                }
            
            # Prepare context from search results
            context_parts = []
            for i, result in enumerate(search_results):
                text = result['text'][:1000]  # Limit context per result
                context_parts.append(f"Source {i+1} (Relevance: {result['score']:.3f}): {text}")
            
            context = "\n\n".join(context_parts)
            
            # Prepare rubric context
            rubric_parts = []
            for i, chunk in enumerate(rubric_chunks):
                rubric_parts.append(f"Rubric {i+1}: {chunk['text'][:1000]}")
            rubric_context = "\n\n".join(rubric_parts)
            
            # Create grading prompt
            prompt = f"""
You are an expert educator grading a student's answer to a business ethics question. Use the provided sources (which include both textbook content and grading rubrics) to evaluate the student's response.

QUESTION: "{question}"

STUDENT ANSWER: "{student_answer}"

TEXTBOOK SOURCES:
{context}

RUBRIC CRITERIA:
{rubric_context}

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
            return {"error": f"Grading failed: {e}"}

# Initialize Flask app
app = Flask(__name__)

# Initialize grading system only if API keys are available
grading_system = None
if PINECONE_API_KEY and OPENAI_API_KEY:
    try:
        grading_system = RAGGradingSystem()
        print("✅ RAG Grading System initialized successfully")
    except Exception as e:
        print(f"❌ Failed to initialize RAG system: {e}")
        grading_system = None
else:
    print("⚠️ Warning: Missing API keys. Grading functionality will be limited.")
    print(f"PINECONE_API_KEY: {'Set' if PINECONE_API_KEY else 'Missing'}")
    print(f"OPENAI_API_KEY: {'Set' if OPENAI_API_KEY else 'Missing'}")

# HTML template
HTML_TEMPLATE = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>RAG Grading System</title>
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
        
        .grade {
            font-size: 3em;
            font-weight: bold;
            text-align: center;
            margin-bottom: 15px;
        }
        
        .grade.A { color: #28a745; }
        .grade.B { color: #17a2b8; }
        .grade.C { color: #ffc107; }
        .grade.D { color: #fd7e14; }
        .grade.F { color: #dc3545; }
        
        .score {
            text-align: center;
            font-size: 1.5em;
            font-weight: 600;
            margin-bottom: 20px;
        }
        
        .feedback {
            background: white;
            padding: 20px;
            border-radius: 8px;
            margin-bottom: 20px;
        }
        
        .feedback h3 {
            margin-bottom: 15px;
            color: #333;
        }
        
        .strengths, .weaknesses, .missing, .correct, .suggestions {
            margin-bottom: 15px;
        }
        
        .strengths h4 { color: #28a745; }
        .weaknesses h4 { color: #dc3545; }
        .missing h4 { color: #fd7e14; }
        .correct h4 { color: #17a2b8; }
        .suggestions h4 { color: #6f42c1; }
        .rubric h4 { color: #20c997; }
        
        ul {
            margin-left: 20px;
        }
        
        li {
            margin-bottom: 5px;
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
            text-align: center;
            font-size: 1.1em;
            margin-bottom: 15px;
            padding: 10px;
            border-radius: 5px;
        }
        
        .confidence.high { background: #d4edda; color: #155724; }
        .confidence.medium { background: #fff3cd; color: #856404; }
        .confidence.low { background: #f8d7da; color: #721c24; }
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>🎓 RAG Grading System</h1>
            <p>AI-Powered Student Answer Evaluation with Textbook & Rubric Context</p>
        </div>
        
        <div class="content">
            <form id="gradingForm">
                <div class="form-group">
                    <label for="question">Question:</label>
                    <input type="text" id="question" name="question" placeholder="Enter the question here..." required>
                </div>
                
                <div class="form-group">
                    <label for="student_answer">Student Answer:</label>
                    <textarea id="student_answer" name="student_answer" placeholder="Enter the student's answer here..." required></textarea>
                </div>
                
                <button type="submit" class="btn" id="gradeBtn">Grade Answer</button>
            </form>
            
            <div id="result" style="display: none;"></div>
        </div>
    </div>
    
    <script>
        document.getElementById('gradingForm').addEventListener('submit', async function(e) {
            e.preventDefault();
            
            const question = document.getElementById('question').value;
            const studentAnswer = document.getElementById('student_answer').value;
            const gradeBtn = document.getElementById('gradeBtn');
            const resultDiv = document.getElementById('result');
            
            // Show loading
            gradeBtn.disabled = true;
            gradeBtn.textContent = 'Grading...';
            resultDiv.style.display = 'block';
            resultDiv.innerHTML = '<div class="loading">🤔 Analyzing answer...</div>';
            
            try {
                const response = await fetch('/grade', {
                    method: 'POST',
                    headers: {
                        'Content-Type': 'application/json',
                    },
                    body: JSON.stringify({
                        question: question,
                        student_answer: studentAnswer
                    })
                });
                
                const result = await response.json();
                
                if (result.error) {
                    resultDiv.innerHTML = `<div class="error">❌ Error: ${result.error}</div>`;
                } else {
                    const grade = result.grade || 'N/A';
                    const score = result.score || 0;
                    const feedback = result.feedback || 'No feedback available';
                    const confidence = result.confidence || 'unknown';
                    
                    let html = `
                        <div class="result">
                            <div class="grade ${grade}">${grade}</div>
                            <div class="score">Score: ${score}/100</div>
                            <div class="confidence ${confidence}">Confidence: ${confidence.toUpperCase()}</div>
                            
                            <div class="feedback">
                                <h3>📝 Feedback</h3>
                                <p>${feedback}</p>
                            </div>
                    `;
                    
                    if (result.strengths && result.strengths.length > 0) {
                        html += `
                            <div class="strengths">
                                <h4>✅ Strengths</h4>
                                <ul>${result.strengths.map(s => `<li>${s}</li>`).join('')}</ul>
                            </div>
                        `;
                    }
                    
                    if (result.weaknesses && result.weaknesses.length > 0) {
                        html += `
                            <div class="weaknesses">
                                <h4>❌ Areas for Improvement</h4>
                                <ul>${result.weaknesses.map(w => `<li>${w}</li>`).join('')}</ul>
                            </div>
                        `;
                    }
                    
                    if (result.key_points_correct && result.key_points_correct.length > 0) {
                        html += `
                            <div class="correct">
                                <h4>🎯 Correct Points</h4>
                                <ul>${result.key_points_correct.map(p => `<li>${p}</li>`).join('')}</ul>
                            </div>
                        `;
                    }
                    
                    if (result.key_points_missing && result.key_points_missing.length > 0) {
                        html += `
                            <div class="missing">
                                <h4>⚠️ Missing Points</h4>
                                <ul>${result.key_points_missing.map(p => `<li>${p}</li>`).join('')}</ul>
                            </div>
                        `;
                    }
                    
                    if (result.suggestions && result.suggestions.length > 0) {
                        html += `
                            <div class="suggestions">
                                <h4>💡 Suggestions</h4>
                                <ul>${result.suggestions.map(s => `<li>${s}</li>`).join('')}</ul>
                            </div>
                        `;
                    }
                    
                    if (result.rubric_applied) {
                        html += `
                            <div class="rubric">
                                <h4>📋 Rubric Applied</h4>
                                <p>${result.rubric_applied}</p>
                            </div>
                        `;
                    }
                    
                    html += '</div>';
                    resultDiv.innerHTML = html;
                }
                
            } catch (error) {
                resultDiv.innerHTML = `<div class="error">❌ Error: ${error.message}</div>`;
            } finally {
                gradeBtn.disabled = false;
                gradeBtn.textContent = 'Grade Answer';
            }
        });
    </script>
</body>
</html>
"""

@app.route('/')
def index():
    return render_template_string(HTML_TEMPLATE)

@app.route('/health')
def health():
    """Health check endpoint for debugging"""
    return jsonify({
        "status": "healthy",
        "grading_system_initialized": grading_system is not None,
        "pinecone_api_key_set": bool(PINECONE_API_KEY),
        "openai_api_key_set": bool(OPENAI_API_KEY)
    })

@app.route('/grade', methods=['POST'])
def grade():
    try:
        if not grading_system:
            return jsonify({
                "error": "Grading system not initialized. Please check API keys.",
                "grade": "F",
                "score": 0,
                "feedback": "System error: Grading system unavailable."
            })
        
        data = request.get_json()
        question = data.get('question', '')
        student_answer = data.get('student_answer', '')
        
        if not question or not student_answer:
            return jsonify({"error": "Question and student answer are required"})
        
        # Grade the answer
        result = grading_system.grade_student_answer(question, student_answer)
        
        return jsonify(result)
        
    except Exception as e:
        return jsonify({"error": f"Server error: {str(e)}"})

# For Vercel deployment - export the app
app.debug = False

# Add error handling for Vercel
@app.errorhandler(500)
def internal_error(error):
    return jsonify({"error": "Internal server error", "details": str(error)}), 500

@app.errorhandler(404)
def not_found(error):
    return jsonify({"error": "Not found"}), 404

# Only run locally if not on Vercel
if __name__ == '__main__':
    print("🚀 Starting RAG Grading UI...")
    print("📱 Open your browser to: http://localhost:5002")
    print("🎓 Ready to grade student answers!")
    app.run(debug=True) 