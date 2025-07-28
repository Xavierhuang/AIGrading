#!/usr/bin/env python3
"""
RAG Grading UI - Vercel Optimized
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

# Initialize Flask app
app = Flask(__name__)

def initialize_pinecone():
    """Initialize Pinecone client separately"""
    try:
        import pinecone
        # Try the older initialization pattern for serverless compatibility
        pc = pinecone.init(api_key=PINECONE_API_KEY)
        print("✅ Pinecone client initialized successfully")
        return pc
    except Exception as e:
        print(f"❌ Failed to initialize Pinecone: {e}")
        raise e

class RAGGradingSystem:
    def __init__(self):
        try:
            # Initialize clients separately
            self.pc = initialize_pinecone()
            self.openai_client = OpenAI(api_key=OPENAI_API_KEY)
            self.index_name = "aiprofessors"
            print("✅ RAG Grading System initialized successfully")
        except Exception as e:
            print(f"❌ Failed to initialize RAG system: {e}")
            print(f"❌ Error type: {type(e).__name__}")
            import traceback
            print(f"❌ Full traceback: {traceback.format_exc()}")
            raise e
    
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
        """Fetch the top N rubric chunks by ID"""
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
You are an expert educator grading a student's answer to a business ethics question. Use the provided sources to evaluate the student's response.

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
                    {"role": "system", "content": "You are an expert educator who grades student answers using both textbook content and grading rubrics."},
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

# Initialize grading system
grading_system = None
if PINECONE_API_KEY and OPENAI_API_KEY:
    try:
        print("🔧 Attempting to initialize RAGGradingSystem...")
        grading_system = RAGGradingSystem()
        print("✅ RAG Grading System initialized successfully")
    except Exception as e:
        print(f"❌ Failed to initialize RAG system: {e}")
        print(f"❌ Error type: {type(e).__name__}")
        import traceback
        print(f"❌ Full traceback: {traceback.format_exc()}")
        grading_system = None
else:
    print("⚠️ Warning: Missing API keys. Grading functionality will be limited.")
    print(f"PINECONE_API_KEY: {'Set' if PINECONE_API_KEY else 'Missing'}")
    print(f"OPENAI_API_KEY: {'Set' if OPENAI_API_KEY else 'Missing'}")
    if PINECONE_API_KEY:
        print(f"PINECONE_API_KEY length: {len(PINECONE_API_KEY)}")
    if OPENAI_API_KEY:
        print(f"OPENAI_API_KEY length: {len(OPENAI_API_KEY)}")

# Simple health check
@app.route('/health')
def health():
    """Health check endpoint"""
    return jsonify({
        "status": "healthy",
        "grading_system_initialized": grading_system is not None,
        "pinecone_api_key_set": bool(PINECONE_API_KEY),
        "openai_api_key_set": bool(OPENAI_API_KEY)
    })

# Simple test endpoint
@app.route('/test')
def test():
    """Test endpoint"""
    return jsonify({"message": "Flask app is working!"})

# Debug endpoint
@app.route('/debug')
def debug():
    """Debug endpoint to check environment variables"""
    return jsonify({
        "pinecone_api_key_set": bool(PINECONE_API_KEY),
        "openai_api_key_set": bool(OPENAI_API_KEY),
        "pinecone_key_length": len(PINECONE_API_KEY) if PINECONE_API_KEY else 0,
        "openai_key_length": len(OPENAI_API_KEY) if OPENAI_API_KEY else 0,
        "pinecone_key_start": PINECONE_API_KEY[:10] + "..." if PINECONE_API_KEY else "None",
        "openai_key_start": OPENAI_API_KEY[:10] + "..." if OPENAI_API_KEY else "None",
        "grading_system_initialized": grading_system is not None
    })

# Test initialization endpoint
@app.route('/test-init')
def test_init():
    """Test endpoint to try initializing the grading system"""
    try:
        if PINECONE_API_KEY and OPENAI_API_KEY:
            test_system = RAGGradingSystem()
            return jsonify({
                "success": True,
                "message": "Grading system initialized successfully",
                "pinecone_connected": True,
                "openai_connected": True
            })
        else:
            return jsonify({
                "success": False,
                "message": "Missing API keys",
                "pinecone_api_key_set": bool(PINECONE_API_KEY),
                "openai_api_key_set": bool(OPENAI_API_KEY)
            })
    except Exception as e:
        return jsonify({
            "success": False,
            "message": f"Initialization failed: {str(e)}",
            "error_type": type(e).__name__,
            "pinecone_api_key_set": bool(PINECONE_API_KEY),
            "openai_api_key_set": bool(OPENAI_API_KEY)
        })

# Version endpoint
@app.route('/version')
def version():
    """Check Pinecone version and initialization"""
    try:
        import pinecone
        # Try to get version without pkg_resources
        try:
            pinecone_version = pinecone.__version__
        except AttributeError:
            pinecone_version = "unknown"
        
        # Try to initialize Pinecone
        try:
            pc = pinecone.Pinecone(api_key=PINECONE_API_KEY)
            init_method = "pinecone.Pinecone()"
        except TypeError as e:
            if "proxies" in str(e):
                pc = pinecone.init(api_key=PINECONE_API_KEY)
                init_method = "pinecone.init()"
            else:
                raise e
        
        return jsonify({
            "pinecone_version": pinecone_version,
            "init_method": init_method,
            "api_keys_set": bool(PINECONE_API_KEY and OPENAI_API_KEY)
        })
    except Exception as e:
        return jsonify({
            "error": str(e),
            "error_type": type(e).__name__,
            "pinecone_version": "unknown"
        })

# Main grading endpoint
@app.route('/grade', methods=['POST'])
def grade():
    """Grade a student answer"""
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
        
        # Grade the answer using RAG
        result = grading_system.grade_student_answer(question, student_answer)
        return jsonify(result)
        
    except Exception as e:
        return jsonify({"error": f"Server error: {str(e)}"})

# Main page
@app.route('/')
def index():
    """Main page"""
    return """
    <!DOCTYPE html>
    <html>
    <head>
        <title>RAG Grading System</title>
        <style>
            body { font-family: Arial, sans-serif; margin: 40px; background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); min-height: 100vh; }
            .container { max-width: 800px; margin: 0 auto; background: white; border-radius: 15px; box-shadow: 0 20px 40px rgba(0,0,0,0.1); overflow: hidden; }
            .header { background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); color: white; padding: 30px; text-align: center; }
            .header h1 { font-size: 2.5em; margin-bottom: 10px; }
            .header p { font-size: 1.1em; opacity: 0.9; }
            .content { padding: 40px; }
            .form-group { margin-bottom: 25px; }
            label { display: block; margin-bottom: 8px; font-weight: 600; color: #333; }
            input, textarea { width: 100%; padding: 12px; border: 2px solid #e1e5e9; border-radius: 8px; font-size: 16px; transition: border-color 0.3s ease; }
            textarea { resize: vertical; min-height: 120px; }
            input:focus, textarea:focus { outline: none; border-color: #667eea; }
            button { background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); color: white; padding: 15px 30px; border: none; border-radius: 8px; font-size: 16px; font-weight: 600; cursor: pointer; transition: transform 0.2s ease; }
            button:hover { transform: translateY(-2px); }
            button:disabled { opacity: 0.6; cursor: not-allowed; transform: none; }
            .result { margin-top: 30px; padding: 25px; border-radius: 10px; background: #f8f9fa; border-left: 5px solid #667eea; }
            .grade { font-size: 3em; font-weight: bold; text-align: center; margin-bottom: 15px; }
            .grade.A { color: #28a745; }
            .grade.B { color: #17a2b8; }
            .grade.C { color: #ffc107; }
            .grade.D { color: #fd7e14; }
            .grade.F { color: #dc3545; }
            .score { text-align: center; font-size: 1.5em; font-weight: 600; margin-bottom: 20px; }
            .feedback { background: white; padding: 20px; border-radius: 8px; margin-bottom: 20px; }
            .feedback h3 { margin-bottom: 15px; color: #333; }
            .strengths, .weaknesses, .missing, .correct, .suggestions { margin-bottom: 15px; }
            .strengths h4 { color: #28a745; }
            .weaknesses h4 { color: #dc3545; }
            .missing h4 { color: #fd7e14; }
            .correct h4 { color: #17a2b8; }
            .suggestions h4 { color: #6f42c1; }
            .rubric h4 { color: #20c997; }
            ul { margin-left: 20px; }
            li { margin-bottom: 5px; }
            .loading { text-align: center; padding: 40px; color: #666; }
            .error { background: #f8d7da; color: #721c24; padding: 15px; border-radius: 8px; border: 1px solid #f5c6cb; }
            .confidence { text-align: center; font-size: 1.1em; margin-bottom: 15px; padding: 10px; border-radius: 5px; }
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
                        <input type="text" id="question" placeholder="Enter the question here..." required>
                    </div>
                    <div class="form-group">
                        <label for="studentAnswer">Student Answer:</label>
                        <textarea id="studentAnswer" placeholder="Enter the student's answer here..." required></textarea>
                    </div>
                    <button type="submit" id="gradeBtn">Grade Answer</button>
                </form>
                
                <div id="result" class="result" style="display: none;"></div>
            </div>
        </div>
        
        <script>
            document.getElementById('gradingForm').addEventListener('submit', async function(event) {
                event.preventDefault();
                
                const question = document.getElementById('question').value;
                const studentAnswer = document.getElementById('studentAnswer').value;
                const resultDiv = document.getElementById('result');
                const gradeBtn = document.getElementById('gradeBtn');
                
                resultDiv.style.display = 'block';
                resultDiv.innerHTML = '<div class="loading">🤔 Analyzing answer...</div>';
                gradeBtn.disabled = true;
                gradeBtn.textContent = 'Grading...';
                
                try {
                    const response = await fetch('/grade', {
                        method: 'POST',
                        headers: {
                            'Content-Type': 'application/json'
                        },
                        body: JSON.stringify({ question: question, student_answer: studentAnswer })
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

# Error handlers
@app.errorhandler(500)
def internal_error(error):
    return jsonify({"error": "Internal server error"}), 500

@app.errorhandler(404)
def not_found(error):
    return jsonify({"error": "Not found"}), 404

# For local development
if __name__ == '__main__':
    print("🚀 Starting RAG Grading UI...")
    print("📱 Open your browser to: http://localhost:5002")
    app.run(debug=True, host='0.0.0.0', port=5002) 