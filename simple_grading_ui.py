#!/usr/bin/env python3
"""
Simple AI Grading UI
A web interface for testing the AI grading system
"""

import os
import json
from flask import Flask, render_template, request, jsonify
from pinecone import Pinecone, ServerlessSpec
from openai import OpenAI

# Get environment variables (works both locally and on Render)
PINECONE_API_KEY = os.environ.get('PINECONE_API_KEY')
OPENAI_API_KEY = os.environ.get('OPENAI_API_KEY')

if not PINECONE_API_KEY or not OPENAI_API_KEY:
    print("❌ Error: Please set PINECONE_API_KEY and OPENAI_API_KEY environment variables")
    print("   You can set them in your deployment platform or create a .env file locally")
    exit(1)

app = Flask(__name__)

# Initialize services
try:
    pc = Pinecone(api_key=PINECONE_API_KEY)
    openai_client = OpenAI(api_key=OPENAI_API_KEY)
    index = pc.Index("professorjames-experiment-grading")
    print("✅ Connected to Pinecone and OpenAI")
except Exception as e:
    print(f"❌ Failed to initialize services: {e}")
    index = None
    openai_client = None

def generate_embedding(text):
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

def search_rubric_content(query, top_k=3):
    """Search for relevant rubric content"""
    try:
        # Generate embedding for the query
        query_embedding = generate_embedding(query)
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

def generate_ai_grade(submission_text, rubric_context):
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

@app.route('/')
def index():
    return render_template('grading_ui.html')

@app.route('/grade', methods=['POST'])
def grade_submission():
    try:
        data = request.get_json()
        submission_text = data.get('submission', '')
        
        if not submission_text.strip():
            return jsonify({'error': 'No submission text provided'})
        
        # Search for relevant rubric content
        rubric_matches = search_rubric_content(
            "grading criteria for short answer questions and essay structure"
        )
        
        if rubric_matches:
            rubric_context = "\n\n".join([match.metadata.get('text', '') for match in rubric_matches])
        else:
            rubric_context = "Use standard academic grading criteria."
        
        # Generate AI grade
        grade_result = generate_ai_grade(submission_text, rubric_context)
        
        return jsonify(grade_result)
        
    except Exception as e:
        return jsonify({'error': f'Grading failed: {str(e)}'})

if __name__ == '__main__':
    # Create templates directory if it doesn't exist
    os.makedirs('templates', exist_ok=True)
    
    # Create the HTML template
    html_template = '''
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>AI Grading System</title>
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
            border-radius: 20px;
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
            font-size: 1.2em;
            opacity: 0.9;
        }
        
        .content {
            padding: 40px;
        }
        
        .input-section {
            margin-bottom: 40px;
        }
        
        .input-section h2 {
            color: #333;
            margin-bottom: 20px;
            font-size: 1.8em;
        }
        
        .form-group {
            margin-bottom: 20px;
        }
        
        label {
            display: block;
            margin-bottom: 8px;
            font-weight: 600;
            color: #555;
        }
        
        textarea {
            width: 100%;
            min-height: 300px;
            padding: 15px;
            border: 2px solid #e1e5e9;
            border-radius: 10px;
            font-size: 16px;
            font-family: inherit;
            resize: vertical;
            transition: border-color 0.3s ease;
        }
        
        textarea:focus {
            outline: none;
            border-color: #667eea;
        }
        
        .button {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            border: none;
            padding: 15px 30px;
            border-radius: 10px;
            font-size: 16px;
            font-weight: 600;
            cursor: pointer;
            transition: transform 0.2s ease;
        }
        
        .button:hover {
            transform: translateY(-2px);
        }
        
        .button:disabled {
            opacity: 0.6;
            cursor: not-allowed;
            transform: none;
        }
        
        .results-section {
            margin-top: 40px;
            display: none;
        }
        
        .results-section h2 {
            color: #333;
            margin-bottom: 20px;
            font-size: 1.8em;
        }
        
        .grade-summary {
            background: #f8f9fa;
            padding: 20px;
            border-radius: 10px;
            margin-bottom: 20px;
        }
        
        .grade-score {
            font-size: 2em;
            font-weight: bold;
            color: #667eea;
        }
        
        .grade-letter {
            font-size: 1.5em;
            color: #764ba2;
            margin-left: 10px;
        }
        
        .confidence {
            color: #666;
            font-size: 0.9em;
            margin-top: 5px;
        }
        
        .feedback-section {
            margin-top: 20px;
        }
        
        .feedback-item {
            margin-bottom: 15px;
        }
        
        .feedback-item h3 {
            color: #333;
            margin-bottom: 10px;
            font-size: 1.2em;
        }
        
        .feedback-list {
            list-style: none;
            padding-left: 0;
        }
        
        .feedback-list li {
            padding: 8px 0;
            border-bottom: 1px solid #eee;
        }
        
        .feedback-list li:before {
            content: "•";
            color: #667eea;
            font-weight: bold;
            margin-right: 8px;
        }
        
        .criterion-analysis {
            background: #f8f9fa;
            padding: 20px;
            border-radius: 10px;
            margin-top: 20px;
        }
        
        .criterion-item {
            margin-bottom: 15px;
            padding: 15px;
            background: white;
            border-radius: 8px;
            border-left: 4px solid #667eea;
        }
        
        .criterion-header {
            display: flex;
            justify-content: space-between;
            align-items: center;
            margin-bottom: 10px;
        }
        
        .criterion-name {
            font-weight: 600;
            color: #333;
        }
        
        .criterion-score {
            font-weight: bold;
            color: #667eea;
        }
        
        .loading {
            text-align: center;
            padding: 40px;
            color: #666;
        }
        
        .spinner {
            border: 4px solid #f3f3f3;
            border-top: 4px solid #667eea;
            border-radius: 50%;
            width: 40px;
            height: 40px;
            animation: spin 1s linear infinite;
            margin: 0 auto 20px;
        }
        
        @keyframes spin {
            0% { transform: rotate(0deg); }
            100% { transform: rotate(360deg); }
        }
        
        .error {
            background: #fee;
            color: #c33;
            padding: 15px;
            border-radius: 8px;
            margin-top: 20px;
        }
        
        .sample-submission {
            background: #e8f4fd;
            padding: 15px;
            border-radius: 8px;
            margin-top: 10px;
            font-size: 0.9em;
            color: #666;
        }
        
        .sample-submission h4 {
            color: #333;
            margin-bottom: 10px;
        }
        
        .sample-submission pre {
            background: white;
            padding: 10px;
            border-radius: 5px;
            overflow-x: auto;
            font-size: 0.8em;
        }
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>🎓 AI Grading System</h1>
            <p>Ethics in the Workplace Midterm Grading</p>
        </div>
        
        <div class="content">
            <div class="input-section">
                <h2>📝 Student Submission</h2>
                <div class="form-group">
                    <label for="submission">Enter the student's midterm response:</label>
                    <textarea id="submission" placeholder="Paste the student's submission here..."></textarea>
                </div>
                
                <button class="button" onclick="gradeSubmission()" id="gradeBtn">
                    🎯 Grade Submission
                </button>
                
                <div class="sample-submission">
                    <h4>💡 Sample Submission</h4>
                    <p>Try this sample response to test the system:</p>
                    <pre>PART 1: SHORT ANSWER QUESTIONS

1. What's the difference between honesty and fidelity?
Honesty is being true to others by giving a correct representation of the world. Fidelity is being true to oneself and one's own oaths or contracts.

2. Why might a libertarian disobey zoning laws?
Libertarianism focuses on freedom maximization. Libertarians believe property is an extension of themselves and would dislike zoning laws as restrictions on their land use.

PART 2: ESSAY - CASE 1
I would answer "No" to using prediction systems. From a Rights Theory perspective, while I might freely choose to enter initially, this would limit my future freedom. The system would make decisions for me, reducing my sequential freedom.</pre>
                </div>
            </div>
            
            <div class="results-section" id="resultsSection">
                <h2>📊 Grading Results</h2>
                
                <div class="loading" id="loading">
                    <div class="spinner"></div>
                    <p>Analyzing submission and generating grade...</p>
                </div>
                
                <div id="gradeResults" style="display: none;">
                    <div class="grade-summary">
                        <div>
                            <span class="grade-score" id="overallScore">-</span>
                            <span class="grade-letter" id="gradeLetter">-</span>
                        </div>
                        <div class="confidence" id="confidence">-</div>
                    </div>
                    
                    <div class="feedback-section">
                        <div class="feedback-item">
                            <h3>💬 Overall Feedback</h3>
                            <p id="overallFeedback">-</p>
                        </div>
                        
                        <div class="feedback-item">
                            <h3>✅ Strengths</h3>
                            <ul class="feedback-list" id="strengthsList"></ul>
                        </div>
                        
                        <div class="feedback-item">
                            <h3>⚠️ Areas for Improvement</h3>
                            <ul class="feedback-list" id="weaknessesList"></ul>
                        </div>
                        
                        <div class="feedback-item">
                            <h3>💡 Suggestions</h3>
                            <ul class="feedback-list" id="suggestionsList"></ul>
                        </div>
                    </div>
                    
                    <div class="criterion-analysis">
                        <h3>📋 Detailed Criterion Analysis</h3>
                        <div id="criterionAnalysis"></div>
                    </div>
                </div>
            </div>
        </div>
    </div>
    
    <script>
        async function gradeSubmission() {
            const submission = document.getElementById('submission').value.trim();
            if (!submission) {
                alert('Please enter a submission to grade.');
                return;
            }
            
            // Show loading state
            document.getElementById('resultsSection').style.display = 'block';
            document.getElementById('loading').style.display = 'block';
            document.getElementById('gradeResults').style.display = 'none';
            document.getElementById('gradeBtn').disabled = true;
            document.getElementById('gradeBtn').textContent = '⏳ Grading...';
            
            try {
                const response = await fetch('/grade', {
                    method: 'POST',
                    headers: {
                        'Content-Type': 'application/json',
                    },
                    body: JSON.stringify({ submission: submission })
                });
                
                const result = await response.json();
                
                // Hide loading
                document.getElementById('loading').style.display = 'none';
                document.getElementById('gradeResults').style.display = 'block';
                
                if (result.error) {
                    document.getElementById('gradeResults').innerHTML = `
                        <div class="error">
                            <h3>❌ Error</h3>
                            <p>${result.error}</p>
                            ${result.raw_response ? `<pre>${result.raw_response}</pre>` : ''}
                        </div>
                    `;
                } else {
                    // Display results
                    document.getElementById('overallScore').textContent = result.overall_score || 'N/A';
                    document.getElementById('gradeLetter').textContent = result.grade_letter || 'N/A';
                    document.getElementById('confidence').textContent = `Confidence: ${(result.confidence * 100).toFixed(0)}%`;
                    document.getElementById('overallFeedback').textContent = result.overall_feedback || 'No feedback provided';
                    
                    // Display lists
                    displayList('strengthsList', result.strengths || []);
                    displayList('weaknessesList', result.weaknesses || []);
                    displayList('suggestionsList', result.improvement_suggestions || []);
                    
                    // Display criterion analysis
                    displayCriterionAnalysis(result.criterion_analysis || []);
                }
                
            } catch (error) {
                document.getElementById('loading').style.display = 'none';
                document.getElementById('gradeResults').innerHTML = `
                    <div class="error">
                        <h3>❌ Error</h3>
                        <p>Failed to grade submission: ${error.message}</p>
                    </div>
                `;
            } finally {
                document.getElementById('gradeBtn').disabled = false;
                document.getElementById('gradeBtn').textContent = '🎯 Grade Submission';
            }
        }
        
        function displayList(elementId, items) {
            const element = document.getElementById(elementId);
            element.innerHTML = '';
            items.forEach(item => {
                const li = document.createElement('li');
                li.textContent = item;
                element.appendChild(li);
            });
        }
        
        function displayCriterionAnalysis(criteria) {
            const container = document.getElementById('criterionAnalysis');
            container.innerHTML = '';
            
            criteria.forEach(criterion => {
                const div = document.createElement('div');
                div.className = 'criterion-item';
                div.innerHTML = `
                    <div class="criterion-header">
                        <span class="criterion-name">${criterion.criterion}</span>
                        <span class="criterion-score">${criterion.score}/${criterion.max_score}</span>
                    </div>
                    <p><strong>Feedback:</strong> ${criterion.feedback}</p>
                    ${criterion.strengths && criterion.strengths.length > 0 ? 
                        `<p><strong>Strengths:</strong> ${criterion.strengths.join(', ')}</p>` : ''}
                    ${criterion.weaknesses && criterion.weaknesses.length > 0 ? 
                        `<p><strong>Weaknesses:</strong> ${criterion.weaknesses.join(', ')}</p>` : ''}
                    ${criterion.suggestions && criterion.suggestions.length > 0 ? 
                        `<p><strong>Suggestions:</strong> ${criterion.suggestions.join(', ')}</p>` : ''}
                `;
                container.appendChild(div);
            });
        }
    </script>
</body>
</html>
    '''
    
    # Write the HTML template
    with open('templates/grading_ui.html', 'w') as f:
        f.write(html_template)
    
    print("🚀 Starting AI Grading UI...")
    print("📱 Open your browser and go to: http://localhost:5000")
    print("🎯 The UI is ready to test your AI grading system!")
    
    # Get port from environment (for Render) or use default
    port = int(os.environ.get('PORT', 5000))
    app.run(debug=False, host='0.0.0.0', port=port) 