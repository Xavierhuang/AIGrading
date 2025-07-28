# AI Grading System

A sophisticated AI-powered grading system that uses Retrieval-Augmented Generation (RAG) to evaluate student answers against textbook content and grading rubrics.

## Features

- **RAG-Powered Grading**: Uses Pinecone vector database with hosted embeddings for accurate content retrieval
- **Multi-Source Evaluation**: Combines textbook content and grading rubrics for comprehensive assessment
- **Intelligent Context**: Automatically fetches relevant textbook material and rubric criteria
- **Detailed Feedback**: Provides grades, scores, strengths, weaknesses, and improvement suggestions
- **Web Interface**: User-friendly Flask web UI for easy interaction

## Technology Stack

- **Backend**: Python Flask
- **Vector Database**: Pinecone with hosted `llama-text-embed-v2` embeddings
- **LLM**: OpenAI GPT-4o for answer synthesis and grading
- **Text Processing**: PyPDF2 for PDF extraction and semantic chunking

## Quick Start

1. **Install Dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

2. **Set Environment Variables**:
   ```bash
   export PINECONE_API_KEY="your_pinecone_api_key"
   export OPENAI_API_KEY="your_openai_api_key"
   ```

3. **Run the Grading UI**:
   ```bash
   python rag_grading_ui.py
   ```

4. **Access the Web Interface**:
   Open your browser to `http://localhost:5002`

## Usage

1. Enter the question being graded
2. Paste the student's answer
3. Click "Grade Answer" to get comprehensive feedback
4. Review the detailed evaluation including:
   - Grade (A-F) and numerical score
   - Strengths and areas for improvement
   - Key points covered and missing
   - Suggestions for enhancement
   - Rubric criteria applied

## Architecture

The system uses a sophisticated RAG architecture:
- **Content Indexing**: Textbook and rubric content are chunked and embedded in Pinecone
- **Smart Retrieval**: Queries are routed to appropriate content sources
- **Context Enhancement**: Explicit rubric fetching ensures grading criteria are always considered
- **LLM Synthesis**: GPT-4o generates comprehensive evaluations using retrieved context

## Deployment

The system is configured for deployment on Render/Vercel with the updated `render.yaml` configuration.

## Files

- `rag_grading_ui.py`: Main grading interface with RAG integration
- `use_existing_index.py`: Core RAG system for content retrieval
- `upload_grading_content.py`: Script for uploading rubrics and grading materials
- `requirements.txt`: Python dependencies 