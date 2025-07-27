# 🎓 AI Grading System

A simple, powerful AI grading system for Ethics in the Workplace Midterm exams using OpenAI and Pinecone.

## 🚀 Quick Start

### 1. Install Dependencies
```bash
python -m pip install flask pinecone-client openai tiktoken PyPDF2
```

### 2. Set Up Your API Keys
Create a `.env` file with your API keys:
```
PINECONE_API_KEY=your_pinecone_api_key
OPENAI_API_KEY=your_openai_api_key
```

### 3. Create Pinecone Index
```bash
python create_new_index.py
```

### 4. Process Textbook (Optional)
```bash
python improved_chunk_textbook.py
```

### 5. Add Grading Rubric
```bash
python add_grading_rubric_simple.py
```

### 6. Start the Web UI
```bash
python simple_grading_ui.py
```

Open your browser and go to: **http://localhost:5000**

## 📁 Project Structure

### Core Files
- `simple_grading_ui.py` - **Main web application** (UI + API + AI grading)
- `improved_chunk_textbook.py` - Process textbook content for RAG
- `add_grading_rubric_simple.py` - Upload grading rubric to Pinecone
- `create_new_index.py` - Create Pinecone index
- `test_simple_grading.py` - Test the AI grading system

### Data Files
- `BinderTextbooks_first300.pdf` - Textbook content
- `textbook_chunking.log` - Processing logs
- `templates/` - Web UI templates (auto-generated)

## 🎯 Features

- **🤖 AI-Powered Grading**: Uses GPT-4 with your specific rubric
- **🔍 RAG Integration**: Searches relevant rubric content from Pinecone
- **📊 Detailed Analysis**: Provides scores, feedback, strengths, and suggestions
- **🎨 Beautiful UI**: Modern, responsive web interface
- **⚡ Real-time**: Instant grading with loading animations

## 🧪 Testing

Test the system with sample submissions:
```bash
python test_simple_grading.py
```

## 🔧 How It Works

1. **Textbook Processing**: Chunks and embeds textbook content in Pinecone
2. **Rubric Upload**: Stores detailed grading criteria in Pinecone
3. **Web UI**: Students submit responses through the web interface
4. **AI Grading**: 
   - Searches relevant rubric content from Pinecone
   - Uses GPT-4 to analyze submissions against the rubric
   - Returns comprehensive grades with detailed feedback

## 🎓 Grading Criteria

The system uses the Ethics in the Workplace Midterm rubric including:
- **Short Answer Questions** (20 points)
- **Theory Definition & Application** (15 points)
- **Argument Development & Persuasiveness** (15 points)
- **Variety & Integration of Theories** (10 points)
- **Essay Structure & Clarity** (10 points)

## 🛠️ Technology Stack

- **Backend**: Flask (Python)
- **AI**: OpenAI GPT-4 + text-embedding-3-large
- **Vector Database**: Pinecone
- **Frontend**: HTML/CSS/JavaScript
- **Text Processing**: tiktoken, PyPDF2

## 📝 Usage

1. Start the web UI: `python simple_grading_ui.py`
2. Open http://localhost:5000 in your browser
3. Paste a student submission
4. Click "Grade Submission"
5. View comprehensive AI grading results

## 🎉 Success!

Your AI grading system is now ready to grade Ethics midterm submissions with professional-level analysis and feedback! 