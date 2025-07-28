#!/usr/bin/env python3
"""
Upload Grading Content to Dedicated Grading Index
Uploads rubrics, grading criteria, and other grading materials
"""

import os
import re
import PyPDF2
from pinecone import Pinecone
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

class GradingContentUploader:
    def __init__(self):
        self.pc = Pinecone(api_key=os.environ.get('PINECONE_API_KEY'))
        self.index_name = "aiprofessorgrading"
        
    def extract_rubric_text(self, file_path: str) -> str:
        """Extract text from PDF rubric"""
        try:
            with open(file_path, 'rb') as file:
                reader = PyPDF2.PdfReader(file)
                text = ""
                for page in reader.pages:
                    text += page.extract_text()
                
                print(f"✅ Extracted {len(text)} characters from rubric")
                return text
                
        except Exception as e:
            print(f"❌ Failed to extract text from {file_path}: {e}")
            return ""
    
    def create_rubric_chunks(self, text: str) -> list:
        """Create semantic chunks from rubric text"""
        chunks = []
        
        # Split by sections (common rubric patterns)
        section_patterns = [
            r'GRADING\s+CRITERIA',
            r'RUBRIC',
            r'SCORING\s+GUIDE',
            r'EVALUATION\s+CRITERIA',
            r'ASSESSMENT\s+STANDARDS',
            r'POINTS?\s*:?\s*\d+',
            r'SCORE\s*:?\s*\d+',
            r'GRADE\s*:?\s*[A-F]',
            r'EXCELLENT\s*\([^)]+\)',
            r'GOOD\s*\([^)]+\)',
            r'FAIR\s*\([^)]+\)',
            r'POOR\s*\([^)]+\)',
            r'CRITERIA\s+\d+',
            r'QUESTION\s+\d+',
            r'PART\s+[A-Z]',
            r'[A-Z]\s*\.\s*[A-Z]',  # A. B. C. etc.
        ]
        
        # Find sections
        sections = []
        current_section = ""
        
        lines = text.split('\n')
        for line in lines:
            line = line.strip()
            if not line:
                continue
                
            # Check if this line starts a new section
            is_section_header = any(re.search(pattern, line, re.IGNORECASE) for pattern in section_patterns)
            
            if is_section_header and current_section:
                sections.append(current_section.strip())
                current_section = line
            else:
                current_section += " " + line
        
        # Add the last section
        if current_section:
            sections.append(current_section.strip())
        
        # If no clear sections found, split by paragraphs
        if not sections:
            paragraphs = text.split('\n\n')
            sections = [p.strip() for p in paragraphs if p.strip()]
        
        # Create chunks from sections
        for i, section in enumerate(sections):
            if len(section) < 50:  # Skip very short sections
                continue
                
            # Split large sections into smaller chunks
            if len(section) > 1000:
                # Split by sentences
                sentences = re.split(r'[.!?]+', section)
                current_chunk = ""
                
                for sentence in sentences:
                    sentence = sentence.strip()
                    if not sentence:
                        continue
                        
                    if len(current_chunk) + len(sentence) < 800:
                        current_chunk += sentence + ". "
                    else:
                        if current_chunk:
                            chunks.append({
                                "id": f"rubric_chunk_{len(chunks)}",
                                "text": current_chunk.strip()
                            })
                        current_chunk = sentence + ". "
                
                if current_chunk:
                    chunks.append({
                        "id": f"rubric_chunk_{len(chunks)}",
                        "text": current_chunk.strip()
                    })
            else:
                chunks.append({
                    "id": f"rubric_chunk_{i}",
                    "text": section
                })
        
        print(f"✅ Created {len(chunks)} rubric chunks")
        return chunks
    
    def upload_rubric_chunks(self, chunks: list) -> bool:
        """Upload rubric chunks to grading index"""
        try:
            index = self.pc.Index(self.index_name)
            
            # Prepare records for upload
            records = []
            for chunk in chunks:
                records.append({
                    "id": chunk["id"],
                    "text": chunk["text"]
                })
            
            # Upload in batches
            batch_size = 50
            for i in range(0, len(records), batch_size):
                batch = records[i:i + batch_size]
                index.upsert_records(
                    namespace="textbook",
                    records=batch
                )
                print(f"📤 Uploaded batch {i//batch_size + 1}/{(len(records) + batch_size - 1)//batch_size}")
            
            print(f"✅ Successfully uploaded {len(records)} rubric chunks!")
            return True
            
        except Exception as e:
            print(f"❌ Failed to upload rubric chunks: {e}")
            return False
    
    def upload_rubric(self, file_path: str) -> bool:
        """Upload a rubric file to the grading index"""
        print(f"📋 Processing rubric: {file_path}")
        
        # Extract text
        text = self.extract_rubric_text(file_path)
        if not text:
            return False
        
        # Create chunks
        chunks = self.create_rubric_chunks(text)
        if not chunks:
            print("❌ No chunks created from rubric")
            return False
        
        # Upload chunks
        return self.upload_rubric_chunks(chunks)

def main():
    """Upload grading content"""
    print("📋 Grading Content Upload")
    print("=" * 30)
    
    uploader = GradingContentUploader()
    
    # Get file path from user
    file_path = input("Enter the path to your grading rubric file: ").strip()
    
    if not os.path.exists(file_path):
        print(f"❌ File not found: {file_path}")
        return
    
    # Upload the rubric
    if uploader.upload_rubric(file_path):
        print("\n🎉 Rubric successfully uploaded to grading index!")
        print("📋 Rubric content is now available in the 'textbook' namespace")
        print("🎓 You can now use this for AI-powered grading")
    else:
        print("\n❌ Failed to upload rubric")

if __name__ == "__main__":
    main() 