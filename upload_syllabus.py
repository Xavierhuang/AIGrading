#!/usr/bin/env python3
"""
Upload Syllabus to RAG System
Uploads syllabus content to the existing Pinecone index
"""

import os
import json
import time
import re
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

class SyllabusUploader:
    def __init__(self):
        self.pc = Pinecone(api_key=PINECONE_API_KEY)
        self.openai_client = OpenAI(api_key=OPENAI_API_KEY)
        self.index_name = "aiprofessors"
        
    def extract_syllabus_text(self, file_path: str) -> str:
        """Extract text from syllabus file (PDF or TXT)"""
        try:
            if file_path.lower().endswith('.pdf'):
                import PyPDF2
                text = ""
                with open(file_path, 'rb') as file:
                    pdf_reader = PyPDF2.PdfReader(file)
                    for page in pdf_reader.pages:
                        text += page.extract_text() + "\n"
                return text
            elif file_path.lower().endswith('.txt'):
                with open(file_path, 'r', encoding='utf-8') as file:
                    return file.read()
            else:
                print(f"❌ Unsupported file format: {file_path}")
                return ""
        except Exception as e:
            print(f"❌ Failed to extract text: {e}")
            return ""
    
    def create_syllabus_chunks(self, text: str) -> List[Dict[str, Any]]:
        """Create semantic chunks optimized for syllabus content"""
        import re
        
        # Clean text
        text = re.sub(r'\s+', ' ', text).strip()
        
        # Split into sections based on common syllabus patterns
        sections = []
        
        # Common syllabus section headers
        section_patterns = [
            r'COURSE\s+DESCRIPTION',
            r'LEARNING\s+OBJECTIVES',
            r'COURSE\s+OBJECTIVES',
            r'REQUIRED\s+MATERIALS',
            r'TEXTBOOKS',
            r'GRADING\s+POLICY',
            r'ASSIGNMENTS',
            r'EXAMS',
            r'COURSE\s+SCHEDULE',
            r'WEEKLY\s+SCHEDULE',
            r'WEEK\s+\d+',  # Add week patterns like "Week 9"
            r'FINAL\s+EXAM',
            r'MIDTERM\s+EXAM',
            r'FIRST\s+EXAM',
            r'POLICIES',
            r'ACADEMIC\s+INTEGRITY',
            r'ATTENDANCE',
            r'LATE\s+WORK',
            r'MAKEUP\s+EXAMS',
            r'OFFICE\s+HOURS',
            r'CONTACT\s+INFORMATION',
            r'COURSE\s+OUTLINE',
            r'TOPICS',
            r'MODULES',
            r'UNITS'
        ]
        
        # Find section boundaries
        section_boundaries = []
        for pattern in section_patterns:
            matches = re.finditer(pattern, text, re.IGNORECASE)
            for match in matches:
                section_boundaries.append(match.start())
        
        # Sort boundaries
        section_boundaries.sort()
        
        # Create sections
        if section_boundaries:
            for i, start in enumerate(section_boundaries):
                end = section_boundaries[i + 1] if i + 1 < len(section_boundaries) else len(text)
                section_text = text[start:end].strip()
                if section_text:
                    sections.append(section_text)
        else:
            # If no clear sections, try to capture schedule information
            # Look for week patterns and group them together
            lines = text.split('\n')
            schedule_section = ""
            other_sections = []
            current_section = ""
            
            for line in lines:
                line = line.strip()
                if not line:
                    continue
                    
                # Check if this is a week line
                if re.match(r'Week\s+\d+', line, re.IGNORECASE):
                    if current_section:
                        other_sections.append(current_section)
                    current_section = line
                    schedule_section += line + "\n"
                elif current_section and (line.startswith('•') or 'Exam' in line or 'Review' in line):
                    # Continue schedule section
                    current_section += "\n" + line
                    schedule_section += line + "\n"
                else:
                    if current_section:
                        other_sections.append(current_section)
                        current_section = ""
                    other_sections.append(line)
            
            if current_section:
                other_sections.append(current_section)
            
            # Add schedule section if found
            if schedule_section.strip():
                sections.append(schedule_section.strip())
            
            # Add other sections
            sections.extend(other_sections)
        
        # Create chunks from sections
        chunks = []
        chunk_id = 0
        
        for section in sections:
            # If section is too long, split it
            if len(section) > 1000:
                # Split by sentences for long sections
                sentences = re.split(r'[.!?]+', section)
                sentences = [s.strip() for s in sentences if s.strip()]
                
                current_chunk = ""
                for sentence in sentences:
                    if len(current_chunk) + len(sentence) < 1200:  # Increased chunk size
                        current_chunk += sentence + ". "
                    else:
                        if current_chunk.strip():
                            chunks.append({
                                'id': f"syllabus_chunk_{chunk_id}",
                                'text': current_chunk.strip(),
                                'length': len(current_chunk.strip()),
                                'type': 'syllabus_section'
                            })
                            chunk_id += 1
                        current_chunk = sentence + ". "
                
                # Add the last chunk
                if current_chunk.strip():
                    chunks.append({
                        'id': f"syllabus_chunk_{chunk_id}",
                        'text': current_chunk.strip(),
                        'length': len(current_chunk.strip()),
                        'type': 'syllabus_section'
                    })
                    chunk_id += 1
            else:
                # Use the entire section
                chunks.append({
                    'id': f"syllabus_chunk_{chunk_id}",
                    'text': section,
                    'length': len(section),
                    'type': 'syllabus_section'
                })
                chunk_id += 1
        
        return chunks
    
    def upload_syllabus_chunks(self, chunks: List[Dict[str, Any]]) -> bool:
        """Upload syllabus chunks to the existing index"""
        try:
            index = self.pc.Index(self.index_name)
            
            # Prepare data for the existing index
            records = []
            for i, chunk in enumerate(chunks):
                records.append({
                    "id": f"syllabus_chunk_{i}",
                    "text": chunk['text']
                })
            
            # Upload in batches
            batch_size = 50  # Reduced batch size for stability
            total_chunks = len(records)
            
            for i in range(0, total_chunks, batch_size):
                batch = records[i:i + batch_size]
                
                # Upsert to existing index
                index.upsert_records(
                    namespace="syllabus",  # Use syllabus namespace
                    records=batch
                )
                
                print(f"📤 Uploaded batch {i//batch_size + 1}/{(total_chunks + batch_size - 1)//batch_size}")
                time.sleep(1)  # Rate limiting
            
            print(f"✅ Successfully uploaded {total_chunks} syllabus chunks!")
            return True
            
        except Exception as e:
            print(f"❌ Failed to upload syllabus chunks: {e}")
            return False
    
    def upload_syllabus(self, file_path: str) -> bool:
        """Upload syllabus to the existing index"""
        try:
            print(f"📚 Processing syllabus: {file_path}")
            
            # Extract text from syllabus
            text = self.extract_syllabus_text(file_path)
            if not text:
                return False
            
            print(f"✅ Extracted {len(text)} characters from syllabus")
            
            # Create syllabus chunks
            chunks = self.create_syllabus_chunks(text)
            print(f"✅ Created {len(chunks)} syllabus chunks")
            
            # Upload to existing index
            success = self.upload_syllabus_chunks(chunks)
            
            if success:
                print("🎉 Syllabus successfully uploaded to your existing index!")
                print("📋 Syllabus content is now available in the 'syllabus' namespace")
            
            return success
            
        except Exception as e:
            print(f"❌ Failed to process syllabus: {e}")
            return False

def main():
    """Main function to upload syllabus"""
    uploader = SyllabusUploader()
    
    print("📚 Syllabus Upload to RAG System")
    print("="*40)
    
    # Ask for syllabus file path
    file_path = input("Enter the path to your syllabus file (PDF or TXT): ").strip()
    
    if not file_path:
        print("❌ No file path provided")
        return
    
    if not os.path.exists(file_path):
        print(f"❌ File not found: {file_path}")
        return
    
    # Upload syllabus
    if uploader.upload_syllabus(file_path):
        print("\n🎉 Syllabus upload complete!")
        print("📋 You can now query syllabus content in your RAG system")
        print("🔍 Use the 'syllabus' namespace for syllabus-specific queries")
    else:
        print("\n❌ Syllabus upload failed")

if __name__ == "__main__":
    main() 