#!/usr/bin/env python3
"""
Create New Pinecone Index for Grading System
"""

import os
from pinecone import Pinecone, ServerlessSpec

# Set environment variables
os.environ['PINECONE_API_KEY'] = "pcsk_Rq3uo_3degqLXVnfacY8pQ89dmESt2t1nYNNR4hc7gLnM2J6SmtLKDrnUfvK4xMkRCWFc"

print("🚀 Creating New Pinecone Index for Grading System")
print("="*60)

try:
    # Initialize Pinecone client
    print("1. Initializing Pinecone client...")
    pc = Pinecone(api_key=os.environ['PINECONE_API_KEY'])
    print("   ✅ Pinecone client initialized!")
    
    # Define index name
    index_name = "professorjames-experiment-grading"
    
    print(f"\n2. Checking if index '{index_name}' exists...")
    if pc.has_index(index_name):
        print(f"   ⚠️ Index '{index_name}' already exists!")
        print("   📊 Index details:")
        index = pc.Index(index_name)
        stats = index.describe_index_stats()
        print(f"      - Total vectors: {stats.total_vector_count}")
        print(f"      - Namespaces: {list(stats.namespaces.keys())}")
    else:
        print(f"   ℹ️ Index '{index_name}' does not exist. Creating new index...")
        
        # Create new index with OpenAI embeddings
        print("\n3. Creating new index with OpenAI embeddings...")
        pc.create_index(
            name=index_name,
            dimension=3072,  # OpenAI text-embedding-3-large dimension
            metric="cosine",
            spec=ServerlessSpec(
                cloud="aws",
                region="us-east-1"
            )
        )
        print(f"   ✅ Index '{index_name}' created successfully!")
        
        # Wait a moment for index to be ready
        print("\n4. Waiting for index to be ready...")
        import time
        time.sleep(10)  # Wait 10 seconds
        
        # Get index stats
        print("\n5. Getting index statistics...")
        index = pc.Index(index_name)
        stats = index.describe_index_stats()
        print(f"   📊 Total vectors: {stats.total_vector_count}")
        print(f"   📚 Namespaces: {list(stats.namespaces.keys())}")
    
    print(f"\n🎉 Index '{index_name}' is ready for use!")
    print("="*60)
    
except Exception as e:
    print(f"❌ Error: {e}")
    import traceback
    traceback.print_exc() 