#!/bin/bash

echo "🚀 Deploying RAG Grading System to Vercel..."

# Check if Vercel CLI is installed
if ! command -v vercel &> /dev/null; then
    echo "❌ Vercel CLI not found. Installing..."
    npm install -g vercel
fi

# Login to Vercel (if not already logged in)
echo "🔐 Checking Vercel login status..."
vercel whoami 2>/dev/null || vercel login

# Deploy to Vercel
echo "📦 Deploying to Vercel..."
vercel --prod

echo "✅ Deployment complete!"
echo "🌐 Your app should be live at the URL shown above"
echo ""
echo "📋 Next steps:"
echo "1. Set environment variables in Vercel dashboard:"
echo "   - PINECONE_API_KEY"
echo "   - OPENAI_API_KEY"
echo "2. Test your deployed app"
echo "3. Share the URL with others" 