# Deployment Guide

## Vercel Deployment

This project is configured for deployment on Vercel using the `vercel.json` configuration file.

### Automatic Deployment

1. **Connect to GitHub**: Link your GitHub repository to Vercel
2. **Environment Variables**: Set the following in your Vercel dashboard:
   - `PINECONE_API_KEY`: Your Pinecone API key
   - `OPENAI_API_KEY`: Your OpenAI API key
3. **Deploy**: Vercel will automatically deploy when you push to the `main` branch

### Manual Deployment Steps

1. **Install Vercel CLI**:
   ```bash
   npm i -g vercel
   ```

2. **Login to Vercel**:
   ```bash
   vercel login
   ```

3. **Deploy**:
   ```bash
   vercel
   ```

4. **Set Environment Variables**:
   ```bash
   vercel env add PINECONE_API_KEY
   vercel env add OPENAI_API_KEY
   ```

### Deployment Configuration

The `vercel.json` file configures:
- **Build**: Uses `@vercel/python` for Python applications
- **Entry Point**: `rag_grading_ui.py`
- **Routes**: All requests routed to the Flask app
- **Python Version**: 3.12

### Environment Variables Required

- `PINECONE_API_KEY`: For vector database access
- `OPENAI_API_KEY`: For LLM processing
- `PYTHON_VERSION`: Set to 3.12 (configured in vercel.json)

### Updating Deployment

1. **Make Changes**: Edit your local files
2. **Commit Changes**: `git add . && git commit -m "Update message"`
3. **Push to GitHub**: `git push origin main`
4. **Auto-Deploy**: Vercel will automatically redeploy

### Local Development

For local development, use:
```bash
python rag_grading_ui.py
```

The application will be available at `http://localhost:5002`

### Vercel-Specific Considerations

- **Serverless Functions**: Vercel uses serverless functions
- **Cold Starts**: First request may be slower
- **Timeout Limits**: Functions have execution time limits
- **Environment Variables**: Set in Vercel dashboard or via CLI

### Troubleshooting

- **Build Failures**: Check the build logs in Vercel dashboard
- **Runtime Errors**: Check the function logs
- **Environment Variables**: Ensure all required API keys are set
- **Timeout Issues**: Optimize function execution time

### Production Considerations

- **Security**: Ensure API keys are properly secured
- **Scaling**: Vercel automatically handles scaling
- **Monitoring**: Use Vercel's built-in analytics
- **Backups**: Regularly backup your Pinecone index data 