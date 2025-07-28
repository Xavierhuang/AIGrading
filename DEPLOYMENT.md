# Deployment Guide

## Render Deployment

This project is configured for deployment on Render using the `render.yaml` configuration file.

### Automatic Deployment

1. **Connect to GitHub**: Link your GitHub repository to Render
2. **Environment Variables**: Set the following in your Render dashboard:
   - `PINECONE_API_KEY`: Your Pinecone API key
   - `OPENAI_API_KEY`: Your OpenAI API key
3. **Deploy**: Render will automatically deploy when you push to the `main` branch

### Manual Deployment Steps

1. **Fork/Clone the Repository**:
   ```bash
   git clone https://github.com/yourusername/AIGrading.git
   cd AIGrading
   ```

2. **Set Environment Variables**:
   ```bash
   export PINECONE_API_KEY="your_pinecone_api_key"
   export OPENAI_API_KEY="your_openai_api_key"
   ```

3. **Install Dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

4. **Run Locally** (for testing):
   ```bash
   python rag_grading_ui.py
   ```

### Deployment Configuration

The `render.yaml` file configures:
- **Service Type**: Web service
- **Environment**: Python 3.12.0
- **Build Command**: `pip install -r requirements.txt`
- **Start Command**: `python rag_grading_ui.py`
- **Port**: Automatically detected by Render

### Environment Variables Required

- `PINECONE_API_KEY`: For vector database access
- `OPENAI_API_KEY`: For LLM processing
- `PYTHON_VERSION`: Set to 3.12.0 (configured in render.yaml)

### Updating Deployment

1. **Make Changes**: Edit your local files
2. **Commit Changes**: `git add . && git commit -m "Update message"`
3. **Push to GitHub**: `git push origin main`
4. **Auto-Deploy**: Render will automatically redeploy

### Troubleshooting

- **Build Failures**: Check the build logs in Render dashboard
- **Runtime Errors**: Check the service logs
- **Environment Variables**: Ensure all required API keys are set
- **Port Issues**: Render automatically handles port configuration

### Local Development

For local development, use:
```bash
python rag_grading_ui.py
```

The application will be available at `http://localhost:5002`

### Production Considerations

- **Security**: Ensure API keys are properly secured
- **Scaling**: Render automatically handles scaling
- **Monitoring**: Use Render's built-in monitoring tools
- **Backups**: Regularly backup your Pinecone index data 