FROM python:3.10-slim

# Set working directory
WORKDIR /app

# Install dependencies first for Docker layer caching
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy the full project
COPY . .

# HuggingFace Spaces uses port 7860 by default
EXPOSE 7860

# Health check so HF Space knows when the container is ready
HEALTHCHECK --interval=30s --timeout=10s --start-period=15s --retries=3 \
  CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:7860/health')" || exit 1

# Start the FastAPI server (uvicorn)
# To run inference instead: docker run <image> python inference.py
CMD ["uvicorn", "server.app:app", "--host", "0.0.0.0", "--port", "7860", "--workers", "1"]
