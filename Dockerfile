# Dockerfile
FROM python:3.10-slim

# Set working directory
WORKDIR /app

# Install system dependencies with timeouts
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    git \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Configure pip for better network handling
RUN pip config set global.timeout 1000
RUN pip config set global.retries 3

# Copy runtime requirements and install only what the app actually needs
COPY requirements.runtime.txt .

# Upgrade pip first and configure timeouts
RUN pip install --upgrade pip

# Install general runtime dependencies from PyPI
RUN pip install --no-cache-dir \
    --timeout 1000 \
    --retries 3 \
    --default-timeout=1000 \
    -r requirements.runtime.txt

# Install CPU-only PyTorch from the PyTorch wheel index
RUN pip install --no-cache-dir \
    --timeout 1000 \
    --retries 3 \
    --default-timeout=1000 \
    --index-url https://download.pytorch.org/whl/cpu \
    torch==2.0.0+cpu

# Copy application code
COPY . .

# Create results directories
RUN mkdir -p results/clustering results/causal results/llm_responses

# Expose ports
EXPOSE 8501 8000

# Set Python path
ENV PYTHONPATH=/app:/app/helpers:/app/LOCI:/app/ROCHE:/app/LINGAM:/app/DATA

# Make run script executable
RUN chmod +x run.sh

# Start the application
CMD ["./run.sh"]