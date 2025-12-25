# Base Image
FROM python:3.10-slim

# Set working directory
WORKDIR /app

# Install system dependencies for OpenCV
RUN apt-get update && apt-get install -y \
    libgl1-mesa-glx \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first (for caching)
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy source code
COPY . .

# Create necessary directories
RUN mkdir -p data/input_videos results logs

# Set environment variables
ENV PYTHONUNBUFFERED=1

# Entrypoint
# Run in headless mode by default
ENTRYPOINT ["python", "pipeline/run_inference.py", "--headless"]
