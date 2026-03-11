FROM python:3.12-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY main.py .
COPY models/ models/
COPY db/ db/
COPY simulator/ simulator/
COPY profiler/ profiler/
COPY scripts/ scripts/
COPY data/ data/

# Expose default port (Railway overrides via $PORT)
EXPOSE 8001

# Use shell form so $PORT is expanded at runtime
CMD uvicorn main:app --host 0.0.0.0 --port ${PORT:-8001} --workers 2
