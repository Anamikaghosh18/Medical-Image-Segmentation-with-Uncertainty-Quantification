# Use a lightweight Python image
FROM python:3.11-slim

# Prevent Python from writing .pyc files and buffer logs immediately
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

# Create user to run the app as non-root (Hugging Face requirement)
RUN useradd -m -u 1000 user

# Install system dependencies required by OpenCV, matplotlib, and PIL
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc \
    g++ \
    libgl1 \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Switch to the non-root user
USER user
ENV HOME=/home/user \
    PATH=/home/user/.local/bin:$PATH

# Set working directory
WORKDIR $HOME/app

# Copy dependency file first to leverage Docker layer caching
COPY --chown=user requirements.txt .

# Upgrade pip and install Python dependencies
RUN pip install --user --no-cache-dir --upgrade pip && \
    pip install --user --no-cache-dir -r requirements.txt

# Copy the entire project
COPY --chown=user . .

# Hugging Face Spaces expects the app to listen on port 7860
EXPOSE 7860

# Start the FastAPI application
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "7860"]