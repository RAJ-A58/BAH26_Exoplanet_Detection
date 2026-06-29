FROM python:3.10-slim

# Set up a non-root user (required by Hugging Face Spaces)
RUN useradd -m -u 1000 user
USER user
ENV PATH="/home/user/.local/bin:$PATH"

WORKDIR /app

# Install system dependencies (required for some Python packages)
USER root
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*
USER user

# Copy requirements and install
COPY --chown=user:user requirements.txt .
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Copy the rest of the application
COPY --chown=user:user . .

# Hugging Face Spaces expose port 7860 by default
ENV PORT=7860
ENV TF_CPP_MIN_LOG_LEVEL=3
ENV TF_ENABLE_ONEDNN_OPTS=0
ENV MPLCONFIGDIR=/tmp/mplcfg
ENV PYTHONUNBUFFERED=1

EXPOSE 7860

# Start the application
CMD ["python", "app.py"]
