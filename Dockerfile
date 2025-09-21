# AI Documentary Composer - Docker Configuration
FROM python:3.11-slim

# Install system dependencies including build tools for Python packages
RUN apt-get update && apt-get install -y \
    ffmpeg \
    git \
    curl \
    gcc \
    python3-dev \
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# Copy project files
COPY requirements-docker.txt ./requirements.txt
COPY src/ai_doc_composer /app/ai_doc_composer

# Create empty projects directory (actual files will be mounted as volumes)
RUN mkdir -p /app/projects

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Install TTS separately (has complex dependencies)
RUN pip install --no-cache-dir TTS>=0.22.0

# Add the app directory to Python path
ENV PYTHONPATH=/app:$PYTHONPATH

# Expose port for Streamlit
EXPOSE 8501

# Create entrypoint script
RUN echo '#!/bin/bash\n\
if [ "$1" = "ui" ]; then\n\
    exec python -m ai_doc_composer.cli ui\n\
else\n\
    exec python -m ai_doc_composer.cli "$@"\n\
fi' > /app/entrypoint.sh && chmod +x /app/entrypoint.sh

ENTRYPOINT ["/app/entrypoint.sh"]
CMD ["ui"]