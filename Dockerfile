FROM python:3.11-slim

WORKDIR /app

# System dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy project files
COPY .streamlit/ ./.streamlit/
COPY src/ ./src/
COPY models/ ./models/
COPY data/processed/ ./data/processed/
COPY pages/ ./pages/
COPY app.py .

# Create required directories
RUN mkdir -p data/predictions

# Expose Streamlit port
EXPOSE 8501

# Command to run Streamlit app
CMD ["streamlit", "run", "app.py", "--server.port=8501", "--server.address=0.0.0.0"]
