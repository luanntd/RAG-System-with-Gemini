FROM python:3.11-slim AS base

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1    
ENV POETRY_NO_INTERACTION=1

# ---------------- Main Application Stage -----------------
FROM base

# Set the working directory in the container
WORKDIR /app

# Install dependencies
RUN pip install --upgrade pip
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy the application code into the container
COPY main.py .
COPY utils/ ./utils/

# Set environment variable for ChromaDB path *inside* the container
# Data will be mounted to this path using a volume
ENV DB_PATH=chroma_db

# Create the directory for ChromaDB data and declare it as a volume
# This ensures the directory exists and signals it's for persistent data
RUN mkdir -p chroma_db
VOLUME chroma_db

# Expose the port Streamlit runs on
EXPOSE 8501

# Define the command to run the application
# Use 0.0.0.0 to make it accessible from outside the container
CMD ["streamlit", "run", "main.py", "--server.port=8501", "--server.address=0.0.0.0"]
