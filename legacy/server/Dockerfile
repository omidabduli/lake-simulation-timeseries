# LEGACY self-hosting image (Streamlit server). The primary deployment is GitHub Pages.
# Build context is the repository root:
#   docker build -f legacy/server/Dockerfile -t lake-time-series-forecasting .
#
# Use an official Python runtime as a parent image
FROM python:3.11-slim

# Set environment variables to prevent Python from writing pyc files and buffering stdout
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

# Install system dependencies
# libgomp1 is required for XGBoost to work on Linux.
# build-essential is provided for packages that compile C/C++ extensions.
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    libgomp1 \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Set the working directory in the container
WORKDIR /app

# Copy the requirements file to leverage Docker's cache behavior
COPY requirements.txt .

# Install dependencies
RUN pip install --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Copy the application, modules, and Streamlit theme configuration
COPY modules/ ./modules/
COPY app.py .
COPY .streamlit/ ./.streamlit/

# Create logs folder for structured runtime logging
RUN mkdir -p logs

# Expose port 8501 for Streamlit
EXPOSE 8501

# Configure Streamlit behavior via environment variables
ENV STREAMLIT_SERVER_PORT=8501
ENV STREAMLIT_SERVER_ADDRESS=0.0.0.0
ENV STREAMLIT_SERVER_HEADLESS=true

# Let Docker and orchestration platforms verify that the app is responsive
HEALTHCHECK --interval=30s --timeout=5s --start-period=30s --retries=3 \
    CMD curl --fail http://localhost:8501/_stcore/health || exit 1

# Command to run Streamlit
CMD ["streamlit", "run", "app.py"]
