# Use Python 3.12-slim as the base image
FROM python:3.12-slim

# LABELS #
LABEL author="Sizon Nayak"
LABEL description="Dockerfile for BitBirch Clustering Application with RDKit support"

# Install system dependencies for RDKit (including libXrender) and the missing libraries
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
    libxrender1 \
    libxext6 \
    libpng-dev \
    libfreetype6 \
    libssl-dev \
    libexpat1 \
    libgomp1 && \
    apt-get clean && rm -rf /var/lib/apt/lists/*

# Set the working directory in the container
WORKDIR /app

# Copy requirements.txt into the container
COPY requirements.txt .

# Install dependencies directly from PyPI
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code into the container
COPY . .

# Set the command to run the application
#CMD ["python3", "clustering_run.py"]

