FROM python:3.9-slim

# Install CMake and build essentials
RUN apt-get update && apt-get install -y cmake build-essential \
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# Copy requirements file
COPY requirements.txt .

# Install dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy your application code
COPY . .

# Expose port
EXPOSE 5000

# Command to run the application
CMD ["python", "app.py"]

# Set non-root user
USER 10014
