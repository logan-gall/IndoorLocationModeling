# Use the official Python image from the Docker Hub
FROM python:3.9-slim

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

# Set work directory
WORKDIR /app

# Install dependencies
COPY requirements.txt .
RUN pip install --upgrade pip
RUN pip install --no-cache-dir -r requirements.txt

# Copy the application code
COPY . .

# Expose the port (Cloud Run uses the PORT environment variable)
EXPOSE 8080

# Command to run the application using gunicorn
CMD ["gunicorn", "app:server", "--bind", "0.0.0.0:8080", "--workers", "4"]
