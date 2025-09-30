#!/bin/bash

# Deployment script for Job Summary Generator
set -e

echo "🚀 Starting deployment of Job Summary Generator..."

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Function to print colored output
print_status() {
    echo -e "${GREEN}[INFO]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Check if .env file exists
if [ ! -f ".env" ]; then
    print_warning ".env file not found. Creating from template..."
    cp env.example .env
    print_warning "Please edit .env file with your configuration before running again."
    exit 1
fi

# Check if Docker is installed
if ! command -v docker &> /dev/null; then
    print_error "Docker is not installed. Please install Docker first."
    exit 1
fi

# Check if docker-compose is installed
if ! command -v docker-compose &> /dev/null; then
    print_error "docker-compose is not installed. Please install docker-compose first."
    exit 1
fi

# Build and run with Docker Compose
print_status "Building Docker image..."
docker-compose build

print_status "Starting services..."
docker-compose up -d

print_status "Waiting for service to be ready..."
sleep 10

# Check if service is running
if curl -f http://localhost:8000/health > /dev/null 2>&1; then
    print_status "✅ Service is running successfully!"
    print_status "🌐 Application is available at: http://localhost:8000"
    print_status "📚 API documentation at: http://localhost:8000/docs"
else
    print_error "❌ Service failed to start. Check logs with: docker-compose logs"
    exit 1
fi

print_status "🎉 Deployment completed successfully!"
