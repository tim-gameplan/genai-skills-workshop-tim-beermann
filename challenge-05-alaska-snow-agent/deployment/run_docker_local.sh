#!/bin/bash
# Script to run Docker container locally with proper environment variables

# Get current project from gcloud
PROJECT_ID=$(gcloud config get-value project)

echo "Building Docker image..."
docker build -t alaska-snow-agent .

echo ""
echo "Running container with PROJECT_ID: $PROJECT_ID"
docker run -p 8080:8080 \
  -e PROJECT_ID=$PROJECT_ID \
  -e REGION=us-central1 \
  -e DATASET_ID=alaska_snow_capstone \
  -e GOOGLE_APPLICATION_CREDENTIALS=/tmp/keys/key.json \
  -v ~/.config/gcloud/application_default_credentials.json:/tmp/keys/key.json:ro \
  alaska-snow-agent
