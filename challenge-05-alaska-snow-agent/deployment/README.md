# Alaska Snow Agent - Deployment Package

Production-ready Streamlit application for Challenge 5.

## What's Included

- **app.py** - Main Streamlit application with RAG + security
- **requirements.txt** - Python dependencies
- **Dockerfile** - Container configuration with validation
- **validate_resources.py** - Startup validation script
- **entrypoint.sh** - Container startup script

## Prerequisites

Before deploying, you MUST create the BigQuery resources by running the Challenge 5 notebook:

1. Open `challenge_05_alaska_snow_final.ipynb` in Google Colab
2. Run all cells to create:
   - Dataset: `alaska_snow_capstone`
   - Table: `snow_vectors` (with embeddings)
   - Model: `embedding_model`
   - Model Armor security template

## Docker Improvements

The Docker container now includes:

✅ **Automatic validation** - Checks for required resources on startup  
✅ **Clear error messages** - Tells you exactly what's missing  
✅ **Health checks** - Cloud Run can monitor app health  
✅ **Auto-detection** - Detects PROJECT_ID from environment  

## Environment Variables

### Required
- `PROJECT_ID` - GCP project ID (**required** for Cloud Run deployment)

### Optional
- `REGION` - GCP region (default: `us-central1`)
- `DATASET_ID` - BigQuery dataset name (default: `alaska_snow_capstone`)
- `GOOGLE_MAPS_API_KEY` - Google Geocoding API key (enables location-based queries)

### External API Features

The agent includes integrations with external APIs (matching the notebook implementation):

1. **Google Geocoding API** (`get_coordinates`)
   - Converts addresses to geographic coordinates
   - Requires `GOOGLE_MAPS_API_KEY` environment variable
   - If not set, location-based queries will still work via RAG

2. **National Weather Service API** (`get_weather_forecast`)
   - Provides weather forecasts for specific locations
   - No API key required (free public API)
   - Only covers USA locations

## Local Testing

### Option 1: Run Python directly
```bash
# Set environment variables
export PROJECT_ID=$(gcloud config get-value project)

# Run validation first
python validate_resources.py

# If validation passes
streamlit run app.py
```

### Option 2: Test with Docker (recommended)
```bash
# Build and run with Docker (tests the actual deployment)
./run_docker_local.sh

# Or manually:
docker build -t alaska-snow-agent .
docker run -p 8080:8080 \
  -e PROJECT_ID=$(gcloud config get-value project) \
  -e GOOGLE_APPLICATION_CREDENTIALS=/tmp/keys/key.json \
  -v ~/.config/gcloud/application_default_credentials.json:/tmp/keys/key.json:ro \
  alaska-snow-agent
```

## Cloud Run Deployment

```bash
# IMPORTANT: Must set PROJECT_ID environment variable
gcloud run deploy alaska-snow-agent \
  --source . \
  --region us-central1 \
  --platform managed \
  --allow-unauthenticated \
  --timeout 300 \
  --set-env-vars PROJECT_ID=$(gcloud config get-value project) \
  --quiet
```

The `--set-env-vars PROJECT_ID=...` flag is **required** - the container needs this to connect to BigQuery.
