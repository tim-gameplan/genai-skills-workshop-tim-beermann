#!/usr/bin/env python3
"""
Quick setup script to create minimal BigQuery resources for local testing
This creates just enough to test the Streamlit app
"""

import subprocess
import time
from google.cloud import bigquery, storage

# Configuration
PROJECT_ID = subprocess.check_output("gcloud config get-value project", shell=True).decode().strip()
REGION = "us-central1"
DATASET_ID = "alaska_snow_capstone"
CONNECTION_ID = f"{REGION}.vertex-ai-conn"
SOURCE_BUCKET = "gs://labs.roitraining.com/alaska-dept-of-snow"

print("🚀 Setting up minimal BigQuery resources for testing")
print("=" * 70)
print(f"Project: {PROJECT_ID}")
print(f"Region: {REGION}")
print(f"Dataset: {DATASET_ID}")
print()

# Initialize clients
bq_client = bigquery.Client(project=PROJECT_ID, location=REGION)
storage_client = storage.Client(project=PROJECT_ID)

# Step 1: Enable APIs
print("1️⃣  Enabling required APIs...")
apis = [
    "aiplatform.googleapis.com",
    "bigquery.googleapis.com",
    "geocoding-backend.googleapis.com",
]

for api in apis:
    result = subprocess.run(
        f"gcloud services enable {api} --project={PROJECT_ID}",
        shell=True,
        capture_output=True
    )
    if result.returncode == 0:
        print(f"   ✅ {api}")
    else:
        print(f"   ⚠️  {api} (may already be enabled)")

print()

# Step 2: Create dataset
print("2️⃣  Creating BigQuery dataset...")
dataset = bigquery.Dataset(f"{PROJECT_ID}.{DATASET_ID}")
dataset.location = REGION

try:
    bq_client.create_dataset(dataset, exists_ok=True)
    print(f"   ✅ Dataset '{DATASET_ID}' created")
except Exception as e:
    print(f"   ❌ Error: {e}")
    exit(1)

print()

# Step 3: Load data
print("3️⃣  Loading FAQ data from Cloud Storage...")
print(f"   Source: {SOURCE_BUCKET}")

# Find CSV file
bucket_name = SOURCE_BUCKET.replace("gs://", "").split("/")[0]
prefix = "/".join(SOURCE_BUCKET.replace("gs://", "").split("/")[1:])

blobs = storage_client.list_blobs(bucket_name, prefix=prefix)
csv_file = None

for blob in blobs:
    if blob.name.endswith(".csv"):
        csv_file = f"gs://{bucket_name}/{blob.name}"
        break

if not csv_file:
    print("   ❌ No CSV file found!")
    exit(1)

print(f"   Found: {csv_file}")

# Load into BigQuery
table_ref = bq_client.dataset(DATASET_ID).table("snow_faqs_raw")
schema = [
    bigquery.SchemaField("question", "STRING"),
    bigquery.SchemaField("answer", "STRING"),
]

job_config = bigquery.LoadJobConfig(
    schema=schema,
    source_format=bigquery.SourceFormat.CSV,
    skip_leading_rows=1,
    write_disposition=bigquery.WriteDisposition.WRITE_TRUNCATE
)

load_job = bq_client.load_table_from_uri(csv_file, table_ref, job_config=job_config)
load_job.result()

print(f"   ✅ Loaded {load_job.output_rows} rows")
print()

# Step 4: Create BigQuery connection
print("4️⃣  Setting up BigQuery Cloud Resource Connection...")
connection_name = "vertex-ai-conn"

check_conn = subprocess.run(
    f"bq show --connection --project_id={PROJECT_ID} --location={REGION} {connection_name}",
    shell=True,
    capture_output=True
)

if check_conn.returncode != 0:
    subprocess.run(
        f"bq mk --connection --connection_type=CLOUD_RESOURCE "
        f"--project_id={PROJECT_ID} --location={REGION} {connection_name}",
        shell=True,
        check=True,
        capture_output=True
    )
    print(f"   ✅ Connection '{connection_name}' created")
else:
    print(f"   ✅ Connection '{connection_name}' already exists")

# Grant IAM permissions
import json
conn_info = subprocess.run(
    f"bq show --format=json --connection --project_id={PROJECT_ID} --location={REGION} {connection_name}",
    shell=True,
    capture_output=True,
    text=True
)

if conn_info.returncode == 0:
    conn_sa = json.loads(conn_info.stdout)["cloudResource"]["serviceAccountId"]
    subprocess.run(
        f"gcloud projects add-iam-policy-binding {PROJECT_ID} "
        f"--member='serviceAccount:{conn_sa}' "
        f"--role='roles/aiplatform.user' --quiet",
        shell=True,
        capture_output=True
    )
    print(f"   ✅ IAM permissions granted to {conn_sa}")
    print("   ⏳ Waiting 15 seconds for IAM propagation...")
    time.sleep(15)

print()

# Step 5: Create embedding model
print("5️⃣  Creating text-embedding-004 model...")
create_model_sql = f"""
CREATE OR REPLACE MODEL `{PROJECT_ID}.{DATASET_ID}.embedding_model`
REMOTE WITH CONNECTION `{PROJECT_ID}.{CONNECTION_ID}`
OPTIONS (ENDPOINT = 'text-embedding-004');
"""

try:
    model_job = bq_client.query(create_model_sql, location=REGION)
    model_job.result()
    print("   ✅ Embedding model created")
except Exception as e:
    print(f"   ❌ Error: {e}")
    exit(1)

print()

# Step 6: Generate embeddings
print("6️⃣  Generating vector embeddings (this may take 1-2 minutes)...")
index_sql = f"""
CREATE OR REPLACE TABLE `{PROJECT_ID}.{DATASET_ID}.snow_vectors` AS
SELECT
  base.question,
  base.answer,
  emb.ml_generate_embedding_result as embedding
FROM ML.GENERATE_EMBEDDING(
  MODEL `{PROJECT_ID}.{DATASET_ID}.embedding_model`,
  (
    SELECT
      question,
      answer,
      CONCAT('Question: ', question, ' Answer: ', answer) as content
    FROM `{PROJECT_ID}.{DATASET_ID}.snow_faqs_raw`
  )
) as emb
JOIN `{PROJECT_ID}.{DATASET_ID}.snow_faqs_raw` as base
ON emb.question = base.question;
"""

try:
    index_job = bq_client.query(index_sql, location=REGION)
    index_job.result()

    # Get count
    count_result = list(bq_client.query(
        f"SELECT COUNT(*) as total FROM `{PROJECT_ID}.{DATASET_ID}.snow_vectors`",
        location=REGION
    ).result())[0]

    print(f"   ✅ Vector index created with {count_result.total} embeddings")
except Exception as e:
    print(f"   ❌ Error: {e}")
    exit(1)

print()
print("=" * 70)
print("✅ Setup complete! All resources ready for testing.")
print()
print("Next step: Run the Streamlit app")
print("   streamlit run app.py")
print()
