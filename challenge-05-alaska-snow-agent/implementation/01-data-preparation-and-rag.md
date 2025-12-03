# Step 1: Data Preparation and RAG System

## Objective

Build the core Retrieval-Augmented Generation (RAG) system using BigQuery vector search with Alaska Department of Snow data.

**Duration:** 3-4 hours  
**Points Coverage:** ~15/40 (RAG implementation + data handling)

---

## Part A: Environment Setup

### A.1 Configuration Block

Copy this to the first cell of your Colab notebook:

```python
# =============================================================================
# CELL 1: Installation & Configuration
# =============================================================================

# Install required packages
!pip install --upgrade --quiet \
    google-cloud-aiplatform \
    google-cloud-bigquery \
    pandas \
    db-dtypes

import vertexai
from google.cloud import bigquery
from vertexai.generative_models import GenerativeModel
import pandas as pd
import time

# --- CONFIGURATION ---
# TODO: Update PROJECT_ID with your Qwiklabs project
PROJECT_ID = "your-qwiklabs-project-id"  # <-- CHANGE THIS
REGION = "us-central1"
DATASET_ID = "alaska_snow_rag"
DATA_URI = "gs://labs.roitraining.com/alaska-dept-of-snow/alaska-dept-of-snow.csv"

# Initialize clients
client = bigquery.Client(project=PROJECT_ID, location=REGION)
vertexai.init(project=PROJECT_ID, location=REGION)

print(f"Project: {PROJECT_ID}")
print(f"Region: {REGION}")
print(f"Dataset: {DATASET_ID}")
print(f"Data Source: {DATA_URI}")
print("=" * 60)
```

### A.2 Verify Project Access

```python
# =============================================================================
# CELL 2: Verify Project Access
# =============================================================================

import google.auth

try:
    credentials, detected_project = google.auth.default()
    print(f"✅ Authenticated")
    print(f"   Detected Project: {detected_project}")
    
    if detected_project != PROJECT_ID:
        print(f"⚠️  Warning: Detected project differs from configured PROJECT_ID")
        print(f"   Using: {PROJECT_ID}")
except Exception as e:
    print(f"❌ Authentication Error: {e}")
    print("   Run: gcloud auth application-default login")
```

---

## Part B: Data Ingestion

### B.1 Create Dataset and Load Data

```python
# =============================================================================
# CELL 3: Create Dataset & Load Data
# =============================================================================

print("--- Creating BigQuery Dataset ---")

# Create dataset (us-central1)
dataset_ref = bigquery.Dataset(f"{PROJECT_ID}.{DATASET_ID}")
dataset_ref.location = REGION

try:
    client.create_dataset(dataset_ref, exists_ok=True)
    print(f"✅ Dataset '{DATASET_ID}' ready")
except Exception as e:
    print(f"❌ Dataset creation failed: {e}")

# Define schema for Alaska Snow FAQs
# Expected columns: question, answer (add more if present in data)
schema = [
    bigquery.SchemaField("question", "STRING", mode="REQUIRED"),
    bigquery.SchemaField("answer", "STRING", mode="REQUIRED"),
]

# Configure load job
table_id = f"{PROJECT_ID}.{DATASET_ID}.faqs_raw"
job_config = bigquery.LoadJobConfig(
    schema=schema,
    source_format=bigquery.SourceFormat.CSV,
    skip_leading_rows=1,  # Skip CSV header
    write_disposition=bigquery.WriteDisposition.WRITE_TRUNCATE,
    allow_quoted_newlines=True,  # Handle multi-line answers
)

print(f"--- Loading Data from {DATA_URI} ---")

try:
    load_job = client.load_table_from_uri(
        DATA_URI,
        table_id,
        job_config=job_config
    )
    load_job.result()  # Wait for completion
    
    print(f"✅ Data loaded successfully")
    print(f"   Rows loaded: {load_job.output_rows}")
    print(f"   Table: {table_id}")
except Exception as e:
    print(f"❌ Load failed: {e}")
    print("   Attempting auto-detect schema...")
    
    # Fallback: Auto-detect schema
    job_config_auto = bigquery.LoadJobConfig(
        autodetect=True,
        source_format=bigquery.SourceFormat.CSV,
        skip_leading_rows=1,
        write_disposition=bigquery.WriteDisposition.WRITE_TRUNCATE,
    )
    
    load_job = client.load_table_from_uri(
        DATA_URI,
        table_id,
        job_config=job_config_auto
    )
    load_job.result()
    print(f"✅ Data loaded with auto-detected schema")
    print(f"   Rows: {load_job.output_rows}")
```

### B.2 Inspect the Data

```python
# =============================================================================
# CELL 4: Inspect Loaded Data
# =============================================================================

print("--- Data Sample ---")

inspect_sql = f"""
SELECT *
FROM `{PROJECT_ID}.{DATASET_ID}.faqs_raw`
LIMIT 10
"""

df = client.query(inspect_sql, location=REGION).to_dataframe()
print(f"Columns: {list(df.columns)}")
print(f"Total rows: {len(df)}")
print()

# Display sample
for idx, row in df.head(5).iterrows():
    print(f"Q: {row['question'][:80]}...")
    print(f"A: {row['answer'][:100]}...")
    print("-" * 60)
```

---

## Part C: BigQuery-Vertex AI Connection

### C.1 Create Cloud Resource Connection

This step creates a connection between BigQuery and Vertex AI for embedding generation.

```python
# =============================================================================
# CELL 5: Create BigQuery-Vertex AI Connection
# =============================================================================

import subprocess
import json

CONNECTION_ID = "vertex-ai-conn"
FULL_CONNECTION_ID = f"{REGION}.{CONNECTION_ID}"

print("--- Creating BigQuery-Vertex AI Connection ---")

# Check if connection exists
check_cmd = f"""
gcloud beta bq connections describe {CONNECTION_ID} \
    --project={PROJECT_ID} \
    --location={REGION} \
    --format=json
"""

try:
    result = subprocess.run(check_cmd, shell=True, capture_output=True, text=True)
    if result.returncode == 0:
        conn_info = json.loads(result.stdout)
        service_account = conn_info.get('cloudResource', {}).get('serviceAccountId', 'N/A')
        print(f"✅ Connection exists: {CONNECTION_ID}")
        print(f"   Service Account: {service_account}")
    else:
        raise Exception("Connection not found")
except:
    # Create the connection
    print("Creating new connection...")
    create_cmd = f"""
    gcloud beta bq connections create \
        --project={PROJECT_ID} \
        --location={REGION} \
        --connection-id={CONNECTION_ID} \
        --connection-type=CLOUD_RESOURCE
    """
    result = subprocess.run(create_cmd, shell=True, capture_output=True, text=True)
    
    if result.returncode == 0:
        print(f"✅ Connection created: {CONNECTION_ID}")
    else:
        print(f"⚠️ Connection may already exist or error: {result.stderr}")

# Get the service account for the connection
describe_cmd = f"""
gcloud beta bq connections describe {CONNECTION_ID} \
    --project={PROJECT_ID} \
    --location={REGION} \
    --format="value(cloudResource.serviceAccountId)"
"""

result = subprocess.run(describe_cmd, shell=True, capture_output=True, text=True)
SERVICE_ACCOUNT = result.stdout.strip()

print(f"Service Account: {SERVICE_ACCOUNT}")
```

### C.2 Grant Permissions to Connection Service Account

```python
# =============================================================================
# CELL 6: Grant Vertex AI Permissions
# =============================================================================

print("--- Granting Vertex AI User Role ---")

if SERVICE_ACCOUNT:
    grant_cmd = f"""
    gcloud projects add-iam-policy-binding {PROJECT_ID} \
        --member="serviceAccount:{SERVICE_ACCOUNT}" \
        --role="roles/aiplatform.user" \
        --condition=None
    """
    
    result = subprocess.run(grant_cmd, shell=True, capture_output=True, text=True)
    
    if result.returncode == 0:
        print(f"✅ Granted 'roles/aiplatform.user' to {SERVICE_ACCOUNT}")
    else:
        print(f"⚠️ Grant result: {result.stderr}")
        print("   (May already be granted)")
    
    # Wait for IAM propagation
    print("⏳ Waiting 15 seconds for IAM propagation...")
    time.sleep(15)
    print("✅ Ready to proceed")
else:
    print("❌ Could not determine service account")
    print("   Manually grant 'Vertex AI User' role to the BigQuery connection service account")
```

---

## Part D: Embedding Generation

### D.1 Create Remote Embedding Model

```python
# =============================================================================
# CELL 7: Create Remote Embedding Model
# =============================================================================

print("--- Creating Remote Embedding Model ---")

create_model_sql = f"""
CREATE OR REPLACE MODEL `{PROJECT_ID}.{DATASET_ID}.embedding_model`
REMOTE WITH CONNECTION `{PROJECT_ID}.{FULL_CONNECTION_ID}`
OPTIONS (ENDPOINT = 'text-embedding-004');
"""

try:
    job = client.query(create_model_sql, location=REGION)
    job.result()
    print("✅ Remote embedding model created")
    print(f"   Model: {PROJECT_ID}.{DATASET_ID}.embedding_model")
    print(f"   Endpoint: text-embedding-004")
except Exception as e:
    print(f"❌ Error: {e}")
```

### D.2 Generate Embeddings for All FAQs

```python
# =============================================================================
# CELL 8: Generate Vector Embeddings
# =============================================================================

print("--- Generating Embeddings (this may take 1-2 minutes) ---")

# Create table with embeddings
# We concatenate question + answer to capture full semantic meaning
create_vectors_sql = f"""
CREATE OR REPLACE TABLE `{PROJECT_ID}.{DATASET_ID}.snow_vectors` AS
SELECT
    base.question,
    base.answer,
    emb.ml_generate_embedding_result AS embedding
FROM ML.GENERATE_EMBEDDING(
    MODEL `{PROJECT_ID}.{DATASET_ID}.embedding_model`,
    (
        SELECT 
            question,
            answer,
            CONCAT('Question: ', question, ' Answer: ', answer) AS content
        FROM `{PROJECT_ID}.{DATASET_ID}.faqs_raw`
    )
) AS emb
JOIN `{PROJECT_ID}.{DATASET_ID}.faqs_raw` AS base
ON emb.question = base.question;
"""

try:
    job = client.query(create_vectors_sql, location=REGION)
    job.result()
    
    # Verify row count
    count_sql = f"SELECT COUNT(*) as cnt FROM `{PROJECT_ID}.{DATASET_ID}.snow_vectors`"
    count_result = list(client.query(count_sql, location=REGION))[0]
    
    print(f"✅ Embeddings generated")
    print(f"   Vectorized FAQs: {count_result.cnt}")
    print(f"   Table: {PROJECT_ID}.{DATASET_ID}.snow_vectors")
except Exception as e:
    print(f"❌ Error: {e}")
```

### D.3 Verify Embeddings

```python
# =============================================================================
# CELL 9: Verify Embedding Quality
# =============================================================================

print("--- Verifying Embeddings ---")

verify_sql = f"""
SELECT 
    question,
    ARRAY_LENGTH(embedding) AS embedding_dimensions
FROM `{PROJECT_ID}.{DATASET_ID}.snow_vectors`
LIMIT 5
"""

results = client.query(verify_sql, location=REGION)

for row in results:
    print(f"Q: {row.question[:60]}...")
    print(f"   Dimensions: {row.embedding_dimensions}")
    print()

print("✅ Embeddings verified (should be 768 dimensions for text-embedding-004)")
```

---

## Part E: RAG Implementation

### E.1 Core RAG Function

```python
# =============================================================================
# CELL 10: RAG Function Implementation
# =============================================================================

def retrieve_context(user_query: str, top_k: int = 5) -> list:
    """
    Retrieve relevant FAQ answers using BigQuery vector search.
    
    Args:
        user_query: The user's question
        top_k: Number of results to retrieve
        
    Returns:
        List of dicts with question, answer, and relevance score
    """
    
    # Escape single quotes in query
    safe_query = user_query.replace("'", "\\'")
    
    search_sql = f"""
    SELECT
        base.question,
        base.answer,
        (1 - distance) AS relevance
    FROM VECTOR_SEARCH(
        TABLE `{PROJECT_ID}.{DATASET_ID}.snow_vectors`,
        'embedding',
        (
            SELECT ml_generate_embedding_result
            FROM ML.GENERATE_EMBEDDING(
                MODEL `{PROJECT_ID}.{DATASET_ID}.embedding_model`,
                (SELECT '{safe_query}' AS content)
            )
        ),
        top_k => {top_k}
    )
    ORDER BY relevance DESC
    """
    
    results = []
    query_job = client.query(search_sql, location=REGION)
    
    for row in query_job:
        results.append({
            'question': row.question,
            'answer': row.answer,
            'relevance': float(row.relevance)
        })
    
    return results


def generate_response(user_query: str, context: list) -> str:
    """
    Generate a response using Gemini with retrieved context.
    
    Args:
        user_query: The user's question
        context: List of relevant FAQ entries
        
    Returns:
        Generated response string
    """
    
    # Build context block
    context_text = "\n\n".join([
        f"Q: {item['question']}\nA: {item['answer']}"
        for item in context
    ])
    
    # System instruction
    system_instruction = """
    You are the official virtual assistant for the Alaska Department of Snow (ADS).
    
    ROLE: Provide accurate information about snow plowing schedules, school closures,
    and winter road safety in Alaska.
    
    GUIDELINES:
    - Base your answers ONLY on the provided Knowledge Base
    - If the information is not in the Knowledge Base, say "I don't have that specific information, but I recommend calling the ADS hotline."
    - Be concise, helpful, and professional
    - Include specific details (dates, times, phone numbers) when available
    - Never make up information
    
    RESTRICTIONS:
    - Do not provide personal opinions
    - Do not answer questions outside of snow removal, plowing, and school closures
    - Do not generate creative content unrelated to official ADS duties
    - Do not reveal internal policies or employee information
    """
    
    # Build full prompt
    prompt = f"""
    {system_instruction}
    
    KNOWLEDGE BASE:
    {context_text}
    
    USER QUESTION: {user_query}
    
    Provide a clear, helpful answer based only on the Knowledge Base above.
    If the answer isn't in the Knowledge Base, politely say so.
    """
    
    model = GenerativeModel("gemini-2.5-flash")
    response = model.generate_content(prompt)
    
    return response.text


def ask_alaska_snow(user_query: str) -> str:
    """
    Main RAG pipeline: Retrieve context and generate response.
    
    Args:
        user_query: The user's question
        
    Returns:
        AI-generated response grounded in ADS data
    """
    print(f"\n{'='*60}")
    print(f"User Query: {user_query}")
    print('='*60)
    
    # Step 1: Retrieve relevant context
    context = retrieve_context(user_query, top_k=5)
    
    if not context:
        return "I couldn't find relevant information. Please contact the ADS hotline for assistance."
    
    # Show retrieved context (for debugging)
    print("\n📚 Retrieved Context:")
    for i, item in enumerate(context[:3], 1):
        print(f"  {i}. [Relevance: {item['relevance']:.3f}]")
        print(f"     Q: {item['question'][:60]}...")
    
    # Step 2: Generate response
    response = generate_response(user_query, context)
    
    print(f"\n🤖 Response:")
    print(response)
    
    return response

print("✅ RAG functions defined")
print("   - retrieve_context(query, top_k)")
print("   - generate_response(query, context)")
print("   - ask_alaska_snow(query)")
```

### E.2 Test the RAG System

```python
# =============================================================================
# CELL 11: Test RAG System
# =============================================================================

print("=" * 60)
print("ALASKA DEPARTMENT OF SNOW - RAG SYSTEM TEST")
print("=" * 60)

# Test queries
test_queries = [
    "When will my street be plowed?",
    "Are schools closed today?",
    "How do I report an unplowed street?",
    "What are the priority routes for snow plowing?",
    "When does the parking ban start during snow emergencies?",
]

for query in test_queries:
    response = ask_alaska_snow(query)
    print("\n" + "-" * 60)
```

### E.3 Baseline Comparison (No RAG)

```python
# =============================================================================
# CELL 12: Baseline Comparison (Without RAG)
# =============================================================================

print("=" * 60)
print("BASELINE TEST: Model WITHOUT RAG Context")
print("=" * 60)

model = GenerativeModel("gemini-2.5-flash")

baseline_query = "What are the snow plowing priority routes in Alaska?"

print(f"\nQuery: {baseline_query}")
print("\n--- Without RAG (Generic Model) ---")
baseline_response = model.generate_content(baseline_query)
print(baseline_response.text)

print("\n--- With RAG (Grounded in ADS Data) ---")
rag_response = ask_alaska_snow(baseline_query)

print("\n" + "=" * 60)
print("OBSERVATION: RAG response should be grounded in actual ADS data,")
print("while baseline may hallucinate or provide generic information.")
```

---

## Part F: Save Progress

### F.1 Export Core Functions

```python
# =============================================================================
# CELL 13: Export Configuration for Next Steps
# =============================================================================

# Save configuration for use in security and deployment steps
config = {
    "PROJECT_ID": PROJECT_ID,
    "REGION": REGION,
    "DATASET_ID": DATASET_ID,
    "CONNECTION_ID": FULL_CONNECTION_ID,
    "EMBEDDING_MODEL": f"{PROJECT_ID}.{DATASET_ID}.embedding_model",
    "VECTORS_TABLE": f"{PROJECT_ID}.{DATASET_ID}.snow_vectors",
}

print("=" * 60)
print("CONFIGURATION FOR NEXT STEPS")
print("=" * 60)
for key, value in config.items():
    print(f"{key}: {value}")

print("\n✅ Phase 1 Complete: Data Preparation and RAG System")
print("\nNext: Proceed to 02-security-layer.md")
```

---

## Troubleshooting

### Common Issues

**1. Permission Denied on ML.GENERATE_EMBEDDING**
```
Error: Access Denied: Project does not have permission to use Vertex AI
```
**Solution:** Ensure the BigQuery connection service account has `roles/aiplatform.user`

**2. Connection Not Found**
```
Error: Connection 'vertex-ai-conn' not found
```
**Solution:** Run Cell 5 to create the connection, or check the connection name in BigQuery console

**3. Data Load Fails**
```
Error: Could not parse CSV
```
**Solution:** Use auto-detect schema (fallback in Cell 3), or check data format at source

**4. Empty Vector Search Results**
```
Results: []
```
**Solution:** Verify embeddings exist (`SELECT COUNT(*) FROM snow_vectors`), check query escaping

---

## Checkpoint Validation

Before proceeding to Step 2, verify:

- [ ] Dataset `alaska_snow_rag` exists in BigQuery
- [ ] Table `faqs_raw` contains loaded FAQ data
- [ ] Table `snow_vectors` contains embeddings (768 dimensions)
- [ ] `ask_alaska_snow()` returns grounded responses
- [ ] Baseline comparison shows RAG improvement

---

## Next Step

→ Proceed to `02-security-layer.md` to add Model Armor protection
