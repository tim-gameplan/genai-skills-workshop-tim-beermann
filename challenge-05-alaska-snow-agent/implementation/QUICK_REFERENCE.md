# Quick Reference Card

## Essential Commands

### Project Configuration
```python
PROJECT_ID = "your-qwiklabs-project-id"  # CHANGE THIS!
REGION = "us-central1"
DATASET_ID = "alaska_snow_rag"
DATA_URI = "gs://labs.roitraining.com/alaska-dept-of-snow/alaska-dept-of-snow.csv"
SECURITY_TEMPLATE_ID = "alaska-snow-security"
```

### BigQuery Connection Setup
```sql
-- Create remote model for embeddings
CREATE OR REPLACE MODEL `{PROJECT}.{DATASET}.embedding_model`
REMOTE WITH CONNECTION `{PROJECT}.{REGION}.vertex-ai-conn`
OPTIONS (ENDPOINT = 'text-embedding-004');
```

### Vector Search Query
```sql
SELECT question, answer, (1 - distance) AS relevance
FROM VECTOR_SEARCH(
    TABLE `{PROJECT}.{DATASET}.snow_vectors`, 'embedding',
    (SELECT ml_generate_embedding_result
     FROM ML.GENERATE_EMBEDDING(
         MODEL `{PROJECT}.{DATASET}.embedding_model`,
         (SELECT '{query}' AS content))),
    top_k => 5
) ORDER BY relevance DESC;
```

### Model Armor Template (REST API)
```bash
curl -X POST \
  "https://modelarmor.us-central1.rep.googleapis.com/v1/projects/${PROJECT}/locations/us-central1/templates?templateId=alaska-snow-security" \
  -H "Authorization: Bearer $(gcloud auth print-access-token)" \
  -H "Content-Type: application/json" \
  -d '{
    "filterConfig": {
      "piAndJailbreakFilterSettings": {
        "filterEnforcement": "ENABLED",
        "confidenceLevel": "LOW_AND_ABOVE"
      },
      "sdpSettings": {
        "basicConfig": {"filterEnforcement": "ENABLED"}
      }
    }
  }'
```

### Cloud Run Deploy
```bash
gcloud run deploy alaska-snow-agent \
    --source . \
    --region us-central1 \
    --allow-unauthenticated \
    --set-env-vars "PROJECT_ID=$PROJECT_ID,REGION=us-central1" \
    --memory 1Gi
```

### Grant Permissions
```bash
# For BigQuery connection service account
gcloud projects add-iam-policy-binding $PROJECT_ID \
    --member="serviceAccount:${SERVICE_ACCOUNT}" \
    --role="roles/aiplatform.user"
```

### Run Tests
```bash
pytest -v test_alaska_snow_agent.py
```

### Evaluation Metrics
```python
from vertexai.evaluation import EvalTask

metrics = ["groundedness", "fluency", "coherence", "safety", "fulfillment"]
task = EvalTask(dataset=df, metrics=metrics, experiment="alaska-eval")
result = task.evaluate(model=model)
```

---

## Checklist

### Phase 1: Data & RAG
- [ ] Create BigQuery dataset `alaska_snow_rag`
- [ ] Load data from GCS
- [ ] Create Vertex AI connection
- [ ] Grant `aiplatform.user` to connection service account
- [ ] Create embedding model
- [ ] Generate embeddings table `snow_vectors`
- [ ] Test `ask_alaska_snow()` function

### Phase 2: Security
- [ ] Create Model Armor template
- [ ] Implement `sanitize_input()`
- [ ] Implement `sanitize_output()`
- [ ] Create `interaction_logs` table
- [ ] Test prompt injection blocking
- [ ] Verify logging works

### Phase 3: Testing & Eval
- [ ] Create `test_alaska_snow_agent.py`
- [ ] Run pytest (18+ tests)
- [ ] Create evaluation dataset
- [ ] Run EvalTask with 5+ metrics
- [ ] Compare 3 prompt variants
- [ ] Export results to CSV

### Phase 4: Deployment
- [ ] Create Flask `app.py`
- [ ] Create `index.html` template
- [ ] Create `requirements.txt`
- [ ] Create `Dockerfile`
- [ ] Deploy to Cloud Run
- [ ] Grant service account permissions
- [ ] Test public URL
- [ ] Verify all endpoints

### Phase 5: Documentation
- [ ] Create architecture diagram
- [ ] Write README.md
- [ ] Export notebooks
- [ ] Create GitHub repository
- [ ] Push all files
- [ ] Share URLs with instructor

---

## Estimated Points

| Component | Points | Status |
|-----------|--------|--------|
| Architecture Diagram | 5 | [ ] |
| RAG Implementation | 10 | [ ] |
| Security | 8 | [ ] |
| Unit Tests | 7 | [ ] |
| Evaluation | 5 | [ ] |
| Deployment | 5 | [ ] |
| **Total** | **40** | |

**Target:** 36+ points for excellence
