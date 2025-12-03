# Challenge 5: Implementation Overview

## Quick Reference

| Item | Value |
|------|-------|
| **Data Source** | `gs://labs.roitraining.com/alaska-dept-of-snow` |
| **Dataset ID** | `alaska_snow_rag` |
| **Region** | `us-central1` |
| **Model** | `gemini-2.5-flash` |
| **Embedding** | `text-embedding-004` |
| **Points** | 40 (36% of workshop) |

---

## Implementation Sequence

| Step | File | Duration | Description |
|------|------|----------|-------------|
| 1 | `01-data-preparation-and-rag.md` | 3-4 hours | Load data, create embeddings, build RAG |
| 2 | `02-security-layer.md` | 1-2 hours | Model Armor, input/output filtering |
| 3 | `03-testing-and-evaluation.md` | 2-3 hours | pytest + Vertex AI Evaluation |
| 4 | `04-deployment.md` | 2-3 hours | Cloud Run web deployment |
| 5 | `05-architecture-diagram.md` | 1 hour | System diagram + documentation |

**Total Estimated Time:** 10-14 hours

---

## Prerequisites Checklist

Before starting, ensure you have:

- [ ] Google Cloud project with billing enabled
- [ ] Qwiklabs/Cloud Skills Boost credentials (if using lab environment)
- [ ] APIs enabled:
  - [ ] Vertex AI API
  - [ ] BigQuery API
  - [ ] Cloud Run API (for deployment)
  - [ ] Model Armor API (for security)
  - [ ] Cloud Logging API

---

## Directory Structure (Final)

```
challenge-05-alaska-snow-agent/
├── IMPLEMENTATION_PLAN.md          # Original plan
├── implementation/
│   ├── 00-overview.md              # This file
│   ├── 01-data-preparation-and-rag.md
│   ├── 02-security-layer.md
│   ├── 03-testing-and-evaluation.md
│   ├── 04-deployment.md
│   ├── 05-architecture-diagram.md
│   └── notebooks/
│       ├── 01_data_and_rag.ipynb
│       ├── 02_security.ipynb
│       ├── 03_evaluation.ipynb
│       └── 04_full_agent.ipynb
├── src/
│   ├── agent.py                    # AlaskaSnowAgent class
│   ├── security.py                 # Security utilities
│   └── app.py                      # Flask/FastAPI web app
├── tests/
│   └── test_agent.py               # pytest tests
├── deployment/
│   ├── Dockerfile
│   ├── requirements.txt
│   └── cloudbuild.yaml
├── static/
│   ├── index.html
│   └── style.css
└── README.md                       # Final documentation
```

---

## Quick Start Commands

```bash
# Set project (replace with your project ID)
export PROJECT_ID="your-qwiklabs-project-id"
gcloud config set project $PROJECT_ID

# Enable APIs
gcloud services enable aiplatform.googleapis.com
gcloud services enable bigquery.googleapis.com
gcloud services enable run.googleapis.com
gcloud services enable cloudbuild.googleapis.com

# Verify authentication
gcloud auth list
```

---

## Key Patterns from Previous Challenges

### From Challenge 1 (Security)
```python
# Model Armor sanitization pattern
from google.cloud import modelarmor_v1

armor_client = modelarmor_v1.ModelArmorClient(
    client_options={"api_endpoint": f"modelarmor.{REGION}.rep.googleapis.com"}
)
```

### From Challenge 2 (RAG)
```python
# Vector search pattern
VECTOR_SEARCH_SQL = """
SELECT base.answer, (1 - distance) AS similarity
FROM VECTOR_SEARCH(
    TABLE `{project}.{dataset}.vectors`, 'embedding',
    (SELECT ml_generate_embedding_result FROM ML.GENERATE_EMBEDDING(...)),
    top_k => 5
)
"""
```

### From Challenge 3 (Evaluation)
```python
# EvalTask pattern
from vertexai.evaluation import EvalTask

task = EvalTask(
    dataset=eval_dataset,
    metrics=["groundedness", "fluency", "coherence", "safety"],
    experiment="alaska-snow-eval"
)
result = task.evaluate(model=model)
```

---

## Success Criteria

### Minimum (32/40 points)
- ✅ RAG system returns grounded answers
- ✅ Basic security (input filtering)
- ✅ 10+ unit tests passing
- ✅ Evaluation metrics documented
- ✅ Website deployed and accessible

### Excellence (36-40/40 points)
- ✅ Model Armor + DLP integration
- ✅ Comprehensive logging
- ✅ 20+ tests, prompt comparison
- ✅ Professional UI
- ✅ Complete documentation

---

## Next Step

→ Proceed to `01-data-preparation-and-rag.md`
