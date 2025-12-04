# Architecture Diagrams

System architecture diagrams for the Alaska Department of Snow Virtual Assistant.

## Files

| File | Format | Description | Viewer |
|------|--------|-------------|--------|
| `architecture.mmd` | Mermaid | Interactive flowchart diagram | [Mermaid Live Editor](https://mermaid.live/), GitHub, VS Code |
| `architecture.txt` | ASCII | Text-based architecture diagram | Any text editor |

## Architecture Overview

The Alaska Snow Agent uses a multi-component architecture:

```
User Query → Security Check → RAG Retrieval → LLM Generation → Security Check → Response
              (Model Armor)    (BigQuery)      (Gemini 2.5)    (Model Armor)
```

### Components

1. **Frontend**: Streamlit web application
2. **Security Layer**: Google Model Armor for prompt injection protection
3. **RAG System**: BigQuery vector search with text-embedding-004
4. **LLM**: Gemini 2.5 Flash for response generation
5. **External APIs**:
   - Google Geocoding API (optional, with fallback coordinates)
   - National Weather Service API (free, public)
6. **Logging**: BigQuery interaction logs for audit trail

### Data Flow

1. User submits query through Streamlit chat interface
2. **Input Security**: Model Armor sanitizes user input
3. **Retrieval**: Query embedded and searched against BigQuery vector index (top-k=3)
4. **Context Building**: Retrieved FAQs combined with system instructions
5. **Function Calling**: Gemini decides if weather API should be called
6. **Generation**: Gemini generates response using RAG context
7. **Output Security**: Model Armor sanitizes model response
8. **Logging**: Interaction logged to BigQuery
9. **Response**: Answer displayed to user

## Viewing the Diagrams

### Mermaid Diagram (architecture.mmd)

**Option 1: Mermaid Live Editor**
1. Copy contents of `architecture.mmd`
2. Open https://mermaid.live/
3. Paste and view interactive diagram

**Option 2: VS Code**
1. Install "Markdown Preview Mermaid Support" extension
2. Open `architecture.mmd`
3. Click preview icon

**Option 3: GitHub**
GitHub automatically renders Mermaid diagrams in markdown files.

### ASCII Diagram (architecture.txt)

Simply open in any text editor. Best viewed in monospace font.

```bash
cat architecture.txt
```

## Diagram Contents

Both diagrams illustrate:

- **User interaction flow** - From query to response
- **Security checkpoints** - Input and output sanitization
- **RAG pipeline** - Embedding, retrieval, and context building
- **Function calling** - Weather API integration
- **Error handling** - Graceful degradation when services unavailable
- **Logging** - Audit trail in BigQuery

## Related Documentation

- `../deployment/README.md` - Deployment architecture
- `../deployment/docs/DEPLOYMENT.md` - Comprehensive deployment guide
- `../notebook/` - Jupyter notebooks with implementation details

## Regenerating Diagrams

The diagrams were generated in the Jupyter notebook (Cell 9). To regenerate:

1. Open `../notebook/challenge_05_alaska_snow_final.ipynb` in Google Colab
2. Run Cell 9: "Create Architecture Diagrams"
3. Downloads will include updated versions of these files
