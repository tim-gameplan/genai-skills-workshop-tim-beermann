# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Repository Purpose

This repository is for a 2-day (16-hour) Google Cloud Generative AI and Contact Center AI (CCAI) bootcamp. The focus is hands-on demonstration of building an end-to-end GenAI solution, not lectures.

## Bootcamp Challenges & Grading

**Total Points: 110 | Passing Score: 80 points**

| Challenge | Points |
|-----------|--------|
| Challenge One: Gemini Prompt Security | 20 |
| Challenge Two: Programming a RAG System in BigQuery | 20 |
| Challenge Three: Testing and Evaluation | 20 |
| Bonus Challenge Four: Using Agent Builder | 10 |
| Challenge Five: Alaska Department of Snow Online Agent | 40 |

The final challenge (Alaska Department of Snow Online Agent) is the capstone project worth the most points and likely integrates all previous learnings.

## Expected Project Architecture

The likely capstone project is a **Customer Service Agent** with the following components:

### Frontend: Conversational Interface
- **Dialogflow CX**: Stateful conversational agent for intent routing and flow management
- **GenAI Integration**: Generators, Generative Fallback, and Data Stores within Dialogflow CX

### Backend: AI Processing & Integration
- **Vertex AI/Gemini**: Core LLM for text generation and multimodal understanding
- **LangChain (Python)**: Framework for orchestrating LLM calls and backend logic
- **Cloud Functions**: Webhooks for dynamic responses from Dialogflow CX

### Data & Knowledge Layer
- **RAG (Retrieval Augmented Generation)**: Grounding LLM responses in specific documents/data
- **BigQuery**: Data warehouse for querying structured information
- **Data Stores**: For enterprise search and document grounding

### Infrastructure
- **Terraform**: Infrastructure as Code for deploying all resources
- **Google Cloud Platform**: All services run on GCP

## Critical Integration Points

When working on code in this repository, pay special attention to:

1. **Dialogflow CX ↔ Cloud Functions**: Webhook integration for dynamic backend logic
2. **LangChain ↔ Vertex AI**: Python code connecting to Google Cloud LLMs
3. **RAG Implementation**: Document ingestion and grounding for factual responses
4. **Function Calling**: LLMs taking actions via tools/APIs
5. **Terraform State Management**: Applying `.tf` files and managing state

## Technology Stack

### Primary Languages & Frameworks
- Python (LangChain, Cloud Functions)
- SQL (BigQuery queries)
- Terraform (HCL configuration)

### Google Cloud Services
- Vertex AI Studio (model testing, prompt engineering)
- Dialogflow CX (conversational agent)
- Cloud Functions (webhooks)
- BigQuery (data analytics)
- Cloud Speech API (optional)
- Gemini API

## Development Workflow

### Vertex AI Studio
- Prototype prompts before coding
- Test models (text, chat, code generation)
- Use the API for production integration

### Dialogflow CX Development
1. Design conversation flows with intent recognition
2. Implement webhooks for backend integration
3. Add GenAI generators for dynamic responses
4. Connect to data stores for grounding

### RAG Implementation
1. Ingest documents (PDF/text)
2. Configure data stores
3. Ground LLM responses to prevent hallucinations
4. Cite sources in responses

### Infrastructure Deployment
- Use provided Terraform scripts
- Understand state file management
- Apply configurations: `terraform apply`
- Deploy resources (servers, load balancers, etc.)

## Key Concepts to Understand

- **Grounding**: Anchoring LLM responses to specific data sources
- **Function Calling**: Extending LLMs to execute tools and take actions
- **Multimodality**: Processing text + images/video with Gemini
- **Rapid Evaluation API**: Testing and validating GenAI apps
- **Conversation Design**: Crafting human-like chat experiences

## Track Focus

This repository follows the **Developer Track** (not ML Engineer track), which is required for partners executing a Sprint. The emphasis is on:
- GAD series courses (Vertex AI, LangChain, Gemini Integration)
- CCAI/Dialogflow CX implementation
- Security, operations, and evaluation practices
