"""
Alaska Department of Snow - Virtual Assistant
Streamlit Web Application
"""

import streamlit as st
import vertexai
from google.cloud import bigquery, modelarmor_v1
from vertexai.generative_models import GenerativeModel
import os
import datetime

# =============================================================================
# CONFIGURATION
# =============================================================================

PROJECT_ID = os.environ.get("PROJECT_ID", "qwiklabs-gcp-03-ba43f2730b93")
REGION = os.environ.get("REGION", "us-central1")
DATASET_ID = "alaska_snow_capstone"

# =============================================================================
# PAGE CONFIGURATION
# =============================================================================

st.set_page_config(
    page_title="Alaska Department of Snow",
    page_icon="❄️",
    layout="centered",
    initial_sidebar_state="collapsed"
)

# Custom CSS
st.markdown("""
<style>
    .stApp {
        background-color: #f0f8ff;
    }
    .stChatMessage {
        background-color: white;
        border-radius: 10px;
        padding: 10px;
        margin: 5px 0;
    }
</style>
""", unsafe_allow_html=True)

# =============================================================================
# HEADER
# =============================================================================

st.title("❄️ Alaska Department of Snow")
st.markdown("### Virtual Assistant for Plowing & Closure Information")

st.markdown("""
**Ask me about:**
- Snow plowing schedules
- Priority routes
- School closures
- Parking bans
- Reporting unplowed streets
""")

st.divider()

# =============================================================================
# AGENT CLASS
# =============================================================================

class AlaskaSnowAgentEnhanced:
    """
    Production-grade RAG agent for Alaska Department of Snow.
    """

    def __init__(self, project_id, region, dataset_id):
        # Initialize Vertex AI
        vertexai.init(project=project_id, location=region)

        # Gemini 2.5 Flash for generation
        self.model = GenerativeModel("gemini-2.5-flash")

        # BigQuery client
        self.bq_client = bigquery.Client(project=project_id, location=region)
        self.project_id = project_id
        self.region = region
        self.dataset_id = dataset_id

        # Model Armor client for security
        self.armor_client = modelarmor_v1.ModelArmorClient(
            client_options={"api_endpoint": f"modelarmor.{region}.rep.googleapis.com"}
        )
        self.armor_template = f"projects/{project_id}/locations/{region}/templates/basic-security-template"

        # System instruction for consistent behavior
        self.system_instruction = """
        You are the official virtual assistant for the Alaska Department of Snow (ADS).

        ROLE:
        - Answer citizen questions about snow plowing schedules
        - Provide information on road conditions and closures
        - Inform about school closures due to weather

        GUIDELINES:
        - Base ALL answers on the provided CONTEXT ONLY
        - Be concise, professional, and helpful
        - If information is not in the context, say: "I don't have that information. Please call the ADS hotline at 555-SNOW."
        """

    def _log(self, step, message):
        """Simple logging for debugging."""
        timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        print(f"[{timestamp}] [{step}] {message}")

    def sanitize(self, text, check_type="input"):
        """
        Security wrapper using Model Armor API.

        Checks for:
        - Prompt injection attempts (jailbreaks)
        - Malicious URIs
        - PII (Personally Identifiable Information)
        """
        try:
            if check_type == "input":
                request = modelarmor_v1.SanitizeUserPromptRequest(
                    name=self.armor_template,
                    user_prompt_data=modelarmor_v1.DataItem(text=text)
                )
                response = self.armor_client.sanitize_user_prompt(request=request)
            else:
                request = modelarmor_v1.SanitizeModelResponseRequest(
                    name=self.armor_template,
                    model_response_data=modelarmor_v1.DataItem(text=text)
                )
                response = self.armor_client.sanitize_model_response(request=request)

            is_safe = response.sanitization_result.filter_match_state == 1

            if not is_safe:
                self._log("SECURITY", f"⚠️  {check_type.upper()} BLOCKED")
                return False

            return True

        except Exception as e:
            self._log("WARN", f"Security check skipped: {e}")
            return True  # Fail open

    def retrieve(self, query):
        """
        Retrieve relevant FAQs using BigQuery vector search.
        """
        safe_query = query.replace("'", "\\'")

        sql = f"""
        SELECT
          base.answer,
          (1 - distance) as relevance_score
        FROM VECTOR_SEARCH(
          TABLE `{self.project_id}.{self.dataset_id}.snow_vectors`,
          'embedding',
          (
            SELECT ml_generate_embedding_result
            FROM ML.GENERATE_EMBEDDING(
              MODEL `{self.project_id}.{self.dataset_id}.embedding_model`,
              (SELECT '{safe_query}' AS content)
            )
          ),
          top_k => 3
        )
        ORDER BY relevance_score DESC
        """

        rows = self.bq_client.query(sql, location=self.region).result()
        context_pieces = []
        for row in rows:
            context_pieces.append(f"- {row.answer}")

        context = "\n".join(context_pieces)
        if not context:
            context = "No relevant records found in the knowledge base."

        self._log("RETRIEVAL", f"Found {len(context_pieces)} relevant entries")
        return context

    def chat(self, user_query):
        """
        Main chat interface - orchestrates the full RAG pipeline.
        """
        self._log("CHAT_START", f"Query: {user_query}")

        # 1. Security Check
        if not self.sanitize(user_query, "input"):
            return "❌ Your request was blocked by our security policy. Please rephrase your question."

        # 2. Retrieval
        context = self.retrieve(user_query)

        # 3. Generation
        full_prompt = f"""
{self.system_instruction}

CONTEXT (from official ADS knowledge base):
{context}

USER QUESTION:
{user_query}

ASSISTANT RESPONSE:
"""

        self._log("GENERATION", "Sending to Gemini 2.5 Flash...")
        response_text = self.model.generate_content(full_prompt).text

        # 4. Output Security
        if not self.sanitize(response_text, "output"):
            return "❌ [REDACTED] - Response contained sensitive information."

        self._log("CHAT_END", "Response sent")
        return response_text

# =============================================================================
# AGENT INITIALIZATION
# =============================================================================

@st.cache_resource
def initialize_agent():
    """Initialize the agent (cached across sessions)."""
    return AlaskaSnowAgentEnhanced(PROJECT_ID, REGION, DATASET_ID)

agent = initialize_agent()

# =============================================================================
# CHAT INTERFACE
# =============================================================================

# Initialize chat history
if "messages" not in st.session_state:
    st.session_state.messages = []
    # Add welcome message
    st.session_state.messages.append({
        "role": "assistant",
        "content": "Hello! I'm the ADS Virtual Assistant. How can I help you with snow removal information today?"
    })

# Display chat history
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# User input
if prompt := st.chat_input("Ask about snow removal..."):
    # Add user message to chat
    st.session_state.messages.append({"role": "user", "content": prompt})

    with st.chat_message("user"):
        st.markdown(prompt)

    # Generate response
    with st.chat_message("assistant"):
        with st.spinner("Checking records..."):
            response = agent.chat(prompt)
            st.markdown(response)

    # Add assistant response to chat
    st.session_state.messages.append({"role": "assistant", "content": response})

# =============================================================================
# FOOTER
# =============================================================================

st.divider()
st.caption("Alaska Department of Snow Virtual Assistant | Powered by Google Gemini & BigQuery")
