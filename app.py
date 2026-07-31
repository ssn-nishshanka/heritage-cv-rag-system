
# =========================================================================
# NHPT Heritage Assistant
# Computer Vision (EfficientNetB0 style classifier) + LangChain RAG
# (FAISS + Ollama) + Conversation Memory
#
# CW01 — Part C: LangChain Conversational System
# Run with: streamlit run app.py
# =========================================================================

import os
import json
import time
from pathlib import Path

import requests
import numpy as np
import streamlit as st

APP_DIR = Path(__file__).parent
CV_MODEL_PATH = APP_DIR / "final_model.keras"
FAISS_INDEX_PATH = APP_DIR / "heritage_faiss_index"

IMG_SIZE = (224, 224)

CLASS_LABELS = [
    "Edwardian architecture",
    "Georgian architecture",
    "Gothic architecture",
    "Queen Anne architecture",
    "Romanesque architecture",
    "Tudor Revival architecture",
]

OLLAMA_BASE_URL = "http://localhost:11434"
OLLAMA_GENERATE_URL = f"{OLLAMA_BASE_URL}/api/generate"
DEFAULT_LLM_MODEL = "gemma3:12b"
EMBEDDING_MODEL = "nomic-embed-text"

CONNECT_TIMEOUT = 10
STREAM_TIMEOUT = 300
HEALTH_TIMEOUT = 5

ALL_STYLE_KEYWORDS = ["gothic", "georgian", "tudor", "edwardian", "romanesque", "queen anne"]

CANDIDATE_POOL_MULTIPLIER = 5
MIN_CANDIDATE_POOL = 30

RAG_PROMPT_TEMPLATE = """You are the NHPT Heritage Assistant, a knowledgeable on-site guide helping visitors understand historic sites and architectural heritage.

Conversation so far:
{chat_history}

Detected structural/visual context from the visitor's photo (if available):
{cv_context}

Retrieved reference material (authoritative source — for your use only, never mention it to the visitor):
{context}

This reference material is the only factual source available. If a requested fact is not present here, say plainly that you don't have that detail — do not guess or use prior knowledge.

Step 1 — Check conversation state:
- Look at {chat_history}. Has the detected style and confidence score already been stated to the visitor in a previous turn?
- If yes: do NOT restate the style name or confidence score, even briefly. This rule applies even when the visitor's question refers to "this style" while asking about something else (its period, features, history, or importance) — that is NOT a request to re-identify. Only restate if the visitor's question is literally asking what the style/building is (e.g. "what style is this," "can you identify this again," "what is this building").
- If no, and the visitor is asking to identify/describe/classify the image: state the style and confidence score once, then continue.
- Do not repeat facts, feature lists, or phrasing you have already given earlier in {chat_history}, even if they are relevant to the new question — find what's new or specific to THIS question instead. Only repeat something if the visitor explicitly asks you to.

Step 2 — Filter retrieved material by relevance:
- Read {context} and discard any sentence that does not directly answer {question}.
- Discard any retrieved sentence that names a different specific building, monument, or site than the one implied by the detected style, unless the visitor explicitly asks for a comparison or names that site themselves.
- Discard any sentence describing a different architectural style than the one detected in {cv_context}, unless explicitly asked to compare.
- Do not include a retrieved fact just because it mentions the same general topic (e.g. "windows," "stone") — it must be about the correct style/building.
- Do not mix in facts about maintenance, inspection, conservation, or unrelated topics unless the visitor's question is about that topic.

Step 3 — Answer like a guide, not a system:
- Use only the retrieved sentences that survived Step 2 and that haven't already been shared.
- If describing the image for the first time, lead with style + confidence, then defining features.
- For all other questions, answer directly with no style/confidence restatement, no unrelated facts.
- Do not combine characteristics from different architectural styles unless the visitor explicitly asks for a comparison.
- Keep responses concise (3-6 sentences). Do not begin with "According to my analysis" or similar.
- Each retrieved item in {context} is tagged with a topic label in brackets, e.g. [Gothic Architecture]. Whenever you state a fact drawn from that material, add a citation right after it in the form (Source: Topic Label), using the exact label from the brackets. Do not cite anything that isn't backed by a labeled item in {context}, and never invent a label.
- NEVER refer to "reference material," "context," "retrieved documents," "provided texts," "knowledge base," or your own limitations as an AI/model. These are internal tools only — the visitor should never know they exist. The one allowed exception is the (Source: Topic Label) citations described above, which the visitor should see.
- If the material doesn't cover what's asked, say so briefly and warmly in your own voice, e.g. "I don't have that detail on hand, but on-site staff or a nearby info panel should be able to help." Then stop — don't elaborate on what topics you do or don't have data on.
- If low_confidence = true in the detected context, note the uncertainty conversationally (e.g. "I'm not fully certain, but this looks like...") before answering.

Visitor question:
{question}

Answer:"""


# =========================================================================
# Ollama helpers
# =========================================================================
def check_ollama_connection() -> bool:
    try:
        r = requests.get(f"{OLLAMA_BASE_URL}/api/tags", timeout=HEALTH_TIMEOUT)
        return r.status_code == 200
    except Exception:
        return False


def get_available_models() -> list:
    try:
        r = requests.get(f"{OLLAMA_BASE_URL}/api/tags", timeout=HEALTH_TIMEOUT)
        if r.status_code == 200:
            return [m["name"] for m in r.json().get("models", [])]
    except Exception:
        pass
    return [DEFAULT_LLM_MODEL]


def stream_ollama(prompt_text: str, model: str, placeholder, temperature: float) -> str:
    """Stream a completion from Ollama's REST API with error handling.

    NOTE: the notebook's tested ask() function called llm.invoke(prompt) via
    LangChain's Ollama() wrapper with no options override, i.e. Ollama's own
    server defaults. We keep temperature user-adjustable (default ~0.7,
    close to Ollama's own default) rather than hardcoding a low value, since
    over-constraining generation was not part of what was tested.
    """
    payload = {
        "model": model,
        "prompt": prompt_text,
        "stream": True,
        "options": {"num_ctx": 4096, "num_predict": 600, "temperature": temperature},
    }
    full_text = ""
    first_token = False
    dots = 0
    try:
        with requests.post(
            OLLAMA_GENERATE_URL, json=payload, stream=True,
            timeout=(CONNECT_TIMEOUT, STREAM_TIMEOUT),
        ) as resp:
            resp.raise_for_status()
            for line in resp.iter_lines():
                if not line:
                    if not first_token:
                        dots = (dots + 1) % 4
                        placeholder.markdown(f"⏳ Thinking{'.' * (dots + 1)}")
                    continue
                try:
                    chunk = json.loads(line.decode("utf-8"))
                except json.JSONDecodeError:
                    continue
                token = chunk.get("response", "")
                if token:
                    first_token = True
                    full_text += token
                    placeholder.markdown(full_text + "▌")
                if chunk.get("done", False):
                    break
        placeholder.markdown(full_text.strip())
        return full_text.strip()
    except requests.exceptions.ConnectTimeout:
        msg = "❌ Connection timeout — is Ollama running? Try `ollama serve`."
        placeholder.error(msg)
        return msg
    except requests.exceptions.ReadTimeout:
        msg = "❌ The model is taking too long to respond. Try a smaller model or ask a shorter question."
        placeholder.error(msg)
        return msg
    except requests.exceptions.ConnectionError:
        msg = "❌ Cannot connect to Ollama. Run `ollama serve` in a terminal first."
        placeholder.error(msg)
        return msg
    except requests.exceptions.HTTPError as e:
        status = e.response.status_code if e.response else "?"
        if status == 404:
            msg = f"❌ Model `{model}` not found locally. Run: `ollama pull {model}`"
        else:
            msg = f"❌ Ollama HTTP error {status}: {e}"
        placeholder.error(msg)
        return msg
    except Exception as e:
        msg = f"❌ Unexpected error: {type(e).__name__}: {e}"
        placeholder.error(msg)
        return msg


# =========================================================================
# Resource loading (cached)
# =========================================================================
@st.cache_resource(show_spinner=False)
def load_vectordb():
    from langchain_community.vectorstores import FAISS
    from langchain_ollama import OllamaEmbeddings

    embedding = OllamaEmbeddings(model=EMBEDDING_MODEL, base_url=OLLAMA_BASE_URL)
    vectordb = FAISS.load_local(
        str(FAISS_INDEX_PATH), embedding, allow_dangerous_deserialization=True
    )
    return vectordb


@st.cache_resource(show_spinner=False)
def load_cv_model():
    from tensorflow import keras
    return keras.models.load_model(str(CV_MODEL_PATH))


@st.cache_resource(show_spinner=False)
def load_class_labels():
    return CLASS_LABELS


# =========================================================================
# CV inference — the CV -> LLM handoff payload
# =========================================================================
def classify_structure(pil_image, confidence_threshold: float = 0.60) -> dict:
    from tensorflow.keras.applications.efficientnet import preprocess_input

    cv_model = load_cv_model()
    class_labels = load_class_labels()

    img = pil_image.convert("RGB").resize(IMG_SIZE)
    x = np.expand_dims(np.array(img, dtype=np.float32), axis=0)
    x = preprocess_input(x)

    probs = cv_model.predict(x, verbose=0)[0]
    pred_idx = int(np.argmax(probs))

    return {
        "predicted_style": class_labels[pred_idx],
        "confidence": float(probs[pred_idx]),
        "all_probabilities": {class_labels[i]: float(p) for i, p in enumerate(probs)},
        "low_confidence": bool(probs[pred_idx] < confidence_threshold),
    }

def retrieve_filtered_docs(question: str, vectordb, cv_result: dict | None, top_k: int = 6):
    predicted_style = cv_result.get("predicted_style", "") if cv_result else ""
    style_keyword = predicted_style.split()[0].lower() if predicted_style else ""

    if style_keyword:
        other_keywords = [kw for kw in ALL_STYLE_KEYWORDS if kw != style_keyword]

        def is_style_pure(doc):
            topic = doc.metadata.get("topic", "").lower()
            mentions_target = style_keyword in topic
            mentions_other = any(kw in topic for kw in other_keywords)
            return mentions_target and not mentions_other

    
        pool_size = max(MIN_CANDIDATE_POOL, top_k * CANDIDATE_POOL_MULTIPLIER)
        wide_retriever = vectordb.as_retriever(
            search_type="similarity", search_kwargs={"k": pool_size}
        )
        candidates = wide_retriever.invoke(question)
        retrieved = [doc for doc in candidates if is_style_pure(doc)][:top_k]

    else:
        retriever = vectordb.as_retriever(
            search_type="similarity", search_kwargs={"k": top_k}
        )
        retrieved = retriever.invoke(question)

    return retrieved


def format_chat_history(history: list) -> str:
    """Mirrors ConversationBufferMemory's default text format."""
    if not history:
        return ""
    lines = []
    for role, msg in history:
        label = "Human" if role == "user" else "AI"
        lines.append(f"{label}: {msg}")
    return "\n".join(lines)


def build_rag_prompt(question, docs, chat_history, cv_result):
    context = "\n".join(f"[{d.metadata.get('topic','')}] {d.page_content}" for d in docs)
    cv_context = str(cv_result) if cv_result else "None provided."
    return RAG_PROMPT_TEMPLATE.format(
        context=context if context else "No matching reference material found.",
        question=question,
        chat_history=chat_history,
        cv_context=cv_context,
    )


# =========================================================================
# Streamlit UI
# =========================================================================
st.set_page_config(page_title="Architectural Heritage Assistant", page_icon="🏛️", layout="wide")

if "history" not in st.session_state:
    st.session_state["history"] = []          # list[(role, message)]
if "cv_result" not in st.session_state:
    st.session_state["cv_result"] = None
if "sources_log" not in st.session_state:
    st.session_state["sources_log"] = []      # sources shown per assistant turn

# ---------------- Sidebar ----------------
with st.sidebar:
    st.header("⚙️ Settings")

    ollama_ok = check_ollama_connection()
    if ollama_ok:
        st.success("✅ Ollama connected")
        available_models = get_available_models()
    else:
        st.error("❌ Ollama offline")
        st.code("ollama serve", language="bash")
        st.caption("Also make sure you've pulled the models:")
        st.code(f"ollama pull {DEFAULT_LLM_MODEL}\nollama pull {EMBEDDING_MODEL}", language="bash")
        st.stop()

    selected_model = st.selectbox(
        "🤖 LLM model",
        options=available_models,
        index=available_models.index(DEFAULT_LLM_MODEL) if DEFAULT_LLM_MODEL in available_models else 0,
    )

    st.divider()
    st.subheader("🔍 Retrieval")
    top_k = st.slider("Documents to use as final context (k)", 1, 10, 6)
    confidence_threshold = st.slider("CV low-confidence threshold", 0.0, 1.0, 0.60, 0.05)
    temperature = st.slider("LLM temperature", 0.0, 1.0, 0.7, 0.05)

    st.divider()
    st.subheader("📸 Upload a site photo")
    uploaded_file = st.file_uploader("Structure / feature photo", type=["jpg", "jpeg", "png"])

    if uploaded_file is not None:
        if not CV_MODEL_PATH.exists():
            st.error(f"CV model not found at `{CV_MODEL_PATH.name}`. Place `final_model.keras` next to app.py.")
        else:
            from PIL import Image
            pil_img = Image.open(uploaded_file)
            st.image(pil_img, caption="Uploaded photo", use_container_width=True)
            with st.spinner("Classifying architectural style..."):
                try:
                    cv_result = classify_structure(pil_img, confidence_threshold)
                    st.session_state["cv_result"] = cv_result
                except Exception as e:
                    st.error(f"CV inference failed: {e}")
                    cv_result = None

            if cv_result:
                if cv_result["low_confidence"]:
                    st.warning(f"⚠️ Low confidence ({cv_result['confidence']:.0%}) — "
                               f"best guess: **{cv_result['predicted_style']}**")
                else:
                    st.success(f"**{cv_result['predicted_style']}** ({cv_result['confidence']:.0%} confidence)")

                with st.expander("Full probability breakdown"):
                    for style, p in sorted(cv_result["all_probabilities"].items(), key=lambda x: -x[1]):
                        st.progress(p, text=f"{style}: {p:.1%}")

    if st.session_state["cv_result"] and st.button("🗑️ Clear uploaded photo context", use_container_width=True):
        st.session_state["cv_result"] = None
        st.rerun()

    st.divider()
    if st.button("🗑️ Clear conversation", use_container_width=True):
        st.session_state["history"] = []
        st.session_state["sources_log"] = []
        st.rerun()

# ---------------- Vector DB load ----------------
if not FAISS_INDEX_PATH.exists():
    st.error(
        f"FAISS index not found at `{FAISS_INDEX_PATH.name}/`. "
        "Run the Part C notebook cells that build `heritage_faiss_index` and "
        "copy the folder next to app.py."
    )
    st.stop()

with st.spinner("📚 Loading heritage knowledge base..."):
    try:
        vectordb = load_vectordb()
    except Exception as e:
        st.error(f"❌ Failed to load FAISS index: {e}")
        st.stop()

# ---------------- Main ----------------
st.title("Architectural Heritage Assistant")
st.caption(
    f"Model: **{selected_model}**  ·  Context docs: **{top_k}**  ·  "
    + ("📷 Photo context active" if st.session_state["cv_result"] else "No photo uploaded yet")
)

for i, (role, msg) in enumerate(st.session_state["history"]):
    with st.chat_message("user" if role == "user" else "assistant"):
        st.markdown(msg)
        if role == "assistant" and i < len(st.session_state["sources_log"]):
            sources = st.session_state["sources_log"][i]
            if sources:
                with st.expander("📚 Retrieved chunks used this turn"):
                    for s in sources:
                        st.caption(f"• {s}")

user_message = st.chat_input("Ask about this site, its style, history, or preservation...")

if user_message:
    user_message = user_message.strip()
    st.session_state["history"].append(("user", user_message))
    with st.chat_message("user"):
        st.markdown(user_message)

    with st.chat_message("assistant"):
        cv_result = st.session_state["cv_result"]

        with st.spinner("🔍 Searching the heritage knowledge base..."):
            try:
                docs = retrieve_filtered_docs(user_message, vectordb, cv_result, top_k=top_k)
            except Exception as e:
                st.error(f"Retrieval error: {e}")
                docs = []

        chat_history_text = format_chat_history(st.session_state["history"][:-1])
        prompt_text = build_rag_prompt(user_message, docs, chat_history_text, cv_result)

        placeholder = st.empty()
        answer = stream_ollama(prompt_text, selected_model, placeholder, temperature)

        sources = sorted({d.metadata.get("topic", "Unknown") for d in docs})
        if sources:
            with st.expander("📚 Retrieved chunks used this turn"):
                for s in sources:
                    st.caption(f"• {s}")

    st.session_state["history"].append(("assistant", answer))
    st.session_state["sources_log"].append(sources)
