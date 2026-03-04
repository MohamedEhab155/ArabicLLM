"""
Run with:  streamlit run app/streamlit_app.py
"""
import json
import time
import os
import sys
import requests
import streamlit as st
from dotenv import load_dotenv

# ── Allow imports from project root ──
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from transformers import AutoTokenizer
from src.tasks.news_details_extraction_task import build_details_extraction_prompt
from src.tasks.translation_task import build_translation_prompt
from src.helper import config

load_dotenv("src/.env")

# ────────────────────────────────────────────────────────────────────────── #
#  Page config
# ────────────────────────────────────────────────────────────────────────── #
st.set_page_config(
    page_title="ArabicLLM · Inference Studio",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ────────────────────────────────────────────────────────────────────────── #
#  Custom CSS
# ────────────────────────────────────────────────────────────────────────── #
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Syne:wght@400;700;800&family=IBM+Plex+Mono&family=Tajawal:wght@400;500;700&display=swap');

html, body, [class*="css"] { font-family: 'Syne', sans-serif; }

.main { background: #0a0b0f; }
.block-container { padding-top: 1.5rem; max-width: 1200px; }

/* Sidebar */
[data-testid="stSidebar"] {
    background: #111318 !important;
    border-right: 1px solid #1e222c;
}

/* Metric cards */
[data-testid="metric-container"] {
    background: #161920;
    border: 1px solid #1e222c;
    border-radius: 12px;
    padding: 16px;
}

/* Buttons */
.stButton > button {
    background: linear-gradient(135deg, #4f8ef7, #a78bfa) !important;
    color: white !important;
    border: none !important;
    border-radius: 10px !important;
    font-family: 'Syne', sans-serif !important;
    font-weight: 700 !important;
    font-size: 15px !important;
    padding: 10px 24px !important;
    width: 100%;
    transition: all 0.2s !important;
}
.stButton > button:hover { transform: translateY(-1px); box-shadow: 0 8px 24px rgba(79,142,247,0.35) !important; }

/* Text areas */
.stTextArea textarea {
    font-family: 'Tajawal', sans-serif !important;
    font-size: 15px !important;
    direction: rtl;
    background: #111318 !important;
    border: 1px solid #1e222c !important;
    border-radius: 8px !important;
    color: #e8eaf0 !important;
}

/* Code / JSON output */
.stCode { border-radius: 10px !important; }

/* Select boxes */
.stSelectbox select { background: #111318 !important; color: #e8eaf0 !important; }

/* Headers */
h1 { font-family: 'Syne', sans-serif !important; font-weight: 800 !important; font-size: 28px !important; }
h2, h3 { font-family: 'Syne', sans-serif !important; font-weight: 700 !important; }

/* Status badge */
.status-online {
    display: inline-flex; align-items: center; gap: 6px;
    padding: 4px 12px;
    background: rgba(52,211,153,0.1);
    border: 1px solid rgba(52,211,153,0.3);
    border-radius: 20px;
    font-size: 12px; color: #34d399;
    font-family: 'IBM Plex Mono', monospace;
}
.dot { width: 7px; height: 7px; background: #34d399; border-radius: 50%; display: inline-block; }

.output-box {
    background: #0d1117;
    border: 1px solid #1e222c;
    border-radius: 10px;
    padding: 18px;
    font-family: 'IBM Plex Mono', monospace;
    font-size: 13px;
    line-height: 1.8;
    white-space: pre-wrap;
    color: #cdd9e5;
    min-height: 180px;
}
.arch-tag {
    display: inline-flex; align-items: center; gap: 4px;
    padding: 3px 10px; border-radius: 5px;
    font-size: 11px; font-family: 'IBM Plex Mono', monospace;
    margin: 2px;
}
</style>
""", unsafe_allow_html=True)


# ────────────────────────────────────────────────────────────────────────── #
#  Session state
# ────────────────────────────────────────────────────────────────────────── #
if "history" not in st.session_state:
    st.session_state.history = []
if "total_requests" not in st.session_state:
    st.session_state.total_requests = 0
if "total_latency_ms" not in st.session_state:
    st.session_state.total_latency_ms = 0
if "total_input_tokens" not in st.session_state:
    st.session_state.total_input_tokens = 0
if "total_output_tokens" not in st.session_state:
    st.session_state.total_output_tokens = 0


# ────────────────────────────────────────────────────────────────────────── #
#  Helpers
# ────────────────────────────────────────────────────────────────────────── #
@st.cache_resource
def get_tokenizer():
    return AutoTokenizer.from_pretrained(config.BASE_MODEL_ID)


def call_vllm(prompt: str, endpoint: str, model_id: str, temperature: float, max_tokens: int) -> dict:
    resp = requests.post(
        f"{endpoint}/v1/completions",
        json={"model": model_id, "prompt": prompt, "max_tokens": max_tokens, "temperature": temperature},
        timeout=120,
    )
    resp.raise_for_status()
    return resp.json()


def build_chat_prompt(messages, tokenizer) -> str:
    return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)


# ────────────────────────────────────────────────────────────────────────── #
#  Sidebar
# ────────────────────────────────────────────────────────────────────────── #
with st.sidebar:
    st.markdown("## ⚡ ArabicLLM")
    st.markdown('<span class="status-online"><span class="dot"></span> vLLM connected</span>', unsafe_allow_html=True)
    st.divider()

    st.markdown("### ⚙️ Server Config")
    vllm_endpoint = st.text_input("vLLM Endpoint", value=os.getenv("VLLM_ENDPOINT", "http://localhost:8000"))
    vllm_model_id = st.text_input("Model ID", value=os.getenv("VLLM_MODEL_ID", "news-lora"))

    st.markdown("### 🎛️ Generation")
    temperature = st.slider("Temperature", 0.0, 1.0, 0.3, 0.05)
    max_tokens  = st.slider("Max Tokens",   64, 2048, 1000, 64)

    st.divider()
    st.markdown("### 📊 Session Stats")
    st.metric("Requests",      st.session_state.total_requests)
    st.metric("Avg Latency",   f"{int(st.session_state.total_latency_ms / max(1, st.session_state.total_requests))} ms")
    st.metric("Input Tokens",  st.session_state.total_input_tokens)
    st.metric("Output Tokens", st.session_state.total_output_tokens)

    st.divider()
    st.caption("Base model: Qwen/Qwen2.5-1.5B-Instruct\nAdapter: news-lora (LoRA)")


# ────────────────────────────────────────────────────────────────────────── #
#  Main tabs
# ────────────────────────────────────────────────────────────────────────── #
st.markdown("# ⚡ Inference Studio")
st.caption("vLLM · LoRA · Arabic NLP")

tab_infer, tab_translate, tab_history, tab_arch = st.tabs([
    "📰 Details Extraction", "🔤 Translation", "🕓 History", "📐 Architecture"
])


# ══════════════════════════════════════════════════════════════════════════ #
#  TAB 1 — Details Extraction
# ══════════════════════════════════════════════════════════════════════════ #
with tab_infer:
    col_left, col_right = st.columns([1, 1], gap="large")

    with col_left:
        st.markdown("### Arabic Story")

        SAMPLE = (
            "ذكرت مجلة فوربس أن العائلة تلعب دورا محوريا في تشكيل علاقة الأفراد بالمال، "
            "حيث تتأثر هذه العلاقة بأنماط السلوك المالي المتوارثة عبر الأجيال. "
            "التقرير يستند إلى أبحاث الأستاذ الجامعي شاين إنيت حول الرفاه المالي."
        )

        c1, c2 = st.columns(2)
        if c1.button("📄 Load Sample"):
            st.session_state["story_text"] = SAMPLE
        if c2.button("🗑️ Clear"):
            st.session_state["story_text"] = ""

        story = st.text_area(
            "Story Content (Arabic)",
            value=st.session_state.get("story_text", SAMPLE),
            height=180,
            label_visibility="collapsed",
            key="story_input",
        )

        run = st.button("▶  Run Details Extraction", use_container_width=True)

    with col_right:
        st.markdown("### Extracted Details")
        output_placeholder = st.empty()
        metrics_placeholder = st.empty()

    if run and story.strip():
        with st.spinner("Calling vLLM…"):
            try:
                tok = get_tokenizer()
                messages = build_details_extraction_prompt(story)
                prompt   = build_chat_prompt(messages, tok)

                t0 = time.time()
                resp = call_vllm(prompt, vllm_endpoint, vllm_model_id, temperature, max_tokens)
                latency_ms = int((time.time() - t0) * 1000)

                raw_text = resp["choices"][0]["text"]
                in_tokens  = len(tok.encode(prompt))
                out_tokens = len(tok.encode(raw_text))

                # Update state
                st.session_state.total_requests     += 1
                st.session_state.total_latency_ms   += latency_ms
                st.session_state.total_input_tokens  += in_tokens
                st.session_state.total_output_tokens += out_tokens
                st.session_state.history.append({
                    "task": "details_extraction",
                    "story": story[:80] + "…",
                    "output": raw_text,
                    "latency_ms": latency_ms,
                    "in_tokens": in_tokens,
                    "out_tokens": out_tokens,
                })

                # Try to parse JSON
                try:
                    parsed = json.loads(raw_text.strip().lstrip("```json").rstrip("```"))
                    display = json.dumps(parsed, ensure_ascii=False, indent=2)
                except Exception:
                    display = raw_text

                with col_right:
                    output_placeholder.code(display, language="json")
                    metrics_placeholder.markdown(
                        f"⏱️ **{latency_ms} ms** &nbsp;·&nbsp; "
                        f"🔢 {in_tokens} in / {out_tokens} out tokens"
                    )

            except requests.exceptions.ConnectionError:
                st.error("❌ Cannot reach vLLM server — is it running on " + vllm_endpoint + " ?")
            except Exception as e:
                st.error(f"Error: {e}")


# ══════════════════════════════════════════════════════════════════════════ #
#  TAB 2 — Translation
# ══════════════════════════════════════════════════════════════════════════ #
with tab_translate:
    st.markdown("### Translation")
    col_a, col_b = st.columns([1, 1], gap="large")

    with col_a:
        src_lang = st.selectbox("Source Language", ["Arabic (ar)", "English (en)"], index=0)
        tgt_lang = st.selectbox("Target Language", ["English (en)", "Arabic (ar)", "French (fr)"], index=0)
        trans_story = st.text_area("Input Text", height=200, placeholder="أدخل النص هنا…")
        run_trans = st.button("▶  Translate", use_container_width=True)

    with col_b:
        st.markdown("### Translation Output")
        trans_output = st.empty()

    if run_trans and trans_story.strip():
        tgt_code = tgt_lang.split("(")[1].rstrip(")")
        with st.spinner("Translating…"):
            try:
                tok = get_tokenizer()
                messages = build_translation_prompt(trans_story, tgt_code)
                prompt   = build_chat_prompt(messages, tok)

                t0 = time.time()
                resp = call_vllm(prompt, vllm_endpoint, vllm_model_id, temperature, max_tokens)
                latency_ms = int((time.time() - t0) * 1000)

                raw_text = resp["choices"][0]["text"]

                try:
                    parsed  = json.loads(raw_text.strip().lstrip("```json").rstrip("```"))
                    display = json.dumps(parsed, ensure_ascii=False, indent=2)
                except Exception:
                    display = raw_text

                with col_b:
                    trans_output.code(display, language="json")
                    st.caption(f"⏱️ {latency_ms} ms · {len(tok.encode(prompt))} in / {len(tok.encode(raw_text))} out tokens")

            except requests.exceptions.ConnectionError:
                st.error("❌ Cannot reach vLLM server.")
            except Exception as e:
                st.error(f"Error: {e}")


# ══════════════════════════════════════════════════════════════════════════ #
#  TAB 3 — History
# ══════════════════════════════════════════════════════════════════════════ #
with tab_history:
    st.markdown("### Inference History")

    if not st.session_state.history:
        st.info("No inferences yet — run some tasks first.")
    else:
        for i, item in enumerate(reversed(st.session_state.history)):
            with st.expander(f"#{len(st.session_state.history) - i}  ·  {item['task']}  ·  {item['latency_ms']} ms"):
                st.markdown(f"**Story preview:** {item['story']}")
                st.markdown(f"**Tokens:** {item['in_tokens']} in / {item['out_tokens']} out")
                st.code(item["output"], language="json")


# ══════════════════════════════════════════════════════════════════════════ #
#  TAB 4 — Architecture
# ══════════════════════════════════════════════════════════════════════════ #
with tab_arch:
    st.markdown("### New Architecture")

    st.markdown("""
    ```
    ArabicLLM/
    ├── src/
    │   ├── inference/
    │   │   ├── vllm_client.py         # NEW – vLLM HTTP wrapper
    │   │   └── tokenizer_utils.py     # NEW – token counting utils
    │   ├── models/
    │   │   ├── base_model.py
    │   │   ├── finetunnig_model.py
    │   │   ├── openai_model.py
    │   │   ├── vllm_model.py          # NEW – vLLM model class
    │   │   └── initialize_model.py    # UPDATED (+vllm_model type)
    │   ├── load_testing/
    │   │   ├── locust_runner.py       # NEW
    │   │   └── token_analyzer.py      # NEW
    │   └── tasks/ ...
    ├── app/                           # NEW – Streamlit UI
    │   ├── streamlit_app.py
    │   └── components/
    │       ├── inference_tab.py
    │       └── loadtest_tab.py
    └── scripts/ ...
    ```
    """)

    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown("**Inference Pipeline**")
        st.markdown("Arabic text → Prompt Builder → Tokenizer → vLLM HTTP → LoRA Adapter → Pydantic Schema → JSON Output")
    with col2:
        st.markdown("**New Model Types**")
        st.code("model_type='vllm_model'", language="python")
        st.markdown("Uses `VLLMModel` class, calls `/v1/completions`")
    with col3:
        st.markdown("**Load Testing**")
        st.code("locust --headless -f locust.py\n  --host=http://localhost:8000\n  -u 20 -r 1 -t 60s", language="bash")

    st.divider()
    st.markdown("### Updated `.env` keys")
    st.code("""
VLLM_ENDPOINT=http://localhost:8000
VLLM_MODEL_ID=news-lora
VLLM_TEMPERATURE=0.3
VLLM_MAX_TOKENS=1000
OPENAI_ENDPOINT=...
OPENAI_MODEL_NAME=...
OPENAI_API_KEY=...
""", language="bash")