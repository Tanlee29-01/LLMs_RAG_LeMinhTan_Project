import os
import tempfile
import inspect

import requests
import streamlit as st

from models.llm_model import load_model_and_tokenizer, build_hf_llm, DEFAULT_MODEL_NAME
from src.rag.file_loader import Loader
from src.rag.vectorstrore import VectorDB
from src.rag.offline_rag import Offline_RAG


# =========================
# Page config
# =========================
st.set_page_config(page_title="RAG Chatbot - LeMinhTan", layout="wide")
st.title("🤖 Chat với tài liệu PDF của bạn")
st.markdown(
    """
    <style>
    :root {
        --bg-dark: #1e1e24;
        --bg-darker: #2b2d35;
        --border-gray: rgb(107, 114, 128, 0.2);
        --text-gray: rgb(209, 213, 219);
        --text-muted: rgb(156, 163, 175);
    }

    .mode-container {
        max-width: 100%;
        background-color: var(--bg-dark);
        border-radius: 16px;
        border: 1px solid var(--border-gray);
        padding: 24px;
        margin: 16px 0;
        box-shadow: 0 20px 25px -5px rgba(0, 0, 0, 0.3);
    }

    .mode-header {
        display: flex;
        flex-direction: column;
        gap: 12px;
        margin-bottom: 20px;
    }

    .mode-badge {
        display: inline-block;
        padding: 6px 12px;
        border-radius: 999px;
        font-size: 0.75rem;
        font-weight: 600;
        letter-spacing: 0.05em;
        text-transform: uppercase;
        border: 1px solid;
        width: fit-content;
    }

    .mode-badge.extraction {
        background-color: rgba(99, 102, 241, 0.1);
        color: #a5b4fc;
        border-color: rgba(99, 102, 241, 0.2);
    }

    .mode-badge.qa {
        background-color: rgba(251, 146, 60, 0.1);
        color: #fed7aa;
        border-color: rgba(251, 146, 60, 0.2);
    }

    .mode-badge.summary {
        background-color: rgba(96, 165, 250, 0.1);
        color: #bfdbfe;
        border-color: rgba(96, 165, 250, 0.2);
    }

    .mode-title {
        font-size: 1.125rem;
        font-weight: 600;
        color: #f3f4f6;
        letter-spacing: -0.01em;
        margin: 0;
    }

    .mode-description {
        font-size: 0.875rem;
        color: var(--text-muted);
        margin: 0;
    }

    .input-wrapper {
        position: relative;
    }

    .prompt-textarea {
        width: 100%;
        background-color: var(--bg-darker);
        border: 1px solid var(--border-gray);
        border-radius: 12px;
        padding: 16px;
        min-height: 120px;
        color: #e5e7eb;
        font-family: inherit;
        font-size: 0.95rem;
        resize: none;
        transition: all 0.2s;
        line-height: 1.5;
    }

    .prompt-textarea::placeholder {
        color: #9ca3af;
    }

    .prompt-textarea:focus {
        outline: none;
        border-color: #6366f1;
        box-shadow: 0 0 0 3px rgba(99, 102, 241, 0.1);
    }

    .submit-button {
        margin-top: 16px;
        width: 100%;
        background-color: #6366f1;
        color: white;
        font-weight: 500;
        padding: 12px 20px;
        border-radius: 12px;
        border: none;
        cursor: pointer;
        transition: all 0.2s;
        display: flex;
        justify-content: center;
        align-items: center;
        gap: 8px;
        font-size: 0.95rem;
        box-shadow: 0 10px 15px -3px rgba(99, 102, 241, 0.2);
    }

    .submit-button:hover {
        background-color: #4f46e5;
        box-shadow: 0 15px 20px -5px rgba(99, 102, 241, 0.3);
    }

    .submit-button:active {
        transform: scale(0.98);
    }

    /* Streamlit-specific overrides */
    [data-testid="stTextArea"] textarea {
        background-color: var(--bg-darker) !important;
        border: 1px solid var(--border-gray) !important;
        border-radius: 12px !important;
        color: #e5e7eb !important;
    }

    [data-testid="stTextArea"] textarea:focus {
        border-color: #6366f1 !important;
        box-shadow: 0 0 0 3px rgba(99, 102, 241, 0.1) !important;
    }

    [data-testid="stForm"] {
        border: none !important;
        padding: 0 !important;
    }

    .stButton > button {
        background-color: #6366f1 !important;
        color: white !important;
        font-weight: 500 !important;
        width: 100% !important;
        border-radius: 12px !important;
        font-size: 0.95rem !important;
        box-shadow: 0 10px 15px -3px rgba(99, 102, 241, 0.2) !important;
    }

    .stButton > button:hover {
        background-color: #4f46e5 !important;
        box-shadow: 0 15px 20px -5px rgba(99, 102, 241, 0.3) !important;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


@st.cache_data(ttl=10)
def get_ollama_runtime_status(model_name: str) -> dict:
    base_url = os.getenv("OLLAMA_BASE_URL", "http://127.0.0.1:11434").rstrip("/")
    is_ollama_model = ":" in model_name and "/" not in model_name
    status = {
        "is_ollama_model": is_ollama_model,
        "base_url": base_url,
        "server_up": False,
        "model_exists": False,
        "error": None,
    }
    if not is_ollama_model:
        return status

    try:
        response = requests.get(f"{base_url}/api/tags", timeout=2)
        response.raise_for_status()
        payload = response.json()
        models = payload.get("models", []) if isinstance(payload, dict) else []
        model_names = {
            item.get("name")
            for item in models
            if isinstance(item, dict) and isinstance(item.get("name"), str)
        }
        status["server_up"] = True
        status["model_exists"] = model_name in model_names
    except requests.RequestException as exc:
        status["error"] = str(exc)

    return status


# =========================
# LLM (cached)
# =========================
@st.cache_resource
def load_llms():
    model, tokenizer = load_model_and_tokenizer()
    active_model_name = getattr(model, "_copilot_model_name", DEFAULT_MODEL_NAME)
    
    # BẮT BUỘC: do_sample=False và repetition_penalty=1.0 để khóa mõm ảo giác
    llm_extraction = build_hf_llm(
        model, tokenizer,
        max_new_tokens=256,
        repetition_penalty=1.0, 
        no_repeat_ngram_size=0,
        do_sample=False
    )
    llm_qa_summary = build_hf_llm(
        model, tokenizer,
        max_new_tokens=300,
        repetition_penalty=1.0,
        no_repeat_ngram_size=0,
        do_sample=False
    )
    llm_summary_document = build_hf_llm(
        model, tokenizer,
        max_new_tokens=350,
        repetition_penalty=1.0,
        no_repeat_ngram_size=0,
        do_sample=False
    )
    return llm_extraction, llm_qa_summary, llm_summary_document, active_model_name

llm_extraction, llm_qa_summary, llm_summary_document, active_model_name = load_llms()


# =========================
# Sidebar: data processing
# =========================
with st.sidebar:
    st.header("Cấu hình dữ liệu")
    st.caption(f"Model yêu cầu: {DEFAULT_MODEL_NAME}")
    st.caption(f"Model đang dùng: {active_model_name}")

    ollama_status = get_ollama_runtime_status(active_model_name)
    if ollama_status["is_ollama_model"]:
        st.caption(f"Ollama URL: {ollama_status['base_url']}")
        if not ollama_status["server_up"]:
            st.error("Ollama server chưa chạy. Mở terminal và chạy: ollama serve")
        elif not ollama_status["model_exists"]:
            st.warning(
                f"Ollama đang chạy nhưng chưa có model {active_model_name}. Chạy: ollama pull {active_model_name}"
            )
        else:
            st.success(f"Ollama sẵn sàng với model {active_model_name}")

    if active_model_name != DEFAULT_MODEL_NAME:
        st.warning("7B chưa tải xong, ứng dụng đang tự dùng model fallback để tránh treo lúc khởi động.")
    uploaded_files = st.file_uploader(
        "Upload tài liệu PDF",
        type=["pdf"],
        accept_multiple_files=True
    )
    process_button = st.button("Xử lý tài liệu", type="primary")
    clear_chat_button = st.button("Xóa lịch sử chat")


# =========================
# Session state
# =========================
if "vector_db" not in st.session_state:
    st.session_state.vector_db = None

if "messages_extraction" not in st.session_state:
    st.session_state.messages_extraction = []

if "messages_qa" not in st.session_state:
    st.session_state.messages_qa = []

if "messages_summary" not in st.session_state:
    st.session_state.messages_summary = []

if "mode_supported_checked" not in st.session_state:
    st.session_state.mode_supported_checked = False

if "supports_mode_param" not in st.session_state:
    st.session_state.supports_mode_param = False


# =========================
# Optional: clear messages
# =========================
if clear_chat_button:
    st.session_state.messages_extraction = []
    st.session_state.messages_qa = []
    st.session_state.messages_summary = []
    st.success("Đã xóa lịch sử chat.")


# =========================
# Process uploaded docs
# =========================
if process_button and uploaded_files:
    with st.spinner("Đang đọc và phân tích tài liệu..."):
        with tempfile.TemporaryDirectory() as temp_dir:
            file_paths = []
            for uploaded_file in uploaded_files:
                file_path = os.path.join(temp_dir, uploaded_file.name)
                with open(file_path, "wb") as f:
                    f.write(uploaded_file.getbuffer())
                file_paths.append(file_path)

            loader = Loader(
                file_type="pdf",
                strategy="standard",
                split_kwargs={"chunk_size": 1200, "chunk_overlap": 150}
            )
            workers = 1 if os.name == "nt" else 3
            doc_split = loader.load(file_paths, workers=workers)

            # Cách 2: lưu vector_db, không lưu rag_chain
            st.session_state.vector_db = VectorDB(documents=doc_split)

            # Reset lịch sử để tránh lẫn với dữ liệu cũ
            st.session_state.messages_extraction = []
            st.session_state.messages_qa = []
            st.session_state.messages_summary = []

            st.success("Đã xử lý xong! Bạn có thể bắt đầu đặt câu hỏi theo từng mode.")


# =========================
# Check once: get_chain supports mode param?
# =========================
if not st.session_state.mode_supported_checked:
    try:
        sig = inspect.signature(Offline_RAG.get_chain)
        st.session_state.supports_mode_param = "mode" in sig.parameters
    except Exception:
        st.session_state.supports_mode_param = False
    st.session_state.mode_supported_checked = True


def render_chat_history(messages):
    for message in messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

            # Chi render citation cho assistant va khi co du lieu sources
            if message["role"] == "assistant":
                sources = message.get("sources", [])
                if sources:
                    with st.expander(f"Nguồn tham khảo ({len(sources)})", expanded=False):
                        for i, c in enumerate(sources, start=1):
                            page_str = (
                                f"Trang {c['page'] + 1}"
                                if c.get("page") is not None
                                else "Không rõ trang"
                            )
                            head_str = f" | {c.get('headers', '')}" if c.get("headers") else ""
                            st.markdown(f"**[{i}] {c.get('file', 'Không rõ nguồn')} - {page_str}{head_str}**")
                            snippet = c.get("snippet", "")
                            if snippet:
                                st.caption(snippet + "...")



MODE_UI = {
    "extraction": {
        "badge": "Extraction",
        "title": "Trích xuất thông tin chính xác từ tài liệu",
        "description": "Phù hợp với định nghĩa, điều khoản, số liệu, hoặc câu hỏi cần trích nguyên văn ngắn gọn.",
        "placeholder": "Ví dụ: Điều khoản X quy định gì? Định nghĩa Y là gì?",
        "submit_label": "Gửi câu hỏi Extraction",
        "k": 4,
    },
    "qa": {
        "badge": "QA",
        "title": "Hỏi đáp về nội dung tài liệu",
        "description": "Phù hợp với câu hỏi cần diễn giải, so sánh, hoặc liên kết nhiều ý trong tài liệu.",
        "placeholder": "Ví dụ: So sánh A và B. Vai trò của X là gì?",
        "submit_label": "Gửi câu hỏi QA",
        "k": 4,
    },
    "summary": {
        "badge": "Summary",
        "title": "Tóm tắt tài liệu",
        "description": "Phù hợp với yêu cầu tóm tắt, tổng hợp nội dung chính hoặc các ý chính của tài liệu.",
        "placeholder": "Ví dụ: Tóm tắt tài liệu trong 5 ý chính. Nội dung chính của chương này là gì?",
        "submit_label": "Gửi yêu cầu Tóm tắt",
        "k": 6,
    },
}


def render_mode_header(mode: str):
    config = MODE_UI[mode]
    st.markdown(
        f"""
        <div class="mode-container">
            <div class="mode-header">
                <div class="mode-badge {mode}">{config['badge']}</div>
                <h3 class="mode-title">{config['title']}</h3>
                <p class="mode-description">{config['description']}</p>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_prompt_form(mode: str, form_key: str, input_key: str):
    config = MODE_UI[mode]
    has_documents = st.session_state.vector_db is not None

    if not has_documents:
        st.info("Hãy upload PDF và bấm 'Xử lý tài liệu' trước khi bắt đầu chat.")

    with st.form(form_key, clear_on_submit=True):
        st.markdown(
            '<div class="input-wrapper">', unsafe_allow_html=True
        )
        prompt = st.text_area(
            "label",
            key=input_key,
            height=120,
            placeholder=config["placeholder"],
            disabled=not has_documents,
            label_visibility="collapsed",
        )
        st.markdown(
            '</div>', unsafe_allow_html=True
        )
        
        col1, col2 = st.columns([3, 1]) if not has_documents else (None, None)
        
        submitted = st.form_submit_button(
            config["submit_label"],
            type="primary",
            use_container_width=True,
            disabled=not has_documents,
        )

    if submitted:
        cleaned_prompt = prompt.strip()
        if not cleaned_prompt:
            st.warning("Nhập câu hỏi trước khi gửi.")
            return
        ask_with_mode(
            prompt=cleaned_prompt,
            mode=mode,
            messages_key=f"messages_{mode}",
            k=config["k"],
        )


def ask_with_mode(prompt, mode, messages_key, k):
    if st.session_state.vector_db is None:
        st.error("Vui lòng upload và nhấn 'Xử lý tài liệu' trước khi hỏi!")
        return

    st.session_state[messages_key].append({"role": "user", "content": prompt})
    st.rerun()  # Rerun để render chat history với user message
    
    # Đoạn code này không bao giờ được chạy do st.rerun() ở trên


def process_llm_response(prompt, mode, messages_key, k):
    """Process LLM response without early rerun"""
    if mode == "extraction":
        rag_builder = Offline_RAG(llm_extraction)
        retriever = st.session_state.vector_db.get_retriever(search_kwargs={"k": k})
        chain = rag_builder.get_chain(retriever=retriever, mode=mode)
    elif mode == "summary":
        rag_builder = Offline_RAG(llm_summary_document)
        summary_docs = st.session_state.vector_db.get_summary_documents(max_docs=8, max_chars=7000)
        chain = rag_builder.get_chain_from_documents(documents=summary_docs, mode="summary")
    else:  # qa
        rag_builder = Offline_RAG(llm_qa_summary)
        retriever = st.session_state.vector_db.get_retriever(search_kwargs={"k": k})
        chain = rag_builder.get_chain(retriever=retriever, mode=mode)

    result = chain.invoke(prompt)

    if isinstance(result, dict):
        source_docs = result.get("context", []) or []
        if not source_docs:
            answer = "Không tìm thấy thông tin này trong tài liệu."
        else:
            answer = result.get("answer", "")
    else:
        answer = str(result)
        source_docs = []

    # Làm trích dẫn (citations)
    citations = []
    seen = set()
    for doc in source_docs:
        meta = doc.metadata or {}
        source = meta.get("source", "")
        file_name = os.path.basename(source) if source else "Không rõ nguồn"
        page = meta.get("page", None)
        h1 = meta.get("Header 1", "")
        h2 = meta.get("Header 2", "")
        headers = " > ".join([x for x in [h1, h2] if x])

        key = (file_name, page, headers)
        if key in seen:
            continue
        seen.add(key)

        snippet = (doc.page_content or "").replace("\n", " ").strip()[:200]
        citations.append({"file": file_name, "page": page, "headers": headers, "snippet": snippet})

    return answer, citations


def render_prompt_form_with_response(mode: str, form_key: str, input_key: str):
    """Render form and handle response with spinner"""
    config = MODE_UI[mode]
    has_documents = st.session_state.vector_db is not None
    messages_key = f"messages_{mode}"
    
    # Check if we're waiting for response (last message is from user)
    is_waiting_for_response = (
        len(st.session_state[messages_key]) > 0 and 
        st.session_state[messages_key][-1]["role"] == "user"
    )
    
    # If waiting for response, show spinner and process
    if is_waiting_for_response:
        with st.spinner("Đang đọc tài liệu và soạn câu trả lời..."):
            answer, citations = process_llm_response(
                prompt=st.session_state[messages_key][-1]["content"],
                mode=mode,
                messages_key=messages_key,
                k=config["k"],
            )
            st.session_state[messages_key].append({
                "role": "assistant",
                "content": answer,
                "sources": citations
            })
            st.rerun()
    else:
        # Show input form (only when NOT waiting for response)
        if not has_documents:
            st.info("Hãy upload PDF và bấm 'Xử lý tài liệu' trước khi bắt đầu chat.")

        with st.form(form_key, clear_on_submit=True):
            st.markdown(
                '<div class="input-wrapper">', unsafe_allow_html=True
            )
            prompt = st.text_area(
                "label",
                key=input_key,
                height=120,
                placeholder=config["placeholder"],
                disabled=not has_documents,
                label_visibility="collapsed",
            )
            st.markdown(
                '</div>', unsafe_allow_html=True
            )
            
            submitted = st.form_submit_button(
                config["submit_label"],
                type="primary",
                use_container_width=True,
                disabled=not has_documents,
            )

        if submitted:
            cleaned_prompt = prompt.strip()
            if not cleaned_prompt:
                st.warning("Nhập câu hỏi trước khi gửi.")
                return
            ask_with_mode(
                prompt=cleaned_prompt,
                mode=mode,
                messages_key=messages_key,
                k=config["k"],
            )


# =========================
# UI: 3 tabs
# =========================
tab_extraction, tab_qa, tab_summary = st.tabs(["Extraction", "QA", "Summary"])

with tab_extraction:
    render_mode_header("extraction")
    render_chat_history(st.session_state.messages_extraction)
    render_prompt_form_with_response(
        mode="extraction",
        form_key="prompt_form_extraction",
        input_key="prompt_input_extraction",
    )

with tab_qa:
    render_mode_header("qa")
    render_chat_history(st.session_state.messages_qa)
    render_prompt_form_with_response(
        mode="qa",
        form_key="prompt_form_qa",
        input_key="prompt_input_qa",
    )

with tab_summary:
    render_mode_header("summary")
    render_chat_history(st.session_state.messages_summary)
    render_prompt_form_with_response(
        mode="summary",
        form_key="prompt_form_summary",
        input_key="prompt_input_summary",
    )

if not st.session_state.supports_mode_param:
    st.info("Offline_RAG.get_chain hiện chưa có tham số mode. App vẫn chạy, nhưng cả 2 tab sẽ dùng cùng một prompt cho đến khi bạn cập nhật Offline_RAG.")