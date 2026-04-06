import streamlit as st
import time
import os
import tempfile
import requests
import uuid
from langchain_ollama import ChatOllama


# =====================================================================
# 2. CẤU HÌNH GIAO DIỆN & CSS (Dark Mode chuẩn Open WebUI)
# =====================================================================
st.set_page_config(
    page_title="RAG Workspace",
    page_icon="✨",
    layout="wide",
    initial_sidebar_state="expanded"
)

custom_css = """
<style>
    /* Tùy chỉnh thanh cuộn cho mượt mà */
    ::-webkit-scrollbar { width: 6px; height: 6px; }
    ::-webkit-scrollbar-track { background: transparent; }
    ::-webkit-scrollbar-thumb { background: #444; border-radius: 3px; }
    ::-webkit-scrollbar-thumb:hover { background: #666; }
    
    /* Chỉnh padding ô chat input */
    .stChatFloatingInputContainer { padding-bottom: 20px; }
</style>
"""
st.markdown(custom_css,unsafe_allow_html=True)

# =====================================================================
# 3. QUẢN LÝ TRẠNG THÁI (SESSION STATE)
# =====================================================================
if "messages" not in st.session_state:
    st.session_state.messages = []
if "vectorstore_ready" not in st.session_state:
    st.session_state.vectorstore_ready = False
if "chat_id" not in st.session_state:
    st.session_state.chat_id = uuid.uuid4().hex

# =====================================================================
# 4. CÁC HÀM XỬ LÝ LÕI 
# =====================================================================
def process_uploaded_files(uploaded_files):
    """Upload files to backend API"""
    try:
        # Chuẩn bị dữ liệu gửi
        files_to_upload = [
            ("files", (file.name, file.getbuffer(), "application/pdf")) 
            for file in uploaded_files
        ]
        
        response = requests.post(
            'http://127.0.0.1:5051/upload',
            files=files_to_upload,
            data={'chat_id': st.session_state.chat_id},
            timeout=120
        )
        
        if response.status_code == 200:
            data = response.json()
            uploaded_count = len(data.get('uploaded_files', []))
            chain_status = data.get('chain_status', {})
            st.session_state.vectorstore_ready = True
            return uploaded_count, chain_status
        else:
            return 0, {"status": "error", "message": f"Server error: {response.status_code}"}
    except requests.exceptions.ConnectionError:
        return 0, {"status": "error", "message": "Cannot connect to server on port 5051. Please start: uvicorn src.app:app --host 127.0.0.1 --port 5051"}
    except Exception as e:
        return 0, {"status": "error", "message": str(e)}

def get_rag_response(user_query, model_name="llama3.1:8b", temperature=0.1):
    """Hàm gọi API backend RAG thật"""
    try:
        # Gọi FastAPI endpoint
        response = requests.post(
            'http://127.0.0.1:5051/generative_ai',
            json={'question': user_query, 'chat_id': st.session_state.chat_id},
            timeout=90
        )
        
        if response.status_code == 200:
            data = response.json()
            answer = data.get('answer', 'Không có câu trả lời')
            source = data.get('source', 'model_knowledge')
            
            # Hiển thị badge nguồn
            if source == 'web':
                answer = f"🌐 **[Từ Web]** {answer}"
            elif source == 'documents':
                answer = f"📄 **[Từ Tài liệu]** {answer}"
            else:
                answer = f"🤖 **[Từ Model]** {answer}"
            
            return answer
        else:
            return f"❌ Lỗi từ server: {response.status_code}"
    
    except requests.exceptions.Timeout:
        return "⏱️ Hết thời gian chờ. Vui lòng đảm bảo server RAG đang chạy trên port 5051."
    except requests.exceptions.ConnectionError:
        return "🔌 Không thể kết nối đến server. Vui lòng chạy: `uvicorn src.app:app --host 127.0.0.1 --port 5051`"
    except Exception as e:
        return f"❌ Lỗi: {str(e)}"

# =====================================================================
# 5. GIAO DIỆN THANH BÊN (SIDEBAR)
# =====================================================================
with st.sidebar:
    st.title("🗂️ Workspace RAG")
    
    st.markdown("### 1. Dữ liệu (Knowledge Base)")
    uploaded_files = st.file_uploader("Kéo thả PDF / TXT vào đây", accept_multiple_files=True)
    
    if uploaded_files:
        if st.button("🚀 Nhúng tài liệu (Embed)", use_container_width=True, type="primary"):
            with st.spinner("Đang tải lên và xử lý tài liệu..."):
                uploaded_count, result = process_uploaded_files(uploaded_files)
                if result.get('status') == 'success':
                    st.success(f"✅ Đã xử lý {uploaded_count} tài liệu!")
                else:
                    st.error(f"❌ Lỗi: {result.get('message', 'Unknown error')}")
    
    st.divider()
    
    st.markdown("### 2. Cài đặt Mô hình")
    # ĐÃ THÊM LLAMA 3.1 8B THEO YÊU CẦU
    selected_model = st.selectbox(
        "Lựa chọn LLM", 
        ["llama3.1:8b", "gpt-4o", "gemini-1.5-pro", "claude-3-5-sonnet"]
    )
    temperature = st.slider("Temperature (Độ sáng tạo)", 0.0, 1.0, 0.1, 0.1)
    
    st.divider()
    
    if st.button("🗑️ Xóa lịch sử chat", use_container_width=True):
        st.session_state.messages = []
        st.rerun()

# =====================================================================
# 6. GIAO DIỆN CHAT CHÍNH (MAIN CHAT INTERFACE)
# =====================================================================
st.markdown("<h2 style='text-align: center; margin-bottom: 2rem;'>Trợ lý ảo AI - Minh Tân</h2>", unsafe_allow_html=True)

# Hiển thị cảnh báo nếu chưa nhúng tài liệu
if not st.session_state.vectorstore_ready:
    st.info("💡 Vui lòng tải lên tài liệu và bấm 'Nhúng tài liệu' ở Sidebar trước khi chat để RAG có thể hoạt động.")

# In ra lịch sử tin nhắn
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Ô nhập liệu (chỉ cho phép nhập nếu đã nhúng xong tài liệu, hoặc bạn có thể bỏ 'disabled' đi nếu muốn chat không cần tài liệu)
if prompt := st.chat_input("Hỏi bất cứ điều gì về tài liệu của bạn..."):
    
    # Thêm câu hỏi của User vào UI và Database
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # Khởi tạo khung chứa câu trả lời của AI
    with st.chat_message("assistant"):
        message_placeholder = st.empty()
        
        # Trạng thái xoay xoay khi AI đang load / truy xuất VectorDB
        with st.spinner(f"Đang phân tích bằng {selected_model}..."):
            # Lấy kết quả từ hàm RAG
            response = get_rag_response(prompt, selected_model, temperature)
            
        # Hiệu ứng gõ chữ (Typewriter effect) cho xịn
        full_response = ""
        for chunk in response.split(" "):
            full_response += chunk + " "
            time.sleep(0.04) # Tốc độ hiện chữ
            message_placeholder.markdown(full_response + "▌") # Con trỏ nhấp nháy
            
        # Kết thúc hiệu ứng gõ chữ
        message_placeholder.markdown(full_response)
        
    # Lưu câu trả lời của AI vào bộ nhớ
    st.session_state.messages.append({"role": "assistant", "content": full_response})