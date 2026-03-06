import streamlit as st
import os
import sys
import tempfile

from models.llm_model import get_hf_llm
from src.rag.file_loader import Loader
from src.rag.vectorstrore import VectorDB
from src.rag.offline_rag import Offline_RAG




#Copy -> streamlit run streamlit_app.py -> past to terminal

# Cấu hình trang
st.set_page_config(page_title="RAG Chatbot - LeMinhTan", layout="wide")
st.title("🤖 Chat với tài liệu PDF của bạn")

# Khởi tạo LLM (Sử dụng cache để không load lại model mỗi lần nhấn nút)
@st.cache_resource
def load_llm():
    return get_hf_llm(model_name="microsoft/Phi-3.5-mini-instruct", temperature=0.0)

llm = load_llm()

# Sidebar để upload file
with st.sidebar:
    st.header("Cấu hình dữ liệu")
    uploaded_files = st.file_uploader("Upload tài liệu PDF", type=["pdf"], accept_multiple_files=True)
    process_button = st.button("Xử lý tài liệu")

# Khởi tạo session state để lưu trữ chain

if "rag_chain" not in st.session_state:
    st.session_state.rag_chain = None
if "messages" not in st.session_state:
    st.session_state.messages = []

# Xử lý khi nhấn nút "Xử lý tài liệu"
if process_button and uploaded_files:
    with st.spinner("Đang đọc và phân tích tài liệu..."):
        # Tạo thư mục tạm để lưu file upload
        with tempfile.TemporaryDirectory() as temp_dir:
            file_paths = []
            for uploaded_file in uploaded_files:
                file_path = os.path.join(temp_dir, uploaded_file.name)
                with open(file_path, "wb") as f:
                    f.write(uploaded_file.getbuffer())
                file_paths.append(file_path)
            
            # Sử dụng Loader của bạn để xử lý file
            loader = Loader(file_type="pdf", split_kwargs={"chunk_size": 1000, "chunk_overlap": 200})
            doc_split = loader.load(file_paths, workers=2)
            
            # Tạo Vector Database
            vector_db = VectorDB(documents=doc_split)
            retriever = vector_db.get_retriever(search_kwargs={"k": 5})
            
            # Tạo RAG Chain
            st.session_state.rag_chain = Offline_RAG(llm).get_chain(retriever)
            st.success("Đã xử lý xong! Bạn có thể bắt đầu đặt câu hỏi.")

# Giao diện Chat
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

if prompt := st.chat_input("Hỏi gì đó về tài liệu..."):
    if st.session_state.rag_chain is None:
        st.error("Vui lòng upload và nhấn 'Xử lý tài liệu' trước khi hỏi!")
    else:
        # Hiển thị câu hỏi người dùng
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        # AI trả lời
        with st.chat_message("assistant"):
            with st.spinner("Đang suy nghĩ..."):
                response = st.session_state.rag_chain.invoke(prompt)
                st.markdown(response)
        
        st.session_state.messages.append({"role": "assistant", "content": response})