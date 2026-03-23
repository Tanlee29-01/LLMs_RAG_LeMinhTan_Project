# LLMs RAG Workspace

Dự án **LLMs RAG Workspace** là một hệ thống Trợ lý ảo AI ứng dụng kỹ thuật RAG (Retrieval-Augmented Generation) tiên tiến, cho phép người dùng hỏi đáp trực tiếp dựa trên các tài liệu cá nhân (PDF, TXT) hoặc thông tin thời gian thực từ Internet. Dự án được phát triển bởi LeMinhTan.

## 🌟 Tính năng nổi bật

* **Giao diện người dùng trực quan**: Được xây dựng bằng `Streamlit` với chế độ Dark Mode chuẩn Open WebUI, hỗ trợ tải lên nhiều tài liệu cùng lúc và hiệu ứng gõ chữ (Typewriter effect) chuyên nghiệp.
* **Backend API mạnh mẽ**: Sử dụng `FastAPI` và `LangServe` để xử lý các luồng dữ liệu, hỗ trợ playground để kiểm thử trực tiếp.
* **Hỗ trợ đa mô hình (Multi-LLMs)**: Hỗ trợ tích hợp đa dạng các mô hình LLM từ chạy cục bộ (Ollama, HuggingFace) đến các API thương mại (gpt-4o, gemini-1.5-pro, claude-3-5-sonnet). Mô hình mặc định là `llama3.1:8b`.
* **Xử lý tài liệu chuyên sâu**: Tích hợp `Docling` và `pymupdf4llm` để chuyển đổi PDF thành Markdown, kết hợp cùng `MarkdownHeaderTextSplitter` để giữ nguyên cấu trúc ngữ nghĩa của tài liệu khi cắt nhỏ.
* **Cơ sở dữ liệu Vector (Vector DB)**: Lưu trữ và truy xuất ngữ nghĩa bằng `Chroma` hoặc `FAISS`, sử dụng mô hình nhúng mặc định `nomic-embed-text` qua giao thức của Ollama.
* **RAG Lai tự động điều hướng (Hybrid/Routing RAG)**: Tự động phân tích ý định trong câu hỏi:
  * Trích xuất tài liệu: Trả lời dựa trên các file PDF/TXT người dùng tải lên.
  * Tìm kiếm Web: Tự động kích hoạt tìm kiếm thời gian thực (qua DuckDuckGo) nếu phát hiện các từ khóa thời sự như "hôm nay", "thời tiết", "giá cả"....

## 📁 Cấu trúc thư mục

```text
├── .streamlit/
│   └── config.toml          # Cấu hình giao diện Streamlit (Dark mode)
├── data/
│   └── generative_ai/       # Nơi lưu trữ tự động các file tài liệu người dùng tải lên
├── docs/                    # Tài liệu hướng dẫn (MkDocs) & Lộ trình RAG
├── models/
│   └── llm_model.py         # Module quản lý việc tải và giao tiếp với các mô hình LLM
├── src/
│   ├── app.py               # Chứa Backend FastAPI chính
│   └── rag/
│       ├── file_loader.py   # Code trích xuất và cắt nhỏ văn bản (Docling, PyMuPDF)
│       ├── main.py          # Khởi tạo RAG Chain
│       ├── offline_rag.py   # Lõi xử lý logic RAG, Prompting và Routing
│       ├── vectorstrore.py  # Quản lý Vector Database (Chroma/FAISS)
│       └── web_search.py    # Module tìm kiếm DuckDuckGo (HTML/DDGS)
├── app_ui.py                # File chạy giao diện Frontend Streamlit
├── Makefile                 # Các lệnh setup tự động tiện ích
└── requirements.txt         # Danh sách các thư viện cần thiết

⚙️ Cài đặt
1. Clone dự án và thiết lập môi trường
Bạn nên sử dụng môi trường ảo (Virtual Environment). Bạn có thể tự động tạo môi trường bằng lệnh Make:

Bash
make create_environment
2. Cài đặt các thư viện phụ thuộc
Chạy lệnh Make để cài đặt requirements.txt và nâng cấp pip:

Bash
make requirements
(Lưu ý: Bạn cũng cần cài đặt và khởi chạy ứng dụng Ollama trên máy tính của mình với các model llama3.1:8b và nomic-embed-text để sử dụng mặc định).

🚀 Khởi chạy hệ thống
Hệ thống hoạt động dựa trên cơ chế Client-Server, do đó bạn cần khởi chạy Backend trước khi mở Frontend.

Bước 1: Chạy Backend (FastAPI)
Mở terminal và chạy lệnh sau để khởi động API Server ở cổng 5051:

Bash
uvicorn src.app:app --host 127.0.0.1 --port 5051 --reload
API Server sẽ cung cấp các endpoint như tải file (/upload) và truy vấn AI (/generative_ai).

Bước 2: Chạy Frontend (Streamlit)
Mở một terminal mới và chạy file giao diện:

Bash
streamlit run app_ui.py
Trình duyệt sẽ tự động mở trang web tại địa chỉ http://localhost:8501.

💡 Hướng dẫn sử dụng
1. Tại thanh bên (Sidebar), kéo thả các file PDF hoặc TXT chứa kiến thức của bạn.

2. Bấm nút "🚀 Nhúng tài liệu (Embed)" và chờ hệ thống phân tích, lưu trữ vào Vector DB.

3. Ở khung bên dưới cùng, chọn LLM mong muốn và tùy chỉnh thông số "Độ sáng tạo" (Temperature).

4. Bắt đầu chat với trợ lý ảo Minh Tân để hỏi đáp các kiến thức có trong tài liệu hoặc các tin tức mới nhất từ internet!
