***
# 🚀 LLMs RAG Project - Trợ lý AI phân tích tài liệu

Dự án này là một hệ thống **RAG (Retrieval-Augmented Generation)** cho phép người dùng trò chuyện, đặt câu hỏi và khai thác thông tin từ các tài liệu cá nhân (PDF, TXT,...) thông qua giao diện **Telegram Bot** tiện lợi. Hệ thống cũng cung cấp sẵn giao diện Web bằng Streamlit.

---

## ✨ Tính năng nổi bật

- **Giao diện Telegram Bot:** Tương tác mượt mà, tiện lợi ngay trên điện thoại và máy tính.
- **Hỗ trợ đa mô hình (Multi-Models):** Có thể linh hoạt chuyển đổi giữa các LLM mạnh mẽ như Llama 3.1 8B, GPT-4o,...
- **Xử lý tài liệu thông minh:** Kéo thả trực tiếp file PDF, TXT vào khung chat để bot nhúng vào Vector Database và học nội dung.
- **Tuỳ chỉnh tham số:** Hỗ trợ lệnh `/settings` trên Telegram để điều chỉnh độ sáng tạo (Temperature).
- **Backend độc lập:** Sử dụng FastAPI làm lõi trung tâm, dễ dàng mở rộng và tích hợp với nhiều nền tảng khác.

---

## 📂 Cấu trúc thư mục chính

```text
LLMs_RAG_LeMinhTan_Project/
│
├── src/                # Chứa mã nguồn Backend (FastAPI) và Logic RAG
│   ├── app.py          # File khởi chạy API Server
│   └── rag/            # Các module xử lý VectorDB, Loader, Web Search,...
│
├── models/             # Chứa mã nguồn cấu hình LLM
├── telegram_bot.py     # Giao diện Chatbot trên Telegram
├── app_ui.py           # Giao diện Web (Streamlit) - Phương án dự phòng
├── requirements.txt    # Danh sách các thư viện Python cần thiết
├── .env.example        # Mẫu file chứa các biến môi trường (Cần tạo file .env từ đây)
└── README.md           # Tài liệu hướng dẫn sử dụng
```

---

## ⚙️ Hướng dẫn cài đặt

**1. Clone dự án và di chuyển vào thư mục:**
```bash
git clone <đường-link-repo-của-bạn>
cd LLMs_RAG_LeMinhTan_Project
```

**2. Tạo và kích hoạt môi trường ảo (Khuyến nghị):**
* **Windows:** `python -m venv .venv` sau đó `.\.venv\Scripts\activate`
* **Mac/Linux:** `python3 -m venv .venv` sau đó `source .venv/bin/activate`

**3. Cài đặt các thư viện cần thiết:**
```bash
pip install -r requirements.txt
```

**4. Cấu hình biến môi trường:**
* Tạo một file tên là `.env` ở thư mục gốc của dự án.
* Thêm Token của Telegram Bot và các API Key (như OpenAI Key, Groq Key...) vào file `.env`:
```env
TELEGRAM_BOT_TOKEN="123456789:ABCdefGHIjklmNOPqrs..."
# Thêm các API Key khác của bạn ở dưới (Ví dụ: OPENAI_API_KEY,...)
```

---

## 🚀 Hướng dẫn khởi chạy

Để hệ thống hoạt động, bạn cần chạy Backend trước, sau đó chạy Frontend (Telegram Bot hoặc Streamlit).

### Bước 1: Khởi động Backend (Bắt buộc)
Mở một terminal mới, đảm bảo đã kích hoạt môi trường ảo và chạy lệnh sau:
```bash
uvicorn src.app:app --host 127.0.0.1 --port 5051 --reload
```
*(Backend sẽ chạy ở địa chỉ http://127.0.0.1:5051)*

### Bước 2: Khởi động Giao diện (Chọn 1 trong 2)

**Lựa chọn A: Dùng Telegram Bot (Khuyến nghị)**
Mở một terminal **mới** (giữ nguyên terminal Backend đang chạy) và gõ:
```bash
python telegram_bot.py
```
*Sau khi thấy thông báo "Bot đã sẵn sàng", hãy lên Telegram tìm `@username_bot_cua_ban` và nhấn /start.*

**Lựa chọn B: Dùng Giao diện Web Streamlit**
Mở một terminal **mới** và gõ:
```bash
streamlit run app_ui.py
```
*Trình duyệt sẽ tự động mở giao diện web ở địa chỉ http://localhost:8501.*

---

## 📝 Cách sử dụng Bot Telegram
- **`/start`**: Bắt đầu trò chuyện và khởi tạo bot.
- **`/settings`**: Mở menu chọn Model AI và tùy chỉnh độ sáng tạo.
- **Gửi File**: Kéo thả trực tiếp file PDF/TXT vào chat để bot đọc dữ liệu.
- **Chat trực tiếp**: Nhập câu hỏi bất kỳ để truy vấn thông tin từ tài liệu đã gửi.

---

**Tác giả:** Lê Minh Tân  
**Phiên bản:** 1.0.0

**Lưu ý nhỏ:**\
 Ở phần **"4. Cấu hình biến môi trường"**, tui có nhắc đến file `.env`. Nếu trước đó bạn chưa dùng file `.env` mà chỉ set biến môi trường trực tiếp trên Terminal thì tui khuyên bạn nên tạo file `.env` và tải thêm thư viện `python-dotenv` nhé. Việc lưu Token vào file `.env` (và nhớ **không up file này lên Github**, đã có trong `.gitignore`) là chuẩn mực bảo mật khi làm dự án thực tế đó!
