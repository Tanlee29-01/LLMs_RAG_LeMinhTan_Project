import asyncio
import os
from typing import Any

import httpx
from telegram import InlineKeyboardButton, InlineKeyboardMarkup, Update, constants
from telegram.ext import CallbackQueryHandler, CommandHandler, ContextTypes, Application, MessageHandler, filters

TELEGRAM_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN", "Your_Token")
FASTAPI_URL = os.getenv("FASTAPI_URL", "http://127.0.0.1:5051")
MAX_UPLOAD_MB = int(os.getenv("MAX_UPLOAD_MB", "50"))
DEFAULT_MODEL = os.getenv("DEFAULT_MODEL", "llama3.1:8b")
DEFAULT_TEMPERATURE = float(os.getenv("DEFAULT_TEMPERATURE", "0.1"))


def _get_chat_id(update: Update) -> str:
    return str(update.effective_chat.id)


def _get_user_settings(context: ContextTypes.DEFAULT_TYPE) -> dict[str, Any]:
    settings = context.user_data.setdefault(
        "settings",
        {"model": DEFAULT_MODEL, "temperature": DEFAULT_TEMPERATURE},
    )
    return settings


def _build_settings_keyboard(context: ContextTypes.DEFAULT_TYPE) -> InlineKeyboardMarkup:
    settings = _get_user_settings(context)
    model_label = settings.get("model", DEFAULT_MODEL)
    temperature = float(settings.get("temperature", DEFAULT_TEMPERATURE))

    keyboard = [
        [
            InlineKeyboardButton("🤖 Model: Llama 3.1 8B", callback_data="model_llama"),
            InlineKeyboardButton("🌐 Model: GPT-4o", callback_data="model_gpt4"),
        ],
        [
            InlineKeyboardButton("➖ Giảm Sáng tạo", callback_data="temp_down"),
            InlineKeyboardButton(f"🌡️ Temp: {temperature:.1f}", callback_data="temp_info"),
            InlineKeyboardButton("➕ Tăng Sáng tạo", callback_data="temp_up"),
        ],
        [
            InlineKeyboardButton("🗑️ Xóa lịch sử chat", callback_data="clear_data"),
        ],
    ]

    if model_label == "gpt-4o":
        keyboard[0][0] = InlineKeyboardButton("🤖 Model: Llama 3.1 8B", callback_data="model_llama")
        keyboard[0][1] = InlineKeyboardButton("🌐 Model: GPT-4o ✓", callback_data="model_gpt4")
    else:
        keyboard[0][0] = InlineKeyboardButton("🤖 Model: Llama 3.1 8B ✓", callback_data="model_llama")
        keyboard[0][1] = InlineKeyboardButton("🌐 Model: GPT-4o", callback_data="model_gpt4")

    return InlineKeyboardMarkup(keyboard)


async def _post_json(url: str, payload: dict[str, Any]) -> httpx.Response:
    async with httpx.AsyncClient(timeout=120) as client:
        return await client.post(url, json=payload)


async def _post_files(url: str, files: list[tuple[str, tuple[str, bytes, str]]], data: dict[str, str]) -> httpx.Response:
    multipart_files = {
        name: (filename, file_content, content_type)
        for name, (filename, file_content, content_type) in files
    }
    async with httpx.AsyncClient(timeout=120) as client:
        return await client.post(url, files=multipart_files, data=data)


async def start_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text(
        "👋 Chào bạn. Gửi PDF/TXT vào đây hoặc dùng /settings để mở bảng điều khiển.",
    )


async def settings_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    reply_markup = _build_settings_keyboard(context)
    await update.message.reply_text(
        "⚙️ Bảng điều khiển hệ thống RAG:\nChọn cấu hình bạn muốn thay đổi.",
        reply_markup=reply_markup,
        parse_mode=constants.ParseMode.MARKDOWN,
    )


async def button_handler(update: Update, context: ContextTypes.DEFAULT_TYPE):
    query = update.callback_query
    await query.answer()

    settings = _get_user_settings(context)

    if query.data == "clear_data":
        context.user_data.clear()
        await query.edit_message_text(
            "✅ Đã xóa lịch sử chat và cấu hình của phiên này.\n"
            "Lưu ý: tài liệu đã nạp ở backend vẫn còn, nếu muốn xóa dữ liệu server thì cần thêm endpoint riêng.",
        )
        return

    if query.data == "temp_down":
        settings["temperature"] = max(0.0, float(settings.get("temperature", DEFAULT_TEMPERATURE)) - 0.1)
    elif query.data == "temp_up":
        settings["temperature"] = min(1.0, float(settings.get("temperature", DEFAULT_TEMPERATURE)) + 0.1)
    elif query.data == "model_llama":
        settings["model"] = "llama3.1:8b"
    elif query.data == "model_gpt4":
        settings["model"] = "gpt-4o"
    elif query.data == "temp_info":
        pass

    await query.edit_message_text(
        "⚙️ Bảng điều khiển hệ thống RAG:\nChọn cấu hình bạn muốn thay đổi.",
        reply_markup=_build_settings_keyboard(context),
        parse_mode=constants.ParseMode.MARKDOWN,
    )


async def handle_document(update: Update, context: ContextTypes.DEFAULT_TYPE):
    chat_id = _get_chat_id(update)
    document = update.message.document
    file_name = document.file_name or "uploaded_file"

    if not file_name.lower().endswith((".pdf", ".txt")):
        await update.message.reply_text("❌ Tôi chỉ hỗ trợ file .PDF hoặc .TXT.")
        return

    if document.file_size and document.file_size > MAX_UPLOAD_MB * 1024 * 1024:
        await update.message.reply_text(
            f"❌ File quá lớn. Giới hạn hiện tại là {MAX_UPLOAD_MB}MB để tránh quá tải máy local."
        )
        return

    status_msg = await update.message.reply_text(f"⏳ Đang tải `{file_name}` từ Telegram...", parse_mode=constants.ParseMode.MARKDOWN)

    try:
        file_info = await context.bot.get_file(document.file_id)
        file_bytes = await file_info.download_as_bytearray()

        await status_msg.edit_text(
            f"⏳ Đang nhúng `{file_name}` vào Vector DB...",
            parse_mode=constants.ParseMode.MARKDOWN,
        )

        payload_files = [
            (
                "files",
                (
                    file_name,
                    bytes(file_bytes),
                    "application/pdf" if file_name.lower().endswith(".pdf") else "text/plain",
                ),
            )
        ]
        payload_data = {"chat_id": chat_id}

        response = await _post_files(f"{FASTAPI_URL}/upload", payload_files, payload_data)

        if response.status_code == 200:
            await status_msg.edit_text(
                f"✅ Đã học xong tài liệu: `{file_name}`. Bạn có thể đặt câu hỏi ngay.",
                parse_mode=constants.ParseMode.MARKDOWN,
            )
        else:
            await status_msg.edit_text(f"❌ Lỗi từ máy chủ AI: {response.text}")
    except Exception as exc:
        await status_msg.edit_text(f"❌ Lỗi xử lý tài liệu: {exc}")


async def handle_message(update: Update, context: ContextTypes.DEFAULT_TYPE):
    chat_id = _get_chat_id(update)
    question = update.message.text
    settings = _get_user_settings(context)

    await context.bot.send_chat_action(chat_id=chat_id, action=constants.ChatAction.TYPING)
    wait_msg = await update.message.reply_text("🤖 Trợ lý đang suy nghĩ và lục tìm tài liệu...")

    try:
        payload = {
            "question": question,
            "chat_id": chat_id,
            "model": settings.get("model", DEFAULT_MODEL),
            "temperature": settings.get("temperature", DEFAULT_TEMPERATURE),
        }
        response = await _post_json(f"{FASTAPI_URL}/generative_ai", payload)

        if response.status_code == 200:
            data = response.json()
            answer = data.get("answer", "Không có câu trả lời")
            source = data.get("source", "model_knowledge")

            if source == "web":
                answer = f"🌐 **[Trích xuất từ Internet]**\n\n{answer}"
            elif source == "documents":
                answer = f"📄 **[Dựa trên tài liệu của bạn]**\n\n{answer}"
            else:
                answer = f"🤖 **[Kiến thức AI]**\n\n{answer}"

            await wait_msg.edit_text(answer, parse_mode=constants.ParseMode.MARKDOWN)
        else:
            await wait_msg.edit_text(f"❌ Lỗi truy xuất AI: {response.status_code}\n{response.text}")
    except Exception as exc:
        await wait_msg.edit_text(f"🔌 Mất kết nối tới Backend FastAPI. Chi tiết: {exc}")


def main():
    if not TELEGRAM_TOKEN or TELEGRAM_TOKEN == "YOUR_BOT_TOKEN_HERE":
        raise RuntimeError("Missing TELEGRAM_BOT_TOKEN environment variable or placeholder token still set.")

    print("🚀 Đang khởi động Telegram Bot...")
    app = Application.builder().token(TELEGRAM_TOKEN).build()

    app.add_handler(CommandHandler("start", start_command))
    app.add_handler(CommandHandler("settings", settings_command))
    app.add_handler(CallbackQueryHandler(button_handler))
    app.add_handler(MessageHandler(filters.Document.ALL, handle_document))
    app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, handle_message))

    print("✅ Bot đã sẵn sàng nhận lệnh!")
    app.run_polling()


if __name__ == "__main__":
    main()
