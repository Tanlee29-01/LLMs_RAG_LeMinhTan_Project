from langchain_core.runnables import RunnablePassthrough, RunnableParallel, RunnableLambda
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
import re


class Str_OutputParser(StrOutputParser):
    mode: str = "qa_summary"

    def _trim_extraction_style(self, text: str) -> str:
        sentences = re.split(r"(?<=[\.\!\?])\s+", text.strip())
        sentences = [sentence.strip() for sentence in sentences if sentence.strip()]
        return " ".join(sentences[:2]).strip()

    def _trim_to_sentences(self, text: str, max_sentences: int) -> str:
        sentences = re.split(r"(?<=[\.\!\?])\s+", text.strip())
        sentences = [sentence.strip() for sentence in sentences if sentence.strip()]
        return " ".join(sentences[:max_sentences]).strip()

    def _normalize_summary_output(self, text: str) -> str:
        lines = [line.strip() for line in text.splitlines() if line.strip()]
        if not lines:
            return text

        normalized_lines = []
        bullet_index = 1
        in_list = False

        for line in lines:
            line = re.sub(r"^(Tóm tắt|Kết quả trả lời)\s*:?\s*", "", line, flags=re.IGNORECASE)
            if not line:
                continue

            numbered_match = re.match(r"^\d+[\.)]\s*(.+)$", line)
            bullet_match = re.match(r"^[-*•]\s*(.+)$", line)

            if numbered_match:
                normalized_lines.append(f"{bullet_index}. {numbered_match.group(1).strip()}")
                bullet_index += 1
                in_list = True
                continue

            if bullet_match:
                normalized_lines.append(f"{bullet_index}. {bullet_match.group(1).strip()}")
                bullet_index += 1
                in_list = True
                continue

            if in_list and normalized_lines:
                normalized_lines[-1] = f"{normalized_lines[-1]} {line}"
                continue

            normalized_lines.append(line)

        return "\n".join(normalized_lines).strip()

    def _extract_requested_points(self, question: str | None) -> int | None:
        if not question:
            return None

        match = re.search(r"(\d+)\s*(?:ý|y|điểm|diem)", question.lower())
        if match:
            return int(match.group(1))
        return None

    def _is_definition_question(self, question: str | None) -> bool:
        if not question:
            return False

        normalized = question.lower()
        keywords = ["là gì", "la gi", "định nghĩa", "dinh nghia", "khái niệm", "khai niem"]
        return any(keyword in normalized for keyword in keywords)

    def clean(self, text: str, question: str | None = None):
        # 1. Dọn dẹp các token rác và lỗi font thường gặp
        cleaned = str(text).replace("<|im_start|>", "").replace("<|im_end|>", "").replace("<|endoftext|>", "").strip()
        cleaned = re.sub(r"^(assistant|Assistant|Hệ thống|Trợ lý)[\s\:\n]+", "", cleaned, flags=re.IGNORECASE).strip()
        cleaned = re.sub(r"^(Answer|Câu trả lời|Tóm tắt|Kết quả)[\s\:\n]+", "", cleaned, flags=re.IGNORECASE).strip()
       
        # 2. Xử lý theo từng mode
        if self.mode == "extraction":
            cleaned = self._trim_extraction_style(cleaned)
        elif self.mode in ("qa", "qa_summary") and self._is_definition_question(question):
            cleaned = self._trim_to_sentences(cleaned, max_sentences=3)
        elif self.mode in ("summary", "summary_document"):
            cleaned = self._normalize_summary_output(cleaned)
            target_points = self._extract_requested_points(question) or 5
            summary_lines = [line for line in cleaned.splitlines() if line.strip()]
            if summary_lines:
                cleaned = "\n".join(summary_lines[:target_points]).strip()

        return cleaned

    def parse(self, text: str):
        return self.clean(text)


class Offline_RAG:

    def __init__(self, llm) -> None:
        self.llm = llm
        self.str_parser = Str_OutputParser()
        
    def _get_prompt(self, mode: str) -> ChatPromptTemplate:
        if mode == "extraction":
            return ChatPromptTemplate.from_messages(
                [
                    (
                        "system",
                        """Bạn là một trợ lý AI có nhiệm vụ trích xuất thông tin từ tài liệu.

LUẬT TỐI CAO:
1. NẾU NGỮ CẢNH TRỐNG HOẶC KHÔNG LIÊN QUAN: Bắt buộc trả lời "Không tìm thấy thông tin này trong tài liệu." Tuyệt đối không tự bịa ra thông tin.
2. CHỈ sử dụng thông tin nằm trong thẻ <tai_lieu>.
3. BẮT BUỘC trả lời 100% bằng TIẾNG VIỆT. TUYỆT ĐỐI KHÔNG dùng tiếng Trung.
4. Trả lời cực kỳ ngắn gọn, đi thẳng vào vấn đề."""
                    ),
                    (
                        "human",
                        """<tai_lieu>
{context}
</tai_lieu>

Câu hỏi: {question}

Yêu cầu: Trích xuất thông tin ngắn gọn, 100% bằng Tiếng Việt."""
                    ),
                ]
            )

        if mode in ("qa", "qa_summary"):
            return ChatPromptTemplate.from_messages(
                [
                    (
                        "system",
                        """Bạn là một trợ lý AI chuyên trả lời câu hỏi dựa trên tài liệu.

LUẬT TỐI CAO:
1. NẾU NGỮ CẢNH TRỐNG: Bạn PHẢI trả lời "Không tìm thấy thông tin này trong tài liệu." Không được sáng tác.
2. Mọi câu trả lời phải rút ra từ thẻ <tai_lieu>.
3. Trả lời 100% bằng TIẾNG VIỆT, không dùng tiếng Trung.
4. Trình bày rõ ràng, dễ hiểu."""
                    ),
                    (
                        "human",
                        """<tai_lieu>
{context}
</tai_lieu>

Câu hỏi: {question}

Yêu cầu: Trả lời rõ ràng, 100% bằng Tiếng Việt."""
                    ),
                ]
            )

        if mode in ("summary", "summary_document"):
            return ChatPromptTemplate.from_messages(
                [
                    (
                        "system",
                        """Bạn là một trợ lý AI chuyên tóm tắt tài liệu.

LUẬT TỐI CAO:
1. Tóm tắt CHỈ dựa vào thẻ <tai_lieu>. Nếu không có nội dung, trả lời "Không có thông tin để tóm tắt".
2. Trả lời 100% bằng TIẾNG VIỆT, không tiếng Trung.
3. Trả lời bằng danh sách đánh số, ĐÚNG SỐ LƯỢNG ý người dùng yêu cầu."""
                    ),
                    (
                        "human",
                        """<tai_lieu>
{context}
</tai_lieu>

Yêu cầu tóm tắt: {question}
Số lượng ý cần tóm tắt: {target_points}

Yêu cầu: Trả lời bằng danh sách đánh số, 100% bằng Tiếng Việt."""
                    ),
                ]
            )

        raise ValueError(f"Unsupported mode: {mode}")

    def _build_chain(self, context_source, mode: str):
        prompt = self._get_prompt(mode)
        parser = Str_OutputParser(mode=mode)
        base_input = RunnablePassthrough.assign(
            context=lambda x: self.format_docs(x["context"]),
            target_points=lambda x: parser._extract_requested_points(x["question"]) or 5,
        )
        raw_answer_chain = prompt | self.llm.bind(stop=["<|im_end|>", "<|endoftext|>"])

        retriever_step = RunnableParallel(
            context=context_source,
            question=RunnablePassthrough(),
        )
        answer_chain = (
            base_input
            | RunnablePassthrough.assign(raw_answer=raw_answer_chain)
            | RunnableLambda(
    lambda x: parser.clean(
        x["raw_answer"].content if hasattr(x["raw_answer"], "content") else x["raw_answer"],
        x["question"],
    )
)
        )

        return retriever_step.assign(answer=answer_chain)

    def get_chain(self, retriever, mode: str = "qa"):
        return self._build_chain(retriever, mode)

    def get_chain_from_documents(self, documents, mode: str = "summary"):
        return self._build_chain(lambda _: documents, mode)

    def format_docs(self, docs):
        blocks = []
        for i, doc in enumerate(docs, start=1):
            meta = doc.metadata or {}
            source = meta.get("source", "Không rõ nguồn")
            page = meta.get("page", None)
            page_str = f"trang {page + 1}" if page is not None else "Không rõ trang"
            h1 = meta.get("Header 1", "")
            h2 = meta.get("Header 2", "")
            headers = " > ".join([x for x in [h1, h2] if x])
            head_str = f" | {headers}" if headers else ""
            blocks.append(f"[Doan {i} | {source} | {page_str}{head_str}]\n{doc.page_content}")
        formatted_context = "\n\n---\n\n".join(blocks)
        # BẬT CAMERA AN NINH: In nội dung tài liệu ra Terminal để bạn kiểm tra
        print("\n" + "="*50)
        print("NGỮ CẢNH ĐƯỢC GỬI CHO LLM:")
        print(formatted_context)
        print("="*50 + "\n")
    
        return formatted_context
