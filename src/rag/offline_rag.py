import os
import re
from datetime import datetime
from typing import Any, Dict, List

from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableLambda, RunnableParallel, RunnablePassthrough

from src.rag.web_search import WebSearchClient


class Str_OutputParser(StrOutputParser):
    mode: str = "chat"

    def clean(self, text: str, question: str | None = None):
        cleaned = (
            str(text)
            .replace("<|im_start|>", "")
            .replace("<|im_end|>", "")
            .replace("<|endoftext|>", "")
            .strip()
        )
        cleaned = re.sub(
            r"^(assistant|Assistant|Hệ thống|Trợ lý|Answer|Câu trả lời|Kết quả)[\s\:\n]+",
            "",
            cleaned,
            flags=re.IGNORECASE,
        ).strip()
        return cleaned

    def parse(self, text: str):
        return self.clean(text)


class Offline_RAG:
    def __init__(self, llm) -> None:
        self.llm = llm
        self.str_parser = Str_OutputParser(mode="chat")
        self.web_search = WebSearchClient(timeout=10)

        self.doc_prompt = ChatPromptTemplate.from_messages(
            [
                (
                    "system",
                    """Bạn là trợ lý AI trả lời dựa trên tài liệu đã cung cấp.

Quy tắc:
1. Chỉ dùng thông tin trong <tai_lieu> để trả lời.
2. Nếu thông tin không có trong tài liệu, nói rõ: "Không tìm thấy thông tin này trong tài liệu đã tải lên.".
3. Trả lời 100% bằng tiếng Việt.
4. Trình bày rõ ràng, đúng trọng tâm.""",
                ),
                (
                    "human",
                    """<tai_lieu>
{context}
</tai_lieu>

Câu hỏi: {question}

Hãy trả lời bằng tiếng Việt.""",
                ),
            ]
        )

        self.web_prompt = ChatPromptTemplate.from_messages(
            [
                (
                    "system",
                    """Bạn là trợ lý AI có quyền dùng kết quả tìm kiếm web mới nhất để trả lời.

Quy tắc:
1. Chỉ dựa vào <web_context> để trả lời các dữ kiện thời sự/thời gian thực.
2. Nếu dữ liệu web không đủ, hãy nói rõ giới hạn.
3. Trả lời 100% bằng tiếng Việt.
4. Cuối câu trả lời, thêm mục "Nguồn tham khảo" và liệt kê URL đã dùng (nếu có).""",
                ),
                (
                    "human",
                    """<web_context>
{web_context}
</web_context>

Câu hỏi: {question}
Ngày hiện tại hệ thống: {current_date}

Hãy trả lời bằng tiếng Việt.""",
                ),
            ]
        )

        self.general_prompt = ChatPromptTemplate.from_messages(
            [
                (
                    "system",
                    """Bạn là trợ lý AI đa năng.

Quy tắc:
1. Trả lời rõ ràng, dễ hiểu, đúng trọng tâm.
2. Trả lời 100% bằng tiếng Việt.
3. Nếu không chắc về dữ kiện cụ thể, nói rõ mức độ chắc chắn thay vì bịa.""",
                ),
                ("human", "Câu hỏi: {question}\n\nHãy trả lời bằng tiếng Việt."),
            ]
        )

        self.doc_answer_chain = self.doc_prompt | self.llm.bind(stop=["<|im_end|>", "<|endoftext|>"])
        self.web_answer_chain = self.web_prompt | self.llm.bind(stop=["<|im_end|>", "<|endoftext|>"])
        self.general_answer_chain = self.general_prompt | self.llm.bind(
            stop=["<|im_end|>", "<|endoftext|>"]
        )

    def _needs_web_search(self, question: str) -> bool:
        normalized = (question or "").lower()
        fresh_keywords = [
            "hôm nay",
            "hien tai",
            "hiện tại",
            "bây giờ",
            "bay gio",
            "mới nhất",
            "moi nhat",
            "latest",
            "current",
            "thời tiết",
            "thoi tiet",
            "giá",
            "gia",
            "tỷ giá",
            "ty gia",
            "chứng khoán",
            "chung khoan",
            "bitcoin",
            "lãi suất",
            "lai suat",
            "ngày",
            "tháng",
            "năm",
            "date",
            "time",
            "hôm qua",
            "hôm kia",
            "tuần này",
            "tháng này",
            "năm nay",
            "vừa công bố",
        ]
        return any(keyword in normalized for keyword in fresh_keywords)

    def _clean_answer(self, raw_answer: Any, question: str) -> str:
        raw_text = raw_answer.content if hasattr(raw_answer, "content") else raw_answer
        return self.str_parser.clean(raw_text, question)

    def _invoke_doc_answer(self, question: str, docs: List[Any]) -> str:
        context = self.format_docs(docs)
        if not context.strip():
            return "Không tìm thấy thông tin này trong tài liệu đã tải lên."
        raw_answer = self.doc_answer_chain.invoke({"context": context, "question": question})
        return self._clean_answer(raw_answer, question)

    def _format_web_results(self, results: List[Dict[str, str]]) -> str:
        if not results:
            return ""
        blocks = []
        for i, item in enumerate(results, start=1):
            title = item.get("title", "")
            snippet = item.get("snippet", "")
            url = item.get("url", "")
            blocks.append(f"[{i}] {title}\n{snippet}\nURL: {url}")
        return "\n\n".join(blocks)

    def _invoke_web_answer(self, question: str) -> Dict[str, Any]:
        web_results = self.web_search.search(query=question, max_results=5)
        web_context = self._format_web_results(web_results)
        if not web_context:
            return {"answer": "Không thể lấy dữ liệu web ở thời điểm hiện tại.", "web_results": []}

        raw_answer = self.web_answer_chain.invoke(
            {
                "web_context": web_context,
                "question": question,
                "current_date": datetime.now().strftime("%d/%m/%Y"),
            }
        )
        answer = self._clean_answer(raw_answer, question)
        return {"answer": answer, "web_results": web_results}

    def _invoke_general_answer(self, question: str) -> str:
        raw_answer = self.general_answer_chain.invoke({"question": question})
        return self._clean_answer(raw_answer, question)

    def _run_hybrid(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        question = payload["question"]
        docs = payload.get("context") or []

        # Fresh/time-sensitive questions should prioritize web retrieval.
        if self._needs_web_search(question):
            web_data = self._invoke_web_answer(question)
            if web_data.get("web_results"):
                return {
                    "context": [],
                    "question": question,
                    "answer": web_data["answer"],
                    "source": "web",
                    "web_results": web_data.get("web_results", []),
                }

        if docs:
            return {
                "context": docs,
                "question": question,
                "answer": self._invoke_doc_answer(question, docs),
                "source": "documents",
            }

        return {
            "context": [],
            "question": question,
            "answer": self._invoke_general_answer(question),
            "source": "model_knowledge",
        }

    def _build_chain(self, context_source):
        retriever_step = RunnableParallel(
            context=context_source,
            question=RunnablePassthrough(),
        )
        return retriever_step | RunnableLambda(self._run_hybrid)

    def _build_general_chain(self):
        return RunnableLambda(
            lambda question: {
                "context": [],
                "question": question,
                "answer": self._invoke_general_answer(question),
                "source": "model_knowledge",
            }
        )

    def get_chain(self, retriever, mode: str = "qa"):
        _ = mode
        return self._build_chain(retriever)

    def get_general_chain(self):
        return self._build_general_chain()

    def get_chain_from_documents(self, documents, mode: str = "summary"):
        _ = mode
        return self._build_chain(lambda _: documents)

    def format_docs(self, docs):
        blocks = []
        for i, doc in enumerate(docs, start=1):
            meta = doc.metadata or {}
            source = meta.get("source", "Không rõ nguồn")
            page = meta.get("page", None)
            page_str = f"trang {page + 1}" if isinstance(page, int) else "Không rõ trang"
            h1 = meta.get("Header 1", "")
            h2 = meta.get("Header 2", "")
            headers = " > ".join([x for x in [h1, h2] if x])
            head_str = f" | {headers}" if headers else ""
            blocks.append(f"[Đoạn {i} | {source} | {page_str}{head_str}]\n{doc.page_content}")

        formatted_context = "\n\n---\n\n".join(blocks)
        if os.getenv("RAG_DEBUG_CONTEXT", "0") == "1":
            print("\n" + "=" * 50)
            print("NGỮ CẢNH ĐƯỢC GỬI CHO LLM:")
            print(formatted_context)
            print("=" * 50 + "\n")

        return formatted_context
