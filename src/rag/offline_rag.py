import re
from langchain_classic import hub
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate
from langchain_classic.retrievers import MultiQueryRetriever



"""
output: tach nhung cai minh can
"""
#1. Hau xu dua ra output
class Str_OutputParser(StrOutputParser):
    def __init__(self) -> None:
        super().__init__()

    def parse(self, text:str):
        return self.extract_answer(text)

    def extract_answer(
            self,
            text_response:str,
            pattern:str = r"Answer:\s*(.*)"
    )-> str:
        match = re.search(pattern,text_response,re.DOTALL)
        if match:
            answer_text = match.group(1).strip()
            return answer_text
        else:
            return text_response

# 2. Format cua cau prompt
class Offline_RAG:
    def __init__(self,llm) -> None:
        self.llm = llm
        #self.prompt = hub.pull("rlm/rag-prompt")
        template = """You are a helpful and precise assistant. Use the following pieces of retrieved context to answer the question. 
                If you don't know the answer, just say that you don't know. 
                Answer concisely, directly, and DO NOT generate additional questions or explanations.

                Context: {context}

                Question: {question}

                Helpful Answer:
                """
        
        self.prompt = PromptTemplate.from_template(template)
        self.str_parser = Str_OutputParser()

    def get_chain(self,retriever):
        advanced_retriever = MultiQueryRetriever.from_llm(
            retriever=retriever,
            llm=self.llm
        )

        input_data ={
            "context": advanced_retriever | self.format_docs,
            "question": RunnablePassthrough()
        }

        rag_chain =(
            input_data|
            self.prompt|
            self.llm|
            self.str_parser
        )
        return rag_chain
    
    def format_docs(self,docs):
        return "\n\n".join(doc.page_content for doc in docs)