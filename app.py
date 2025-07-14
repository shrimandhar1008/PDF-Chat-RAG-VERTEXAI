import streamlit as st
from langchain_google_vertexai import ChatVertexAI
from langchain_community.vectorstores import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.document_loaders import PyPDFLoader
from langchain import hub
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains import create_retrieval_chain
from dotenv import load_dotenv
import os
os.environ["LANGCHAIN_TRACING_V2"] = "false"
load_dotenv()

class RagPdf:
    def __init__(self):
        self.PDF_PATH     = r"E:\Shrimandhar\DS_interview\Aditya Y. Bhargava - Grokking Algorithms_ An illustrated guide for programmers and other curious people-Manning Publications (2016).pdf"
        self.docs = PyPDFLoader(self.PDF_PATH).load()
        self.embeddings = HuggingFaceEmbeddings(
            model_name='E:\\Shrimandhar\\Generative AI Projects\\legal-helper\\all-MiniLM-L6-v2')
        self.vectorstore = Chroma.from_documents(self.docs, self.embeddings)

    def build_rag_chain(self):
        # 2. Build RAG chain
        llm  = ChatVertexAI(model_name=os.environ["GEN_MODEL"], project=os.environ["PROJECT_ID"], location=os.environ["REGION"],temperature=0.2)
        rag_prompt = hub.pull("langchain-ai/retrieval-qa-chat")

        doc_chain   = create_stuff_documents_chain(llm, rag_prompt)
        return create_retrieval_chain(self.vectorstore.as_retriever(),
                                             doc_chain)

    def streamui(self):
        rag_chain = self.build_rag_chain()
        # 3. Streamlit UI
        st.title("📄🔎 PDF RAG Chatbot (Vertex AI)")
        if 'history' not in st.session_state:
            st.session_state.history = []

        for msg in st.session_state.history:
            st.chat_message(msg["role"]).markdown(msg["content"])

        prompt = st.chat_input("Ask me anything about the PDF…")
        if prompt:
            st.chat_message("user").markdown(prompt)
            result = rag_chain.invoke({"input": prompt})
            answer = result["answer"]
            st.chat_message("assistant").markdown(answer)
            st.session_state.history.append({"role": "user", "content": prompt})
            st.session_state.history.append({"role": "assistant", "content": answer})


if __name__=="__main__":
    obj = RagPdf()
    obj.streamui()
