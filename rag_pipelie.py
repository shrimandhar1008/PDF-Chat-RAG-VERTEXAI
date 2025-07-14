import pathlib, uuid, json
from langchain_community.document_loaders import PyPDFLoader              # :contentReference[oaicite:0]{index=0}
from langchain.text_splitter import RecursiveCharacterTextSplitter
# from langchain_google_vertexai import VertexAIEmbeddings
from langchain.prompts import SemanticSimilarityExampleSelector
# from langchain.embeddings import HuggingFaceEmbeddings
# from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_huggingface import HuggingFaceEmbeddings # :contentReference[oaicite:1]{index=1}
from langchain_google_vertexai.vectorstores import VectorSearchVectorStore, VectorSearchVectorStoreDatastore  # :contentReference[oaicite:2]{index=2}
# from google.cloud import aiplatform
# from langchain.vectorstores import Chroma
from langchain_community.vectorstores import Chroma
from dotenv import load_dotenv
import os
load_dotenv()
os.environ['PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION'] = 'python'
PROJECT_ID   = "rag-chatbot-465616"
REGION       = 'us-west1'
INDEX_NAME   = f"pdf‑index‑{uuid.uuid4().hex[:8]}"
EMBED_MODEL  = "text-embedding-005"   # or textembedding-gecko@005 (gecko@003 retires 14 May 2025) :contentReference[oaicite:3]{index=3}
PDF_PATH     = r"E:\Shrimandhar\DS_interview\Aditya Y. Bhargava - Grokking Algorithms_ An illustrated guide for programmers and other curious people-Manning Publications (2016).pdf"
BUCKET = 'pdf-chatbot'


# 1. Load & chunk
docs = PyPDFLoader(PDF_PATH).load()
splitter = RecursiveCharacterTextSplitter(chunk_size=1024, chunk_overlap=128)
chunks   = splitter.split_documents(docs)
embeddings = HuggingFaceEmbeddings(model_name='E:\\Shrimandhar\\Generative AI Projects\\legal-helper\\all-MiniLM-L6-v2')
# 2. Embed chunks
# embeddings = VertexAIEmbeddings(model_name=EMBED_MODEL, project=PROJECT_ID, location=REGION)
# 3. Push to Vertex Vector Search (auto‑creates Cloud Index if absent)
# vectorstore = VectorSearchVectorStore.from_components(
#     embedding=embeddings,
#     project_id=PROJECT_ID,
#     location=REGION,
#     index_id=INDEX_NAME,
#     region=REGION,# one‑time creation ≈ 25‑60 min
#     gcs_bucket_name = BUCKET,
#     endpoint_id=ENDPOINT
#     # `dimensions`, `distance_type` etc.
# )
# vectorstore.add_texts(texts=chunks)
vectorstore = Chroma.from_documents(docs, embeddings, persist_directory="./chroma_store")
vectorstore.persist()
# retriever    = vectorstore.as_retriever()
