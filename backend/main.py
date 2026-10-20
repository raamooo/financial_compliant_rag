import os
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from dotenv import load_dotenv
load_dotenv()

from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser

app = FastAPI(title="Regulatory Compliance RAG API", version="1.0")

# Enable CORS so your frontend can communicate with the backend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, specify your frontend domain
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# =====================================================================
# RAG Initialization on Startup
# =====================================================================
pdf_dir = "./regulatory_docs/"
if not os.path.exists(pdf_dir):
    os.makedirs(pdf_dir, exist_ok=True)

print("Loading PDFs and initializing vector store...")
loader = PyPDFDirectoryLoader(pdf_dir)
raw_documents = loader.load()

text_splitter = RecursiveCharacterTextSplitter(chunk_size=350, chunk_overlap=50)
docs = text_splitter.split_documents(raw_documents)

embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
vectorstore = Chroma.from_documents(
    docs, 
    embeddings, 
    collection_name="fastapi_compliance",
    collection_metadata={"hnsw:space": "cosine"}
)
retriever = vectorstore.as_retriever(search_kwargs={"k": 2})

llm = ChatOpenAI(
    model="openai/gpt-oss-20b",
    temperature=0,
    base_url="https://api.groq.com/openai/v1",
    api_key=os.environ.get("GROQ_API_KEY", "")
)

template = """
You are an expert Chief Compliance Officer and Regulatory AI Assistant. 
Answer the user's compliance question accurately using ONLY the provided policy text context. 
For every rule or threshold you state, you MUST explicitly quote the source document name, section number, and explain any cross-document differences or updates.

Context:
{context}

Question: {question}

Compliance Analysis & Answer:
"""
prompt = ChatPromptTemplate.from_template(template)

def format_docs(retrieved_docs):
    formatted = []
    for doc in retrieved_docs:
        source_file = os.path.basename(doc.metadata.get('source', 'Unknown Document'))
        page_num = doc.metadata.get('page', 0) + 1
        formatted.append(f"[Source Document: {source_file} | Page: {page_num}]\n{doc.page_content}")
    return "\n\n".join(formatted)

rag_chain = (
    {"context": retriever | format_docs, "question": RunnablePassthrough()}
    | prompt
    | llm
    | StrOutputParser()
)

# Pydantic Schema for Request Body
class QueryRequest(BaseModel):
    question: str

@app.post("/chat")
def chat_endpoint(request: QueryRequest):
    try:
        query = request.question
        
        # 1. Fetch similarity metrics
        scored_results = vectorstore.similarity_search_with_score(query, k=2)
        metrics = []
        for rank, (doc, score) in enumerate(scored_results, 1):
            metrics.append({
                "rank": rank,
                "distance": round(score, 4),
                "source": os.path.basename(doc.metadata.get('source', 'Unknown')),
                "page": doc.metadata.get('page', 0) + 1
            })
            
        # 2. Run RAG chain
        answer = rag_chain.invoke(query)
        
        return {
            "response": answer,
            "metrics": metrics
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/")
def root():
    return {"status": "Compliance RAG API is running"}