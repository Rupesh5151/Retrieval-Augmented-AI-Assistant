# -*- coding: utf-8 -*-
"""
RAG PDF Chatbot
Based on: https://github.com/debashish0404/rag-pdf-chatbot

A Retrieval-Augmented Generation (RAG) based chatbot that enables users
to ask questions from PDF documents and receive context-aware answers
using open-source models (no paid APIs required).
"""
import sys
sys.stdout.reconfigure(encoding='utf-8')

# ─── 1. Load PDF Document ─────────────────────────────────────────────────────
from langchain_community.document_loaders import PyPDFLoader

# You can replace this URL with a local path like: PyPDFLoader("sample.pdf")
loader = PyPDFLoader("https://arxiv.org/pdf/1706.03762.pdf")
docs = loader.load()

print(f"[+] Loaded {len(docs)} pages from PDF.")

# ─── 2. Split Text into Chunks ────────────────────────────────────────────────
from langchain_text_splitters import RecursiveCharacterTextSplitter

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,
    chunk_overlap=50
)

splits = text_splitter.split_documents(docs)
print(f"[+] Split into {len(splits)} chunks.")

# ─── 3. Create Embeddings ─────────────────────────────────────────────────────
from langchain_community.embeddings import HuggingFaceEmbeddings

embedding = HuggingFaceEmbeddings()
print("[+] Embeddings model loaded.")

# ─── 4. Store in Vector Database (ChromaDB) ───────────────────────────────────
from langchain_community.vectorstores import Chroma

vectorstore = Chroma.from_documents(
    documents=splits,
    embedding=embedding
)

retriever = vectorstore.as_retriever()
print("[+] Vector store created and retriever ready.")

# ─── 5. Load Language Model (GPT-2 via HuggingFace) ──────────────────────────
from transformers import pipeline
from langchain_community.llms import HuggingFacePipeline

pipe = pipeline(
    "text-generation",
    model="gpt2",
    max_new_tokens=200
)

llm = HuggingFacePipeline(pipeline=pipe)
print("[+] LLM loaded.")

# ─── 6. Build RAG Chain ───────────────────────────────────────────────────────
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

prompt = ChatPromptTemplate.from_template("""
Answer the question based only on the context:

{context}

Question: {question}
""")

def format_docs(docs):
    return "\n\n".join([doc.page_content for doc in docs])

rag_chain = (
    {"context": retriever | format_docs, "question": lambda x: x}
    | prompt
    | llm
    | StrOutputParser()
)

print("[+] RAG chain ready.\n")

# ─── 7. Interactive Chatbot Loop ──────────────────────────────────────────────
print("=" * 50)
print("  RAG PDF Chatbot  (type 'exit' to quit)")
print("=" * 50)

while True:
    question = input("\nAsk a question: ").strip()
    if question.lower() in ("exit", "quit", "q"):
        print("Goodbye!")
        break
    if not question:
        continue

    print("\nThinking...\n")
    result = rag_chain.invoke(question)
    print(f"Answer:\n{result}")
