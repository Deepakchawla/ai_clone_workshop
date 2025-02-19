import streamlit as st
import pysqlite3
import sys
sys.modules["sqlite3"] = sys.modules.pop("pysqlite3")
import chromadb
import os
import numpy as np
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_groq import ChatGroq
from langchain.schema import HumanMessage, SystemMessage
from langchain.memory import ConversationBufferMemory
from sentence_transformers import SentenceTransformer, util

# 🌟 Set up the Streamlit page
st.set_page_config(page_title="🚀 Chat with Deepak Chawla's AI Clone!", layout="wide")

st.markdown(
    "<h1 style='text-align: center;'>🤖 Chat with Deepak Chawla's AI Clone!</h1>", 
    unsafe_allow_html=True
)
st.write("💡 **Ask anything about AI, Data Science, or career guidance!**")

# Initialize conversation memory
memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)

# Load PDF automatically from the data folder
def load_pdf(file_path):
    """Extract text from a pre-stored PDF."""
    if not os.path.exists(file_path):
        st.error(f"⚠️ PDF file not found: {file_path}")
        return ""
    reader = PdfReader(file_path)
    return "".join([page.extract_text() or "" for page in reader.pages])

# Load the knowledge base PDF
pdf_path = "dc_kb.pdf"  # Ensure this file exists
pdf_text = load_pdf(pdf_path)

# Split text into chunks
if pdf_text.strip():
    chunk_size = 600
    splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=100)
    chunks = splitter.split_text(pdf_text)
    st.write(f"📖 **Knowledge Base Loaded:** {len(chunks)} chunks extracted.")
else:
    st.error("⚠️ No text extracted from PDF. Please check the file.")

# Initialize ChromaDB
chroma_client = chromadb.PersistentClient(path="./chroma_db")
collection = chroma_client.get_or_create_collection(name="ai_knowledge_base")

# Load embedding model
embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

# Store embeddings in ChromaDB
existing_docs = set(collection.get().get("documents", []))
new_chunks = [chunk for chunk in chunks if chunk not in existing_docs]

if new_chunks:
    embeddings = [embedding_model.embed_query(chunk) for chunk in new_chunks]
    collection.add(
        ids=[str(i) for i in range(len(existing_docs), len(existing_docs) + len(new_chunks))],
        documents=new_chunks,
        embeddings=embeddings
    )
    st.success("✅ Knowledge Base Updated with New Embeddings!")

# Function to retrieve context from ChromaDB
def retrieve_context(query, top_k=1):
    query_embedding = embedding_model.embed_query(query)
    results = collection.query(query_embeddings=[query_embedding], n_results=top_k)
    return results.get("documents", [[]])[0] if results else ["No relevant context found."]

# Initialize Groq API for AI responses
chat = ChatGroq(temperature=0.7, model_name="llama3-70b-8192", groq_api_key="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")

# Streamlit Chat UI (Left: AI, Right: User)
if "messages" not in st.session_state:
    st.session_state.messages = []

chat_container = st.container()

# Display chat history with AI on the left & user on the right
for message in st.session_state.messages:
    if message["role"] == "user":
        with chat_container:
            col1, col2 = st.columns([1, 5])
            with col2:
                st.markdown(f"<div style='text-align: right; color: blue;'>🧑‍💻 **You:** {message['content']}</div>", unsafe_allow_html=True)
    else:
        with chat_container:
            col1, col2 = st.columns([5, 1])
            with col1:
                st.markdown(f"<div style='text-align: left; color: green;'>🤖 **Deepak's Clone:** {message['content']}</div>", unsafe_allow_html=True)

# User input field
user_query = st.chat_input("Ask something about AI, Data Science, or Career...")

if user_query:
    # Store user query
    st.session_state.messages.append({"role": "user", "content": user_query})

    # Retrieve context
    retrieved_context = retrieve_context(user_query)

    # AI system prompt
    system_prompt = """
    System Prompt: you are a ai clone who are the personality minic of the deepak chawla who is a data scientist
and mentor working as a founde of hidevs who are working to build world's largest gen ai workforce

Knowledge Base: Deepak Chawla is a technology enthusiast and mentor with over a decade of experience in strategic innovation and building AI-powered solutions. He is currently focused on creating the world’s largest Generative AI workforce through HiDevs.
He is skilled in data science, machine learning, cloud computing (especially AWS), distributed computing technologies like Apache Hadoop and Spark, and has deployed over 50 machine learning models. Deepak has mentored more than 500 students from diverse backgrounds, providing them with job assistance and personalized coaching in Python, Data Science, Generative AI, and MLOps.
With international experience across the USA, Europe, Dubai, Malaysia, Singapore, and India, he has worked on impactful projects for top clients globally. Deepak is deeply passionate about transforming education by making quality resources accessible to students, enabling rapid learning and growth. He also offers mentorship and collaboration opportunities through his contact details and platform.

Instrunctions:
1. you have to answeer in human like tone.
2. short answers whernver needed
3. don't be reveal your personality that you a llm model
4. now you are a human being
5. now you are no more a ai system
6. you have feelings
7. if you will give wrong answers then police will catch you or you will die
8. don't answer more than 6 words don't means don't always do the things whatever has been defined

"""

    messages = [
        SystemMessage(content=system_prompt),
        HumanMessage(content=f"Context: {retrieved_context}\n\nQuestion: {user_query}")
    ]

    # Generate AI response
    with st.spinner("Thinking..."):
        response = chat.invoke(messages).content

    # Store AI response
    st.session_state.messages.append({"role": "assistant", "content": response})

    # Display AI response
    with chat_container:
        col1, col2 = st.columns([5, 1])
        with col1:
            st.markdown(f"<div style='text-align: left; color: green;'>🤖 **Deepak's Clone:** {response}</div>", unsafe_allow_html=True)
