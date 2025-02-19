__import__('pysqlite3')
import sys
sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')
import chromadb
import streamlit as st
import sys
import os
import numpy as np
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_groq import ChatGroq
from langchain.schema import HumanMessage, SystemMessage
from langchain.memory import ConversationBufferMemory
from sentence_transformers import SentenceTransformer, util

# 🔹 Set Page Title
st.set_page_config(page_title="Deepak Chawla AI Clone - Ask Me Anything", layout="wide")

# 🔹 Streamlit App Title
st.markdown("<h1 style='text-align: center;'>💬 Chat with Deepak Chawla's AI Clone</h1>", unsafe_allow_html=True)

# ✅ Initialize ChromaDB
chroma_client = chromadb.PersistentClient(path="./chroma_db")
collection = chroma_client.get_collection(name="ai_knowledge_base")

# ✅ Load Embedding Model
embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

# ✅ Initialize Chat Model (Llama3 via Groq)
chat = ChatGroq(
    temperature=0.7, 
    model_name="llama3-70b-8192", 
    groq_api_key="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"
)

# ✅ Initialize Memory for Chat History
memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)

# ✅ Function to Retrieve Context from ChromaDB
def retrieve_context(query, top_k=1):
    query_embedding = embedding_model.embed_query(query)
    results = collection.query(query_embeddings=[query_embedding], n_results=top_k)
    return results.get("documents", [[]])[0] if results else [""]

# ✅ Function to Handle User Queries
def query_llama3(user_query):
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

    # Retrieve Context from ChromaDB
    retrieved_context = retrieve_context(user_query)

    # Create Message History
    messages = [
        SystemMessage(content=system_prompt),
        HumanMessage(content=f"Context: {retrieved_context}\n\nQuestion: {user_query}")
    ]

    try:
        response = chat.invoke(messages)
        memory.save_context({"input": user_query}, {"output": response.content})
        return response.content if response else "I don't have an answer for that."
    except Exception as e:
        return f"Error: {str(e)}"

# ✅ Streamlit UI for Chat
st.markdown("<style>div.stTextInput>div>div>input {text-align: right;}</style>", unsafe_allow_html=True)

# Initialize Chat History in Session State
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

# Display Chat Messages
for message in st.session_state.chat_history:
    if message["role"] == "user":
        st.markdown(f"<div style='text-align: right; background-color: #DCF8C6; padding: 10px; border-radius: 10px; margin: 5px;'>"
                    f"<b>You:</b> {message['content']}</div>", unsafe_allow_html=True)
    else:
        st.markdown(f"<div style='text-align: left; background-color: #EAEAEA; padding: 10px; border-radius: 10px; margin: 5px;'>"
                    f"<b>Deepak's Clone:</b> {message['content']}</div>", unsafe_allow_html=True)

# User Input Field
user_query = st.text_input("Type your message here and press Enter...", key="user_input")

if user_query:
    response = query_llama3(user_query)

    # Update Chat History
    st.session_state.chat_history.append({"role": "user", "content": user_query})
    st.session_state.chat_history.append({"role": "assistant", "content": response})

    # Refresh UI (Chat History)
    st.rerun()
