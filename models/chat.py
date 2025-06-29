from transformers import pipeline
from utils.embedding_utils import load_or_create_index
import numpy as np
import streamlit as st

@st.cache_resource
def load_resources():
    chatbot_index, chatbot_df, chatbot_embed_model = load_or_create_index(
        "chatbot_db.csv",
        "chatbot_index.faiss",
        ["question", "answer"]
    )
    qa_pipeline = pipeline("text2text-generation", model="google/flan-t5-large")
    return chatbot_index, chatbot_df, chatbot_embed_model, qa_pipeline

chatbot_index, chatbot_df, chatbot_embed_model, qa_pipeline = load_resources()


# Similar resolution retriever
def get_similar_resolution(query, k=2):
    query_vector = np.array(chatbot_embed_model.encode([query])).astype("float32")
    D, I = chatbot_index.search(query_vector, k)

    results = []
    for idx in I[0]:
        ticket = chatbot_df.iloc[idx]
        results.append({
            'id': ticket['id'],
            'question': ticket['question'],
            'answer': ticket['answer'],
        })
    return results

# Chatbot response generator
def chatbot_respond(user_input):
    context_chunks = get_similar_resolution(user_input)
    context_text = "\n".join([chunk['answer'].strip() for chunk in context_chunks])

    prompt = f"""
    You are a helpful airline support assistant.

    Using only the information below, answer the user's question clearly and politely, including any important policies or instructions.  
    If the information is not available, reply exactly: "I'm sorry, I don’t have that information right now. Let me escalate this to a support agent."

    Past Resolutions:
    {context_text}

    User question: {user_input}

    Answer:
    """
    response = qa_pipeline(prompt.strip(), max_new_tokens=256, num_beams=4)[0]["generated_text"]
    return response.strip()