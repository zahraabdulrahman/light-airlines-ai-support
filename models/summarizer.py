from transformers import pipeline
import streamlit as st

@st.cache_resource(show_spinner=False)
def load_summarizer():
    return pipeline("summarization", model="facebook/bart-large-cnn")

summarizer = load_summarizer()

def summarize(text, min_words=12):
    if len(text.split()) <= min_words:
        return text  # It's already short — no need to summarize
    else:
        summary = summarizer("summarize: Extract only the main customer complaint or issue from this support ticket. "
        "Ignore greetings, context, or emotional tone. Write one concise sentence.\n\n"
        +text, max_length=15, min_length=5, do_sample=False)[0]['summary_text']
        return summary