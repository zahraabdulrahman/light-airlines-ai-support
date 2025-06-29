from models.summarizer import summarize
from models.classifier import classify
from utils.translator import translate_to_english
from models.retriever import retrieve_resolution

def handle_ticket(raw_text):
    text = translate_to_english(raw_text) # if not english, gets translated. otherwise return text
    summary = summarize(text)
    category, team, confidence = classify(summary)
    suggestions = retrieve_resolution(summary, team)

    return {
        "summary": summary,
        "category": category,
        "team": team,
        "confidence": confidence,
        "suggestions": suggestions
    }