from transformers import pipeline
import streamlit as st

@st.cache_resource(show_spinner=False)
def load_classifier():
    return pipeline("zero-shot-classification", model="facebook/bart-large-mnli")

classifier = load_classifier()

labels = [
    "Baggage Issues – lost or delayed luggage",
    "Flight Changes – schedule modifications or missed flights",
    "App Errors – login issues, crashes, bugs",
    "Payments – overcharges, double charges, refunds",
    "Booking Problems – issues with booking, seat selection, passenger details",
    "Cancellation – ticket cancellation or fees"
]

category_to_team = {
    "Baggage Issues": "Baggage Support Team",
    "Flight Changes": "Flight Operations",
    "App Errors": "Tech Support",
    "Payments": "Finance Team",
    "Booking Problems": "Reservation Desk",
    "Cancellation": "Customer Care"
}

def classify(text):
    result = classifier(text, candidate_labels=labels)
    predicted_label = result["labels"][0].split(" – ")[0]  # Clean label
    confidence = result["scores"][0]
    assigned_team = category_to_team.get(predicted_label, "General Support")

    return predicted_label, assigned_team, confidence

