import joblib
import pandas as pd
import streamlit as st 

@st.cache_resource
def load_models():
    scaler = joblib.load("artifacts/scaler.joblib")
#    kmeans = joblib.load("artifacts/kmeans.joblib")
    return scaler

scaler = load_models()

EXPECTED_FEATURES = [
    'tier_Gold', 'tier_Platinum', 'tier_Silver',
    'contact_Email', 'contact_SMS',
    'Aisle seat', 'Extra baggage', 'Lounge access', 'No preference',
    'Non-veg', 'Priority boarding', 'Vegetarian meal', 'Window seat',
    'avg_booking_frequency_scaled', 'avg_flight_delay_scaled',
    'last_ticket_resolution_time_scaled', 'satisfaction_score_scaled'
]


def preprocess_customer_row(row):
    #One-hot encode tier, contact_method, preferences
    tier_df = pd.get_dummies(pd.Series([row["tier"]]), prefix="tier")
    contact_df = pd.get_dummies(pd.Series([row["contact_method"]]), prefix="contact")
    
    prefs = ";".join([p.strip() for p in str(row["preferences"]).split(";") if p.strip()])
    pref_df = pd.Series([prefs]).str.get_dummies(sep=";")
    
    # Scale numerical features
    num = scaler.transform([[row["avg_booking_frequency"],
                             row["avg_flight_delay"],
                             row["last_ticket_resolution_time"],
                             row["satisfaction_score"]]])
    
    num_df = pd.DataFrame(num, columns=[
        "avg_booking_frequency_scaled", "avg_flight_delay_scaled",
        "last_ticket_resolution_time_scaled", "satisfaction_score_scaled"
    ])
    
    # Combine all features
    feats = pd.concat([tier_df, contact_df, pref_df, num_df], axis=1)
    
    # Add missing columns as zeros
    for col in EXPECTED_FEATURES:
        if col not in feats.columns:
            feats[col] = 0
    
    #Ensure correct column order
    feats = feats[EXPECTED_FEATURES]
    
    # IMPORTANT: Reset index and ensure numeric dtype
    feats = feats.astype(float)
    feats.reset_index(drop=True, inplace=True)
    
    return feats

def get_customer_cluster(customer_row):
    return customer_row.get("cluster_label", None)

def generate_cluster_message(customer, flight):

    cluster_label = get_customer_cluster(customer)

    base_msg = f"Your flight {flight['flight_no']} from {flight['origin']} to {flight['dest']} " \
               f"scheduled at {flight['scheduled_time']} is delayed by {flight['delay_minutes']} minutes."

    # Tier-based personalization
    perks = {
        "Silver": "priority boarding and standard support.",
        "Gold": "priority boarding, extra baggage allowance, and fast track support.",
        "Platinum": "lounge access, fast track, and flexible rebooking."
    }
    tier_msg = f"As a {customer['tier']} member, you're eligible for {perks.get(customer['tier'], 'standard support')}"

    # Preference
    pref_msg = ""
    if customer['preferences'] and customer['preferences'] != "No preference":
        pref_msg = f" Your preferences: {customer['preferences']}."

    # Clustering-based personalization
    cluster_messages = {
        0: "We appreciate your loyalty and are here to ensure a smooth experience.",
        1: "We know delays are frustrating. We're working on improvements and appreciate your patience.",
        2: "Thanks for flying with us. Your premium support options are ready if you need help."
    }
    cluster_msg = f" {cluster_messages.get(cluster_label, '')}"

    return f"{base_msg} {tier_msg}{pref_msg}{cluster_msg}"
