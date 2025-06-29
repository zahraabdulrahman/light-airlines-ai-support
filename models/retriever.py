from utils.embedding_utils import load_or_create_index
import numpy as np
import streamlit as st

@st.cache_resource
def load_ticket_resources():
    return load_or_create_index(
        "ticket_db.csv",
        "ticket_index.faiss",
        ["issue_summary", "resolution_summary"]
    )

ticket_index, ticket_df, ticket_embed_model = load_ticket_resources()


def get_similar_resolution(query, k=3):
    # Encode the query
    query_vector = ticket_embed_model.encode([query])
    
    # Search in FAISS index
    D, I = ticket_index.search(np.array(query_vector), k)

    results = []
    for idx in I[0]:
        ticket = ticket_df.iloc[idx]
        results.append({
            'ticket_id': ticket['ticket_id'],
            'issue': ticket['issue_summary'],
            'resolution': ticket['resolution_summary'],
            'team': ticket['team'],
            'tags': ticket['tags']
        })
    return results

def retrieve_resolution(summary, team=None):
    # Retrieve top-k similar tickets
    similar_tickets = get_similar_resolution(summary)

    if team:
        # Filter tickets that match the same team
        filtered = [t for t in similar_tickets if t['team'].lower() == team.lower()]
        if filtered:
            similar_tickets = filtered  # Use team-matched results if available

    # Extract and return the top resolution summaries
    resolutions = [ticket['resolution'] for ticket in similar_tickets]
    return resolutions


