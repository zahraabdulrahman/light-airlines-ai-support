import streamlit as st, pandas as pd
from chatbot import chatbot_reply
from ticket_handler import handle_ticket
from models.cluster_cus import generate_cluster_message, get_customer_cluster, preprocess_customer_row
from sklearn.decomposition import PCA
import plotly.express as px

st.set_page_config(page_title="Airlines Assistant ✈️", layout="wide")

@st.cache_data
def load_customers():
    return pd.read_csv("data/expanded_customers_with_clusters.csv")

@st.cache_data
def load_flights():
    return pd.read_csv("data/flights.csv")

@st.cache_data
def preprocess_all_customers(df):
    features_list = []
    for idx, row in df.iterrows():
        feats = preprocess_customer_row(row)
        features_list.append(feats)
    return pd.concat(features_list, ignore_index=True)

@st.cache_data
def predict_clusters(df):
    clusters = []
    for idx, row in df.iterrows():
        cluster_label = get_customer_cluster(row)
        clusters.append(cluster_label)
    return clusters

customers = load_customers()
flights = load_flights()

selected_id = st.sidebar.selectbox(
    "👤 Select customer",
    customers["customer_id"].tolist()
)

customer = customers.set_index("customer_id").loc[selected_id]

tabs = st.tabs(["Chatbot", "Ticket Triage", "Alerts", "Segments"])

with tabs[0]:
    st.title("Airline Support Chatbot")
    user_input = st.text_input("Ask a question about flights, baggage, booking, etc.")

    if st.button("Send") and user_input.strip():
        with st.spinner("Getting answer..."):
            answer = chatbot_reply(user_input)
        st.markdown("**Answer:**")
        st.success(answer)

with tabs[1]:
    st.title("Ticket Triage System")
    txt = st.text_area(f"Enter {customer['name']} ticket description:")

    if st.button("Submit") and txt.strip():
        st.write("Tier:", customer['tier'])
        st.write("Preferences:", customer['preferences'])
        with st.spinner("Processing ticket..."):
            result = handle_ticket(txt)

        st.markdown("### ✏️ Summary")
        st.success(result["summary"])

        st.markdown("### 🏷️ Issue Category")
        st.info(f"{result['category']} → {result['team']} ({result['confidence']*100:.1f}%)")

        st.markdown("### 💡 Suggested Resolutions")
        suggestions = result.get("suggestions", [])
        if suggestions:
            for res in suggestions:
                st.markdown(f"- {res}")

    else:
        st.info("Please enter a ticket description above and click Submit.")

with tabs[2]:
    st.header("Generate Delay Alerts")
    #st.title("Personalize Delay Alerts")
    delayed_flights = flights[flights['delay_minutes'] > 30]  

    for _, flight in delayed_flights.iterrows():
        affected_customers = customers[customers["flight_no"] == flight["flight_no"]]

        for _, cust in affected_customers.iterrows():
            message = generate_cluster_message(cust, flight)
            st.markdown(f"**{cust['name']}** ({cust['tier']})")
            st.code(message)
            if st.button(f"Send to {cust['name']}", key=f"{cust['customer_id']}"):
                st.success(f"✅ Alert sent to {cust['name']}")

with tabs[3]:
    st.header("Customer Segments")

    df = customers.copy()

    if 'cluster_label' in df.columns and 'cluster' not in df.columns:
        df['cluster'] = df['cluster_label']

    features = preprocess_all_customers(df)

    # Apply PCA for 2D visualization
    pca = PCA(n_components=2)
    reduced = pca.fit_transform(features)
    df['pca_x'], df['pca_y'] = reduced[:, 0], reduced[:, 1]

    st.write("➤ df ready for plotting (head):", df[['pca_x','pca_y','cluster']].head())
    
    # plot clusters with interactive hover info
    fig = px.scatter(
        df,
        x="pca_x",
        y="pca_y",
        color=df["cluster"].astype(str),
        hover_data=["name", "tier", "preferences"]
    )

    st.plotly_chart(fig, use_container_width=True)
    st.markdown("Hover over points to view passenger details.")

    cluster_descriptions = {
        0: """
         **Cluster 0 – Reliable Regulars**  
        - Moderate booking frequency, low delays  
        - Mostly Silver/Gold tier, prefer SMS  
        - Window seat & priority boarding preferences  
        - Consistently satisfied — stable group
        """,
        1: """
         **Cluster 1 – High-Maintenance Frequent Flyers**  
        - Fly often but face more delays & resolution times  
        - More Platinum members, prefer Email  
        - Aisle seats, extra baggage, non-veg meals  
        - Lower satisfaction — needs proactive support
        """,
        2: """
         **Cluster 2 – Occasional Luxury Seekers**  
        - Fly less often, receive quicker support  
        - Lounge access, vegetarian meals, no strong preference  
        - Email communication  
        - Highest satisfaction — potential upsell group
        """
    }

    st.markdown("---")
    st.subheader("Segment Summaries")

    for cluster_id in sorted(cluster_descriptions.keys()):
        st.markdown(cluster_descriptions[cluster_id])