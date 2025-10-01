# dashboard.py

import streamlit as st
import pandas as pd
import json
from collections import Counter

LOG_FILE = "analytics.log"

st.set_page_config(page_title="SentioBot Analytics", page_icon="📊", layout="wide")

st.title("📊 SentioBot Analytics Dashboard")

def load_data():
    """Loads and parses the JSON log file."""
    logs = []
    try:
        with open(LOG_FILE, "r", encoding="utf-8") as f:
            for line in f:
                try:
                    logs.append(json.loads(line))
                except json.JSONDecodeError:
                    continue # Skip malformed lines
    except FileNotFoundError:
        st.warning(f"Log file not found at '{LOG_FILE}'. Please run the chatbot first to generate logs.")
        return pd.DataFrame()

    # Create a DataFrame from the initial logs
    df = pd.DataFrame([log for log in logs if 'feedback' not in log])
    
    # Process feedback logs separately
    feedback_logs = {log['interaction_id']: log['feedback'] for log in logs if 'feedback' in log}
    
    # Map feedback to the main DataFrame
    df['feedback'] = df['interaction_id'].map(feedback_logs).fillna(0).astype(int)
    
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    return df

data = load_data()

if not data.empty:
    st.sidebar.header("Filters")
    user_filter = st.sidebar.multiselect("Filter by User ID:", options=data['user_id'].unique(), default=data['user_id'].unique())
    
    filtered_data = data[data['user_id'].isin(user_filter)]

    # --- Key Metrics ---
    st.header("Key Metrics")
    total_queries = len(filtered_data)
    positive_feedback = (filtered_data['feedback'] == 1).sum()
    negative_feedback = (filtered_data['feedback'] == -1).sum()

    col1, col2, col3 = st.columns(3)
    col1.metric("Total Queries", total_queries)
    col2.metric("👍 Positive Feedback", f"{positive_feedback}")
    col3.metric("👎 Negative Feedback", f"{negative_feedback}")

    # --- Most Frequent Questions ---
    st.header("Most Frequent Questions")
    question_counts = filtered_data['user_query'].value_counts().nlargest(10)
    st.bar_chart(question_counts)
    
    # --- Queries with No Documents Found ---
    st.header("Queries with No Documents Found")
    no_docs_df = filtered_data[filtered_data['retrieved_docs'].apply(lambda x: not x)]
    if not no_docs_df.empty:
        st.dataframe(no_docs_df[['timestamp', 'user_id', 'user_query']], use_container_width=True)
    else:
        st.success("No queries resulted in zero documents found. Great retriever performance!")

    # --- Detailed Interaction Log ---
    with st.expander("View Full Interaction Log", expanded=False):
        st.dataframe(filtered_data)
else:
    st.info("No data to display. Start a conversation with SentioBot to generate logs.")