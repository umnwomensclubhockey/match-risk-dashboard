import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

st.set_page_config(page_title="BBBS Match Risk Dashboard", layout="wide")

st.title("🧠 BBBS Match Risk & Sentiment Explorer")

# Load risk predictions
@st.cache_data

def load_data():
    return pd.read_csv("TestSet_RiskPredictions.csv")

risk_df = load_data()

# Sidebar filters
st.sidebar.header("🔍 Filter Matches")
min_days = st.sidebar.slider("Minimum Days Active", 0, int(risk_df["Note Duration (days)"].max()), 0)
show_risk = st.sidebar.checkbox("Show Only At-Risk Matches", value=False)

filtered_df = risk_df[risk_df["Note Duration (days)"] >= min_days]
if show_risk:
    filtered_df = filtered_df[filtered_df["Predicted At Risk"] == 1]

st.subheader("📋 Match Summary")
st.dataframe(filtered_df, use_container_width=True)

# Match selection
match_id = st.selectbox("Select a Match ID to View Sentiment Timeline:", filtered_df["Match ID 18Char"].unique())

# Load full note history (from training + test sets)
@st.cache_data

def load_full_notes():
    df1 = pd.read_csv("Training-Restated.xlsx - Sheet1.csv", dtype=str, low_memory=False)
    df2 = pd.read_csv("Test-Truncated-Restated.xlsx - Sheet1.csv", dtype=str, low_memory=False)
    df = pd.concat([df1, df2], ignore_index=True)
    df["Completion Date"] = pd.to_datetime(df["Completion Date"], errors="coerce")
    return df.dropna(subset=["Match Support Contact Notes"])

notes_df = load_full_notes()
notes_df = notes_df[notes_df["Match ID 18Char"] == match_id].copy()
notes_df.sort_values("Completion Date", inplace=True)

from textblob import TextBlob
notes_df["Sentiment"] = notes_df["Match Support Contact Notes"].apply(lambda x: TextBlob(x).sentiment.polarity)

# Plot timeline
st.subheader("📈 Sentiment Over Time")
fig, ax = plt.subplots(figsize=(10, 4))
sns.lineplot(data=notes_df, x="Completion Date", y="Sentiment", marker="o", ax=ax)
plt.axhline(0, linestyle="--", color="gray")
plt.title(f"Sentiment Timeline for Match {match_id}")
plt.ylabel("Polarity (-1 to 1)")
plt.xlabel("Date")
st.pyplot(fig)

st.markdown("---")
st.caption("Built with 💙 for MinneMUDAC by the DataBells")
