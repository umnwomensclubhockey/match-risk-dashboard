import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from textblob import TextBlob
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import LabelEncoder
from wordcloud import WordCloud
from datetime import datetime

# Optional survival analysis
try:
    from lifelines import KaplanMeierFitter
    LIFELINES_AVAILABLE = True
except ImportError:
    LIFELINES_AVAILABLE = False

st.set_page_config(page_title="BBBS Match Risk Dashboard", layout="wide")
st.title("🧠 BBBS Match Risk & Sentiment Explorer")
st.markdown("""
Welcome to the **Big Brothers Big Sisters Twin Cities Dashboard**, designed for the 2025 MinneMUDAC Challenge.

This interactive app explores match longevity, closure reasons, emotional patterns, and predictive factors of success.
Use the filters and visuals to dive into insights and guide future mentoring success.
""")

# -------------------- LOAD DATA --------------------
@st.cache_data
def load_data():
    df = pd.read_csv("TestSet_RiskPredictions.csv")
    df.dropna(subset=["Note Duration (days)", "Match Length"], inplace=True)
    return df

risk_df = load_data()

# -------------------- SIDEBAR FILTERS --------------------
st.sidebar.header("🔍 Filter Matches")
min_days = st.sidebar.slider("Minimum Days Active", 0, int(risk_df["Note Duration (days)"].max()), 0)
show_risk = st.sidebar.checkbox("Show Only At-Risk Matches", value=False)

filtered_df = risk_df[risk_df["Note Duration (days)"] >= min_days]
if show_risk:
    filtered_df = filtered_df[filtered_df["Predicted At Risk"] == 1]

# -------------------- MATCH SUMMARY --------------------
st.subheader("📋 Match Summary Table")
st.dataframe(filtered_df, use_container_width=True)

# -------------------- MATCH SENTIMENT TIMELINE --------------------
st.subheader("📈 Emotional Tone Timeline for Individual Matches")
match_id = st.selectbox("Choose a Match ID to Track Emotional Shifts:", filtered_df["Match ID 18Char"].unique())

@st.cache_data
def load_full_notes():
    df1 = pd.read_csv("Training-Restated.xlsx - Sheet1.csv", dtype=str, low_memory=False)
    df2 = pd.read_csv("Test-Truncated-Restated.xlsx - Sheet1.csv", dtype=str, low_memory=False)
    df = pd.concat([df1, df2], ignore_index=True)
    df["Completion Date"] = pd.to_datetime(df["Completion Date"], errors="coerce")
    return df.dropna(subset=["Match Support Contact Notes"])

notes_df = load_full_notes()
match_notes = notes_df[notes_df["Match ID 18Char"] == match_id].copy()
match_notes.sort_values("Completion Date", inplace=True)
match_notes["Sentiment"] = match_notes["Match Support Contact Notes"].apply(lambda x: TextBlob(x).sentiment.polarity)

fig, ax = plt.subplots(figsize=(10, 4))
sns.lineplot(data=match_notes, x="Completion Date", y="Sentiment", marker="o", ax=ax)
plt.axhline(0, linestyle="--", color="gray")
plt.title(f"Sentiment Timeline for Match {match_id}")
plt.ylabel("Polarity (-1 to 1)")
plt.xlabel("Date")
st.pyplot(fig)

# -------------------- SENTIMENT SHIFT METRIC --------------------
st.subheader("📉 Early vs. Late Sentiment Shift")
if len(match_notes) >= 2:
    first_half = match_notes.iloc[:len(match_notes)//2]["Sentiment"].mean()
    second_half = match_notes.iloc[len(match_notes)//2:]["Sentiment"].mean()
    shift_value = round(second_half - first_half, 3)
    st.metric(label="Sentiment Change (End - Start)", value=shift_value)
    if shift_value < -0.1:
        st.warning("⚠️ Sentiment declined over time. This match may be at risk.")
    elif shift_value > 0.1:
        st.success("💡 Sentiment improved over time!")
    else:
        st.info("ℹ️ Sentiment remained relatively stable.")

# -------------------- TEXT ANALYSIS: WORD CLOUD --------------------
st.subheader("☁️ Common Themes in Match Notes")
all_notes = notes_df["Match Support Contact Notes"].astype(str)
text_blob = " ".join(all_notes)
wordcloud = WordCloud(width=800, height=400, background_color='white').generate(text_blob)
fig, ax = plt.subplots(figsize=(15, 7))
ax.imshow(wordcloud, interpolation='bilinear')
ax.axis("off")
plt.title("Most Frequent Words in Match Support Notes")
st.pyplot(fig)

# -------------------- FEATURE IMPACT VISUALIZATIONS --------------------
st.subheader("📊 Match Length Insights by Group")
if "Year" not in risk_df.columns:
    risk_df["Completion Date"] = pd.to_datetime(risk_df["Completion Date"], errors="coerce")
    risk_df["Year"] = risk_df["Completion Date"].dt.year

group_features = [
    ("Program Type", "Match Length by Program Type"),
    ("Same Gender", "Match Length by Same Gender"),
    ("Year", "Match Length Over Time")
]
for col, title in group_features:
    if col in risk_df.columns:
        fig, ax = plt.subplots()
        sns.boxplot(data=risk_df, x=col, y="Match Length", ax=ax)
        plt.xticks(rotation=45)
        plt.title(title)
        st.pyplot(fig)

if "Closure Reason" in risk_df.columns:
    fig, ax = plt.subplots(figsize=(10, 5))
    sns.countplot(data=risk_df, y="Closure Reason", order=risk_df["Closure Reason"].value_counts().index, ax=ax)
    plt.title("Top Closure Reasons")
    st.pyplot(fig)

# -------------------- PREDICTED MATCH LENGTH HISTOGRAM --------------------
if 'Predicted Match Length' in risk_df.columns:
    st.subheader("🔮 Distribution of Predicted Match Lengths")
    fig, ax = plt.subplots()
    sns.histplot(risk_df['Predicted Match Length'], bins=20, kde=True, ax=ax)
    ax.set_title("Forecasted Match Durations")
    ax.set_xlabel("Months")
    st.pyplot(fig)

# -------------------- RANDOM FOREST PREDICTION MODEL --------------------
st.subheader("🤖 Match Length Prediction Feature Importances")
@st.cache_data
def train_model():
    df = pd.read_csv("Training-Restated.xlsx - Sheet1.csv", low_memory=False)
    df = df.dropna(subset=["Match Length", "Big Age", "Little Gender", "Big Gender", "Program Type"])
    df["Same Gender"] = df["Big Gender"] == df["Little Gender"]
    le_cols = ["Little Gender", "Big Gender", "Program Type", "Same Gender"]
    for col in le_cols:
        df[col] = LabelEncoder().fit_transform(df[col].astype(str))
    features = ["Big Age"] + le_cols
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(df[features], df["Match Length"])
    importances = pd.Series(model.feature_importances_, index=features).sort_values(ascending=False)
    return model, importances

model, importances = train_model()
fig, ax = plt.subplots()
importances.plot(kind='barh', ax=ax)
plt.title("Top Features for Match Length Prediction")
st.pyplot(fig)

# -------------------- SURVIVAL ANALYSIS --------------------
if LIFELINES_AVAILABLE:
    st.subheader("📉 Kaplan-Meier Match Survival Curve")
    df = pd.read_csv("Training-Restated.xlsx - Sheet1.csv", low_memory=False)
    df = df.dropna(subset=["Match Length", "Stage"])
    df["Event"] = df["Stage"] == "Closed"
    kmf = KaplanMeierFitter()
    kmf.fit(df["Match Length"], event_observed=df["Event"])
    fig, ax = plt.subplots()
    kmf.plot_survival_function(ax=ax)
    ax.set_title("Probability of Match Staying Active Over Time")
    ax.set_xlabel("Months")
    ax.set_ylabel("Survival Probability")
    st.pyplot(fig)
else:
    st.warning("Install 'lifelines' to view survival analysis (pip install lifelines).")

# -------------------- MATCH COMPARISON PANEL --------------------
st.subheader("🧩 Compare Two Matches")
match_ids = filtered_df["Match ID 18Char"].unique()
col1, col2 = st.columns(2)
with col1:
    compare1 = st.selectbox("Match A", match_ids, key="compare1")
with col2:
    compare2 = st.selectbox("Match B", match_ids, key="compare2")

for match in [compare1, compare2]:
    st.markdown(f"### 📊 Summary for {match}")
    subset = notes_df[notes_df["Match ID 18Char"] == match].copy()
    subset.sort_values("Completion Date", inplace=True)
    subset["Sentiment"] = subset["Match Support Contact Notes"].apply(lambda x: TextBlob(x).sentiment.polarity)
    if len(subset):
        fig, ax = plt.subplots(figsize=(6, 2.5))
        sns.lineplot(data=subset, x="Completion Date", y="Sentiment", ax=ax, marker="o")
        plt.axhline(0, linestyle="--", color="gray")
        plt.title(f"Sentiment for Match {match}")
        st.pyplot(fig)
    else:
        st.info(f"No sentiment data available for match {match}.")

# -------------------- EXPORT DATA OPTION --------------------
st.subheader("📁 Export Filtered Match Data")
st.download_button("Download Filtered Data as CSV", data=filtered_df.to_csv(index=False), file_name="filtered_matches.csv", mime="text/csv")

# -------------------- SURVEY QUESTION IDEAS --------------------
st.subheader("📋 Suggested Call Check-In Questions")
st.markdown("""
These dropdown-style questions can be asked during support calls to identify at-risk matches early:

- **How frequently do you and your Little meet in a month?**  
  (Weekly, Biweekly, Monthly, Rarely)
- **How would you describe your relationship with your Little?**  
  (Very Strong, Good, Fair, Weak)
- **Have there been any scheduling conflicts or missed meetings recently?**  
  (Yes, No)
- **Are there shared interests between you and your Little?**  
  (Yes, Some, No)
- **Is your Little facing any life transitions (e.g. school changes, family changes)?**  
  (Yes, No)
""")

# -------------------- EMOTIONAL SHIFT EXPLANATION --------------------
st.subheader("💬 How We Measure Emotional Tone")
st.markdown("""
We analyze the **Match Support Contact Notes** using TextBlob, a natural language processing library.
Each note is assigned a **polarity score** from -1 (very negative) to +1 (very positive).

📉 Sudden drops in sentiment may reflect conflict, disengagement, or emotional difficulty.  
📈 Positive sentiment trends often indicate a strong or improving relationship.

These patterns help staff anticipate problems before formal closure occurs.
""")

st.markdown("---")
st.caption("Built with 💙 for MinneMUDAC by the DataBells")



