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
    df = pd.read_csv("test_risk_predictions_FIXED.csv")
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
    df1 = pd.read_csv("https://drive.google.com/uc?id=1a_Y8_7G-KLtxHpPM5lnYYR4_CNMg-Sxy", dtype=str, low_memory=False)
    df2 = pd.read_csv("https://drive.google.com/uc?id=16AJpW7UF0avDtjZ7Lzdte3p6NpyBzp8Q", dtype=str, low_memory=False)
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
st.markdown("""
This metric compares the average sentiment from the first half of a match’s support notes
with the second half to assess how the tone of the relationship has evolved.
""")
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

# -------------------- PREDICTED MATCH LENGTH HISTOGRAM --------------------
st.subheader("🔮 Distribution of Predicted Match Lengths")
st.markdown("""
This chart shows how long matches are predicted to last using our machine learning model.
It gives an overview of how sustainable most matches are based on their attributes.
""")
if 'Predicted Match Length' in risk_df.columns:
    fig, ax = plt.subplots()
    sns.histplot(risk_df['Predicted Match Length'], bins=20, kde=True, ax=ax)
    ax.set_title("Forecasted Match Durations")
    ax.set_xlabel("Months")
    st.pyplot(fig)

# -------------------- MATCH COMPARISON PANEL --------------------
st.subheader("🧩 Compare Two Matches")
st.markdown("""
This tool allows coordinators to visually compare the sentiment trends of two separate matches.
It is useful for identifying subtle differences in relationship development.
""")
match_ids = filtered_df["Match ID 18Char"].unique()
col1, col2 = st.columns(2)
with col1:
    compare1 = st.selectbox("Match A", match_ids, key="compare1")
with col2:
    compare2 = st.selectbox("Match B", match_ids, key="compare2")

for match in [compare1, compare2]:
    st.markdown(f"### 📊 Sentiment Summary for {match}")
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
st.markdown("""
Download the matches currently visible in the dashboard as a CSV file.
This is helpful for offline analysis or to share targeted lists of at-risk matches with coordinators.
""")
st.download_button("Download Filtered Data as CSV", data=filtered_df.to_csv(index=False), file_name="filtered_matches.csv", mime="text/csv")

# -------------------- ADVANCED MODELS: RANDOM FOREST & LDA --------------------
st.subheader("🧠 Advanced Predictive Modeling")
st.markdown("""
We applied **Random Forest Regression** to predict match length using demographic and match attributes.
We also included **Linear Discriminant Analysis (LDA)** to visually explore how features separate matches that
closed successfully vs. unsuccessfully.
""")

from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, accuracy_score

@st.cache_data
def build_models():
    df = pd.read_csv("training_data.csv")
    df = df.dropna(subset=["Match Length", "Big Age", "Little Gender", "Big Gender", "Program Type", "Stage"])
    df["Same Gender"] = df["Big Gender"] == df["Little Gender"]
    df["Closed Successfully"] = df["Closure Reason"].fillna("").str.contains("Successful", case=False)

    le = LabelEncoder()
    for col in ["Little Gender", "Big Gender", "Program Type"]:
        df[col] = le.fit_transform(df[col].astype(str))

    features = ["Big Age", "Little Gender", "Big Gender", "Program Type", "Same Gender"]
    X = df[features]
    y_reg = df["Match Length"]
    y_class = df["Closed Successfully"].astype(int)

    X_train, X_test, y_train, y_test = train_test_split(X, y_reg, test_size=0.2, random_state=42)
    model_rf = RandomForestRegressor(n_estimators=200, max_depth=10, random_state=42)
    model_rf.fit(X_train, y_train)
    y_pred = model_rf.predict(X_test)
    rmse = round(np.sqrt(mean_squared_error(y_test, y_pred)), 2)

    lda = LinearDiscriminantAnalysis()
    X_lda = lda.fit_transform(X, y_class)

    return model_rf, lda, features, rmse, X_lda, y_class

model_rf, lda_model, features, rf_rmse, X_lda, y_class = build_models()

st.markdown(f"**📈 Random Forest RMSE:** {rf_rmse} months")
fig, ax = plt.subplots()
pd.Series(model_rf.feature_importances_, index=features).sort_values().plot(kind="barh", ax=ax)
ax.set_title("Feature Importance in Match Length Prediction")
st.pyplot(fig)

st.subheader("🎯 LDA: Match Closure Class Separation")
st.markdown("""
This plot shows how well the input features can distinguish matches that were closed successfully vs. not.
Each point represents a match projected onto the LDA axis of best class separation.
""")
fig, ax = plt.subplots()
ax.scatter(X_lda[:, 0], np.zeros_like(X_lda[:, 0]), c=y_class, cmap="coolwarm", edgecolors="k")
ax.set_yticks([])
ax.set_title("LDA Projection: Match Closure Outcome")
ax.set_xlabel("LDA Component 1")
st.pyplot(fig)

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
