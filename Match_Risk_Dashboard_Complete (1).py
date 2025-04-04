
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
import warnings

# Optional survival analysis
try:
    from lifelines import KaplanMeierFitter
    LIFELINES_AVAILABLE = True
except ImportError:
    LIFELINES_AVAILABLE = False

warnings.filterwarnings("ignore")

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
    try:
        df = pd.read_csv("test_risk_predictions_FIXED.csv")
        df.dropna(subset=["Note Duration (days)", "Match Length"], inplace=True)
        return df
    except Exception as e:
        st.error(f"Failed to load match data: {e}")
        return pd.DataFrame()

risk_df = load_data()
if risk_df.empty:
    st.stop()

# -------------------- SIDEBAR FILTERS --------------------
st.sidebar.header("🔍 Filter Matches")
min_days = st.sidebar.slider("Minimum Days Active", 0, int(risk_df["Note Duration (days)"].max()), 0)
show_risk = st.sidebar.checkbox("Show Only At-Risk Matches", value=False)

filtered_df = risk_df[risk_df["Note Duration (days)"] >= min_days]
if show_risk:
    filtered_df = filtered_df[filtered_df["Predicted At Risk"] == 1]

# -------------------- MATCH SUMMARY --------------------
st.subheader("📋 Match Summary Table")
if filtered_df.empty:
    st.warning("No matches meet your filter criteria.")
else:
    st.dataframe(filtered_df, use_container_width=True)

# -------------------- SENTIMENT TIMELINE --------------------
st.subheader("📈 Emotional Tone Timeline for Individual Matches")
if not filtered_df.empty:
    match_id = st.selectbox("Choose a Match ID to Track Emotional Shifts:", filtered_df["Match ID 18Char"].unique())

    @st.cache_data
    def load_full_notes():
        try:
            df1 = pd.read_csv("https://drive.google.com/uc?id=1a_Y8_7G-KLtxHpPM5lnYYR4_CNMg-Sxy", dtype=str, low_memory=False)
            df2 = pd.read_csv("https://drive.google.com/uc?id=16AJpW7UF0avDtjZ7Lzdte3p6NpyBzp8Q", dtype=str, low_memory=False)
            df = pd.concat([df1, df2], ignore_index=True)
            df["Completion Date"] = pd.to_datetime(df["Completion Date"], errors="coerce")
            return df.dropna(subset=["Match Support Contact Notes"])
        except Exception as e:
            st.error(f"Error loading notes: {e}")
            return pd.DataFrame()

    notes_df = load_full_notes()
    match_notes = notes_df[notes_df["Match ID 18Char"] == match_id].copy()
    match_notes.sort_values("Completion Date", inplace=True)
    match_notes["Sentiment"] = match_notes["Match Support Contact Notes"].apply(lambda x: TextBlob(x).sentiment.polarity)

    if not match_notes.empty:
        fig, ax = plt.subplots(figsize=(10, 4))
        sns.lineplot(data=match_notes, x="Completion Date", y="Sentiment", marker="o", ax=ax)
        plt.axhline(0, linestyle="--", color="gray")
        plt.title(f"Sentiment Timeline for Match {match_id}")
        plt.ylabel("Polarity (-1 to 1)")
        plt.xlabel("Date")
        st.pyplot(fig)
    else:
        st.info("No sentiment data available for this match.")

# Additional dashboard sections appended below

# -------------------- SENTIMENT SHIFT METRIC --------------------
if not match_notes.empty:
    st.subheader("📉 Early vs. Late Sentiment Shift")
    st.markdown("This compares average sentiment in the first half vs. second half of a match's notes.")
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
            st.info("ℹ️ Sentiment remained stable.")

# -------------------- PREDICTED MATCH LENGTH HISTOGRAM --------------------
if 'Predicted Match Length' in risk_df.columns:
    st.subheader("🔮 Distribution of Predicted Match Lengths")
    fig, ax = plt.subplots()
    sns.histplot(risk_df['Predicted Match Length'], bins=20, kde=True, ax=ax)
    ax.set_title("Forecasted Match Durations")
    ax.set_xlabel("Months")
    st.pyplot(fig)

# -------------------- MATCH COMPARISON PANEL --------------------
if not notes_df.empty:
    st.subheader("🧩 Compare Two Matches")
    match_ids = filtered_df["Match ID 18Char"].unique()
    if len(match_ids) >= 2:
        col1, col2 = st.columns(2)
        with col1:
            compare1 = st.selectbox("Match A", match_ids, key="compare1")
        with col2:
            compare2 = st.selectbox("Match B", match_ids, key="compare2")
        for match in [compare1, compare2]:
            st.markdown(f"### 📊 Sentiment Summary for {match}")
            subset = notes_df[notes_df["Match ID 18Char"] == match].copy()
            subset.sort_values("Completion Date", inplace=True)
            if not subset.empty:
                subset["Sentiment"] = subset["Match Support Contact Notes"].apply(lambda x: TextBlob(x).sentiment.polarity)
                fig, ax = plt.subplots(figsize=(6, 2.5))
                sns.lineplot(data=subset, x="Completion Date", y="Sentiment", ax=ax, marker="o")
                plt.axhline(0, linestyle="--", color="gray")
                plt.title(f"Sentiment for Match {match}")
                st.pyplot(fig)

# -------------------- EXPORT DATA OPTION --------------------
if not filtered_df.empty:
    st.subheader("📁 Export Filtered Match Data")
    st.download_button("Download Filtered Data as CSV", data=filtered_df.to_csv(index=False),
                       file_name="filtered_matches.csv", mime="text/csv")

# -------------------- ADVANCED MODELS: RANDOM FOREST & LDA --------------------
try:
    st.subheader("🧠 Advanced Predictive Modeling")

    from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import mean_squared_error

    @st.cache_data
    def build_models():
        df = pd.read_csv("training_data.csv")
        df = df.dropna(subset=["Match Length", "Big Age", "Little Gender", "Big Gender", "Program Type", "Stage"])
        df["Same Gender"] = df["Big Gender"] == df["Little Gender"]
        df["Closed Successfully"] = df["Closure Reason"].fillna("").str.contains("Successful", case=False)
        for col in ["Little Gender", "Big Gender", "Program Type"]:
            df[col] = LabelEncoder().fit_transform(df[col].astype(str))
        features = ["Big Age", "Little Gender", "Big Gender", "Program Type", "Same Gender"]
        X = df[features]
        y_reg = df["Match Length"]
        y_class = df["Closed Successfully"].astype(int)
        X_train, X_test, y_train, y_test = train_test_split(X, y_reg, test_size=0.2, random_state=42)
        model_rf = RandomForestRegressor(n_estimators=200, max_depth=10, random_state=42)
        model_rf.fit(X_train, y_train)
        rmse = round(np.sqrt(mean_squared_error(y_test, model_rf.predict(X_test))), 2)
        lda = LinearDiscriminantAnalysis()
        X_lda = lda.fit_transform(X, y_class)
        return model_rf, lda, features, rmse, X_lda, y_class

    model_rf, lda, features, rmse, X_lda, y_class = build_models()
    st.markdown(f"**📈 Random Forest RMSE:** {rmse} months")
    fig, ax = plt.subplots()
    pd.Series(model_rf.feature_importances_, index=features).sort_values().plot(kind="barh", ax=ax)
    ax.set_title("Feature Importance in Match Length Prediction")
    st.pyplot(fig)

    st.subheader("🎯 LDA: Match Closure Class Separation")
    fig, ax = plt.subplots()
    ax.scatter(X_lda[:, 0], np.zeros_like(X_lda[:, 0]), c=y_class, cmap="coolwarm", edgecolors="k")
    ax.set_yticks([])
    ax.set_xlabel("LDA Component 1")
    ax.set_title("LDA Projection: Match Closure Outcome")
    st.pyplot(fig)

except Exception as e:
    st.error(f"⚠️ Error in model training or visualization: {e}")

# -------------------- SURVEY QUESTIONS --------------------
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
