import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from textblob import TextBlob
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import export_text
import numpy as np

st.set_page_config(page_title="BBBS Match Risk Dashboard", layout="wide", page_icon="🌼")

st.title("BBBS Match Risk & Sentiment Explorer – Team DataBells")

# Load risk predictions
@st.cache_data
def load_data():
    return pd.read_csv("test_risk_predictions.csv")

risk_df = load_data()

# Sidebar filters
st.sidebar.header("Filter Matches")
min_days = st.sidebar.slider("Minimum Days Active", 0, int(risk_df["Note Duration (days)"].max()), 0)
show_risk = st.sidebar.checkbox("Show Only At-Risk Matches", value=False)

filtered_df = risk_df[risk_df["Note Duration (days)"] >= min_days]
if show_risk:
    filtered_df = filtered_df[filtered_df["Predicted At Risk"] == 1]

st.subheader("Match Summary")
st.dataframe(filtered_df, use_container_width=True, hide_index=True)

# Match selection
filtered_df = filtered_df.reset_index(drop=True)
filtered_df["Match Label"] = filtered_df.index.map(lambda i: f"Match #{i+1}")
match_lookup = dict(zip(filtered_df["Match Label"], filtered_df["Match ID 18Char"]))
match_id = st.selectbox("Select a Match to View Sentiment Timeline:", filtered_df["Match Label"])
match_id = match_lookup[match_id]

# Load full note history (from training + test sets)
@st.cache_data
def load_full_notes():
    df1 = pd.read_csv("https://drive.google.com/uc?id=1Wptg4A4-6Hdh_IuId3wI6tv8rUl5hGJH", dtype=str, low_memory=False)
    df2 = pd.read_csv("test_data.csv", dtype=str, low_memory=False)
    df = pd.concat([df1, df2], ignore_index=True)
    df["Completion Date"] = pd.to_datetime(df["Completion Date"], errors="coerce")
    return df.dropna(subset=["Match Support Contact Notes"])

notes_df = load_full_notes()
notes_df = notes_df[notes_df["Match ID 18Char"] == match_id].copy()
notes_df.sort_values("Completion Date", inplace=True)
notes_df["Sentiment"] = notes_df["Match Support Contact Notes"].apply(lambda x: TextBlob(x).sentiment.polarity)

with st.expander("Sentiment Timeline for Selected Match", expanded=True):
    fig, ax = plt.subplots(figsize=(10, 4))
    sns.lineplot(data=notes_df, x="Completion Date", y="Sentiment", marker="o", ax=ax)
    plt.axhline(0, linestyle="--", color="gray")
    plt.title(f"Sentiment Timeline for Match {match_id}")
    plt.ylabel("Polarity (-1 to 1)")
    plt.xlabel("Date")
    st.pyplot(fig)

# Feature Engineering
if "Big Gender" in risk_df.columns and "Little Gender" in risk_df.columns:
    risk_df["Same Gender"] = (risk_df["Big Gender"] == risk_df["Little Gender"]).astype(int)
else:
    risk_df["Same Gender"] = 0
if "Big Hobbies" in risk_df.columns and "Little Hobbies" in risk_df.columns:
    risk_df["Shared Hobby"] = risk_df.apply(lambda row: int(any(hobby in str(row.get("Big Hobbies", "")).lower() for hobby in str(row.get("Little Hobbies", "")).lower().split(","))), axis=1)
else:
    risk_df["Shared Hobby"] = 0
if "Big Age" in risk_df.columns and "Little Age" in risk_df.columns:
    risk_df["Age Difference"] = abs(pd.to_numeric(risk_df["Big Age"], errors='coerce') - pd.to_numeric(risk_df["Little Age"], errors='coerce'))
else:
    risk_df["Age Difference"] = 0
risk_df["Age Difference"] = risk_df["Age Difference"].fillna(0)
if "Big Ethnicity" in risk_df.columns and "Little Ethnicity" in risk_df.columns:
    risk_df["Same Ethnicity"] = (risk_df["Big Ethnicity"] == risk_df["Little Ethnicity"]).astype(int)
else:
    risk_df["Same Ethnicity"] = 0
if "Big Zip" in risk_df.columns and "Little Zip" in risk_df.columns:
    risk_df["Same Zip Code"] = (risk_df["Big Zip"] == risk_df["Little Zip"]).astype(int)
else:
    risk_df["Same Zip Code"] = 0
if "Big Years Volunteering" in risk_df.columns:
    risk_df["Big Years Volunteering"] = pd.to_numeric(risk_df["Big Years Volunteering"], errors='coerce').fillna(0)
else:
    risk_df["Big Years Volunteering"] = 0
if "Little Match Count" in risk_df.columns:
    risk_df["Little Match Count"] = pd.to_numeric(risk_df["Little Match Count"], errors='coerce').fillna(0)
else:
    risk_df["Little Match Count"] = 0

features = ["Early", "Late", "Shift", "Note Duration (days)", "Same Gender", "Shared Hobby", "Age Difference", "Same Ethnicity", "Same Zip Code", "Big Years Volunteering", "Little Match Count"]
model_df = risk_df.dropna(subset=features + ["Predicted At Risk"])
X = model_df[features]
y = model_df["Predicted At Risk"]
clf = RandomForestClassifier(n_estimators=100, random_state=42)
clf.fit(X, y)

with st.expander("Feature Importance and Model Insights", expanded=True):
    importances = clf.feature_importances_
    imp_df = pd.DataFrame({"Feature": features, "Importance": importances}).sort_values(by="Importance", ascending=False)
    fig2, ax2 = plt.subplots(figsize=(8, 4))
    sns.barplot(data=imp_df, x="Importance", y="Feature", ax=ax2)
    plt.title("Random Forest Feature Importance")
    st.pyplot(fig2)

    tree_rules = export_text(clf.estimators_[0], feature_names=features)
    st.code(tree_rules, language="text")

with st.expander("Manual Risk Prediction", expanded=False):
    st.markdown("Answer the questions below to evaluate a hypothetical match:")
    input_early = st.slider("1. What was the early sentiment?", -1.0, 1.0, 0.1, 0.01)
    input_late = st.slider("2. What was the recent sentiment?", -1.0, 1.0, 0.0, 0.01)
    input_shift = input_late - input_early
    input_days = st.slider("3. How many days has this match been active?", 0, 1000, 180)
    input_gender = st.selectbox("4. Do the Big and Little identify as the same gender?", ["Yes", "No"])
    input_hobby = st.selectbox("5. Do they share any hobbies?", ["Yes", "No"])
    input_age_diff = st.slider("6. What's their age difference?", 0, 40, 5)
    input_same_ethnicity = 1 if st.selectbox("7. Do they identify with the same ethnicity?", ["Yes", "No"]) == "Yes" else 0
    input_same_zip = 1 if st.selectbox("8. Do they live in the same zip code area?", ["Yes", "No"]) == "Yes" else 0
    input_big_years = st.slider("9. How many years has the Big been volunteering?", 0, 20, 2)
    input_little_matches = st.slider("10. How many previous matches has the Little had?", 0, 10, 1)
    input_same_gender = 1 if input_gender == "Yes" else 0
    input_shared_hobby = 1 if input_hobby == "Yes" else 0

    input_data = pd.DataFrame([{
        "Early": input_early,
        "Late": input_late,
        "Shift": input_shift,
        "Note Duration (days)": input_days,
        "Same Gender": input_same_gender,
        "Shared Hobby": input_shared_hobby,
        "Age Difference": input_age_diff,
        "Same Ethnicity": input_same_ethnicity,
        "Same Zip Code": input_same_zip,
        "Big Years Volunteering": input_big_years,
        "Little Match Count": input_little_matches
    }])

    predicted_risk = clf.predict(input_data)[0]
    proba = clf.predict_proba(input_data)[0][1]
    st.markdown(f"**Prediction:** {'At Risk' if predicted_risk else 'Not At Risk'}")
    st.markdown(f"**Confidence:** {proba*100:.1f}%")

with st.expander("Visual Summaries", expanded=True):

    st.subheader("Explore Individual Feature Effects")

    # Clean categorical data for consistent analysis
    for col in ["Big Gender", "Little Gender", "Big Hobbies", "Little Hobbies", "Big Ethnicity", "Little Ethnicity"]:
        if col in risk_df.columns:
            risk_df[col] = risk_df[col].astype(str).str.strip().str.lower()

    # Same Gender vs Risk
    if "Same Gender" in risk_df.columns:
        st.markdown("**Same Gender vs At-Risk Prediction**")
        fig7, ax7 = plt.subplots()
        sns.barplot(data=risk_df, x="Same Gender", y="Predicted At Risk", estimator=np.mean, ax=ax7)
        ax7.set_xticklabels(["Different", "Same"])
        ax7.set_ylabel("% Predicted At Risk")
        st.pyplot(fig7)
        st.caption("🔎 Matches with same gender appear slightly less likely to be at risk, though this effect is modest.")

    # Shared Hobby vs Risk
    if "Shared Hobby" in risk_df.columns:
        st.markdown("**Shared Hobby vs At-Risk Prediction**")
        fig8, ax8 = plt.subplots()
        sns.barplot(data=risk_df, x="Shared Hobby", y="Predicted At Risk", estimator=np.mean, ax=ax8)
        ax8.set_xticklabels(["No Shared", "Shared"])
        st.pyplot(fig8)
        st.caption("🔎 Having a shared hobby may help reduce risk slightly, but this varies depending on match context.")

    # Same Ethnicity
    if "Same Ethnicity" in risk_df.columns:
        st.markdown("**Same Ethnicity vs At-Risk Prediction**")
        fig9, ax9 = plt.subplots()
        sns.barplot(data=risk_df, x="Same Ethnicity", y="Predicted At Risk", estimator=np.mean, ax=ax9)
        ax9.set_xticklabels(["Different", "Same"])
        st.pyplot(fig9)
        st.caption("🔎 Shared ethnicity shows minimal difference in predicted risk — suggesting the model values other factors more.")

    # Same Zip Code
    if "Same Zip Code" in risk_df.columns:
        st.markdown("**Same Zip Code vs At-Risk Prediction**")
        fig10, ax10 = plt.subplots()
        sns.barplot(data=risk_df, x="Same Zip Code", y="Predicted At Risk", estimator=np.mean, ax=ax10)
        ax10.set_xticklabels(["Different", "Same"])
        st.pyplot(fig10)
        st.caption("🔎 Living in the same zip code does not significantly change risk, likely due to urban density or matching policies.")

    # Age Difference
    if "Age Difference" in risk_df.columns:
        st.markdown("**Age Difference vs At-Risk Prediction**")
        fig11, ax11 = plt.subplots()
        sns.violinplot(data=risk_df, x="Predicted At Risk", y="Age Difference", ax=ax11)
        ax11.set_xticklabels(["Not At Risk", "At Risk"])
        st.pyplot(fig11)
        st.caption("🔎 Larger age differences are slightly more common in higher-risk matches — especially above 10 years.")

    # Big Years Volunteering
    if "Big Years Volunteering" in risk_df.columns:
        st.markdown("**Big Years Volunteering by Risk**")
        fig12, ax12 = plt.subplots()
        sns.boxplot(data=risk_df, x="Predicted At Risk", y="Big Years Volunteering", ax=ax12)
        ax12.set_xticklabels(["Not At Risk", "At Risk"])
        st.pyplot(fig12)
        st.caption("🔎 Bigs with more volunteering years tend to have lower-risk matches — experience matters!")

    # Little Match Count
    if "Little Match Count" in risk_df.columns:
        st.markdown("**Little Match Count by Risk**")
        fig13, ax13 = plt.subplots()
        sns.boxplot(data=risk_df, x="Predicted At Risk", y="Little Match Count", ax=ax13)
        ax13.set_xticklabels(["Not At Risk", "At Risk"])
        st.pyplot(fig13)
        st.caption("🔎 Littles with multiple previous matches show slightly higher risk, possibly reflecting historical challenges.")
    st.subheader("Distribution of Predicted Risk")
    risk_counts = risk_df["Predicted At Risk"].value_counts().rename(index={0: "Not At Risk", 1: "At Risk"})
    fig3, ax3 = plt.subplots(figsize=(6, 4))
    sns.barplot(x=risk_counts.index, y=risk_counts.values, palette="Set2", ax=ax3)
    ax3.set_ylabel("Number of Matches")
    ax3.set_title("Predicted Match Risk Breakdown")
    st.pyplot(fig3)

    st.subheader("Sentiment Shift vs. Match Duration")
    fig4, ax4 = plt.subplots(figsize=(7, 5))
    sns.scatterplot(data=risk_df, x="Shift", y="Note Duration (days)", hue="Predicted At Risk", palette="coolwarm", ax=ax4)
    ax4.set_title("Emotional Shift vs Duration")
    ax4.axvline(0, color="gray", linestyle="--")
    st.pyplot(fig4)

    if "Closure Reason" in risk_df.columns:
        st.subheader("Closure Reason Distribution")
        closure_counts = risk_df["Closure Reason"].value_counts().head(10)
        fig5, ax5 = plt.subplots(figsize=(8, 4))
        sns.barplot(x=closure_counts.values, y=closure_counts.index, ax=ax5)
        ax5.set_title("Top Closure Reasons")
        ax5.set_xlabel("Number of Matches")
        st.pyplot(fig5)

    st.subheader("Feature Correlation Heatmap")
    correlation_matrix = risk_df[features + ["Predicted At Risk"]].corr()
    fig6, ax6 = plt.subplots(figsize=(10, 8))
    sns.heatmap(correlation_matrix, annot=True, cmap="coolwarm", fmt=".2f", ax=ax6)
    ax6.set_title("Correlation Matrix for Model Features")
    st.pyplot(fig6)

with st.expander("Upload Your Own Matches for Prediction", expanded=False):
    uploaded_file = st.file_uploader("Upload a CSV file with the same columns used in the model:", type=["csv"])
    if uploaded_file is not None:
        user_matches = pd.read_csv(uploaded_file)
        try:
            user_matches = user_matches[features]
            predictions = clf.predict(user_matches)
            user_matches["Predicted At Risk"] = predictions
            st.success("Predictions complete!")
            st.dataframe(user_matches)
        except Exception as e:
            st.error(f"Error processing file: {e}")

st.caption("Built for MinneMUDAC by Team DataBells")


