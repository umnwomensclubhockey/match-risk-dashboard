# -------------------- IMPORTS --------------------
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import LabelEncoder
from wordcloud import WordCloud
from textblob import TextBlob

# Optional survival analysis
try:
    from lifelines import KaplanMeierFitter
    LIFELINES_AVAILABLE = True
except ImportError:
    LIFELINES_AVAILABLE = False
    print("lifelines not installed – skipping survival analysis.")

# -------------------- LOAD DATA --------------------
print("📦 Loading files...")
training_df = pd.read_csv("Training-Restated.xlsx - Sheet1.csv", low_memory=False, dtype=str)
test_df = pd.read_csv("Test-Truncated-Restated.xlsx - Sheet1.csv", low_memory=False, dtype=str)

# Convert numeric columns
for col in ["Match Length", "Big Age"]:
    training_df[col] = pd.to_numeric(training_df[col], errors="coerce")

# Convert date columns
training_df["Completion Date"] = pd.to_datetime(training_df["Completion Date"], errors='coerce')
test_df["Completion Date"] = pd.to_datetime(test_df["Completion Date"], errors='coerce')

print("✅ Files loaded successfully.")

# -------------------- EDA --------------------
eda_df = training_df.dropna(subset=["Match Length", "Stage", "Closure Reason"])

# Distribution of Match Length
plt.figure(figsize=(10, 5))
sns.histplot(eda_df["Match Length"], kde=True, bins=30)
plt.title("Distribution of Match Length")
plt.xlabel("Months")
plt.ylabel("Count")
plt.tight_layout()
plt.show()

# Closure Reason Distribution
plt.figure(figsize=(12, 6))
sns.countplot(y="Closure Reason", data=eda_df,
              order=eda_df["Closure Reason"].value_counts().index)
plt.title("Closure Reason Distribution")
plt.xlabel("Count")
plt.tight_layout()
plt.show()

# Match Length by Program Type
if "Program Type" in eda_df.columns:
    plt.figure(figsize=(10, 5))
    sns.boxplot(x="Program Type", y="Match Length", data=eda_df)
    plt.title("Match Length by Program Type")
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()

# Match Length Over Time
training_df["Year"] = training_df["Completion Date"].dt.year
plt.figure(figsize=(10, 5))
sns.boxplot(x="Year", y="Match Length", data=training_df)
plt.title("Match Length Over Time")
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

# -------------------- SAME GENDER ANALYSIS --------------------
if "Big Gender" in training_df.columns and "Little Gender" in training_df.columns:
    training_df["Same Gender"] = training_df["Big Gender"] == training_df["Little Gender"]
    plt.figure(figsize=(6, 4))
    sns.boxplot(x="Same Gender", y="Match Length", data=training_df)
    plt.title("Match Length vs Same Gender")
    plt.tight_layout()
    plt.show()

# -------------------- SURVIVAL ANALYSIS --------------------
if LIFELINES_AVAILABLE:
    kmf = KaplanMeierFitter()
    match_lengths = eda_df["Match Length"]
    closed = eda_df["Stage"] == "Closed"
    kmf.fit(match_lengths, event_observed=closed)
    kmf.plot_survival_function()
    plt.title("Match Survival Function")
    plt.xlabel("Months")
    plt.ylabel("Survival Probability")
    plt.tight_layout()
    plt.show()

# -------------------- WORD CLOUD --------------------
notes = training_df["Match Support Contact Notes"].dropna()
text = " ".join(notes.astype(str))
wordcloud = WordCloud(width=800, height=400, background_color='white').generate(text)
plt.figure(figsize=(15, 7))
plt.imshow(wordcloud, interpolation='bilinear')
plt.axis('off')
plt.title("Word Cloud from Match Notes")
plt.tight_layout()
plt.show()

# -------------------- SENTIMENT ANALYSIS --------------------
def get_sentiment(text):
    return TextBlob(text).sentiment.polarity

if not notes.empty:
    sample_notes = notes.sample(500, random_state=1).astype(str)
    sentiment_scores = sample_notes.apply(get_sentiment)
    plt.figure(figsize=(10, 4))
    sns.histplot(sentiment_scores, bins=30, kde=True)
    plt.title("Sentiment Polarity of Match Notes")
    plt.xlabel("Polarity (-1 = negative, 1 = positive)")
    plt.tight_layout()
    plt.show()

# -------------------- RANDOM FOREST PREDICTION --------------------
features = ["Big Age", "Little Gender", "Big Gender", "Same Gender", "Program Type"]
model_df = training_df.dropna(subset=["Match Length"]).copy()

# Encode categorical features
label_encoders = {}
for col in ["Little Gender", "Big Gender", "Program Type"]:
    if col in model_df.columns:
        le = LabelEncoder()
        model_df[col] = le.fit_transform(model_df[col].astype(str))
        label_encoders[col] = le

X = model_df[features]
y = model_df["Match Length"]

X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)
y_pred = model.predict(X_val)
rmse = mean_squared_error(y_val, y_pred, squared=False)

print("\n✅ Random Forest Model RMSE on Validation Set:", round(rmse, 2))

