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
