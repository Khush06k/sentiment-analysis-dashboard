# -*- coding: utf-8 -*-
"""
Created on Wed Sep 10 12:49:08 2025
@author: Asus
"""
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import os
import io

# --------------- NLTK Preprocessing Imports -----------------
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import string

# Sentiment Analyzer (VADER)
from nltk.sentiment.vader import SentimentIntensityAnalyzer

# Download required NLTK datasets (only if not already downloaded)
nltk.download('punkt', quiet=True)
nltk.download('punkt_tab', quiet=True) 
nltk.download('stopwords', quiet=True)
nltk.download('wordnet', quiet=True)
nltk.download('omw-1.4', quiet=True) 
nltk.download('vader_lexicon', quiet=True)

analyzer = SentimentIntensityAnalyzer()

# ---------------- Negation Fix Function --------------------
def preprocess_review(text: str) -> str:
    """
    Replace common negations so VADER handles them correctly
    Example: "not good" -> "not_good"
    """
    text = text.replace("not good", "not_good")
    text = text.replace("not bad", "not_bad")
    text = text.replace("not happy", "not_happy")
    text = text.replace("not tasty", "not_tasty")
    return text

# ---------------- Preprocessing Function --------------------
def preprocess_text(text):
    """
    Clean review text:
    - Lowercase
    - Tokenize
    - Remove punctuation/numbers
    - Remove stopwords
    - Lemmatize
    """
    text = text.lower()
    tokens = word_tokenize(text)
    tokens = [t for t in tokens if t.isalpha()]  # keep words only
    stop_words = set(stopwords.words('english'))
    tokens = [t for t in tokens if t not in stop_words]
    lemmatizer = WordNetLemmatizer()
    lemmas = [lemmatizer.lemmatize(token) for token in tokens]
    return " ".join(lemmas)

# ---------------- Advanced Negation-Aware Sentiment ----------------
import re

def get_sentiment(review):
    # ------------------- PREPROCESSING -------------------
    raw_lower = review.lower()
    cleaned_review = preprocess_text(review)

    # Negation words list
    negation_words = [
        "not", "no", "never", "without", "lacking", "lacks", 
        "none", "cannot", "can't", "dont", "don't", "hardly", "rarely"
    ]

    # Extended custom phrase-based adjustments
    custom_negations = {
        # Taste / Food related
        "without taste": -0.6,
        "without flavor": -0.6,
        "no taste": -0.6,
        "no flavor": -0.6,
        "tasteless": -0.7,
        "bland taste": -0.6,

        # Quality / Value related
        "lacks quality": -0.7,
        "lacks value": -0.7,
        "not worth the money": -0.8,
        "not worth it": -0.7,
        "waste of money": -0.9,
        "no value for money": -0.8,
        "poor quality": -0.7,

        # Features / Performance
        "missing features": -0.6,
        "lacks performance": -0.7,
        "no improvement": -0.6,
        "without features": -0.6,
        "not functional": -0.7,
        "not reliable": -0.7,
        "lacks support": -0.7,
        "without support": -0.6,
        "no durability": -0.8,
        "lacks durability": -0.7,

        # Experience / Service
        "not user friendly": -0.7,
        "poor service": -0.8,
        "no customer support": -0.9,
        "not satisfied": -0.7,
        "never again": -0.9,
        "not at all good": -0.8,
        "not helpful": -0.6,
        "not clean": -0.6
    }

    # ------------------- VADER Sentiment -------------------
    scores = analyzer.polarity_scores(cleaned_review)
    compound = scores['compound']

    # ------------------- Regex Negation Handling -------------------
    for neg in negation_words:
        # Capture words after negation until punctuation
        pattern = re.compile(rf"\b{neg}\b\s+([a-z\s]+?)([.,!?]|$)")
        matches = pattern.findall(raw_lower)
        for match in matches:
            phrase = match[0].strip()
            if phrase:
                phrase_score = analyzer.polarity_scores(phrase)['compound']
                if phrase_score != 0:
                    compound -= phrase_score  # flip sentiment

    # ------------------- Custom Phrase Adjustment -------------------
    for phrase, adj_score in custom_negations.items():
        if phrase in raw_lower:
            compound += adj_score

    # ------------------- Final Sentiment -------------------
    if compound >= 0.10:
        return "Positive", compound, cleaned_review
    elif compound <= -0.10:
        return "Negative", compound, cleaned_review
    else:
        return "Neutral", compound, cleaned_review



# ------------------------------------------------------------
# Streamlit Dashboard
# ------------------------------------------------------------
st.set_page_config(
    page_title="Customer Review Sentiment Dashboard",
    page_icon="💬",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.sidebar.title("⚙️ Dashboard Controls")
uploaded_file = st.sidebar.file_uploader("📂 Upload Excel File", type=["xlsx"])
st.sidebar.markdown("---")

st.title("💬 Sentiment Analysis of Customer Reviews")
st.markdown("This dashboard analyzes customer reviews and classifies them as **Positive**, **Negative**, or **Neutral**.")

# --------------- Uploaded File Sentiment Pipeline ----------
if uploaded_file:
    df = pd.read_excel(uploaded_file)
    if "Customer_Review" not in df.columns:
        st.error("❌ Excel file must contain a column named 'Customer_Review'")
    else:
        st.success("✅ File uploaded successfully!")

        sentiments = []
        compounds = []
        cleaned_reviews = []

        for review in df["Customer_Review"]:
            sentiment, score, clean_text = get_sentiment(str(review))
            sentiments.append(sentiment)
            compounds.append(score)
            cleaned_reviews.append(clean_text)

        df["Cleaned_Review"] = cleaned_reviews
        df["Compound_Score"] = compounds
        df["Sentiment"] = sentiments

        # ---------------- Show Table ----------------
        st.subheader("📊 Sample Data with Sentiment (with Preprocessing)")
        num_rows = st.slider("Select number of reviews to display:", 10, len(df), 10)
        st.dataframe(df.head(num_rows))   # shows selected rows
        st.markdown(f"**Total Reviews Analyzed: {len(df)}**")

        # ---------------- Charts ----------------
        st.subheader("📈 Sentiment Distribution (All Reviews)")
        sentiment_counts = df.head(num_rows)['Sentiment'].value_counts(normalize=True) * 100

        col1, col2 = st.columns(2)

        with col1:
            st.markdown("### 📊 Bar Chart")
            fig1, ax1 = plt.subplots()
            sentiment_counts.plot(kind="bar", ax=ax1, color=["green", "red", "gray"])
            ax1.set_ylabel("Percentage (%)")
            ax1.set_xlabel("Sentiment")
            ax1.set_title("Sentiment Distribution (Bar Chart)")
            st.pyplot(fig1)

        with col2:
            st.markdown("### 🥧 Pie Chart")
            fig2, ax2 = plt.subplots()
            ax2.pie(
                sentiment_counts,
                labels=sentiment_counts.index,
                autopct="%1.1f%%",
                colors=["green", "red", "gray"],
                startangle=90
            )
            ax2.set_title("Sentiment Distribution (Pie Chart)")
            st.pyplot(fig2)

        # ---------------- Download Button ----------------
        output = io.BytesIO()
        df.to_excel(output, index=False)
        st.download_button(
            label="⬇️ Download full results as Excel",
            data=output.getvalue(),
            file_name="sentiment_results.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
        )

# --------------- User Input Preprocessing & Saving -----------
st.subheader("🧪 Test Your Own Review")
user_input = st.text_area("📝 Enter a review or paragraph here:")

if user_input:
    sentiment_result, polarity, clean_input = get_sentiment(user_input)

    color = "green" if sentiment_result == "Positive" else "red" if sentiment_result == "Negative" else "gray"

    st.markdown(
        f"**Result:** <span style='color:{color}; font-weight:bold;'>{sentiment_result}</span> "
        f"(Compound Score: {polarity:.2f})",
        unsafe_allow_html=True
    )

    if st.button("Save this review"):
        save_dir = r"C:\Users\Asus\Scripts"
        save_file = os.path.join(save_dir, "saved_reviews.xlsx")

        if not os.path.exists(save_dir):
            os.makedirs(save_dir)

        new_entry = pd.DataFrame([{
            "Customer_Review": user_input,
            "Cleaned_Review": clean_input,
            "Sentiment": sentiment_result,
            "Compound_Score": polarity
        }])

        if os.path.exists(save_file):
            old = pd.read_excel(save_file)
            combined = pd.concat([old, new_entry], ignore_index=True)
            combined.to_excel(save_file, index=False)
        else:
            new_entry.to_excel(save_file, index=False)

        st.success(f"✅ Review and sentiment saved to {save_file}!")
