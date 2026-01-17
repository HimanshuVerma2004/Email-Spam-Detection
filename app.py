import streamlit as st
import pickle
import nltk
import string
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import CountVectorizer

# ---------- SAME PROCESS FUNCTION ----------
def process(text):
    nopunc = [char for char in text if char not in string.punctuation]
    nopunc = ''.join(nopunc)

    clean = [word for word in nopunc.split() if word.lower() not in stopwords.words('english')]
    return clean
# ------------------------------------------

# Load dataset to rebuild vectorizer
import pandas as pd
dataset = pd.read_csv("dataset/emails.csv")   # keep same dataset used in notebook

# Recreate vectorizer EXACTLY like training
@st.cache_resource
def load_objects():
    vectorizer = CountVectorizer(analyzer=process)
    vectorizer.fit(dataset['text'])
    model = pickle.load(open("spam_model.pkl","rb"))
    return vectorizer, model

vectorizer, model = load_objects()


# Load only model
model = pickle.load(open("spam_model.pkl","rb"))

def predict_spam(text):
    transformed = vectorizer.transform([text])
    result = model.predict(transformed)[0]
    return "SPAM" if result == 1 else "NOT SPAM"

# ------------- UI ----------------
st.title("Email Spam Detection using NLP")

user_input = st.text_area("Enter Email Text")

if st.button("Predict"):
    result = predict_spam(user_input)

    if result == "SPAM":
        st.error("🚨 Spam Mail")
    else:
        st.success("✅ Not Spam")
