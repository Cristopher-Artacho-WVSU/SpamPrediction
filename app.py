import streamlit as st
import joblib
import re
import string
import nltk
from nltk.corpus import stopwords


# LOAD THE MODEL AND VECTORIZERS
model = joblib.load("spam_classifier.pkl")
vectorizer = joblib.load("tfidf_vectorizer.pkl")


nltk.download("stopwords")


# REDUCE THE INPUT TO ITS MOST BASIC FORM
def preprocess_text(text):
    text = text.lower()
    text = re.sub(r"\d+", "", text) 
    text = text.translate(str.maketrans("", "", string.punctuation))
    words = text.split()
    words = [word for word in words if word not in stopwords.words("english")]
    return " ".join(words)


# STREAMLIT APP
st.title("📩 Spam Detector App")
st.write("Enter a message below to check if it's **Spam** or **Not Spam**.")


user_input = st.text_area("Enter your message:")

if st.button("Check Spam"):
    if user_input.strip():
        processed_input = preprocess_text(user_input) 
        input_vector = vectorizer.transform([processed_input])  
        prediction = model.predict(input_vector)  
        
        result = "Spam" if prediction[0] == 1 else "Not Spam"
        st.success(f"Prediction: {result}")
    else:
        st.warning("Please enter a message to check.")