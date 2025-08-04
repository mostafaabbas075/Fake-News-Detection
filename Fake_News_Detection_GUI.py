import streamlit as st
import joblib
import re
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords

# Download NLTK resources (run once)
import nltk
nltk.download('punkt')
nltk.download('wordnet')
nltk.download('omw-1.4')
nltk.download('stopwords')

# Initialize lemmatizer and stop words
lemmatizer = WordNetLemmatizer()
stop_words = set(stopwords.words('english'))

# Load the saved model and vectorizer
svm_model = joblib.load('svm_model.pkl')
tfidf_vectorizer = joblib.load('tfidf_vectorizer.pkl')

# Function to clean text
def clean_text(text):
    text = text.lower()
    text = re.sub(r'[^\w\s]', '', text)
    text = re.sub(r'\d+', '', text)
    words = word_tokenize(text)
    words = [lemmatizer.lemmatize(word) for word in words if word not in stop_words]
    return ' '.join(words)

# Streamlit app
st.title("Fake News Detector")
st.write("Enter a news text to check if it's Fake or True.")

# Text input
user_input = st.text_area("News Text", "")

if st.button("Predict"):
    if user_input:
        # Clean the input text
        cleaned_text = clean_text(user_input)
        
        # Transform the text using the saved vectorizer
        text_tfidf = tfidf_vectorizer.transform([cleaned_text])
        
        # Predict using the SVM model
        prediction = svm_model.predict(text_tfidf)
        
        if prediction[0] == 1:
            st.success("🟢 This news is predicted as: **True**")
        else:
            st.error("🔴 This news is predicted as: **Fake**")


# Optional: Add a sidebar for info
st.sidebar.header("About")
st.sidebar.write("This app uses a trained SVM model to detect fake news based on text input.")