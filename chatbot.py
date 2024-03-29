import streamlit as st
import pickle
import re
from nltk.tokenize import word_tokenize
from sklearn.feature_extraction.text import TfidfVectorizer

# Load trained classifiers for each emotion
def load_classifiers():
    classifiers = {}
    emotions = ['joy', 'sadness', 'anger', 'fear']
    for emotion in emotions:
        try:
            with open(emotion + '_classifier.pkl', 'rb') as file:
                classifiers[emotion] = pickle.load(file)
        except FileNotFoundError:
            print(f"Classifier file not found for {emotion}")
            classifiers[emotion] = None
    return classifiers

# Preprocess user input
def preprocess_text(text):
    text = re.sub(r'[^\w\s]', '', text)  # Remove punctuation marks
    text = text.lower()  # Convert text to lowercase
    tokens = word_tokenize(text)  # Tokenize the text into individual words
    return ' '.join(tokens)

# Predict emotions in user input
def predict_emotions(input_text):
    input_text = preprocess_text(input_text)
    X_vec = vectorizer.transform([input_text])
    emotions = ['joy', 'sadness', 'anger', 'fear']
    predicted_emotions = {}
    for emotion in emotions:
        classifier = classifiers[emotion]
        if classifier:
            predicted_emotions[emotion] = classifier.predict(X_vec)[0]
        else:
            predicted_emotions[emotion] = 0  # Default value if classifier is not available
    return predicted_emotions

# Generate response based on predicted emotions
def generate_response(input_text):
    predicted_emotions = predict_emotions(input_text)
    response = "Based on your input, I detected the following emotions:\n"
    for emotion, value in predicted_emotions.items():
        if value == 1:
            response += f"- {emotion.capitalize()}\n"
    if not any(predicted_emotions.values()):
        response += "I couldn't detect any specific emotion."
    return response

# Load classifiers and TF-IDF vectorizer
classifiers = load_classifiers()
with open('tfidf_vectorizer.pkl', 'rb') as file:
    vectorizer = pickle.load(file)

# Main Streamlit app
def main():
    st.title("Emotion Recognition Chatbot")
    user_input = st.text_input("Enter your message:")
    if st.button("Submit"):
        response = generate_response(user_input)
        st.write("Chatbot:", response)

# Run the app
if __name__ == "__main__":
    main()
