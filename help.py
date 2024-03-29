import pandas as pd
import re
from nltk.tokenize import word_tokenize
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
import pickle

# Load the dataset containing text data and emotion labels
df = pd.read_csv('dev.csv')

# Preprocess the text data
def preprocess_text(text):
    # Remove punctuation marks
    text = re.sub(r'[^\w\s]', '', text)
    # Convert text to lowercase
    text = text.lower()
    # Tokenize the text into individual words
    tokens = word_tokenize(text)
    return tokens

# Convert preprocessed text data into numerical features using TF-IDF vectorization
def vectorize_text(data):
    vectorizer = TfidfVectorizer(max_features=1000)  # Limiting the number of features for simplicity
    X = data.apply(lambda x: ' '.join(x))  # Join tokenized words back into sentences
    X_vec = vectorizer.fit_transform(X)
    return X_vec, vectorizer

# Train SVM classifier for each emotion and save the trained models
def train_classifier(X, y, emotion):
    svm_classifier = SVC(kernel='linear', random_state=42)  # Linear kernel for simplicity
    try:
        svm_classifier.fit(X, y)
        with open(emotion + '_classifier.pkl', 'wb') as file:
            pickle.dump(svm_classifier, file)
        return svm_classifier
    except Exception as e:
        print(f"Failed to train classifier for {emotion}: {e}")
        return None

# Load the trained SVM classifiers for each emotion
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

# Evaluate the model's performance
def evaluate_classifier(classifier, X_test, y_test):
    if classifier:
        y_pred = classifier.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        return accuracy
    else:
        return None

# Main script
if __name__ == "__main__":
    # Preprocess text data
    df['Tweet'] = df['Tweet'].apply(preprocess_text)
    
    # Vectorize text data
    X_vec, vectorizer = vectorize_text(df['Tweet'])
    
    # Iterate over each emotion and train a separate SVM classifier
    emotions = ['joy', 'sadness', 'anger', 'fear']
    trained_classifiers = {}
    for emotion in emotions:
        y = df[emotion]  # Target variable for the current emotion
        y_binary = (y > 0.5).astype(int)  # Convert to binary labels
        
        # Split the dataset into training and testing sets
        X_train, X_test, y_train, y_test = train_test_split(X_vec, y_binary, test_size=0.2, random_state=42)
        
        # Train SVM classifier for the current emotion and save the model
        classifier = train_classifier(X_train, y_train, emotion)
        trained_classifiers[emotion] = classifier
        
        # Evaluate the model's performance
        accuracy = evaluate_classifier(classifier, X_test, y_test)
        if accuracy is not None:
            print(f"Accuracy for {emotion}: {accuracy:.2f}")
