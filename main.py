import pandas as pd
import re
from nltk.tokenize import word_tokenize
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score

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

# Apply preprocessing to the 'Tweet' column
df['Tweet'] = df['Tweet'].apply(preprocess_text)

#print(df)

# Convert preprocessed text data into numerical features using TF-IDF vectorization
vectorizer = TfidfVectorizer(max_features=1000)  # Limiting the number of features for simplicity
X = df['Tweet'].apply(lambda x: ' '.join(x))  # Join tokenized words back into sentences
X_vec = vectorizer.fit_transform(X)

# Iterate over each emotion and train a separate SVM classifier
emotions = ['joy', 'sadness', 'anger', 'fear']
for emotion in emotions:
    y = df[emotion]  # Target variable for the current emotion
    
    # Convert continuous values to discrete class labels (0 or 1) using a threshold
    threshold = 0.5  # Adjust the threshold as needed
    y_binary = (y > threshold).astype(int)
    
    # Split the dataset into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X_vec, y_binary, test_size=0.2, random_state=42)
    
    # Train a Support Vector Machine (SVM) classifier for the current emotion
    svm_classifier = SVC(kernel='linear', random_state=42)  # Linear kernel for simplicity
    svm_classifier.fit(X_train, y_train)
    
    # Predict emotions on the test data
    y_pred = svm_classifier.predict(X_test)
    
    # Evaluate the model's performance
    accuracy = accuracy_score(y_test, y_pred)
    print("Accuracy for {}: {:.2f}".format(emotion, accuracy))
