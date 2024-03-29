import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.multioutput import MultiOutputClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report

# Load the dataset containing text data and emotion labels
df = pd.read_csv('dev.csv')

# Define features (X) and target variable (y)
X = df['Tweet'] 
y = df[['joy', 'sadness', 'anger', 'fear']] 

# Convert continuous multi-output labels to binary labels
y_binary = y.applymap(lambda x: 1 if x > 0 else 0) #otherwise, the accuracy was showing 0.00

# Vectorize the text data
vectorizer = TfidfVectorizer(max_features=1000)
X_vec = vectorizer.fit_transform(X)

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X_vec, y_binary, test_size=0.2, random_state=42)

# Train multi-output SVM classifier
classifier = MultiOutputClassifier(SVC(kernel='linear', random_state=42))
classifier.fit(X_train, y_train)

# Predict emotions on the test data
y_pred = classifier.predict(X_test)

# Evaluate the model's performance
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy:", accuracy)


