import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

df=pd.read_csv('dev.csv')
print(df.head())
print(df.dtypes)
# Compute pairwise correlation of numerical columns
#correlation_matrix = df.corr()

# Print the correlation matrix
#print(correlation_matrix)
# Define features (X) and target variable (y)
X = df[['joy', 'sadness', 'anger', 'fear']]
y = df['Tweet']  

# Split the data into training and testing sets (e.g., 80% training, 20% testing)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
# Step 1: Choose a model (Random Forest Classifier)
model = RandomForestClassifier(random_state=42)

# Step 2: Train the model
model.fit(X_train, y_train)

# Step 3: Evaluate the model
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy:", accuracy)

