
# Step 1: Choose a model (Random Forest Classifier)
model = RandomForestClassifier(random_state=42)
# Step 2: Train the model
model.fit(X_train, y_train)

# Step 3: Evaluate the model
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy:", accuracy)