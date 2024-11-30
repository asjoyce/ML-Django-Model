import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score
import joblib

# Load your real dataset
df = pd.read_csv('/Users/sathwikajoyce/mlops_project/mlops_project/Notebooks/IMDB Dataset.csv')

# Check the column names to make sure we are using the right ones
print(df.columns)

# Inspect the first few rows to understand the data structure
print(df.head())

# Assuming the dataset has columns 'review' for text and 'label' for sentiment
# Adjust the column names based on the output of print(df.columns)
X = df['review']  # Feature: Text data
y = df['sentiment']   # Label: Sentiment

# Split data into train and test sets (80% train, 20% test)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# Build the model pipeline
pipeline = Pipeline([
    ('vectorizer', CountVectorizer()),  # Convert text to numerical features
    ('classifier', MultinomialNB())     # Use Naive Bayes classifier
])

# Train the model
pipeline.fit(X_train, y_train)

# Evaluate the model
y_pred = pipeline.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)

# Print accuracy
print(f"Model accuracy: {accuracy}")

# Save the trained model to a file
joblib.dump(pipeline, 'sentiment_model.pkl')

# Test prediction on some sample text (optional)
sample_text = ["This movie is amazing!", "I hated the film."]
sample_pred = pipeline.predict(sample_text)
print(f"Sample predictions: {sample_pred}")
