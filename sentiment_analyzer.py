import pandas as pd
import numpy as np
import re
import string
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB # A good baseline for text classification
from sklearn.linear_model import LogisticRegression # Another common choice, often performs well
from sklearn.metrics import accuracy_score, classification_report
from sklearn.pipeline import Pipeline
import joblib # For saving/loading the model and vectorizer

# --- 1. File Paths ---
DATA_FILENAME = 'sentiment_data.csv'
MODEL_FILENAME = 'sentiment_model.pkl'
SAMPLE_PREDICTIONS_FILENAME = 'sample_predictions.txt'
MODEL_REPORT_FILENAME = 'model_report.txt'

# --- 2. Data Loading ---
print(f"--- Loading Data from {DATA_FILENAME} ---")
try:
    df = pd.read_csv(DATA_FILENAME)
    print("Data loaded successfully.")
    print("Dataset head:\n", df.head())
    print(f"\nSentiment distribution:\n{df['sentiment'].value_counts()}")
except FileNotFoundError:
    print(f"Error: Data file '{DATA_FILENAME}' not found. Please run 'python data_generator.py' first.")
    exit() # Exit if data is not available
except Exception as e:
    print(f"Error loading data: {e}")
    exit()

# --- 3. Text Preprocessing Functions ---
print("\n--- Defining Text Preprocessing Steps (Simplified - No NLTK) ---")

def preprocess_text(text):
    text = text.lower() # Lowercasing
    text = re.sub(f'[{re.escape(string.punctuation)}]', '', text) # Remove punctuation
    text = re.sub(r'\d+', '', text) # Remove numbers
    tokens = text.split() # Simple tokenization by whitespace
    # No stop word removal or lemmatization without NLTK
    return ' '.join(tokens)

# Apply preprocessing
print("\n--- Applying Preprocessing to Text Data ---")
df['processed_text'] = df['text'].apply(preprocess_text)
print("Processed text head:\n", df[['text', 'processed_text']].head())


# --- 4. Splitting Data ---
print("\n--- Splitting Data into Training and Testing Sets ---")
X = df['processed_text']
y = df['sentiment']

# Stratify=y ensures that the proportion of sentiment classes is the same in train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

print(f"Training data shape: {X_train.shape}, {y_train.shape}")
print(f"Testing data shape: {X_test.shape}, {y_test.shape}")

# --- 5. Building the NLP Pipeline (Vectorizer + Model) ---
print("\n--- Building and Training the NLP Model Pipeline ---")

# TF-IDF Vectorizer: Converts text into numerical feature vectors
# Model: Multinomial Naive Bayes is a good choice for text classification,
# or uncomment LogisticRegression for an alternative.
text_classifier = Pipeline([
    ('vectorizer', TfidfVectorizer(max_features=5000)), # Consider top 5000 most frequent words
    ('classifier', MultinomialNB()) # Using Multinomial Naive Bayes
    # ('classifier', LogisticRegression(random_state=42, max_iter=1000)) # Alternative: Logistic Regression
])

# Train the model
text_classifier.fit(X_train, y_train)
print("Model training complete!")

# --- 6. Evaluate the Model ---
print("\n--- Evaluating the Model Performance ---")
y_pred = text_classifier.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)
report = classification_report(y_test, y_pred, zero_division=0) # zero_division=0 to handle cases where no true samples for a label

print(f"Accuracy: {accuracy:.2f}")
print("\nClassification Report:\n", report)

# --- 7. Save the Trained Model ---
print(f"\n--- Saving the Trained Model to {MODEL_FILENAME} ---")
joblib.dump(text_classifier, MODEL_FILENAME)
print("Model saved successfully.")

# --- 8. Make Predictions on New Sample Text and Save to File ---
print("\n--- Making Predictions on New Sample Text ---")
new_texts = [
    "This is an absolutely fantastic product, highly recommend it!",
    "The service was incredibly slow and disappointing.",
    "The package arrived, it's just a regular item.",
    "Feeling great about today's progress, very optimistic.",
    "I had a terrible experience with their support, absolutely frustrating.",
    "It's a decent quality for the price, nothing exceptional but it works."
]

predictions = text_classifier.predict(new_texts)

# Write sample predictions to a file
with open(SAMPLE_PREDICTIONS_FILENAME, 'w') as f_pred:
    f_pred.write("--- Sample Sentiment Predictions ---\n\n")
    for i, text in enumerate(new_texts):
        f_pred.write(f"Text: \"{text}\"\n")
        f_pred.write(f"Predicted Sentiment: {predictions[i]}\n\n")
print(f"Sample predictions saved to '{SAMPLE_PREDICTIONS_FILENAME}'")

# --- 9. Generate and Save Model Performance Report ---
print("\n--- Generating Model Performance Report ---")
with open(MODEL_REPORT_FILENAME, 'w') as f_report:
    f_report.write("--- Sentiment Analysis Model Performance Report ---\n\n")
    f_report.write(f"Accuracy: {accuracy:.2f}\n\n")
    f_report.write("Classification Report:\n")
    f_report.write(report)
    f_report.write("\n\nNote: For more detailed insights, analyze precision, recall, and f1-score per class.")
print(f"Model performance report saved to '{MODEL_REPORT_FILENAME}'")

print("\n--- Sentiment Analysis Project Complete! ---")
