import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import SVC
from sklearn.metrics import classification_report, accuracy_score

# Load the dataset
file_path = '/content/netflix_reviews.csv'
df = pd.read_csv(file_path)

# Display the first few rows of the dataframe to understand its structure
print(df.head())

# Check for any missing values
print(df.isnull().sum())

# Drop missing values if any
df.dropna(subset=['content', 'score'], inplace=True)

# Preprocess the text (simple cleaning)
def preprocess_text(text):
    text = text.lower()  # Convert to lowercase
    return text

df['content'] = df['content'].apply(preprocess_text)

# Convert scores to binary sentiment labels (1 for positive, 0 for negative)
df['sentiment'] = df['score'].apply(lambda x: 1 if x >= 3 else 0)

# Split the dataset into training and testing sets
X = df['content']
y = df['sentiment']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Vectorize the text data using TfidfVectorizer
vectorizer = TfidfVectorizer(max_features=5000)
X_train_vec = vectorizer.fit_transform(X_train)
X_test_vec = vectorizer.transform(X_test)

# Train an SVM model
svm = SVC(kernel='linear')
svm.fit(X_train_vec, y_train)

# Make predictions on the test set
y_pred = svm.predict(X_test_vec)

# Evaluate the model
accuracy = accuracy_score(y_test, y_pred)
print(f'Accuracy: {accuracy * 100:.2f}%')
print('Classification Report:')
print(classification_report(y_test, y_pred))
import joblib

# Save the trained model and vectorizer
joblib.dump(svm, 'svm_model.pkl')
joblib.dump(vectorizer, 'vectorizer.pkl')

# Load the trained model and vectorizer
svm = joblib.load('svm_model.pkl')
vectorizer = joblib.load('vectorizer.pkl')

# Preprocess the new review
def preprocess_text(text):
    text = text.lower()  # Convert to lowercase
    return text

#  new reviews
new_reviews = [
    "It was tottly insane.",
    "This movie was terrible. I will never watch it again."
]

# Preprocess and vectorize the new reviews
new_reviews_preprocessed = [preprocess_text(review) for review in new_reviews]
new_reviews_vec = vectorizer.transform(new_reviews_preprocessed)

# Predict the sentiment of the new reviews
predictions = svm.predict(new_reviews_vec)

# Print the predictions
for review, prediction in zip(new_reviews, predictions):
    sentiment = "Positive" if prediction == 1 else "Negative"
    print(f'Review: "{review}"\nSentiment: {sentiment}\n')

