import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
import joblib

# 1️⃣ Load dataset
df = pd.read_csv("Cleaned_FakeNews.csv")
print("✅ Dataset loaded:", df.shape)
print("Columns found in CSV:", df.columns.tolist())

# 2️⃣ Fill missing values
df['text'] = df['text'].fillna("")
df['subject'] = df['subject'].fillna("")

# 3️⃣ Combine text and subject
df['text_data'] = df['text'] + " " + df['subject']

# 4️⃣ Convert labels to 0/1
df['label'] = df['target'].apply(lambda x: 1 if str(x).lower() == 'true' else 0)

# 5️⃣ Take a smaller sample of the dataset for faster training
sample_df = df.sample(n=5000, random_state=42)  # Change n as needed

# 6️⃣ Split dataset
X_train, X_test, y_train, y_test = train_test_split(
    sample_df['text_data'], sample_df['label'], test_size=0.2, random_state=42, stratify=sample_df['label']
)

# 7️⃣ Vectorize text
vectorizer = TfidfVectorizer(max_features=5000, stop_words='english')
X_train_vec = vectorizer.fit_transform(X_train)
X_test_vec = vectorizer.transform(X_test)

# 8️⃣ Train classifier
model = RandomForestClassifier(n_estimators=100, random_state=42)  # Faster training
model.fit(X_train_vec, y_train)

# 9️⃣ Evaluate model
y_pred = model.predict(X_test_vec)
print("Accuracy:", accuracy_score(y_test, y_pred))
print(classification_report(y_test, y_pred))

# 🔟 Save model and vectorizer
joblib.dump(model, "fake_news_model.pkl")
joblib.dump(vectorizer, "tfidf_vectorizer.pkl")
print("✅ Sample model and vectorizer saved successfully!")
