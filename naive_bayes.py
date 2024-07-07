import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from nltk.tokenize import word_tokenize

df = pd.read_csv('./dataset.csv')

df['text'] = df['text'].apply(lambda x: ' '.join(word_tokenize(x.lower())))

label_encoder = LabelEncoder()
df['language'] = label_encoder.fit_transform(df['language'])

X_train, X_test, y_train, y_test = train_test_split(df['text'], df['language'], test_size=0.2, random_state=42)

vectorizer = CountVectorizer()
X_train_vec = vectorizer.fit_transform(X_train)
X_test_vec = vectorizer.transform(X_test)

nb_classifier = MultinomialNB()
nb_classifier.fit(X_train_vec, y_train)

y_pred_train = nb_classifier.predict(X_train_vec)
y_pred_test = nb_classifier.predict(X_test_vec)

train_accuracy = accuracy_score(y_train, y_pred_train)
test_accuracy = accuracy_score(y_test, y_pred_test)

print(f"Training Accuracy: {train_accuracy * 100:.2f}%")
print(f"Test Accuracy: {test_accuracy * 100:.2f}%")

# Classification report
print("\nClassification Report on Test Data:")
print(classification_report(y_test, y_pred_test, target_names=label_encoder.classes_))

def detect_language(input_text):
    input_text = ' '.join(word_tokenize(input_text.lower()))
    input_vec = vectorizer.transform([input_text])
    prediction = nb_classifier.predict(input_vec)
    predicted_language = label_encoder.inverse_transform(prediction)[0]
    return predicted_language

# user_input = input("Enter a text to detect its language: ")
# predicted_language = detect_language(user_input)
# print(f"Predicted Language: {predicted_language}")