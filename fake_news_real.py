import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, classification_report
from tkinter import *
from tkinter import messagebox

# -------------------- LOAD DATASET --------------------
fake = pd.read_csv("Fake.csv")
true = pd.read_csv("True.csv")

fake["label"] = "FAKE"
true["label"] = "REAL"

df = pd.concat([fake, true], axis=0).reset_index(drop=True)
df = df.sample(frac=1).reset_index(drop=True)
df = df[["title", "text", "label"]].dropna()

# -------------------- USE TITLE + TEXT --------------------
X = df["title"] + " " + df["text"]
y = df["label"]

vectorizer = TfidfVectorizer(stop_words="english", max_df=0.7)
X_vector = vectorizer.fit_transform(X)

# -------------------- TRAIN MODEL --------------------
X_train, X_test, y_train, y_test = train_test_split(X_vector, y, test_size=0.2, random_state=42)

model = MultinomialNB()
model.fit(X_train, y_train)

# -------------------- ACCURACY --------------------
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
report = classification_report(y_test, y_pred)

print(f"Model Accuracy: {accuracy*100:.2f}%")
print("Classification Report:\n", report)

# -------------------- PREDICTION FUNCTION --------------------
def predict_news(headline):
    data = vectorizer.transform([headline])
    prediction = model.predict(data)[0]
    confidence = model.predict_proba(data).max() * 100
    return prediction, confidence

# -------------------- SAVE HISTORY --------------------
def save_history(headline, prediction, confidence):
    with open("history.txt", "a") as file:
        file.write(f"{headline} -> {prediction} ({confidence:.2f}%)\n")

# -------------------- GUI --------------------
def on_predict():
    headline = entry.get().strip()
    if headline == "":
        messagebox.showwarning("Input Error", "Please enter a news headline.")
        return

    prediction, confidence = predict_news(headline)

    # Color coding
    color = "green" if prediction == "REAL" else "red"

    # Update label with color and confidence
    result_label.config(
        text=f"Prediction: {prediction} (Confidence: {confidence:.2f}%)",
        fg=color,
        font=("Arial", 14, "bold")
    )

    # Save to history
    save_history(headline, prediction, confidence)

    # Clear input box
    entry.delete(0, END)

# -------------------- SETUP GUI --------------------
root = Tk()
root.title("Fake News Detector (Real Dataset)")
root.geometry("650x300")

Label(root, text="Enter News Headline:", font=("Arial", 14)).pack(pady=10)
entry = Entry(root, width=80, font=("Arial", 12))
entry.pack(pady=5)

Button(root, text="Predict", command=on_predict, font=("Arial", 12), bg="blue", fg="white").pack(pady=10)

result_label = Label(root, text="", font=("Arial", 14))
result_label.pack(pady=10)

root.mainloop()