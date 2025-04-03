from flask import Flask, request, render_template
import numpy as np
import pandas as pd
import pickle
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import train_test_split
import warnings
from feature import FeatureExtraction

warnings.filterwarnings('ignore')

# Load dataset
df = pd.read_csv("phishing.csv")

# Debug: Print column names and shape
print("Dataset columns:", df.columns)
print("Dataset shape:", df.shape)

# Ensure correct feature selection
X = df.drop(columns=["class"])
y = df["class"]

# Drop extra index column if present
if X.shape[1] == 31:
    X = X.iloc[:, 1:]  # Adjust for an extra index column

print("Final feature count for training:", X.shape[1])  # Should be 30

# Split dataset
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train model
gbc = GradientBoostingClassifier()
gbc.fit(X_train, y_train)

# Save model
with open("pickle/model.pkl", "wb") as file:
    pickle.dump(gbc, file)

# Load the trained model
with open("pickle/model.pkl", "rb") as file:
    gbc = pickle.load(file)

app = Flask(__name__)

@app.route("/", methods=["GET", "POST"])
def index():
    if request.method == "POST":
        try:
            url = request.form["url"]
            obj = FeatureExtraction(url)
            features = obj.getFeaturesList()

            print("Extracted feature count:", len(features))  # Debugging

            if len(features) != 30:
                raise ValueError(f"Feature count mismatch: Expected 30, got {len(features)}")

            x = np.array(features).reshape(1, -1)

            y_pred = gbc.predict(x)[0]
            y_pro_phishing = gbc.predict_proba(x)[0, 0]
            y_pro_non_phishing = gbc.predict_proba(x)[0, 1]

            return render_template('index.html', xx=round(y_pro_non_phishing, 2), url=url)

        except Exception as e:
            print("Error:", e)
            return render_template("index.html", xx=-1, error=str(e))

    return render_template("index.html", xx=-1)

if __name__ == "__main__":
    app.run(debug=True)
