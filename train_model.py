import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier

# Load dataset
df = pd.read_csv("WA_Fn-UseC_-Telco-Customer-Churn.csv")


# -----------------------------
# DATA CLEANING
# -----------------------------

df["TotalCharges"] = pd.to_numeric(
    df["TotalCharges"],
    errors="coerce"
)

df.dropna(inplace=True)

df.drop("customerID", axis=1, inplace=True)


# -----------------------------
# CONVERT YES / NO
# -----------------------------

binary_columns = [
    "Partner",
    "Dependents",
    "PhoneService",
    "PaperlessBilling",
    "Churn"
]

for column in binary_columns:
    df[column] = df[column].map({
        "Yes": 1,
        "No": 0
    })


# -----------------------------
# CONVERT GENDER
# -----------------------------

df["gender"] = df["gender"].map({
    "Male": 1,
    "Female": 0
})


# -----------------------------
# ONE-HOT ENCODING
# -----------------------------

df = pd.get_dummies(
    df,
    drop_first=True
)


# -----------------------------
# FEATURES AND TARGET
# -----------------------------

X = df.drop("Churn", axis=1)

y = df["Churn"]

# Split data into training and testing sets

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

print("Training data:", X_train.shape)
print("Testing data:", X_test.shape)
print("I AM RUNNING THE NEW CODE")

# Create Random Forest model

model = RandomForestClassifier(
    n_estimators=100,
    random_state=42
)
# Train the model
model.fit(X_train, y_train)

print("Model training completed!")


# -----------------------------
# DISPLAY INFORMATION
# -----------------------------

print("Preprocessing completed!")

print("\nX shape:", X.shape)

print("y shape:", y.shape)

print("\nFeatures:")
print(X.columns.tolist())

print("\nTarget:")
print(y.value_counts())
# Make predictions
y_pred = model.predict(X_test)

print("Predictions completed!")
print(y_pred[:10])
from sklearn.metrics import accuracy_score

accuracy = accuracy_score(y_test, y_pred)

print("Model Accuracy:", accuracy)
print("Model Accuracy (%):", accuracy * 100)
from sklearn.metrics import classification_report, confusion_matrix

print("\nClassification Report:")
print(classification_report(y_test, y_pred))

print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred))
import joblib

joblib.dump(model, "customer_churn_model.pkl")

print("Model saved successfully!")