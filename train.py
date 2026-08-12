"""
AI-Based Smartphone Decision Support System
train.py -> loads data, cleans it, trains the models, and saves everything
            the app needs (models, scaler, encoders, column list, dataset)
"""

import os
import joblib
import numpy as np
import pandas as pd
import kagglehub

from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics import (mean_absolute_error, mean_squared_error, r2_score,
                             accuracy_score, classification_report, confusion_matrix)

# Anchor the models folder to this file's location, not the current working
# directory. This guarantees train.py and app.py always agree on the path,
# no matter which folder you run the command from.
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(BASE_DIR, "models")
os.makedirs(MODEL_DIR, exist_ok=True)


# ----------------------------------------------------------------------
# 1. Load Dataset
# ----------------------------------------------------------------------
print("Downloading dataset...")
path = kagglehub.dataset_download("githubmasterin/smartphones-cleaned-dataset")
full_file_name = os.path.join(path, "smartphones.csv")

df = pd.read_csv(full_file_name)
print("Number of rows and columns:", df.shape)


# ----------------------------------------------------------------------
# 2. Cleaning
# ----------------------------------------------------------------------
df["front_camera_main_mp"] = df["front_camera_main_mp"].fillna(
    df["front_camera_main_mp"].mean()
)

df.drop(columns=["memory_card_supported", "memory_card_type"], inplace=True)


def phone_category(price):
    if price < 20000:
        return "Budget"
    elif price <= 40000:
        return "Mid-Range"
    else:
        return "Premium"

df["phone_category"] = df["price_inr"].apply(phone_category)

# Keep an untouched copy with readable brand/model names.
# app.py uses this for the rule-based recommender.
phones_original = df.copy()


# ---- ------------------------------------------------------------------
# 3. Encoding
# ----------------------------------------------------------------------
os_encoder = LabelEncoder()
df["os_name"] = os_encoder.fit_transform(df["os_name"])

category_encoder = LabelEncoder()
df["phone_category"] = category_encoder.fit_transform(df["phone_category"])

df = pd.get_dummies(
    df, columns=["smartphone_brand", "processor_brand"], drop_first=True, dtype=int
)

binary_cols = ["has_5g", "has_nfc", "has_ir_blaster", "fast_charging"]
df[binary_cols] = df[binary_cols].astype(int)

df.drop(columns=["processor_name", "model"], inplace=True)


# ----------------------------------------------------------------------
# 4. Features / Targets
# ----------------------------------------------------------------------
X = df.drop(columns=["price_inr", "phone_category"])
y_price = df["price_inr"]
y_category = df["phone_category"]

# The exact column order the models were trained on.
# app.py MUST build new rows using this same order.
feature_columns = X.columns.tolist()

x_train, x_test, y_price_train, y_price_test, y_cat_train, y_cat_test = train_test_split(
    X, y_price, y_category, test_size=0.2, random_state=42, stratify=y_category
)


# ----------------------------------------------------------------------
# 5. Scaling
# ----------------------------------------------------------------------
scaler = StandardScaler()
x_train_scaled = scaler.fit_transform(x_train)
x_test_scaled = scaler.transform(x_test)


# ----------------------------------------------------------------------
# 6. Linear Regression -> predicts price
# ----------------------------------------------------------------------
lr = LinearRegression()
lr.fit(x_train_scaled, y_price_train)
y_pred_lr = lr.predict(x_test_scaled)

print("\n--- Price Regression (Linear Regression) ---")
print("MAE :", mean_absolute_error(y_price_test, y_pred_lr))
print("MSE :", mean_squared_error(y_price_test, y_pred_lr))
print("RMSE:", np.sqrt(mean_squared_error(y_price_test, y_pred_lr)))
print("R2  :", r2_score(y_price_test, y_pred_lr))


# ----------------------------------------------------------------------
# 7. Logistic Regression -> predicts category (Budget/Mid-Range/Premium)
# ----------------------------------------------------------------------
logreg = LogisticRegression(max_iter=1000, random_state=42)
logreg.fit(x_train_scaled, y_cat_train)
y_pred_log = logreg.predict(x_test_scaled)

print("\n--- Category Classification (Logistic Regression) ---")
print("Accuracy:", accuracy_score(y_cat_test, y_pred_log))
print(classification_report(y_cat_test, y_pred_log, target_names=category_encoder.classes_))
print(confusion_matrix(y_cat_test, y_pred_log))


# ----------------------------------------------------------------------
# 8. Save everything the app needs
# ----------------------------------------------------------------------
joblib.dump(lr, os.path.join(MODEL_DIR, "price_model.pkl"))
joblib.dump(logreg, os.path.join(MODEL_DIR, "category_model.pkl"))
joblib.dump(scaler, os.path.join(MODEL_DIR, "scaler.pkl"))
joblib.dump(os_encoder, os.path.join(MODEL_DIR, "os_encoder.pkl"))
joblib.dump(category_encoder, os.path.join(MODEL_DIR, "category_encoder.pkl"))
joblib.dump(feature_columns, os.path.join(MODEL_DIR, "feature_columns.pkl"))

# Lists of valid brand/processor values, used to build dropdowns in app.py
joblib.dump(
    sorted(phones_original["smartphone_brand"].unique().tolist()),
    os.path.join(MODEL_DIR, "brands.pkl"),
)
joblib.dump(
    sorted(phones_original["processor_brand"].unique().tolist()),
    os.path.join(MODEL_DIR, "processor_brands.pkl"),
)
joblib.dump(
    sorted(phones_original["os_name"].unique().tolist()),
    os.path.join(MODEL_DIR, "os_names.pkl"),
)

# Raw, human-readable dataset -> used by the rule-based recommender in app.py
phones_original.to_csv(os.path.join(MODEL_DIR, "phones_original.csv"), index=False)

print(f"\nAll artifacts saved inside the '{MODEL_DIR}/' folder.")
