import warnings
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    mean_absolute_error,
    mean_squared_error,
    r2_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

warnings.filterwarnings("ignore")

# Define Project Paths
BASE_DIR = Path(__file__).resolve().parent
MODELS_DIR = BASE_DIR / "models"
OUTPUTS_DIR = BASE_DIR / "model_outputs"

MODELS_DIR.mkdir(exist_ok=True)
OUTPUTS_DIR.mkdir(exist_ok=True)


def load_data():
    """Load transaction and budget dataset."""
    tx_file = BASE_DIR / "personal_transactions(1).csv"
    if not tx_file.exists():
        tx_file = BASE_DIR / "personal_transactions.csv"

    print(f"Loading data from: {tx_file}")
    df = pd.read_csv(tx_file)
    return df


def preprocess_transactions(df):
    """Clean and feature-engineer dataset."""
    df = df.copy()

    # Normalize column names
    df.columns = df.columns.str.strip().str.lower().str.replace(" ", "_")

    # Date handling
    if "date" in df.columns:
        df["date"] = pd.to_datetime(df["date"], errors="coerce")
        df["day"] = df["date"].dt.day
        df["month"] = df["date"].dt.month
        df["dayofweek"] = df["date"].dt.dayofweek
        df["year"] = df["date"].dt.year

    # Clean text columns
    text_cols = ["description", "category", "transaction_type"]
    for col in text_cols:
        if col in df.columns:
            df[col] = df[col].astype(str).str.strip().str.lower()

    # Clean numeric columns
    if "amount" in df.columns:
        df["amount"] = pd.to_numeric(df["amount"], errors="coerce").fillna(0.0)

    return df


def perform_eda(df):
    """Run quick Exploratory Data Analysis on transactions."""
    print("\n--- Running Exploratory Data Analysis (EDA) ---")
    
    # 1. Check missing values
    print("Missing values per column:")
    print(df.isnull().sum())
    
    # 2. Value counts for categories
    if "category" in df.columns:
        print("\nTransaction count per category:")
        print(df["category"].value_counts())
        
    # 3. Quick visualization of spending distribution
    if "amount" in df.columns:
        plt.figure(figsize=(8, 4))
        sns.histplot(df["amount"], bins=50, kde=True, color="blue")
        plt.title("Transaction Amount Distribution")
        plt.xlabel("Amount ($)")
        plt.ylabel("Count")
        
        # Save plot to outputs directory
        plot_path = OUTPUTS_DIR / "spending_distribution.png"
        plt.savefig(plot_path)
        plt.close()
        print(f"\nSpending distribution plot saved to: {plot_path}")


def build_categorization_model(df):
    """Model 1: Predict Expense Category based on Description & Amount."""
    print("\n--- Training Transaction Categorizer ---")

    if "category" not in df.columns or "description" not in df.columns:
        print("Required columns missing for categorization model.")
        return None

    data = df.dropna(subset=["category"]).copy()

    X = data[["description", "amount"]]
    y = data["category"]

    # Filter out categories with fewer than 2 members to avoid stratification errors
    class_counts = y.value_counts()
    valid_classes = class_counts[class_counts >= 2].index
    mask = y.isin(valid_classes)

    X_filtered = X[mask]
    y_filtered = y[mask]

    print(f"Filtered out {len(X) - len(X_filtered)} rows with rare categories (< 2 samples).")

    if len(y_filtered.unique()) < 2:
        print("Not enough classes with >= 2 members to train classifier.")
        return None

    X_train, X_test, y_train, y_test = train_test_split(
        X_filtered, y_filtered, test_size=0.2, random_state=42, stratify=y_filtered
    )

    # Preprocessing Pipeline for Text + Numerical (with bi-grams enabled)
    preprocessor = ColumnTransformer(
        transformers=[
            ("text", TfidfVectorizer(max_features=1000, stop_words="english", ngram_range=(1, 2)), "description"),
            ("num", Pipeline([
                ("imputer", SimpleImputer(strategy="median")),
                ("scaler", StandardScaler())
            ]), ["amount"])
        ]
    )

    # Classification Pipeline
    classifier_pipeline = Pipeline([
        ("preprocessor", preprocessor),
        ("classifier", RandomForestClassifier(n_estimators=100, random_state=42))
    ])

    classifier_pipeline.fit(X_train, y_train)

    # Evaluate
    y_pred = classifier_pipeline.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    print(f"Categorization Model Accuracy: {acc:.4f}")
    print("\nClassification Report:\n", classification_report(y_test, y_pred))

    # Save Model
    model_path = MODELS_DIR / "category_classifier.joblib"
    joblib.dump(classifier_pipeline, model_path)
    print(f"Model saved to: {model_path}")

    return classifier_pipeline


def build_expense_predictor(df):
    """Model 2: Predict Future Spending Amount based on Category and Time features."""
    print("\n--- Training Spending Forecasting Model ---")

    required_cols = ["amount", "month", "dayofweek", "category"]
    if not all(col in df.columns for col in required_cols):
        print("Required columns missing for expense regression model.")
        return None

    X = df[["month", "dayofweek", "category"]]
    y = df["amount"]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    cat_features = ["category"]
    num_features = ["month", "dayofweek"]

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), num_features),
            ("cat", OneHotEncoder(handle_unknown="ignore"), cat_features)
        ]
    )

    regressor_pipeline = Pipeline([
        ("preprocessor", preprocessor),
        ("regressor", RandomForestRegressor(n_estimators=100, random_state=42))
    ])

    regressor_pipeline.fit(X_train, y_train)

    # Evaluate
    y_pred = regressor_pipeline.predict(X_test)
    mae = mean_absolute_error(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    r2 = r2_score(y_test, y_pred)

    print(f"Mean Absolute Error (MAE): {mae:.2f}")
    print(f"Root Mean Squared Error (RMSE): {rmse:.2f}")
    print(f"R2 Score: {r2:.4f}")

    # Save Model
    model_path = MODELS_DIR / "expense_forecaster.joblib"
    joblib.dump(regressor_pipeline, model_path)
    print(f"Model saved to: {model_path}")

    return regressor_pipeline


def generate_financial_summary(df):
    """Generate budget vs spending analysis report."""
    print("\n--- Financial Insights Summary ---")

    if "amount" in df.columns:
        total_spent = df["amount"].sum()
        avg_spent = df["amount"].mean()
        print(f"Total Transactions Tracked: {len(df)}")
        print(f"Total Spending: ${total_spent:,.2f}")
        print(f"Average Spending per Transaction: ${avg_spent:,.2f}")

    if "category" in df.columns and "amount" in df.columns:
        category_summary = df.groupby("category")["amount"].sum().reset_index()
        category_summary = category_summary.sort_values(by="amount", ascending=False)

        summary_file = OUTPUTS_DIR / "spending_by_category.csv"
        category_summary.to_csv(summary_file, index=False)
        print(f"\nTop Spending Categories saved to: {summary_file}")
        print(category_summary.head(10))


def main():
    print("=== Starting AI Personal Finance Manager ===")
    
    df = load_data()
    df_clean = preprocess_transactions(df)

    # 1. Run EDA & Export Charts
    perform_eda(df_clean)

    # 2. Train Models
    build_categorization_model(df_clean)
    build_expense_predictor(df_clean)

    # 3. Generate Analytics
    generate_financial_summary(df_clean)

    print("\n=== AI Personal Finance Manager Execution Complete ===")


if __name__ == "__main__":
    main()