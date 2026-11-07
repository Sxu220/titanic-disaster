"""
Titanic survival prediction - Python version.

Loads train.csv / test.csv from the data folder, cleans and engineers features,
fits a logistic regression, and reports training and test accuracy.

The data folder defaults to /app/data (the Docker mount point). Override it with
the DATA_DIR environment variable when running outside Docker.
"""

import os
import sys

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

DATA_DIR = os.environ.get("DATA_DIR", "/app/data")
TRAIN_PATH = os.path.join(DATA_DIR, "train.csv")
TEST_PATH = os.path.join(DATA_DIR, "test.csv")
LABELS_PATH = os.path.join(DATA_DIR, "gender_submission.csv")
OUTPUT_PATH = os.path.join(DATA_DIR, "predictions_python.csv")

FEATURES = [
    "Pclass", "Sex_male", "Age", "SibSp", "Parch", "Fare", "FamilySize",
    "IsAlone", "Embarked_Q", "Embarked_S",
    "Title_Miss", "Title_Mrs", "Title_Master", "Title_Rare",
]


def banner(text):
    print("\n" + "=" * 70)
    print(text)
    print("=" * 70)


def load_csv(path, name):
    if not os.path.exists(path):
        print(f"ERROR: could not find {name} at {path}.")
        print("Download it from https://www.kaggle.com/competitions/titanic/data "
              "and place it in src/data/ (see README.md).")
        sys.exit(1)
    df = pd.read_csv(path)
    print(f"Loaded {name}: {df.shape[0]} rows x {df.shape[1]} columns")
    return df


def extract_title(name):
    title = name.split(",")[1].split(".")[0].strip()
    mapping = {"Mlle": "Miss", "Ms": "Miss", "Mme": "Mrs"}
    title = mapping.get(title, title)
    return title if title in {"Mr", "Mrs", "Miss", "Master"} else "Rare"


def prepare(df, age_by_title, fare_median, embarked_mode, label):
    """Apply the same cleaning/feature steps to any dataframe, printing each change."""
    df = df.copy()
    print(f"\n[{label}] Preparing data")

    df["Title"] = df["Name"].apply(extract_title)
    print(f"[{label}] Added 'Title' from Name -> counts: {df['Title'].value_counts().to_dict()}")

    n = df["Age"].isna().sum()
    df["Age"] = df.apply(
        lambda r: age_by_title.get(r["Title"], age_by_title["_all"]) if pd.isna(r["Age"]) else r["Age"],
        axis=1,
    )
    print(f"[{label}] Filled {n} missing 'Age' values with the training median age for each title")

    n = df["Fare"].isna().sum()
    df["Fare"] = df["Fare"].fillna(fare_median)
    print(f"[{label}] Filled {n} missing 'Fare' values with training median {fare_median:.2f}")

    n = df["Embarked"].isna().sum()
    df["Embarked"] = df["Embarked"].fillna(embarked_mode)
    print(f"[{label}] Filled {n} missing 'Embarked' values with training mode '{embarked_mode}'")

    df["FamilySize"] = df["SibSp"] + df["Parch"] + 1
    df["IsAlone"] = (df["FamilySize"] == 1).astype(int)
    print(f"[{label}] Added 'FamilySize' (SibSp + Parch + 1) and 'IsAlone' "
          f"({df['IsAlone'].mean():.1%} travelling alone)")

    df = pd.get_dummies(df, columns=["Sex", "Embarked", "Title"], dtype=int)
    for col in FEATURES:
        if col not in df.columns:
            df[col] = 0
    print(f"[{label}] One-hot encoded Sex, Embarked, Title")

    dropped = [c for c in ["Cabin", "Ticket", "Name", "PassengerId"] if c in df.columns]
    print(f"[{label}] Not using columns: {dropped} (Cabin is 77% missing; Ticket/Name/Id are identifiers)")
    print(f"[{label}] Final feature set ({len(FEATURES)}): {FEATURES}")
    return df


def main():
    banner("STEP 1: Load training data")
    train = load_csv(TRAIN_PATH, "train.csv")
    print("\nFirst 5 rows:")
    print(train.head().to_string())
    print("\nMissing values per column:")
    print(train.isna().sum()[train.isna().sum() > 0].to_string())

    banner("STEP 2: Explore training data")
    print(f"Overall survival rate: {train['Survived'].mean():.1%}")
    print("\nSurvival rate by sex:")
    print(train.groupby("Sex")["Survived"].mean().round(3).to_string())
    print("\nSurvival rate by passenger class:")
    print(train.groupby("Pclass")["Survived"].mean().round(3).to_string())
    print("\nSurvival rate by port of embarkation:")
    print(train.groupby("Embarked")["Survived"].mean().round(3).to_string())

    banner("STEP 3: Clean and engineer features")
    titles = train["Name"].apply(extract_title)
    age_by_title = train.groupby(titles)["Age"].median().to_dict()
    age_by_title["_all"] = train["Age"].median()
    fare_median = train["Fare"].median()
    embarked_mode = train["Embarked"].mode()[0]
    print("Imputation values learned from the training set only:")
    print(f"  Median age by title: { {k: round(v, 1) for k, v in age_by_title.items()} }")
    print(f"  Median fare: {fare_median:.2f}   Most common port: {embarked_mode}")

    train_p = prepare(train, age_by_title, fare_median, embarked_mode, "train")
    X_train = train_p[FEATURES]
    y_train = train_p["Survived"]

    banner("STEP 4: Fit logistic regression")
    model = LogisticRegression(max_iter=1000)
    model.fit(X_train, y_train)
    print(f"Trained on {X_train.shape[0]} passengers with {X_train.shape[1]} features")
    print("\nCoefficients (positive = more likely to survive):")
    coefs = pd.Series(model.coef_[0], index=FEATURES).sort_values(ascending=False)
    print(coefs.round(3).to_string())

    banner("STEP 5: Training accuracy")
    train_acc = accuracy_score(y_train, model.predict(X_train))
    print(f"Training accuracy: {train_acc:.4f} ({train_acc:.1%})")

    banner("STEP 6: Load test data and predict")
    test = load_csv(TEST_PATH, "test.csv")
    test_p = prepare(test, age_by_title, fare_median, embarked_mode, "test")
    test_pred = model.predict(test_p[FEATURES])
    out = pd.DataFrame({"PassengerId": test["PassengerId"], "Survived": test_pred})
    out.to_csv(OUTPUT_PATH, index=False)
    print(f"\nPredicted survival for {len(out)} test passengers "
          f"({test_pred.mean():.1%} predicted to survive)")
    print(f"Saved predictions to {OUTPUT_PATH}")

    banner("STEP 7: Test accuracy")
    if "Survived" in test.columns:
        truth = test["Survived"]
        source = "Survived column in test.csv"
    elif os.path.exists(LABELS_PATH):
        labels = pd.read_csv(LABELS_PATH)
        truth = test[["PassengerId"]].merge(labels, on="PassengerId", how="left")["Survived"]
        source = "gender_submission.csv"
    else:
        truth = None

    if truth is None:
        print("Kaggle's test.csv has no Survived labels, and gender_submission.csv was not found,")
        print("so test accuracy cannot be computed locally. Upload predictions_python.csv to Kaggle")
        print("to get the official score.")
    else:
        test_acc = accuracy_score(truth, test_pred)
        print(f"Test accuracy against {source}: {test_acc:.4f} ({test_acc:.1%})")
        if source == "gender_submission.csv":
            print("NOTE: Kaggle hides the true test labels. gender_submission.csv is Kaggle's sample")
            print("baseline (all women survive, all men die), so this number measures agreement with")
            print("that baseline, not true accuracy. Submit predictions_python.csv to Kaggle for the")
            print("real test score.")

    banner("DONE")


if __name__ == "__main__":
    main()
