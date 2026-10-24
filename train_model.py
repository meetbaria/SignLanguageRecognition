
import json
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

# --------------------------------------------------
# 1. Paths and configuration
# --------------------------------------------------
DATA_PATH = Path("data/landmarks.csv")
MODEL_DIR = Path("models")
REPORT_DIR = Path("reports")

MODEL_DIR.mkdir(parents=True, exist_ok=True)
REPORT_DIR.mkdir(parents=True, exist_ok=True)

RANDOM_STATE = 42
TEST_SIZE = 0.20

# --------------------------------------------------
# 2. Load dataset
# --------------------------------------------------
if not DATA_PATH.exists():
    raise FileNotFoundError(
        "Dataset not found. Run collect_data.py first."
    )

df = pd.read_csv(DATA_PATH)

feature_columns = [f"f{i}" for i in range(63)]

if not set(feature_columns + ["label"]).issubset(df.columns):
    raise ValueError("Dataset does not have the expected columns.")

if df[feature_columns + ["label"]].isnull().any().any():
    raise ValueError("Dataset contains missing values.")

X = df[feature_columns]
y = df["label"].astype(str)

print("Dataset shape:", df.shape)
print("Classes:", sorted(y.unique()))
print("Total samples:", len(df))

# --------------------------------------------------
# 3. Split into training and test sets
# --------------------------------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=TEST_SIZE,
    random_state=RANDOM_STATE,
    stratify=y,
)

print("\nTraining samples:", len(X_train))
print("Testing samples:", len(X_test))

# --------------------------------------------------
# 4. Define classifiers
# --------------------------------------------------
models = {
    "Random Forest": RandomForestClassifier(
        n_estimators=300,
        class_weight="balanced",
        random_state=RANDOM_STATE,
        n_jobs=-1,
    ),

    "SVM": Pipeline([
        ("scaler", StandardScaler()),
        ("classifier", SVC(
            kernel="rbf",
            C=10,
            gamma="scale",
            class_weight="balanced",
            probability=True,
            random_state=RANDOM_STATE,
        )),
    ]),
}

# --------------------------------------------------
# 5. Train, evaluate, and save models
# --------------------------------------------------
results = []
trained_models = {}

for name, model in models.items():
    print(f"\n{'=' * 50}")
    print(f"Training {name}...")
    print("=" * 50)

    model.fit(X_train, y_train)
    predictions = model.predict(X_test)

    accuracy = accuracy_score(y_test, predictions)
    macro_f1 = f1_score(
        y_test, predictions, average="macro"
    )

    report = classification_report(
        y_test,
        predictions,
        output_dict=True,
        zero_division=0,
    )

    print(f"Accuracy: {accuracy:.4f}")
    print(f"Macro F1-score: {macro_f1:.4f}")

    print("\nClassification report:")
    print(classification_report(
        y_test,
        predictions,
        zero_division=0,
    ))

    # Save the model
    model_path = MODEL_DIR / (
        "random_forest.joblib"
        if name == "Random Forest"
        else "svm.joblib"
    )

    joblib.dump(model, model_path)

    # Save detailed classification metrics
    with (REPORT_DIR / (
        "random_forest_report.json"
        if name == "Random Forest"
        else "svm_report.json"
    )).open("w", encoding="utf-8") as file:
        json.dump(report, file, indent=2)

    # Save confusion matrix
    labels = sorted(y.unique())
    cm = confusion_matrix(
        y_test, predictions, labels=labels
    )

    plt.figure(figsize=(9, 7))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=labels,
        yticklabels=labels,
    )
    plt.xlabel("Predicted label")
    plt.ylabel("True label")
    plt.title(f"{name} - Confusion Matrix")
    plt.tight_layout()

    filename = (
        "random_forest_cm.png"
        if name == "Random Forest"
        else "svm_cm.png"
    )
    plt.savefig(REPORT_DIR / filename, dpi=150)
    plt.close()

    results.append({
        "Model": name,
        "Accuracy": accuracy,
        "Macro F1-score": macro_f1,
    })

    trained_models[name] = model

# --------------------------------------------------
# 6. Compare models
# --------------------------------------------------
results_df = pd.DataFrame(results).sort_values(
    "Macro F1-score", ascending=False
)

print("\nModel comparison:")
print(results_df.to_string(index=False))

results_df.to_csv(
    REPORT_DIR / "model_comparison.csv",
    index=False,
)

# Save comparison chart
plt.figure(figsize=(8, 5))
results_df.set_index("Model")[
    ["Accuracy", "Macro F1-score"]
].plot(kind="bar", ylim=(0, 1), rot=0)

plt.title("Random Forest vs SVM")
plt.ylabel("Score")
plt.tight_layout()
plt.savefig(
    REPORT_DIR / "model_comparison.png", dpi=150
)
plt.close()

# --------------------------------------------------
# 7. Save the selected model and metadata
# --------------------------------------------------
best_name = results_df.iloc[0]["Model"]

best_bundle = {
    "model": trained_models[best_name],
    "model_name": best_name,
    "feature_columns": feature_columns,
    "classes": sorted(y.unique()),
}

joblib.dump(
    best_bundle,
    MODEL_DIR / "best_model.joblib",
)

print(f"\nBest model by macro F1-score: {best_name}")
print("Saved models to:", MODEL_DIR.resolve())
print("Saved evaluation reports to:", REPORT_DIR.resolve())
print("\nTraining and evaluation completed.")
