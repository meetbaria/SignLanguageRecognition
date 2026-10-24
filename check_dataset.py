
import pandas as pd
from pathlib import Path

path = Path("data/landmarks.csv")

if not path.exists():
    raise FileNotFoundError(
        "Dataset not found. Run collect_data.py first."
    )

df = pd.read_csv(path)

print("Dataset shape:", df.shape)
print("\nSamples per letter:")
print(df["label"].value_counts().sort_index())

print("\nMissing values:", df.isnull().sum().sum())
print("Duplicate rows:", df.duplicated().sum())

assert df.shape[1] == 64, "Expected 63 features and 1 label."
assert df.isnull().sum().sum() == 0, "Missing values found."
assert set(df["label"]).issubset(
    {"A", "B", "C", "D", "E", "F", "L", "Y"}
), "Unexpected labels found."

print("\nDataset structure verified.")
