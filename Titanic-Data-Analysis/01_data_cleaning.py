import pandas as pd

# ==============================
# 1. Load Dataset
# ==============================

df = pd.read_csv("titanic.csv")

print("First five rows:")
print(df.head())


# ==============================
# 2. View Dataset
# ==============================

print("\nLast five rows:")
print(df.tail())


# ==============================
# 3. Dataset Shape
# ==============================

print("\nDataset shape:")
print(df.shape)


# ==============================
# 4. Column Names
# ==============================

print("\nColumn names:")
print(df.columns.tolist())


# ==============================
# 5. Data Types
# ==============================

print("\nData types:")
print(df.dtypes)


# ==============================
# 6. Dataset Information
# ==============================

print("\nDataset information:")
df.info()


# ==============================
# 7. Descriptive Statistics
# ==============================

print("\nDescriptive statistics:")
print(df.describe())


# ==============================
# 8. Unique Values
# ==============================

print("\nUnique values:")

for col in df.columns:
    print(col, ":", df[col].unique()[:20])


# ==============================
# 9. Number of Unique Values
# ==============================

print("\nNumber of unique values:")
print(df.nunique())


# ==============================
# 10. Missing Values
# ==============================

print("\nMissing values:")
print(df.isnull().sum())


# ==============================
# 11. Missing Value Percentage
# ==============================

print("\nMissing value percentage:")
print((df.isnull().sum() / len(df) * 100).round(2))


# ==============================
# 12. Duplicate Rows
# ==============================

print("\nDuplicate rows:")
print(df.duplicated().sum())


# ==============================
# 13. Categorical Variables
# ==============================

print("\nCategorical variables:")

categorical_columns = [
    "survived",
    "pclass",
    "sex",
    "embarked",
    "class",
    "who",
    "adult_male",
    "embark_town",
    "alive",
    "alone",
]

for col in categorical_columns:
    print("\n", col)
    print(df[col].value_counts(dropna=False))


# ==============================
# 14. Numerical Variables
# ==============================

print("\nNumerical variables:")

numerical_columns = df.select_dtypes(include=["int64", "float64"]).columns

print(numerical_columns.tolist())


# ==============================
# 15. Remove Unnecessary Column
# ==============================

print("\nRemoving deck column...")

df = df.drop(columns=["deck"])

print("New shape:", df.shape)


# ==============================
# 16. Handle Missing Age
# ==============================

print("\nHandling missing age values...")

df["age"] = df["age"].fillna(df["age"].median())

print("Missing age:", df["age"].isnull().sum())


# ==============================
# 17. Handle Missing Embarked
# ==============================

print("\nHandling missing embarked values...")

df["embarked"] = df["embarked"].fillna(df["embarked"].mode()[0])

print("Missing embarked:", df["embarked"].isnull().sum())


# ==============================
# 18. Handle Missing Embark Town
# ==============================

print("\nHandling missing embark town values...")

df["embark_town"] = df["embark_town"].fillna(df["embark_town"].mode()[0])

print("Missing embark town:", df["embark_town"].isnull().sum())


# ==============================
# 19. Final Missing Value Check
# ==============================

print("\nFinal missing value check:")
print(df.isnull().sum())


# ==============================
# 20. Save Cleaned Dataset
# ==============================

df.to_csv("titanic_cleaned.csv", index=False)

print("\nCleaned dataset saved successfully.")
print("File: titanic_cleaned.csv")
