import pandas as pd

# Stage 2: Exploratory Data Analysis

df = pd.read_csv("titanic_cleaned.csv")

print("Data loaded")
print(df.shape)


# ==============================
# 1. Overall Survival Rate
# ==============================

print("\n1. Overall Survival Rate")

survival_rate = df["survived"].mean() * 100
print("Survival rate:", round(survival_rate, 2), "%")


# ==============================
# 2. Survival Count
# ==============================

print("\n2. Survival Count")

print(df["survived"].value_counts())


# ==============================
# 3. Survival Percentage
# ==============================

print("\n3. Survival Percentage")

survival_percent = df["survived"].value_counts(normalize=True) * 100
print(survival_percent.round(2))


# ==============================
# 4. Sex and Survival
# ==============================

print("\n4. Sex and Survival")

print(pd.crosstab(df["sex"], df["survived"]))

sex_rate = df.groupby("sex")["survived"].mean() * 100

print("\nSurvival rate:")
print(sex_rate.round(2))


# ==============================
# 5. Class and Survival
# ==============================

print("\n5. Class and Survival")

print(pd.crosstab(df["pclass"], df["survived"]))

class_rate = df.groupby("pclass")["survived"].mean() * 100

print("\nSurvival rate:")
print(class_rate.round(2))


# ==============================
# 6. Age and Survival
# ==============================

print("\n6. Age and Survival")

age_result = df.groupby("survived")["age"].agg(["mean", "median", "min", "max"])

print(age_result.round(2))


# ==============================
# 7. Fare and Survival
# ==============================

print("\n7. Fare and Survival")

fare_result = df.groupby("survived")["fare"].agg(["mean", "median", "min", "max"])

print(fare_result.round(2))


# ==============================
# 8. Embarked and Survival
# ==============================

print("\n8. Embarked and Survival")

print(pd.crosstab(df["embarked"], df["survived"]))

embarked_rate = df.groupby("embarked")["survived"].mean() * 100

print("\nSurvival rate:")
print(embarked_rate.round(2))


# ==============================
# 9. Alone and Survival
# ==============================

print("\n9. Alone and Survival")

print(pd.crosstab(df["alone"], df["survived"]))

alone_rate = df.groupby("alone")["survived"].mean() * 100

print("\nSurvival rate:")
print(alone_rate.round(2))


# ==============================
# 10. Family Size
# ==============================

print("\n10. Family Size")

df["family_size"] = df["sibsp"] + df["parch"] + 1

print(df["family_size"].describe().round(2))

family_rate = df.groupby("family_size")["survived"].mean() * 100

print("\nSurvival rate:")
print(family_rate.round(2))


# ==============================
# 11. Age Groups
# ==============================

print("\n11. Age Groups")

bins = [0, 12, 17, 30, 50, float("inf")]
labels = ["Child", "Teenager", "Young Adult", "Adult", "Senior"]

df["age_group"] = pd.cut(df["age"], bins=bins, labels=labels)

print(df["age_group"].value_counts().sort_index())

age_group_rate = df.groupby("age_group", observed=True)["survived"].mean() * 100

print("\nSurvival rate:")
print(age_group_rate.round(2))


# ==============================
# 12. Who Category and Survival
# ==============================

print("\n12. Who Category and Survival")

print(pd.crosstab(df["who"], df["survived"]))

who_rate = df.groupby("who")["survived"].mean() * 100

print("\nSurvival rate:")
print(who_rate.round(2))


# ==============================
# 13. Sex and Class
# ==============================

print("\n13. Sex and Class")

result = pd.crosstab([df["sex"], df["pclass"]], df["survived"])

print(result)

rate = df.groupby(["sex", "pclass"])["survived"].mean() * 100

print("\nSurvival rate:")
print(rate.round(2))


# ==============================
# 14. Class and Age Group
# ==============================

print("\n14. Class and Age Group")

result = pd.crosstab([df["pclass"], df["age_group"]], df["survived"])

print(result)

rate = df.groupby(["pclass", "age_group"], observed=True)["survived"].mean() * 100

print("\nSurvival rate:")
print(rate.round(2))


# ==============================
# 15. Sex, Class and Age Group
# ==============================

print("\n15. Sex, Class and Age Group")

result = df.groupby(["sex", "pclass", "age_group"], observed=True)["survived"].agg(
    ["count", "mean"]
)

result["survival_percent"] = result["mean"] * 100

print(result.round(2))


# ==============================
# 16. Family Size and Sex
# ==============================

print("\n16. Family Size and Sex")

result = df.groupby(["family_size", "sex"])["survived"].agg(["count", "mean"])

result["survival_percent"] = result["mean"] * 100

print(result.round(2))


# ==============================
# 17. Family Size Groups
# ==============================

print("\n17. Family Size Groups")

family_bins = [0, 1, 4, 6, float("inf")]
family_labels = ["Alone", "Small Family", "Medium Family", "Large Family"]

df["family_size_group"] = pd.cut(
    df["family_size"], bins=family_bins, labels=family_labels
)

print(df["family_size_group"].value_counts().sort_index())

rate = df.groupby("family_size_group", observed=True)["survived"].mean() * 100

print("\nSurvival rate:")
print(rate.round(2))


# ==============================
# 18. Class and Family Size Group
# ==============================

print("\n18. Class and Family Size Group")

result = df.groupby(["pclass", "family_size_group"], observed=True)["survived"].agg(
    ["count", "mean"]
)

result["survival_percent"] = result["mean"] * 100

print(result.round(2))


# ==============================
# 19. Sex, Class and Family Group
# ==============================

print("\n19. Sex, Class and Family Group")

result = df.groupby(["sex", "pclass", "family_size_group"], observed=True)[
    "survived"
].agg(["count", "mean"])

result["survival_percent"] = result["mean"] * 100

print(result.round(2))


# ==============================
# 20. Fare Groups
# ==============================

print("\n20. Fare Groups")

fare_bins = [-1, 10, 25, 50, 100, float("inf")]
fare_labels = ["Low", "Medium-Low", "Medium", "High", "Very High"]

df["fare_group"] = pd.cut(df["fare"], bins=fare_bins, labels=fare_labels)

print(df["fare_group"].value_counts().sort_index())

rate = df.groupby("fare_group", observed=True)["survived"].mean() * 100

print("\nSurvival rate:")
print(rate.round(2))


# ==============================
# 21. Fare Group and Class
# ==============================

print("\n21. Fare Group and Class")

result = df.groupby(["fare_group", "pclass"], observed=True)["survived"].agg(
    ["count", "mean"]
)

result["survival_percent"] = result["mean"] * 100

print(result.round(2))


# ==============================
# 22. Embarked and Class
# ==============================

print("\n22. Embarked and Class")

result = df.groupby(["embarked", "pclass"])["survived"].agg(["count", "mean"])

result["survival_percent"] = result["mean"] * 100

print(result.round(2))


# ==============================
# 23. Sex and Fare Group
# ==============================

print("\n23. Sex and Fare Group")

result = df.groupby(["sex", "fare_group"], observed=True)["survived"].agg(
    ["count", "mean"]
)

result["survival_percent"] = result["mean"] * 100

print(result.round(2))


# ==============================
# 24. Age Group and Sex
# ==============================

print("\n24. Age Group and Sex")

result = df.groupby(["age_group", "sex"], observed=True)["survived"].agg(
    ["count", "mean"]
)

result["survival_percent"] = result["mean"] * 100

print(result.round(2))


# ==============================
# 25. Class, Sex and Fare Group
# ==============================

print("\n25. Class, Sex and Fare Group")

result = df.groupby(["pclass", "sex", "fare_group"], observed=True)["survived"].agg(
    ["count", "mean"]
)

result["survival_percent"] = result["mean"] * 100

print(result.round(2))


# ==============================
# 26. EDA Summary
# ==============================

print("\n26. EDA Summary")

print("Overall survival rate was around 38%.")
print("Female passengers had a higher survival rate than males.")
print("First class passengers had a higher survival rate.")
print("Higher fare groups generally had higher survival rates.")
print("Family size also showed differences in survival.")
print("Age groups showed different survival patterns.")
print("Small groups should be interpreted carefully.")
print("These results show associations, not cause and effect.")


print("\nEDA completed successfully.")
