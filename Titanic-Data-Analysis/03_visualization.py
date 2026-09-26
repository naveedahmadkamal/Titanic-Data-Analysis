import pandas as pd
import matplotlib.pyplot as plt

# Stage 3: Data Visualization

df = pd.read_csv("titanic_cleaned.csv")

# create required columns
df["family_size"] = df["sibsp"] + df["parch"] + 1

age_bins = [0, 12, 17, 30, 50, float("inf")]
age_labels = ["Child", "Teenager", "Young Adult", "Adult", "Senior"]

df["age_group"] = pd.cut(df["age"], bins=age_bins, labels=age_labels)

family_bins = [0, 1, 4, 6, float("inf")]
family_labels = ["Alone", "Small Family", "Medium Family", "Large Family"]

df["family_size_group"] = pd.cut(
    df["family_size"], bins=family_bins, labels=family_labels
)

fare_bins = [-1, 10, 25, 50, 100, float("inf")]
fare_labels = ["Low", "Medium-Low", "Medium", "High", "Very High"]

df["fare_group"] = pd.cut(df["fare"], bins=fare_bins, labels=fare_labels)


# ==============================
# 1. Survival Distribution
# ==============================

print("\n1. Survival Distribution")

survival_count = df["survived"].value_counts().sort_index()

plt.figure(figsize=(6, 4))
plt.bar(["Did not Survive", "Survived"], survival_count)
plt.title("Survival Distribution")
plt.xlabel("Survival")
plt.ylabel("Number of Passengers")
plt.show()


# ==============================
# 2. Survival Percentage
# ==============================

print("\n2. Survival Percentage")

survival_percent = df["survived"].value_counts(normalize=True).sort_index() * 100

plt.figure(figsize=(6, 4))
plt.bar(["Did not Survive", "Survived"], survival_percent)
plt.title("Survival Percentage")
plt.xlabel("Survival")
plt.ylabel("Percentage")
plt.show()


# ==============================
# 3. Sex and Survival
# ==============================

print("\n3. Sex and Survival")

data = pd.crosstab(df["sex"], df["survived"])

data.plot(kind="bar", figsize=(7, 4))

plt.title("Sex and Survival")
plt.xlabel("Sex")
plt.ylabel("Number of Passengers")
plt.xticks(rotation=0)
plt.legend(["Did not Survive", "Survived"])
plt.tight_layout()
plt.show()


# ==============================
# 4. Class and Survival
# ==============================

print("\n4. Class and Survival")

data = pd.crosstab(df["pclass"], df["survived"])

data.plot(kind="bar", figsize=(7, 4))

plt.title("Passenger Class and Survival")
plt.xlabel("Passenger Class")
plt.ylabel("Number of Passengers")
plt.xticks(rotation=0)
plt.legend(["Did not Survive", "Survived"])
plt.tight_layout()
plt.show()


# ==============================
# 5. Age and Survival
# ==============================

print("\n5. Age and Survival")

plt.figure(figsize=(7, 4))

plt.boxplot(
    [df[df["survived"] == 0]["age"], df[df["survived"] == 1]["age"]],
    tick_labels=["Did not Survive", "Survived"],
)

plt.title("Age Distribution by Survival")
plt.xlabel("Survival")
plt.ylabel("Age")
plt.show()


# ==============================
# 6. Fare and Survival
# ==============================

print("\n6. Fare and Survival")

plt.figure(figsize=(7, 4))

plt.boxplot(
    [df[df["survived"] == 0]["fare"], df[df["survived"] == 1]["fare"]],
    tick_labels=["Did not Survive", "Survived"],
)

plt.title("Fare Distribution by Survival")
plt.xlabel("Survival")
plt.ylabel("Fare")
plt.show()


# ==============================
# 7. Embarked and Survival
# ==============================

print("\n7. Embarked and Survival")

data = pd.crosstab(df["embarked"], df["survived"])

data.plot(kind="bar", figsize=(7, 4))

plt.title("Embarkation and Survival")
plt.xlabel("Embarked")
plt.ylabel("Number of Passengers")
plt.xticks(rotation=0)
plt.legend(["Did not Survive", "Survived"])
plt.tight_layout()
plt.show()


# ==============================
# 8. Alone and Survival
# ==============================

print("\n8. Alone and Survival")

data = pd.crosstab(df["alone"], df["survived"])

data.plot(kind="bar", figsize=(7, 4))

plt.title("Travelling Alone and Survival")
plt.xlabel("Travelling Alone")
plt.ylabel("Number of Passengers")
plt.xticks(rotation=0)
plt.legend(["Did not Survive", "Survived"])
plt.tight_layout()
plt.show()


# ==============================
# 9. Family Size and Survival
# ==============================

print("\n9. Family Size and Survival")

rate = df.groupby("family_size")["survived"].mean() * 100

plt.figure(figsize=(8, 4))
plt.bar(rate.index.astype(str), rate.values)

plt.title("Family Size and Survival Rate")
plt.xlabel("Family Size")
plt.ylabel("Survival Rate (%)")
plt.show()


# ==============================
# 10. Age Group and Survival
# ==============================

print("\n10. Age Group and Survival")

rate = df.groupby("age_group", observed=True)["survived"].mean() * 100

plt.figure(figsize=(8, 4))
plt.bar(rate.index.astype(str), rate.values)

plt.title("Age Group and Survival Rate")
plt.xlabel("Age Group")
plt.ylabel("Survival Rate (%)")
plt.xticks(rotation=20)
plt.tight_layout()
plt.show()


# ==============================
# 11. Who Category and Survival
# ==============================

print("\n11. Who Category and Survival")

data = pd.crosstab(df["who"], df["survived"])

data.plot(kind="bar", figsize=(7, 4))

plt.title("Passenger Category and Survival")
plt.xlabel("Category")
plt.ylabel("Number of Passengers")
plt.xticks(rotation=0)
plt.legend(["Did not Survive", "Survived"])
plt.tight_layout()
plt.show()


# ==============================
# 12. Sex and Class
# ==============================

print("\n12. Sex and Class")

rate = df.groupby(["sex", "pclass"])["survived"].mean() * 100

rate.unstack().plot(kind="bar", figsize=(8, 4))

plt.title("Sex, Class and Survival Rate")
plt.xlabel("Sex")
plt.ylabel("Survival Rate (%)")
plt.xticks(rotation=0)
plt.legend(title="Class")
plt.tight_layout()
plt.show()


# ==============================
# 13. Class and Age Group
# ==============================

print("\n13. Class and Age Group")

rate = df.groupby(["pclass", "age_group"], observed=True)["survived"].mean() * 100

rate.unstack().plot(kind="bar", figsize=(9, 5))

plt.title("Class, Age Group and Survival")
plt.xlabel("Passenger Class")
plt.ylabel("Survival Rate (%)")
plt.xticks(rotation=0)
plt.legend(title="Age Group")
plt.tight_layout()
plt.show()


# ==============================
# 14. Sex, Class and Age Group
# ==============================

print("\n14. Sex, Class and Age Group")

rate = (
    df.groupby(["sex", "pclass", "age_group"], observed=True)["survived"].mean() * 100
)

rate.unstack(level=[1, 2]).plot(kind="bar", figsize=(12, 5))

plt.title("Sex, Class, Age Group and Survival")
plt.xlabel("Sex")
plt.ylabel("Survival Rate (%)")
plt.xticks(rotation=0)
plt.tight_layout()
plt.show()


# ==============================
# 15. Family Size and Sex
# ==============================

print("\n15. Family Size and Sex")

rate = df.groupby(["family_size", "sex"])["survived"].mean() * 100

rate.unstack().plot(kind="bar", figsize=(10, 5))

plt.title("Family Size, Sex and Survival")
plt.xlabel("Family Size")
plt.ylabel("Survival Rate (%)")
plt.xticks(rotation=0)
plt.tight_layout()
plt.show()


# ==============================
# 16. Family Size Group
# ==============================

print("\n16. Family Size Group")

rate = df.groupby("family_size_group", observed=True)["survived"].mean() * 100

plt.figure(figsize=(8, 4))
plt.bar(rate.index.astype(str), rate.values)

plt.title("Family Size Group and Survival")
plt.xlabel("Family Group")
plt.ylabel("Survival Rate (%)")
plt.xticks(rotation=15)
plt.tight_layout()
plt.show()


# ==============================
# 17. Class and Family Size Group
# ==============================

print("\n17. Class and Family Size Group")

rate = (
    df.groupby(["pclass", "family_size_group"], observed=True)["survived"].mean() * 100
)

rate.unstack().plot(kind="bar", figsize=(9, 5))

plt.title("Class, Family Size and Survival")
plt.xlabel("Passenger Class")
plt.ylabel("Survival Rate (%)")
plt.xticks(rotation=0)
plt.tight_layout()
plt.show()


# ==============================
# 18. Sex, Class and Family Group
# ==============================

print("\n18. Sex, Class and Family Group")

rate = (
    df.groupby(["sex", "pclass", "family_size_group"], observed=True)["survived"].mean()
    * 100
)

rate.unstack(level=[1, 2]).plot(kind="bar", figsize=(12, 5))

plt.title("Sex, Class, Family Group and Survival")
plt.xlabel("Sex")
plt.ylabel("Survival Rate (%)")
plt.xticks(rotation=0)
plt.tight_layout()
plt.show()


# ==============================
# 19. Fare Group and Survival
# ==============================

print("\n19. Fare Group and Survival")

rate = df.groupby("fare_group", observed=True)["survived"].mean() * 100

plt.figure(figsize=(8, 4))
plt.bar(rate.index.astype(str), rate.values)

plt.title("Fare Group and Survival")
plt.xlabel("Fare Group")
plt.ylabel("Survival Rate (%)")
plt.xticks(rotation=15)
plt.tight_layout()
plt.show()


# ==============================
# 20. Fare Group and Class
# ==============================

print("\n20. Fare Group and Class")

rate = df.groupby(["fare_group", "pclass"], observed=True)["survived"].mean() * 100

rate.unstack().plot(kind="bar", figsize=(10, 5))

plt.title("Fare Group, Class and Survival")
plt.xlabel("Fare Group")
plt.ylabel("Survival Rate (%)")
plt.xticks(rotation=15)
plt.tight_layout()
plt.show()


# ==============================
# 21. Embarked and Class
# ==============================

print("\n21. Embarked and Class")

rate = df.groupby(["embarked", "pclass"])["survived"].mean() * 100

rate.unstack().plot(kind="bar", figsize=(8, 5))

plt.title("Embarked, Class and Survival")
plt.xlabel("Embarked")
plt.ylabel("Survival Rate (%)")
plt.xticks(rotation=0)
plt.tight_layout()
plt.show()


# ==============================
# 22. Sex and Fare Group
# ==============================

print("\n22. Sex and Fare Group")

rate = df.groupby(["sex", "fare_group"], observed=True)["survived"].mean() * 100

rate.unstack().plot(kind="bar", figsize=(10, 5))

plt.title("Sex, Fare Group and Survival")
plt.xlabel("Sex")
plt.ylabel("Survival Rate (%)")
plt.xticks(rotation=0)
plt.tight_layout()
plt.show()


# ==============================
# 23. Age Group and Sex
# ==============================

print("\n23. Age Group and Sex")

rate = df.groupby(["age_group", "sex"], observed=True)["survived"].mean() * 100

rate.unstack().plot(kind="bar", figsize=(9, 5))

plt.title("Age Group, Sex and Survival")
plt.xlabel("Age Group")
plt.ylabel("Survival Rate (%)")
plt.xticks(rotation=20)
plt.tight_layout()
plt.show()


# ==============================
# 24. Class, Sex and Fare Group
# ==============================

print("\n24. Class, Sex and Fare Group")

rate = (
    df.groupby(["pclass", "sex", "fare_group"], observed=True)["survived"].mean() * 100
)

rate.unstack(level=[1, 2]).plot(kind="bar", figsize=(13, 5))

plt.title("Class, Sex, Fare Group and Survival")
plt.xlabel("Passenger Class")
plt.ylabel("Survival Rate (%)")
plt.xticks(rotation=0)
plt.tight_layout()
plt.show()


# ==============================
# 25. Correlation Heatmap
# ==============================

print("\n25. Correlation Heatmap")

columns = ["survived", "pclass", "age", "sibsp", "parch", "fare", "family_size"]

correlation = df[columns].corr()

plt.figure(figsize=(8, 6))

plt.imshow(correlation, interpolation="nearest")

plt.colorbar()

plt.xticks(range(len(columns)), columns, rotation=45)

plt.yticks(range(len(columns)), columns)

plt.title("Correlation Heatmap")

plt.tight_layout()
plt.show()


# ==============================
# 26. Visualization Summary
# ==============================

print("\n26. Visualization Summary")

print("Survival was higher among female passengers.")
print("First class showed higher survival than third class.")
print("Higher fare groups generally showed higher survival.")
print("Family size showed different survival patterns.")
print("Age groups also showed differences in survival.")
print("The correlation plot shows relationships between numerical variables.")
print("Visual results should be interpreted together with statistical analysis.")

print("\nVisualization completed.")
