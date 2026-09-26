import pandas as pd
import numpy as np
from scipy.stats import pearsonr, spearmanr
from statsmodels.stats.outliers_influence import variance_inflation_factor
from statsmodels.tools.tools import add_constant


# ==============================
# 1. Load Cleaned Dataset
# ==============================

df = pd.read_csv("titanic_cleaned.csv")

# Create family size
df["family_size"] = df["sibsp"] + df["parch"] + 1

print("=" * 60)
print("STAGE 5: CORRELATION ANALYSIS")
print("=" * 60)

print("\nDataset shape:")
print(df.shape)


# ==============================
# 2. Select Numerical Variables
# ==============================

numerical_columns = [
    "survived",
    "pclass",
    "age",
    "sibsp",
    "parch",
    "fare",
    "family_size",
]

numeric_data = df[numerical_columns]

print("\nNumerical variables:")
print(numerical_columns)


# ==============================
# 3. Pearson Correlation
# ==============================

print("\n" + "=" * 60)
print("3. PEARSON CORRELATION")
print("=" * 60)

pearson_correlation = numeric_data.corr(method="pearson")

print("\nPearson correlation matrix:")
print(pearson_correlation.round(3))


# ==============================
# 4. Pearson P-values
# ==============================

print("\n" + "=" * 60)
print("4. PEARSON P-VALUES")
print("=" * 60)

pearson_pvalues = pd.DataFrame(
    np.ones((len(numerical_columns), len(numerical_columns))),
    columns=numerical_columns,
    index=numerical_columns,
)

for column1 in numerical_columns:
    for column2 in numerical_columns:
        if column1 == column2:
            pearson_pvalues.loc[column1, column2] = 0

        else:
            r, p = pearsonr(numeric_data[column1], numeric_data[column2])

            pearson_pvalues.loc[column1, column2] = p

print("\nPearson p-value matrix:")
print(pearson_pvalues.round(4))


# ==============================
# 5. Spearman Correlation
# ==============================

print("\n" + "=" * 60)
print("5. SPEARMAN CORRELATION")
print("=" * 60)

spearman_correlation = numeric_data.corr(method="spearman")

print("\nSpearman correlation matrix:")
print(spearman_correlation.round(3))


# ==============================
# 6. Spearman P-values
# ==============================

print("\n" + "=" * 60)
print("6. SPEARMAN P-VALUES")
print("=" * 60)

spearman_pvalues = pd.DataFrame(
    np.ones((len(numerical_columns), len(numerical_columns))),
    columns=numerical_columns,
    index=numerical_columns,
)

for column1 in numerical_columns:
    for column2 in numerical_columns:
        if column1 == column2:
            spearman_pvalues.loc[column1, column2] = 0

        else:
            rho, p = spearmanr(numeric_data[column1], numeric_data[column2])

            spearman_pvalues.loc[column1, column2] = p

print("\nSpearman p-value matrix:")
print(spearman_pvalues.round(4))


# ==============================
# 7. Correlation with Survival
# ==============================

print("\n" + "=" * 60)
print("7. CORRELATION WITH SURVIVAL")
print("=" * 60)

print("\nPearson correlation with survival:")

for column in numerical_columns:
    if column != "survived":
        r, p = pearsonr(df["survived"], df[column])

        print(f"{column}: r = {r:.4f}, p-value = {p:.6f}")


print("\nSpearman correlation with survival:")

for column in numerical_columns:
    if column != "survived":
        rho, p = spearmanr(df["survived"], df[column])

        print(f"{column}: rho = {rho:.4f}, p-value = {p:.6f}")


# ==============================
# 8. VIF / Multicollinearity
# ==============================

print("\n" + "=" * 60)
print("8. VIF / MULTICOLLINEARITY")
print("=" * 60)

# Do not include survived because it is the target variable.
# We also check the relationship between family_size,
# sibsp and parch.

vif_columns = ["pclass", "age", "fare", "sibsp", "parch", "family_size"]

vif_data = df[vif_columns]

vif_data = add_constant(vif_data)

vif_results = pd.DataFrame()

vif_results["Variable"] = vif_data.columns

vif_results["VIF"] = [
    variance_inflation_factor(vif_data.values, i) for i in range(vif_data.shape[1])
]

print("\nVIF results:")
print(vif_results.round(3))


# ==============================
# 9. Multicollinearity Explanation
# ==============================

print("\n" + "=" * 60)
print("9. MULTICOLLINEARITY EXPLANATION")
print("=" * 60)

print("""
family_size is calculated as:

family_size = sibsp + parch + 1

Therefore, family_size contains the same information
as sibsp and parch.

Because of this exact relationship, very high VIF
values can appear for these variables.

For regression models, it is better to use either:

Option 1:
sibsp + parch

or

Option 2:
family_size

Instead of using all three together.
""")


# ==============================
# 10. Important Correlations
# ==============================

print("=" * 60)
print("10. IMPORTANT CORRELATIONS")
print("=" * 60)

print("\nPearson correlations with survival:")

for column in ["pclass", "age", "sibsp", "parch", "fare", "family_size"]:
    r, p = pearsonr(df["survived"], df[column])

    if abs(r) < 0.10:
        strength = "Very weak"

    elif abs(r) < 0.30:
        strength = "Weak"

    elif abs(r) < 0.50:
        strength = "Moderate"

    else:
        strength = "Strong"

    if p < 0.05:
        significance = "Significant"
    else:
        significance = "Not significant"

    print(f"{column}: r = {r:.4f}, {strength}, {significance}")


# ==============================
# 11. Correlation Interpretation
# ==============================

print("\n" + "=" * 60)
print("11. CORRELATION INTERPRETATION")
print("=" * 60)

print("""
1. Passenger class has a negative correlation with survival.
   Higher class number means a lower passenger class.

2. Fare has a positive correlation with survival.
   Passengers with higher fares generally had higher
   survival rates.

3. Age has a very weak negative correlation with survival.

4. SibSp has a very weak relationship with survival.

5. Parch has a weak positive relationship with survival.

6. Family size has a very weak relationship with survival.

7. Correlation shows association between variables.
   It does not prove that one variable caused another.
""")


# ==============================
# 12. Correlation Summary
# ==============================

print("=" * 60)
print("12. CORRELATION SUMMARY")
print("=" * 60)

print("""
Pearson correlation was used to measure linear relationships.

Spearman correlation was also used because some Titanic
variables are not normally distributed.

The main variables associated with survival were:

- Passenger class
- Fare
- Parch

Age, SibSp and family size showed weaker relationships
with survival.

Strong multicollinearity was found between:

sibsp
parch
family_size

This happens because:

family_size = sibsp + parch + 1

Therefore, family_size should not normally be included
together with sibsp and parch in the same regression model.

Correlation indicates association and should not be
interpreted as proof of causation.
""")


print("\n" + "=" * 60)
print("STAGE 5 COMPLETED")
print("=" * 60)
