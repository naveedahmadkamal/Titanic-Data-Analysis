import pandas as pd
import numpy as np
import statsmodels.api as sm
import statsmodels.formula.api as smf

from scipy.stats import shapiro, chi2_contingency, ttest_ind


# ============================================================
# 1. Load Cleaned Dataset
# ============================================================

df = pd.read_csv("titanic_cleaned.csv")

print("=" * 60)
print("1. LOAD CLEANED DATASET")
print("=" * 60)

print("Dataset shape:", df.shape)

print("\nFirst five rows:")
print(df.head())


# ============================================================
# 2. Descriptive Statistics
# ============================================================

print("\n" + "=" * 60)
print("2. DESCRIPTIVE STATISTICS")
print("=" * 60)

# Family size
df["family_size"] = df["sibsp"] + df["parch"] + 1

numerical_columns = [
    "survived",
    "pclass",
    "age",
    "sibsp",
    "parch",
    "fare",
    "family_size"
]

print("\nMean:")
print(df[numerical_columns].mean())

print("\nMedian:")
print(df[numerical_columns].median())

print("\nVariance:")
print(df[numerical_columns].var())

print("\nComplete descriptive statistics:")
print(df[numerical_columns].describe())


# ============================================================
# 3. Normality Test - Shapiro-Wilk
# ============================================================

print("\n" + "=" * 60)
print("3. NORMALITY TEST - SHAPIRO-WILK")
print("=" * 60)

columns_to_test = ["age", "fare", "family_size"]

for col in columns_to_test:

    statistic, p_value = shapiro(df[col])

    print("\nVariable:", col)
    print("Shapiro statistic:", round(statistic, 4))
    print("p-value:", round(p_value, 6))

    if p_value > 0.05:
        print("Result: Data appears normally distributed.")
    else:
        print("Result: Data does not appear normally distributed.")


# ============================================================
# 4. Simple Linear Regression
# ============================================================

print("\n" + "=" * 60)
print("4. SIMPLE LINEAR REGRESSION")
print("=" * 60)

simple_model = smf.ols(
    "survived ~ fare",
    data=df
).fit()

print(simple_model.summary())

print("\nR-squared:", round(simple_model.rsquared, 4))
print("Fare coefficient:", round(simple_model.params["fare"], 4))
print("Fare p-value:", round(simple_model.pvalues["fare"], 6))

if simple_model.pvalues["fare"] < 0.05:
    print("Result: Fare has a statistically significant linear association with survival.")
else:
    print("Result: Fare does not have a statistically significant linear association with survival.")

print("\nNote: Survival is a binary variable, so logistic regression is more suitable than linear regression.")


# ============================================================
# 5. Multiple Linear Regression
# ============================================================

print("\n" + "=" * 60)
print("5. MULTIPLE LINEAR REGRESSION")
print("=" * 60)

multiple_model = smf.ols(
    "survived ~ pclass + age + sibsp + parch + fare",
    data=df
).fit()

print(multiple_model.summary())

print("\nR-squared:", round(multiple_model.rsquared, 4))
print("Adjusted R-squared:", round(multiple_model.rsquared_adj, 4))

print("\nCoefficients:")
print(multiple_model.params)

print("\nP-values:")
print(multiple_model.pvalues)

print("\nInterpretation:")

for variable in ["pclass", "age", "sibsp", "parch", "fare"]:

    p_value = multiple_model.pvalues[variable]

    if p_value < 0.05:
        print(variable, ": statistically significant")
    else:
        print(variable, ": not statistically significant")


# ============================================================
# 6. Logistic Regression
# ============================================================

print("\n" + "=" * 60)
print("6. LOGISTIC REGRESSION")
print("=" * 60)

logistic_model = smf.logit(
    "survived ~ pclass + age + sibsp + parch + fare",
    data=df
).fit(disp=False)

print(logistic_model.summary())

print("\nLogistic Regression Coefficients:")
print(logistic_model.params)

print("\nP-values:")
print(logistic_model.pvalues)

print("\nOdds Ratios:")
odds_ratios = np.exp(logistic_model.params)
print(odds_ratios)

print("\nPseudo R-squared:", round(logistic_model.prsquared, 4))

print("\nInterpretation:")

for variable in ["pclass", "age", "sibsp", "parch", "fare"]:

    p_value = logistic_model.pvalues[variable]
    odds_ratio = odds_ratios[variable]

    print("\nVariable:", variable)
    print("Odds ratio:", round(odds_ratio, 4))
    print("p-value:", round(p_value, 6))

    if p_value < 0.05:
        print("Result: Statistically significant.")
    else:
        print("Result: Not statistically significant.")


# ============================================================
# 7. One-Way ANOVA
# ============================================================

print("\n" + "=" * 60)
print("7. ONE-WAY ANOVA")
print("=" * 60)

# Compare fare among passenger classes

anova_one = smf.ols(
    "fare ~ C(pclass)",
    data=df
).fit()

anova_table_one = sm.stats.anova_lm(
    anova_one,
    typ=2
)

print("\nOne-Way ANOVA: Fare by Passenger Class")
print(anova_table_one)

# Get p-value using the row name
p_value = anova_table_one.loc["C(pclass)", "PR(>F)"]

print("\np-value:", p_value)

if p_value < 0.05:
    print("Result: Mean fare is significantly different among passenger classes.")
else:
    print("Result: No significant difference in mean fare among passenger classes.")


# ============================================================
# 8. Two-Way ANOVA
# ============================================================

print("\n" + "=" * 60)
print("8. TWO-WAY ANOVA")
print("=" * 60)

# Fare is dependent variable
# Passenger class and sex are independent variables

anova_two = smf.ols(
    "fare ~ C(pclass) + C(sex) + C(pclass):C(sex)",
    data=df
).fit()

anova_table_two = sm.stats.anova_lm(
    anova_two,
    typ=2
)

print("\nTwo-Way ANOVA: Fare by Class and Sex")
print(anova_table_two)

print("\nInterpretation:")

for effect in anova_table_two.index:

    p_value = anova_table_two.loc[effect, "PR(>F)"]

    print("\n", effect)
    print("p-value:", p_value)

    if p_value < 0.05:
        print("Result: Statistically significant.")
    else:
        print("Result: Not statistically significant.")


# ============================================================
# 9. ANCOVA
# ============================================================

print("\n" + "=" * 60)
print("9. ANCOVA")
print("=" * 60)

# Compare fare among passenger classes
# while controlling for age

ancova_model = smf.ols(
    "fare ~ C(pclass) + age",
    data=df
).fit()

ancova_table = sm.stats.anova_lm(
    ancova_model,
    typ=2
)

print("\nANCOVA: Fare by Passenger Class while controlling for Age")
print(ancova_table)

print("\nModel R-squared:", round(ancova_model.rsquared, 4))
print("Adjusted R-squared:", round(ancova_model.rsquared_adj, 4))

print("\nInterpretation:")

for effect in ancova_table.index:

    p_value = ancova_table.loc[effect, "PR(>F)"]

    print("\n", effect)
    print("p-value:", p_value)

    if p_value < 0.05:
        print("Result: Statistically significant.")
    else:
        print("Result: Not statistically significant.")


# ============================================================
# 10. Chi-Square Test
# ============================================================

print("\n" + "=" * 60)
print("10. CHI-SQUARE TEST")
print("=" * 60)

# Sex and survival

contingency_table = pd.crosstab(
    df["sex"],
    df["survived"]
)

print("\nContingency Table:")
print(contingency_table)

chi2, p_value, degrees_of_freedom, expected = chi2_contingency(
    contingency_table
)

print("\nChi-square statistic:", round(chi2, 4))
print("Degrees of freedom:", degrees_of_freedom)
print("p-value:", round(p_value, 6))

print("\nExpected frequencies:")
print(expected)

if p_value < 0.05:
    print("\nResult: There is a statistically significant association between sex and survival.")
else:
    print("\nResult: There is no statistically significant association between sex and survival.")


# ============================================================
# 11. Independent T-Test
# ============================================================

print("\n" + "=" * 60)
print("11. INDEPENDENT T-TEST")
print("=" * 60)

# Compare age between male and female passengers

female_age = df[df["sex"] == "female"]["age"]
male_age = df[df["sex"] == "male"]["age"]

t_statistic, p_value = ttest_ind(
    female_age,
    male_age,
    equal_var=False
)

print("\nFemale mean age:", round(female_age.mean(), 2))
print("Male mean age:", round(male_age.mean(), 2))

print("\nT-statistic:", round(t_statistic, 4))
print("p-value:", round(p_value, 6))

if p_value < 0.05:
    print("Result: Mean age is significantly different between males and females.")
else:
    print("Result: No statistically significant difference in mean age between males and females.")


# ============================================================
# 12. Regression and ANOVA Summary
# ============================================================

print("\n" + "=" * 60)
print("12. REGRESSION AND ANOVA SUMMARY")
print("=" * 60)

print("""
1. Descriptive statistics were calculated for the main numerical variables.

2. Shapiro-Wilk tests showed that age, fare and family size
   do not follow a normal distribution.

3. Simple linear regression showed a statistically significant
   linear association between fare and survival.

4. Multiple linear regression showed statistically significant
   associations for pclass, age, sibsp, parch and fare.

5. Logistic regression is more appropriate for the binary
   survival variable.

6. In logistic regression, pclass, age, sibsp and parch were
   statistically significant at the 0.05 level, while fare
   was not statistically significant at the 0.05 level.

7. One-way ANOVA showed that average fare differs among
   passenger classes.

8. Two-way ANOVA examined passenger class, sex and their
   interaction in relation to fare.

9. ANCOVA examined passenger class and fare while controlling
   for age.

10. Chi-square test showed a statistically significant
    association between sex and survival.

11. Independent t-test was used to compare age between
    male and female passengers.

12. Statistical significance shows an association or difference;
    it does not by itself prove causation.
""")

print("=" * 60)
print("REGRESSION AND ANOVA ANALYSIS COMPLETED")
print("=" * 60)
