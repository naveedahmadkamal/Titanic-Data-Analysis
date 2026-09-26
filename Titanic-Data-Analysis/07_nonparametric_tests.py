import pandas as pd
from scipy.stats import mannwhitneyu
from scipy.stats import kruskal
from scipy.stats import chi2_contingency


# ==============================
# 1. Load Cleaned Dataset
# ==============================

df = pd.read_csv("titanic_cleaned.csv")

print("=" * 60)
print("STAGE 7: NON-PARAMETRIC TESTS")
print("=" * 60)

print("\nDataset shape:")
print(df.shape)


# ==============================
# 2. Prepare Groups
# ==============================

died_fare = df[df["survived"] == 0]["fare"]
survived_fare = df[df["survived"] == 1]["fare"]

died_age = df[df["survived"] == 0]["age"]
survived_age = df[df["survived"] == 1]["age"]

first_age = df[df["pclass"] == 1]["age"]
second_age = df[df["pclass"] == 2]["age"]
third_age = df[df["pclass"] == 3]["age"]

first_fare = df[df["pclass"] == 1]["fare"]
second_fare = df[df["pclass"] == 2]["fare"]
third_fare = df[df["pclass"] == 3]["fare"]


# ==============================
# 3. Mann-Whitney U Test
#    Fare by Survival
# ==============================

print("\n" + "=" * 60)
print("3. MANN-WHITNEY U TEST: FARE BY SURVIVAL")
print("=" * 60)

u_fare, p_fare = mannwhitneyu(died_fare, survived_fare, alternative="two-sided")

print(f"\nDied median fare: {died_fare.median():.2f}")
print(f"Survived median fare: {survived_fare.median():.2f}")
print(f"U-statistic: {u_fare:.4f}")
print(f"P-value: {p_fare:.6f}")

if p_fare < 0.05:
    print("Result: Significant difference in fare distributions.")
else:
    print("Result: No significant difference in fare distributions.")


# ==============================
# 4. Mann-Whitney U Test
#    Age by Survival
# ==============================

print("\n" + "=" * 60)
print("4. MANN-WHITNEY U TEST: AGE BY SURVIVAL")
print("=" * 60)

u_age, p_age = mannwhitneyu(died_age, survived_age, alternative="two-sided")

print(f"\nDied median age: {died_age.median():.2f}")
print(f"Survived median age: {survived_age.median():.2f}")
print(f"U-statistic: {u_age:.4f}")
print(f"P-value: {p_age:.6f}")

if p_age < 0.05:
    print("Result: Significant difference in age distributions.")
else:
    print("Result: No significant difference in age distributions.")


# ==============================
# 5. Kruskal-Wallis Test
#    Age by Passenger Class
# ==============================

print("\n" + "=" * 60)
print("5. KRUSKAL-WALLIS TEST: AGE BY PASSENGER CLASS")
print("=" * 60)

h_age, p_kw_age = kruskal(first_age, second_age, third_age)

print(f"\nFirst class median age: {first_age.median():.2f}")
print(f"Second class median age: {second_age.median():.2f}")
print(f"Third class median age: {third_age.median():.2f}")

print(f"H-statistic: {h_age:.4f}")
print(f"P-value: {p_kw_age:.6f}")

if p_kw_age < 0.05:
    print("Result: Significant difference in age distributions among classes.")
else:
    print("Result: No significant difference among classes.")


# ==============================
# 6. Kruskal-Wallis Test
#    Fare by Passenger Class
# ==============================

print("\n" + "=" * 60)
print("6. KRUSKAL-WALLIS TEST: FARE BY PASSENGER CLASS")
print("=" * 60)

h_fare, p_kw_fare = kruskal(first_fare, second_fare, third_fare)

print(f"\nFirst class median fare: {first_fare.median():.2f}")
print(f"Second class median fare: {second_fare.median():.2f}")
print(f"Third class median fare: {third_fare.median():.2f}")

print(f"H-statistic: {h_fare:.4f}")
print(f"P-value: {p_kw_fare:.6f}")

if p_kw_fare < 0.05:
    print("Result: Significant difference in fare distributions among classes.")
else:
    print("Result: No significant difference among classes.")


# ==============================
# 7. Chi-Square Test
#    Passenger Class vs Survival
# ==============================

print("\n" + "=" * 60)
print("7. CHI-SQUARE TEST: PASSENGER CLASS VS SURVIVAL")
print("=" * 60)

class_survival_table = pd.crosstab(df["pclass"], df["survived"])

print("\nContingency table:")
print(class_survival_table)

chi2_class, p_class, dof_class, expected_class = chi2_contingency(class_survival_table)

print(f"\nChi-square statistic: {chi2_class:.4f}")
print(f"Degrees of freedom: {dof_class}")
print(f"P-value: {p_class:.6f}")

if p_class < 0.05:
    print("Result: Significant association between class and survival.")
else:
    print("Result: No significant association between class and survival.")


# ==============================
# 8. Chi-Square Test
#    Embarked vs Survival
# ==============================

print("\n" + "=" * 60)
print("8. CHI-SQUARE TEST: EMBARKED VS SURVIVAL")
print("=" * 60)

embarked_survival_table = pd.crosstab(df["embarked"], df["survived"])

print("\nContingency table:")
print(embarked_survival_table)

chi2_embarked, p_embarked, dof_embarked, expected_embarked = chi2_contingency(
    embarked_survival_table
)

print(f"\nChi-square statistic: {chi2_embarked:.4f}")
print(f"Degrees of freedom: {dof_embarked}")
print(f"P-value: {p_embarked:.6f}")

if p_embarked < 0.05:
    print("Result: Significant association between embarkation and survival.")
else:
    print("Result: No significant association between embarkation and survival.")


# ==============================
# 9. Chi-Square Effect Size
#    Cramer's V
# ==============================

print("\n" + "=" * 60)
print("9. CRAMER'S V EFFECT SIZE")
print("=" * 60)


def cramers_v(table, chi2):

    n = table.to_numpy().sum()

    rows, columns = table.shape

    minimum_dimension = min(rows - 1, columns - 1)

    return (chi2 / (n * minimum_dimension)) ** 0.5


cramers_v_class = cramers_v(class_survival_table, chi2_class)

cramers_v_embarked = cramers_v(embarked_survival_table, chi2_embarked)

print(f"\nCramer's V - Class vs Survival: {cramers_v_class:.4f}")
print(f"Cramer's V - Embarked vs Survival: {cramers_v_embarked:.4f}")


# ==============================
# 10. Effect Size Interpretation
# ==============================

print("\n" + "=" * 60)
print("10. EFFECT SIZE INTERPRETATION")
print("=" * 60)


def interpret_cramers_v(value):

    if value < 0.10:
        return "Very weak association"
    elif value < 0.30:
        return "Weak association"
    elif value < 0.50:
        return "Moderate association"
    else:
        return "Strong association"


print("Class vs Survival:", interpret_cramers_v(cramers_v_class))

print("Embarked vs Survival:", interpret_cramers_v(cramers_v_embarked))


# ==============================
# 11. Test Results Interpretation
# ==============================

print("\n" + "=" * 60)
print("11. TEST RESULTS INTERPRETATION")
print("=" * 60)

print("""
1. The Mann-Whitney U test showed a significant difference
   in fare distributions between passengers who died and
   passengers who survived.

2. The Mann-Whitney U test for age did not show a
   statistically significant difference in age distributions
   between the two survival groups.

3. The Kruskal-Wallis test showed a significant difference
   in age distributions among the three passenger classes.

4. The Kruskal-Wallis test showed a significant difference
   in fare distributions among the three passenger classes.

5. The Chi-square test showed a significant association
   between passenger class and survival.

6. The Chi-square test also showed a significant association
   between embarkation point and survival.

7. Cramer's V was used to describe the strength of the
   categorical associations.

8. A statistically significant result does not prove
   causation.
""")


# ==============================
# 12. Non-Parametric Tests Summary
# ==============================

print("=" * 60)
print("12. NON-PARAMETRIC TESTS SUMMARY")
print("=" * 60)

print("""
Mann-Whitney U Tests:
- Fare vs Survival: Significant
- Age vs Survival: Not significant

Kruskal-Wallis Tests:
- Age vs Passenger Class: Significant
- Fare vs Passenger Class: Significant

Chi-Square Tests:
- Passenger Class vs Survival: Significant
- Embarked vs Survival: Significant

These tests are useful when normality assumptions are
not satisfied or when the data are ordinal/categorical.

The results show statistical associations or differences,
but they do not establish cause-and-effect relationships.
""")


print("=" * 60)
print("STAGE 7 COMPLETED")
print("=" * 60)
