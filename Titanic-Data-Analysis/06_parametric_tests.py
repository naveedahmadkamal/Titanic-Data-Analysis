import pandas as pd
import numpy as np
from scipy.stats import levene, ttest_ind, f_oneway
import statsmodels.api as sm
from statsmodels.formula.api import ols


# ==============================
# 1. Load Cleaned Dataset
# ==============================

df = pd.read_csv("titanic_cleaned.csv")

print("=" * 60)
print("STAGE 6: PARAMETRIC TESTS")
print("=" * 60)

print("\nDataset shape:")
print(df.shape)


# ==============================
# 2. Prepare Groups
# ==============================

female_age = df[df["sex"] == "female"]["age"]
male_age = df[df["sex"] == "male"]["age"]

died_fare = df[df["survived"] == 0]["fare"]
survived_fare = df[df["survived"] == 1]["fare"]

died_age = df[df["survived"] == 0]["age"]
survived_age = df[df["survived"] == 1]["age"]


# ==============================
# 3. Levene's Test: Age by Sex
# ==============================

print("\n" + "=" * 60)
print("3. LEVENE'S TEST: AGE BY SEX")
print("=" * 60)

levene_age_stat, levene_age_p = levene(female_age, male_age)

print(f"\nLevene statistic: {levene_age_stat:.4f}")
print(f"P-value: {levene_age_p:.6f}")

if levene_age_p < 0.05:
    print("Result: Variances are significantly different.")
else:
    print("Result: No significant difference in variances.")


# ==============================
# 4. Levene's Test: Fare by Survival
# ==============================

print("\n" + "=" * 60)
print("4. LEVENE'S TEST: FARE BY SURVIVAL")
print("=" * 60)

levene_fare_stat, levene_fare_p = levene(died_fare, survived_fare)

print(f"\nLevene statistic: {levene_fare_stat:.4f}")
print(f"P-value: {levene_fare_p:.6f}")

if levene_fare_p < 0.05:
    print("Result: Variances are significantly different.")
    print("Welch's t-test is preferred.")
else:
    print("Result: No significant difference in variances.")


# ==============================
# 5. Welch's T-Test: Fare by Survival
# ==============================

print("\n" + "=" * 60)
print("5. WELCH'S T-TEST: FARE BY SURVIVAL")
print("=" * 60)

fare_t_stat, fare_t_p = ttest_ind(died_fare, survived_fare, equal_var=False)

print(f"\nDied mean fare: {died_fare.mean():.2f}")
print(f"Survived mean fare: {survived_fare.mean():.2f}")
print(f"T-statistic: {fare_t_stat:.4f}")
print(f"P-value: {fare_t_p:.6f}")

if fare_t_p < 0.05:
    print("Result: Significant difference in mean fare.")
else:
    print("Result: No significant difference in mean fare.")


# ==============================
# 6. Welch's T-Test: Age by Survival
# ==============================

print("\n" + "=" * 60)
print("6. WELCH'S T-TEST: AGE BY SURVIVAL")
print("=" * 60)

age_t_stat, age_t_p = ttest_ind(died_age, survived_age, equal_var=False)

print(f"\nDied mean age: {died_age.mean():.2f}")
print(f"Survived mean age: {survived_age.mean():.2f}")
print(f"T-statistic: {age_t_stat:.4f}")
print(f"P-value: {age_t_p:.6f}")

if age_t_p < 0.05:
    print("Result: Significant difference in mean age.")
else:
    print("Result: No significant difference in mean age.")


# ==============================
# 7. One-Way ANOVA: Age by Class
# ==============================

print("\n" + "=" * 60)
print("7. ONE-WAY ANOVA: AGE BY PASSENGER CLASS")
print("=" * 60)

first_class_age = df[df["pclass"] == 1]["age"]
second_class_age = df[df["pclass"] == 2]["age"]
third_class_age = df[df["pclass"] == 3]["age"]

anova_age_stat, anova_age_p = f_oneway(
    first_class_age, second_class_age, third_class_age
)

print(f"\nFirst class mean age: {first_class_age.mean():.2f}")
print(f"Second class mean age: {second_class_age.mean():.2f}")
print(f"Third class mean age: {third_class_age.mean():.2f}")

print(f"F-statistic: {anova_age_stat:.4f}")
print(f"P-value: {anova_age_p:.6f}")

if anova_age_p < 0.05:
    print("Result: Significant difference in mean age among classes.")
else:
    print("Result: No significant difference among classes.")


# ==============================
# 8. Two-Way ANOVA: Age by Class and Sex
# ==============================

print("\n" + "=" * 60)
print("8. TWO-WAY ANOVA: AGE BY CLASS AND SEX")
print("=" * 60)

model = ols("age ~ C(pclass) + C(sex) + C(pclass):C(sex)", data=df).fit()

two_way_anova = sm.stats.anova_lm(model, typ=2)

print("\nTwo-way ANOVA results:")
print(two_way_anova)

print("\nInterpretation:")

for effect in ["C(pclass)", "C(sex)", "C(pclass):C(sex)"]:
    p_value = two_way_anova.loc[effect, "PR(>F)"]

    if p_value < 0.05:
        print(f"{effect}: Significant effect.")
    else:
        print(f"{effect}: Not significant.")


# ==============================
# 9. Cohen's d: Fare by Survival
# ==============================

print("\n" + "=" * 60)
print("9. COHEN'S d: FARE BY SURVIVAL")
print("=" * 60)

n1 = len(died_fare)
n2 = len(survived_fare)

mean1 = died_fare.mean()
mean2 = survived_fare.mean()

std1 = died_fare.std()
std2 = survived_fare.std()

pooled_std = np.sqrt(((n1 - 1) * std1**2 + (n2 - 1) * std2**2) / (n1 + n2 - 2))

cohen_d_fare = (mean1 - mean2) / pooled_std

print(f"\nCohen's d: {cohen_d_fare:.4f}")

if abs(cohen_d_fare) < 0.2:
    print("Effect size: Negligible")
elif abs(cohen_d_fare) < 0.5:
    print("Effect size: Small")
elif abs(cohen_d_fare) < 0.8:
    print("Effect size: Medium")
else:
    print("Effect size: Large")


# ==============================
# 10. Cohen's d: Age by Survival
# ==============================

print("\n" + "=" * 60)
print("10. COHEN'S d: AGE BY SURVIVAL")
print("=" * 60)

n1 = len(died_age)
n2 = len(survived_age)

mean1 = died_age.mean()
mean2 = survived_age.mean()

std1 = died_age.std()
std2 = survived_age.std()

pooled_std = np.sqrt(((n1 - 1) * std1**2 + (n2 - 1) * std2**2) / (n1 + n2 - 2))

cohen_d_age = (mean1 - mean2) / pooled_std

print(f"\nCohen's d: {cohen_d_age:.4f}")

if abs(cohen_d_age) < 0.2:
    print("Effect size: Negligible")
elif abs(cohen_d_age) < 0.5:
    print("Effect size: Small")
elif abs(cohen_d_age) < 0.8:
    print("Effect size: Medium")
else:
    print("Effect size: Large")


# ==============================
# 11. Eta Squared: Age by Class
# ==============================

print("\n" + "=" * 60)
print("11. ETA SQUARED: AGE BY PASSENGER CLASS")
print("=" * 60)

grand_mean = df["age"].mean()

ss_between = (
    len(first_class_age) * (first_class_age.mean() - grand_mean) ** 2
    + len(second_class_age) * (second_class_age.mean() - grand_mean) ** 2
    + len(third_class_age) * (third_class_age.mean() - grand_mean) ** 2
)

ss_total = ((df["age"] - grand_mean) ** 2).sum()

eta_squared = ss_between / ss_total

print(f"\nEta squared: {eta_squared:.4f}")

if eta_squared < 0.01:
    print("Effect size: Very small")
elif eta_squared < 0.06:
    print("Effect size: Small")
elif eta_squared < 0.14:
    print("Effect size: Medium")
else:
    print("Effect size: Large")


# ==============================
# 12. Parametric Tests Summary
# ==============================

print("\n" + "=" * 60)
print("12. PARAMETRIC TESTS SUMMARY")
print("=" * 60)

print("""
1. Levene's test for age by sex showed no significant
   difference in variances.

2. Levene's test for fare by survival showed unequal
   variances, so Welch's t-test was used.

3. Welch's t-test showed a significant difference in
   mean fare between passengers who died and survived.

4. Welch's t-test for age by survival did not show a
   statistically significant difference in mean age.

5. One-way ANOVA showed a significant difference in
   mean age among the three passenger classes.

6. Two-way ANOVA showed significant effects of passenger
   class and sex on age.

7. The interaction between passenger class and sex was
   not statistically significant.

8. Cohen's d was used to measure the effect size of the
   difference in fare and age.

9. Eta squared was used to measure the effect of passenger
   class on age.

10. Statistical significance shows evidence of an
    association or difference. It does not prove causation.
""")


print("=" * 60)
print("STAGE 6 COMPLETED")
print("=" * 60)
