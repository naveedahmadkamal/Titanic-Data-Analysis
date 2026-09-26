import pandas as pd

# ==============================
# STAGE 8: FINAL CONCLUSION
# ==============================

df = pd.read_csv("titanic_cleaned.csv")

print("=" * 70)
print("STAGE 8: FINAL CONCLUSION")
print("=" * 70)


# 1. Dataset Overview
print("\n" + "=" * 70)
print("1. DATASET OVERVIEW")
print("=" * 70)

print("\nNumber of rows:", df.shape[0])
print("Number of columns:", df.shape[1])

print("\nMain variables:")
print("""
survived  - Survival status
pclass    - Passenger class
sex       - Passenger gender
age       - Passenger age
sibsp     - Number of siblings/spouses
parch     - Number of parents/children
fare      - Ticket fare
embarked  - Port of embarkation
""")


# 2. Data Cleaning Findings
print("\n" + "=" * 70)
print("2. DATA CLEANING FINDINGS")
print("=" * 70)

print("""
The original dataset contained missing values.

The following cleaning steps were performed:

1. The deck column was removed because it contained a large
   number of missing values.

2. Missing age values were replaced with the median age.

3. Missing embarked values were replaced with the mode.

4. Missing embark_town values were replaced with the mode.

5. The cleaned dataset was saved as titanic_cleaned.csv.

After cleaning, no missing values remained.
""")


# 3. EDA Findings
print("\n" + "=" * 70)
print("3. EDA FINDINGS")
print("=" * 70)

survival_rate = df["survived"].mean() * 100

female_survival = df[df["sex"] == "female"]["survived"].mean() * 100
male_survival = df[df["sex"] == "male"]["survived"].mean() * 100

first_survival = df[df["pclass"] == 1]["survived"].mean() * 100
second_survival = df[df["pclass"] == 2]["survived"].mean() * 100
third_survival = df[df["pclass"] == 3]["survived"].mean() * 100

print(f"\nOverall survival rate: {survival_rate:.2f}%")

print("\nSurvival rate by sex:")
print(f"Female: {female_survival:.2f}%")
print(f"Male: {male_survival:.2f}%")

print("\nSurvival rate by passenger class:")
print(f"First Class: {first_survival:.2f}%")
print(f"Second Class: {second_survival:.2f}%")
print(f"Third Class: {third_survival:.2f}%")

print("""
EDA showed clear differences in survival rates between
male and female passengers and between passenger classes.

Female passengers had a higher survival rate than male
passengers.

First class passengers had a higher survival rate than
second and third class passengers.

Passengers travelling alone had a lower survival rate
than passengers travelling with family.

Higher fare groups also generally showed higher survival
rates.
""")


# 4. Visualization Findings
print("\n" + "=" * 70)
print("4. VISUALIZATION FINDINGS")
print("=" * 70)

print("""
The visualizations helped to understand the patterns in
the Titanic dataset.

Important patterns shown by the graphs include:

1. Survival and death were not equally distributed.

2. Female survival was much higher than male survival.

3. First class showed a higher survival rate than third class.

4. Higher fare groups generally showed higher survival rates.

5. Family size showed different survival patterns.

6. Age groups showed some differences in survival, although
   the differences were smaller than those seen for sex and class.

The graphs provide visual evidence of associations but do not
prove causation.
""")


# 5. Regression Findings
print("\n" + "=" * 70)
print("5. REGRESSION FINDINGS")
print("=" * 70)

print("""
Simple linear regression showed a positive association between
fare and survival.

Multiple linear regression using:

pclass
age
sibsp
parch
fare

showed that these variables were statistically significant
in the fitted linear model.

However, survival is a binary variable with values 0 and 1.
Therefore, logistic regression is more appropriate for modelling
survival.

The logistic regression results showed:

pclass  -> significant
age     -> significant
sibsp   -> significant
parch   -> significant
fare    -> not significant at the 0.05 level

The logistic model had a pseudo R-squared of about 0.129.

These results describe statistical associations and should not
be interpreted as proof of causation.
""")


# 6. ANOVA Findings
print("\n" + "=" * 70)
print("6. ANOVA FINDINGS")
print("=" * 70)

print("""
One-way ANOVA showed a statistically significant difference
in mean fare between passenger classes.

Two-way ANOVA showed that:

1. Passenger class had a significant effect on age.

2. Sex had a significant effect on age.

3. The interaction between passenger class and sex was
   not statistically significant.

ANCOVA also showed that passenger class and age were
statistically significant in the model for fare.

ANOVA results indicate differences between groups and
do not prove causation.
""")


# 7. Correlation Findings
print("\n" + "=" * 70)
print("7. CORRELATION FINDINGS")
print("=" * 70)

correlations = (
    df[["survived", "pclass", "age", "sibsp", "parch", "fare"]]
    .corr()["survived"]
    .drop("survived")
)

print("\nPearson correlation with survival:")
print(correlations.round(4))

print("""
The strongest Pearson correlation with survival was observed
for passenger class, followed by fare.

Passenger class had a negative correlation with survival
because a higher class number represents a lower passenger
class.

Fare had a positive correlation with survival.

Age, sibsp and family size showed weaker relationships.

Correlation measures association and does not prove causation.
""")


# 8. Parametric Test Findings
print("\n" + "=" * 70)
print("8. PARAMETRIC TEST FINDINGS")
print("=" * 70)

print("""
Welch t-test for fare:

The mean fare of survivors was higher than the mean fare of
passengers who died, and the difference was statistically
significant.

Welch t-test for age:

The difference in age between survivors and non-survivors
was not statistically significant at the 0.05 level.

One-way ANOVA for age by passenger class:

There was a statistically significant difference in age
between passenger classes.

Effect size results:

Cohen's d for fare difference was approximately -0.55,
which represents a medium effect in magnitude.

Cohen's d for age difference by survival was approximately
0.13, which represents a small effect.

Eta squared for age by passenger class was approximately
0.12, showing a meaningful group effect.
""")


# 9. Non-Parametric Test Findings
print("\n" + "=" * 70)
print("9. NON-PARAMETRIC TEST FINDINGS")
print("=" * 70)

print("""
Mann-Whitney U test for fare:

The fare distributions of survivors and non-survivors were
statistically different.

Mann-Whitney U test for age:

The age distributions of survivors and non-survivors were
not statistically different.

Kruskal-Wallis test for age by class:

There was a statistically significant difference in age
distributions between passenger classes.

Kruskal-Wallis test for fare by class:

There was a statistically significant difference in fare
distributions between passenger classes.

Chi-square test for class and survival:

There was a statistically significant association between
passenger class and survival.

Chi-square test for embarked and survival:

There was also a statistically significant association
between embarkation point and survival.

Cramer's V showed a moderate association between class and
survival and a weaker association between embarkation point
and survival.
""")


# 10. Overall Titanic Analysis Conclusion
print("\n" + "=" * 70)
print("10. OVERALL TITANIC ANALYSIS CONCLUSION")
print("=" * 70)

print("""
The Titanic dataset was analysed using data cleaning,
exploratory data analysis, visualization, regression,
ANOVA, correlation, parametric tests and non-parametric tests.

The analysis showed that survival was associated with several
passenger characteristics.

Sex showed a large difference in survival rates, with female
passengers having a much higher survival rate than male
passengers.

Passenger class was also strongly associated with survival.
First class passengers had a higher survival rate, while
third class passengers had a lower survival rate.

Fare was positively associated with survival. Higher fare
groups generally showed higher survival rates.

Family-related variables showed different survival patterns.
Passengers travelling with small families generally showed
higher survival rates than passengers travelling alone,
while very large family groups had small sample sizes and
their results should be interpreted carefully.

Statistical tests supported several of these observed
associations.

Overall, the analysis demonstrates how Python can be used to
clean, explore, visualize and statistically analyse a real
dataset.
""")


# 11. Limitations
print("\n" + "=" * 70)
print("11. LIMITATIONS")
print("=" * 70)

print("""
The analysis has some limitations:

1. The Titanic dataset is a historical dataset.

2. Missing age values were replaced using the median.

3. The deck column was removed because of a large amount
   of missing data.

4. Some subgroups contain very few passengers, so their
   percentages may be unstable.

5. Passenger class and fare are strongly related, so their
   effects should be interpreted together.

6. family_size is calculated from sibsp and parch:

   family_size = sibsp + parch + 1

   Therefore, these variables should not all be used together
   in the same regression model because of multicollinearity.

7. Statistical significance does not mean causation.

8. The analysis does not include every possible historical
   or social factor that may have affected survival.
""")


# 12. Final Summary
print("\n" + "=" * 70)
print("12. FINAL SUMMARY")
print("=" * 70)

print("""
Main findings:

1. Overall survival rate was about 38.38%.

2. Female passengers had a much higher survival rate than
   male passengers.

3. First class passengers had the highest survival rate.

4. Third class passengers had the lowest survival rate.

5. Survivors generally paid higher fares.

6. Passenger class was negatively correlated with survival.

7. Fare was positively correlated with survival.

8. Age had a weak relationship with survival.

9. Class and embarkation point were significantly associated
   with survival according to chi-square tests.

10. Fare distributions differed significantly between
    survivors and non-survivors.

11. Age distributions did not differ significantly between
    survivors and non-survivors in the Mann-Whitney test.

12. Logistic regression was more suitable than linear
    regression for the binary survival variable.

13. The results show associations and group differences,
    not causal relationships.
""")


print("\n" + "=" * 70)
print("STAGE 8 COMPLETED")
print("=" * 70)
