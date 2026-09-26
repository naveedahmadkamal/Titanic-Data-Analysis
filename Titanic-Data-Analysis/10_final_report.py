# ============================================================
# TITANIC DATA ANALYSIS PROJECT
# STAGE 10: FINAL REPORT
# ============================================================
#
# Developed by: Naveed Ahmad
# Project: Titanic Data Analysis
# Dataset: titanic_cleaned.csv
#
# ============================================================


import pandas as pd
import numpy as np

from scipy import stats

import statsmodels.api as sm
import statsmodels.formula.api as smf
from statsmodels.stats.anova import anova_lm


# ============================================================
# PROJECT INFORMATION
# ============================================================

DEVELOPER = "Naveed Ahmad"
PROJECT_NAME = "Titanic Data Analysis Project"
DATASET_NAME = "titanic_cleaned.csv"
OUTPUT_FILE = "Titanic_Final_Report.txt"


# ============================================================
# LOAD DATASET
# ============================================================

try:
    df = pd.read_csv(DATASET_NAME)

except FileNotFoundError:
    print("\n============================================================")
    print("ERROR")
    print("============================================================")
    print(f"'{DATASET_NAME}' was not found.")
    print("Please keep the CSV file in the same folder as this Python file.")
    print("============================================================")

    exit()


# ============================================================
# BASIC PREPROCESSING
# ============================================================

# ------------------------------------------------------------
# Family Size
# ------------------------------------------------------------

if "family_size" not in df.columns:
    df["family_size"] = df["sibsp"] + df["parch"] + 1


# ------------------------------------------------------------
# Age Group
# ------------------------------------------------------------

if "age_group" not in df.columns:
    age_bins = [0, 12, 17, 30, 50, float("inf")]

    age_labels = ["Child", "Teenager", "Young Adult", "Adult", "Senior"]

    df["age_group"] = pd.cut(df["age"], bins=age_bins, labels=age_labels)


# ------------------------------------------------------------
# Family Size Group
# ------------------------------------------------------------

if "family_size_group" not in df.columns:
    family_bins = [0, 1, 4, 6, float("inf")]

    family_labels = ["Alone", "Small Family", "Medium Family", "Large Family"]

    df["family_size_group"] = pd.cut(
        df["family_size"], bins=family_bins, labels=family_labels
    )


# ------------------------------------------------------------
# Fare Group
# ------------------------------------------------------------

if "fare_group" not in df.columns:
    fare_bins = [-1, 10, 25, 50, 100, float("inf")]

    fare_labels = ["Low", "Medium-Low", "Medium", "High", "Very High"]

    df["fare_group"] = pd.cut(df["fare"], bins=fare_bins, labels=fare_labels)


# ============================================================
# HELPER FUNCTIONS
# ============================================================


def safe_mean(series):

    if series.dropna().empty:
        return np.nan

    return series.dropna().mean()


def pvalue_text(p):

    if pd.isna(p):
        return "Not available"

    if p < 0.001:
        return "p < 0.001 (statistically significant)"

    elif p < 0.01:
        return f"p = {p:.4f} (statistically significant)"

    elif p < 0.05:
        return f"p = {p:.4f} (statistically significant)"

    else:
        return f"p = {p:.4f} (not statistically significant)"


def cohen_d(group1, group2):

    x1 = group1.dropna()
    x2 = group2.dropna()

    n1 = len(x1)
    n2 = len(x2)

    if n1 < 2 or n2 < 2:
        return np.nan

    s1 = x1.std(ddof=1)
    s2 = x2.std(ddof=1)

    pooled_sd = np.sqrt(((n1 - 1) * s1**2 + (n2 - 1) * s2**2) / (n1 + n2 - 2))

    if pooled_sd == 0:
        return np.nan

    return (x1.mean() - x2.mean()) / pooled_sd


def cramers_v(chi2, n, shape):

    rows, cols = shape

    minimum_dimension = min(rows - 1, cols - 1)

    if n == 0 or minimum_dimension == 0:
        return np.nan

    return np.sqrt(chi2 / (n * minimum_dimension))


def safe_value(value, digits=2):

    if pd.isna(value):
        return "Not available"

    return f"{value:.{digits}f}"


# ============================================================
# STAGE 1
# DATA LOADING AND UNDERSTANDING
# ============================================================

total_rows = len(df)

total_columns = len(df.columns)

total_passengers = len(df)

survivors = int(df["survived"].sum())

deaths = int(total_passengers - survivors)

if total_passengers > 0:
    overall_survival_rate = (survivors / total_passengers) * 100

else:
    overall_survival_rate = np.nan


average_age = safe_mean(df["age"])

average_fare = safe_mean(df["fare"])


# ============================================================
# STAGE 2
# DATA CLEANING AND PREPROCESSING
# ============================================================

missing_values = int(df.isna().sum().sum())

duplicate_rows = int(df.duplicated().sum())

missing_by_column = df.isna().sum()


# ============================================================
# STAGE 3
# DATA VISUALIZATION
# ============================================================

# ------------------------------------------------------------
# Survival Distribution
# ------------------------------------------------------------

survival_counts = df["survived"].value_counts().sort_index()

survival_percentages = df["survived"].value_counts(normalize=True).sort_index() * 100

survival_rate = survival_percentages.get(1, np.nan)

death_rate = survival_percentages.get(0, np.nan)


# ------------------------------------------------------------
# Sex and Survival
# ------------------------------------------------------------

sex_survival = df.groupby("sex", observed=True)["survived"].mean() * 100

female_survival = sex_survival.get("female", np.nan)

male_survival = sex_survival.get("male", np.nan)


if pd.notna(female_survival) and pd.notna(male_survival):
    if female_survival > male_survival:
        sex_interpretation = (
            f"Female passengers had a higher observed "
            f"survival rate ({female_survival:.2f}%) "
            f"than male passengers ({male_survival:.2f}%)."
        )

    elif male_survival > female_survival:
        sex_interpretation = (
            f"Male passengers had a higher observed "
            f"survival rate ({male_survival:.2f}%) "
            f"than female passengers ({female_survival:.2f}%)."
        )

    else:
        sex_interpretation = (
            "Male and female passengers had the same observed survival rate."
        )

else:
    sex_interpretation = "The survival comparison by sex could not be calculated."


# ------------------------------------------------------------
# Class and Survival
# ------------------------------------------------------------

class_survival = df.groupby("pclass", observed=True)["survived"].mean() * 100


if len(class_survival) > 0:
    highest_class = class_survival.idxmax()

    highest_class_rate = class_survival.max()

    lowest_class = class_survival.idxmin()

    lowest_class_rate = class_survival.min()

    class_interpretation = (
        f"Class {int(highest_class)} had the highest "
        f"observed survival rate ({highest_class_rate:.2f}%), "
        f"while Class {int(lowest_class)} had the lowest "
        f"observed survival rate ({lowest_class_rate:.2f}%)."
    )

else:
    highest_class = "N/A"
    highest_class_rate = np.nan

    lowest_class = "N/A"
    lowest_class_rate = np.nan

    class_interpretation = "Passenger class survival rates could not be calculated."


# ------------------------------------------------------------
# Age Group and Survival
# ------------------------------------------------------------

age_group_survival = df.groupby("age_group", observed=True)["survived"].mean() * 100


if len(age_group_survival) > 0:
    highest_age_group = age_group_survival.idxmax()

    highest_age_rate = age_group_survival.max()

    lowest_age_group = age_group_survival.idxmin()

    lowest_age_rate = age_group_survival.min()

    age_interpretation = (
        f"Survival rates were different across age groups. "
        f"The {highest_age_group} group had the highest "
        f"observed survival rate ({highest_age_rate:.2f}%), "
        f"while the {lowest_age_group} group had the lowest "
        f"({lowest_age_rate:.2f}%)."
    )

else:
    highest_age_group = "N/A"
    highest_age_rate = np.nan

    lowest_age_group = "N/A"
    lowest_age_rate = np.nan

    age_interpretation = "Age group survival rates could not be calculated."


# ------------------------------------------------------------
# Fare and Survival
# ------------------------------------------------------------

fare_survival_means = df.groupby("survived", observed=True)["fare"].mean()

average_fare_died = fare_survival_means.get(0, np.nan)

average_fare_survived = fare_survival_means.get(1, np.nan)


# ------------------------------------------------------------
# Fare Group
# ------------------------------------------------------------

fare_group_survival = df.groupby("fare_group", observed=True)["survived"].mean() * 100


if len(fare_group_survival) > 0:
    highest_fare_group = fare_group_survival.idxmax()

    highest_fare_group_rate = fare_group_survival.max()

    lowest_fare_group = fare_group_survival.idxmin()

    lowest_fare_group_rate = fare_group_survival.min()

    fare_interpretation = (
        f"The {highest_fare_group} fare group had the "
        f"highest observed survival rate "
        f"({highest_fare_group_rate:.2f}%), while the "
        f"{lowest_fare_group} group had the lowest "
        f"({lowest_fare_group_rate:.2f}%)."
    )

else:
    highest_fare_group = "N/A"
    highest_fare_group_rate = np.nan

    lowest_fare_group = "N/A"
    lowest_fare_group_rate = np.nan

    fare_interpretation = "Fare group survival rates could not be calculated."


# ------------------------------------------------------------
# Embarked and Survival
# ------------------------------------------------------------

embarked_survival = df.groupby("embarked", observed=True)["survived"].mean() * 100


if len(embarked_survival) > 0:
    highest_embarked = embarked_survival.idxmax()

    highest_embarked_rate = embarked_survival.max()

    lowest_embarked = embarked_survival.idxmin()

    lowest_embarked_rate = embarked_survival.min()

    embarked_interpretation = (
        f"Among the recorded embarkation points, "
        f"{highest_embarked} had the highest observed "
        f"survival rate ({highest_embarked_rate:.2f}%), "
        f"while {lowest_embarked} had the lowest "
        f"({lowest_embarked_rate:.2f}%)."
    )

else:
    highest_embarked = "N/A"
    highest_embarked_rate = np.nan

    lowest_embarked = "N/A"
    lowest_embarked_rate = np.nan

    embarked_interpretation = "Embarkation survival rates could not be calculated."


# ------------------------------------------------------------
# Alone and Survival
# ------------------------------------------------------------

alone_survival = df.groupby("alone", observed=True)["survived"].mean() * 100


# ------------------------------------------------------------
# Family Size and Survival
# ------------------------------------------------------------

family_survival = df.groupby("family_size", observed=True)["survived"].mean() * 100


if len(family_survival) > 0:
    highest_family_size = family_survival.idxmax()

    highest_family_rate = family_survival.max()

    lowest_family_size = family_survival.idxmin()

    lowest_family_rate = family_survival.min()

    family_interpretation = (
        f"Survival varied across family sizes. "
        f"Family size {int(highest_family_size)} had "
        f"the highest observed survival rate "
        f"({highest_family_rate:.2f}%), while family size "
        f"{int(lowest_family_size)} had the lowest "
        f"({lowest_family_rate:.2f}%)."
    )

else:
    highest_family_size = "N/A"
    highest_family_rate = np.nan

    lowest_family_size = "N/A"
    lowest_family_rate = np.nan

    family_interpretation = "Family-size survival rates could not be calculated."


# ============================================================
# STAGE 4
# STATISTICAL ANALYSIS / REGRESSION / ANOVA
# ============================================================


# ------------------------------------------------------------
# Simple Linear Regression
# Fare -> Survival
# ------------------------------------------------------------

simple_data = df[["survived", "fare"]].dropna()


try:
    simple_linear_model = smf.ols("survived ~ fare", data=simple_data).fit()

    simple_linear_r2 = simple_linear_model.rsquared

    simple_linear_p = simple_linear_model.pvalues.get("fare", np.nan)

except Exception:
    simple_linear_r2 = np.nan
    simple_linear_p = np.nan


# ------------------------------------------------------------
# Multiple Linear Regression
# ------------------------------------------------------------

regression_columns = ["survived", "pclass", "age", "sibsp", "parch", "fare"]

multiple_data = df[regression_columns].dropna()


try:
    multiple_linear_model = smf.ols(
        "survived ~ pclass + age + sibsp + parch + fare", data=multiple_data
    ).fit()

    multiple_linear_r2 = multiple_linear_model.rsquared

except Exception:
    multiple_linear_model = None
    multiple_linear_r2 = np.nan


# ------------------------------------------------------------
# Logistic Regression
# ------------------------------------------------------------

try:
    logistic_model = smf.logit(
        "survived ~ pclass + age + sibsp + parch + fare", data=multiple_data
    ).fit(disp=False)

    logistic_pseudo_r2 = logistic_model.prsquared

    logistic_pvalues = logistic_model.pvalues

except Exception:
    logistic_model = None
    logistic_pseudo_r2 = np.nan
    logistic_pvalues = {}


# ============================================================
# ANOVA
# ============================================================


# ------------------------------------------------------------
# One-Way ANOVA
# Fare among Passenger Classes
# ------------------------------------------------------------

fare_class_groups = []

for class_value in sorted(df["pclass"].dropna().unique()):
    group = df.loc[df["pclass"] == class_value, "fare"].dropna()

    if len(group) > 1:
        fare_class_groups.append(group)


try:
    anova_fare_stat, anova_fare_p = stats.f_oneway(*fare_class_groups)

except Exception:
    anova_fare_stat = np.nan
    anova_fare_p = np.nan


# ------------------------------------------------------------
# Two-Way ANOVA
# Age by Class and Sex
# ------------------------------------------------------------

anova_data = df[["age", "pclass", "sex"]].dropna()


try:
    two_way_model = smf.ols("age ~ C(pclass) * C(sex)", data=anova_data).fit()

    two_way_anova = anova_lm(two_way_model, typ=2)

    class_anova_p = two_way_anova.loc["C(pclass)", "PR(>F)"]

    sex_anova_p = two_way_anova.loc["C(sex)", "PR(>F)"]

    interaction_anova_p = two_way_anova.loc["C(pclass):C(sex)", "PR(>F)"]

except Exception:
    class_anova_p = np.nan
    sex_anova_p = np.nan
    interaction_anova_p = np.nan


# ------------------------------------------------------------
# ANCOVA
# Fare ~ Class + Age
# ------------------------------------------------------------

ancova_data = df[["fare", "pclass", "age"]].dropna()


try:
    ancova_model = smf.ols("fare ~ C(pclass) + age", data=ancova_data).fit()

    ancova_table = anova_lm(ancova_model, typ=2)

    ancova_class_p = ancova_table.loc["C(pclass)", "PR(>F)"]

    ancova_age_p = ancova_table.loc["age", "PR(>F)"]

except Exception:
    ancova_class_p = np.nan
    ancova_age_p = np.nan


# ============================================================
# STAGE 5
# CORRELATION ANALYSIS
# ============================================================

correlation_columns = [
    "survived",
    "pclass",
    "age",
    "sibsp",
    "parch",
    "fare",
    "family_size",
]

correlation_columns = [column for column in correlation_columns if column in df.columns]


correlation_matrix = df[correlation_columns].corr()


survival_correlations = (
    correlation_matrix["survived"]
    .drop("survived")
    .sort_values(key=lambda x: x.abs(), ascending=False)
)


# ============================================================
# STAGE 6
# PARAMETRIC TESTS
# ============================================================

survived_fare = df.loc[df["survived"] == 1, "fare"].dropna()


died_fare = df.loc[df["survived"] == 0, "fare"].dropna()


survived_age = df.loc[df["survived"] == 1, "age"].dropna()


died_age = df.loc[df["survived"] == 0, "age"].dropna()


# ------------------------------------------------------------
# Welch T-Test - Fare
# ------------------------------------------------------------

try:
    fare_t_stat, fare_t_p = stats.ttest_ind(survived_fare, died_fare, equal_var=False)

except Exception:
    fare_t_stat = np.nan
    fare_t_p = np.nan


# ------------------------------------------------------------
# Welch T-Test - Age
# ------------------------------------------------------------

try:
    age_t_stat, age_t_p = stats.ttest_ind(survived_age, died_age, equal_var=False)

except Exception:
    age_t_stat = np.nan
    age_t_p = np.nan


# ------------------------------------------------------------
# Cohen's d
# ------------------------------------------------------------

fare_cohen_d = cohen_d(survived_fare, died_fare)

age_cohen_d = cohen_d(survived_age, died_age)


# ------------------------------------------------------------
# One-Way ANOVA - Age by Class
# ------------------------------------------------------------

age_class_groups = []

for class_value in sorted(df["pclass"].dropna().unique()):
    group = df.loc[df["pclass"] == class_value, "age"].dropna()

    if len(group) > 1:
        age_class_groups.append(group)


try:
    age_anova_stat, age_anova_p = stats.f_oneway(*age_class_groups)

except Exception:
    age_anova_stat = np.nan
    age_anova_p = np.nan


# ------------------------------------------------------------
# Eta Squared
# ------------------------------------------------------------

age_anova_data = df[["age", "pclass"]].dropna()


grand_mean_age = age_anova_data["age"].mean()

ss_between = 0

ss_total = ((age_anova_data["age"] - grand_mean_age) ** 2).sum()


for class_value, group in age_anova_data.groupby("pclass"):
    class_mean = group["age"].mean()

    ss_between += len(group) * (class_mean - grand_mean_age) ** 2


if ss_total != 0:
    eta_squared_age = ss_between / ss_total

else:
    eta_squared_age = np.nan


# ============================================================
# STAGE 7
# NON-PARAMETRIC TESTS
# ============================================================


# ------------------------------------------------------------
# Mann-Whitney U - Fare
# ------------------------------------------------------------

try:
    fare_u_stat, fare_u_p = stats.mannwhitneyu(
        survived_fare, died_fare, alternative="two-sided"
    )

except Exception:
    fare_u_stat = np.nan
    fare_u_p = np.nan


# ------------------------------------------------------------
# Mann-Whitney U - Age
# ------------------------------------------------------------

try:
    age_u_stat, age_u_p = stats.mannwhitneyu(
        survived_age, died_age, alternative="two-sided"
    )

except Exception:
    age_u_stat = np.nan
    age_u_p = np.nan


# ------------------------------------------------------------
# Kruskal-Wallis - Age by Class
# ------------------------------------------------------------

try:
    age_kw_stat, age_kw_p = stats.kruskal(*age_class_groups)

except Exception:
    age_kw_stat = np.nan
    age_kw_p = np.nan


# ------------------------------------------------------------
# Kruskal-Wallis - Fare by Class
# ------------------------------------------------------------

fare_kw_groups = []

for class_value in sorted(df["pclass"].dropna().unique()):
    group = df.loc[df["pclass"] == class_value, "fare"].dropna()

    if len(group) > 1:
        fare_kw_groups.append(group)


try:
    fare_kw_stat, fare_kw_p = stats.kruskal(*fare_kw_groups)

except Exception:
    fare_kw_stat = np.nan
    fare_kw_p = np.nan


# ============================================================
# CHI-SQUARE TESTS
# ============================================================


# ------------------------------------------------------------
# Class vs Survival
# ------------------------------------------------------------

class_survival_table = pd.crosstab(df["pclass"], df["survived"])


try:
    (class_chi2, class_chi_p, class_dof, class_expected) = stats.chi2_contingency(
        class_survival_table
    )

    class_cramers_v = cramers_v(
        class_chi2, class_survival_table.to_numpy().sum(), class_survival_table.shape
    )

except Exception:
    class_chi2 = np.nan
    class_chi_p = np.nan
    class_dof = np.nan
    class_cramers_v = np.nan


# ------------------------------------------------------------
# Embarked vs Survival
# ------------------------------------------------------------

embarked_survival_table = pd.crosstab(df["embarked"], df["survived"])


try:
    (embarked_chi2, embarked_chi_p, embarked_dof, embarked_expected) = (
        stats.chi2_contingency(embarked_survival_table)
    )

    embarked_cramers_v = cramers_v(
        embarked_chi2,
        embarked_survival_table.to_numpy().sum(),
        embarked_survival_table.shape,
    )

except Exception:
    embarked_chi2 = np.nan
    embarked_chi_p = np.nan
    embarked_dof = np.nan
    embarked_cramers_v = np.nan


# ============================================================
# STAGE 8
# EDA SUMMARY
# ============================================================

fare_survival_difference = average_fare_survived - average_fare_died


average_family_size = df["family_size"].mean()

minimum_family_size = df["family_size"].min()

maximum_family_size = df["family_size"].max()


# ============================================================
# STAGE 9
# ADVANCED ANALYSIS / INSIGHTS
# ============================================================


# ------------------------------------------------------------
# Highest Sex Survival
# ------------------------------------------------------------

if len(sex_survival) > 0:
    highest_sex = sex_survival.idxmax()

    highest_sex_rate = sex_survival.max()

else:
    highest_sex = "N/A"
    highest_sex_rate = np.nan


# ------------------------------------------------------------
# Family Size Identity
# ------------------------------------------------------------

family_size_identity = "family_size = sibsp + parch + 1"


try:
    identity_check = np.allclose(
        df["family_size"], df["sibsp"] + df["parch"] + 1, equal_nan=True
    )

except Exception:
    identity_check = False


if identity_check:
    family_size_relationship = (
        "Family Size was calculated directly from "
        "SibSp and Parch using the formula "
        "family_size = sibsp + parch + 1."
    )

else:
    family_size_relationship = (
        "The Family Size relationship could not be verified for every record."
    )


# ============================================================
# STAGE 10
# INTERACTIVE TITANIC ANALYTICS DASHBOARD
# ============================================================

dashboard_description = """

The Stage 10 Interactive Titanic Analytics Dashboard
was developed using Python, Pandas, Matplotlib and Tkinter.

The dashboard presents the important results of the
Titanic analysis in an interactive visual format.

Main dashboard sections include:

1. Executive Overview
2. Demographics
3. Fare and Embarkation
4. Family and Age
5. Correlation and Statistics
6. Analyst Insights

The dashboard contains:

- KPI Cards
- Survival Distribution
- Survival Percentage
- Sex and Survival
- Class and Survival
- Age and Survival
- Fare and Survival
- Embarked and Survival
- Alone and Survival
- Family Size and Survival
- Age Group and Survival
- Passenger Category and Survival
- Sex and Class
- Class and Age Group
- Sex, Class and Age Group
- Family Size and Sex
- Family Size Group
- Class and Family Size Group
- Sex, Class and Family Group
- Fare Group and Survival
- Fare Group and Class
- Embarked and Class
- Sex and Fare Group
- Age Group and Sex
- Class, Sex and Fare Group
- Correlation Heatmap

The dashboard provides a visual summary of the
complete Titanic analysis workflow.

Developed by:

Naveed Ahmad
"""


# ============================================================
# BUILD FINAL REPORT
# ============================================================

report = f"""

================================================================
                 TITANIC DATA ANALYSIS PROJECT
                        FINAL REPORT
================================================================

                    DEVELOPED BY NAVEED AHMAD

Project:
{PROJECT_NAME}

Dataset:
{DATASET_NAME}

Developer:
{DEVELOPER}


================================================================
PROJECT WORKFLOW
================================================================

Stage 1  -> Data Loading & Understanding
Stage 2  -> Data Cleaning & Preprocessing
Stage 3  -> Data Visualization
Stage 4  -> Statistical Analysis / Regression / ANOVA
Stage 5  -> Correlation Analysis
Stage 6  -> Parametric Statistical Tests
Stage 7  -> Non-Parametric Statistical Tests
Stage 8  -> EDA Summary
Stage 9  -> Advanced Analysis & Insights
Stage 10 -> Interactive Titanic Analytics Dashboard
Final    -> Final Report


================================================================
1. DATASET OVERVIEW
================================================================

Total Rows:
{total_rows}

Total Columns:
{total_columns}

Total Passengers:
{total_passengers}

Survived:
{survivors}

Did Not Survive:
{deaths}

Overall Survival Rate:
{safe_value(overall_survival_rate)}%

Average Age:
{safe_value(average_age)}

Average Fare:
{safe_value(average_fare)}


================================================================
2. DATA CLEANING AND PREPROCESSING
================================================================

The dataset was checked and prepared before performing
the main analysis.

The preprocessing work included:

- Checking missing values.
- Checking duplicate records.
- Preparing numerical variables.
- Creating Family Size.
- Creating Age Groups.
- Creating Family Size Groups.
- Creating Fare Groups.

Current Missing Values:
{missing_values}

Duplicate Rows:
{duplicate_rows}


Family Size Formula:

family_size = sibsp + parch + 1


Age Groups:

Child
Teenager
Young Adult
Adult
Senior


Family Size Groups:

Alone
Small Family
Medium Family
Large Family


Fare Groups:

Low
Medium-Low
Medium
High
Very High


================================================================
3. DATA VISUALIZATION
================================================================

The Data Visualization stage was used to understand
the Titanic dataset through charts and graphical
comparisons.

Instead of looking only at numerical values, charts
were used to make the important patterns easier to
see and understand.

The main visualizations included:

- Survival Distribution
- Survival Percentage
- Sex and Survival
- Class and Survival
- Age and Survival
- Fare and Survival
- Embarked and Survival
- Alone and Survival
- Family Size and Survival
- Age Group and Survival
- Passenger Category and Survival
- Sex and Class
- Class and Age Group
- Sex, Class and Age Group
- Family Size and Sex
- Family Size Group
- Class and Family Size Group
- Sex, Class and Family Group
- Fare Group and Survival
- Fare Group and Class
- Embarked and Class
- Sex and Fare Group
- Age Group and Sex
- Class, Sex and Fare Group
- Correlation Heatmap


----------------------------------------------------------------
3.1 SURVIVAL DISTRIBUTION
----------------------------------------------------------------

Survived:
{survival_counts.get(1, 0)}

Did Not Survive:
{survival_counts.get(0, 0)}

Survival Rate:
{safe_value(survival_rate)}%

Non-Survival Rate:
{safe_value(death_rate)}%


INTERPRETATION:

The survival chart gives the basic picture of the
dataset.

It shows the difference between passengers who survived
and passengers who did not survive.

The observed survival rate was {safe_value(survival_rate)}%.
This means that survival was not the outcome for all
passengers and there was a clear difference between the
two groups.


----------------------------------------------------------------
3.2 SEX AND SURVIVAL
----------------------------------------------------------------

Female Survival Rate:
{safe_value(female_survival)}%

Male Survival Rate:
{safe_value(male_survival)}%


INTERPRETATION:

{sex_interpretation}

The chart makes this difference easier to understand
because the two groups can be compared directly.

This is an observed pattern in the Titanic dataset.
It should not be treated as proof that sex alone
caused the difference in survival.


----------------------------------------------------------------
3.3 PASSENGER CLASS AND SURVIVAL
----------------------------------------------------------------

"""

for class_value, rate in class_survival.items():
    report += f"Class {int(class_value)} Survival Rate: {rate:.2f}%\n"


report += f"""

INTERPRETATION:

{class_interpretation}

The visualization shows that survival was not evenly
distributed among the three passenger classes.

Class should also be considered with other variables
such as Sex, Age and Fare because these characteristics
can appear together in the same passenger groups.


----------------------------------------------------------------
3.4 AGE AND SURVIVAL
----------------------------------------------------------------

INTERPRETATION:

The age visualizations compare the ages of passengers
who survived with those who did not survive.

The age-group analysis gives an easier way to understand
the pattern by dividing passengers into Child, Teenager,
Young Adult, Adult and Senior groups.

{age_interpretation}

The result describes the observed pattern in this
dataset and should be considered together with other
passenger characteristics.


----------------------------------------------------------------
3.5 FARE AND SURVIVAL
----------------------------------------------------------------

Average Fare - Did Not Survive:
{safe_value(average_fare_died)}

Average Fare - Survived:
{safe_value(average_fare_survived)}

Difference:
{safe_value(fare_survival_difference)}


INTERPRETATION:

The fare charts show how ticket prices were distributed
among survivors and non-survivors.

{fare_interpretation}

Fare should not be interpreted separately from Passenger
Class because passengers in different classes generally
had different fare distributions.


----------------------------------------------------------------
3.6 EMBARKATION AND SURVIVAL
----------------------------------------------------------------

{embarked_interpretation}


INTERPRETATION:

The chart shows that survival rates were different
between the recorded embarkation points.

However, the embarkation point alone does not provide
a complete explanation of survival. Other variables,
especially Class, Sex and Fare, should also be considered.


----------------------------------------------------------------
3.7 TRAVELLING ALONE AND SURVIVAL
----------------------------------------------------------------

"""

for value, rate in alone_survival.items():
    report += f"{value}: {rate:.2f}% survival\n"


report += """

INTERPRETATION:

The travelling-alone visualization compares passengers
who were travelling alone with those travelling with
family members.

This helps identify whether travelling arrangement
was associated with different observed survival patterns.

The result should be interpreted together with Family
Size, Age, Sex and Passenger Class.


----------------------------------------------------------------
3.8 FAMILY SIZE AND SURVIVAL
----------------------------------------------------------------

Average Family Size:
{average_family_size:.2f}

Minimum Family Size:
{minimum_family_size:.0f}

Maximum Family Size:
{maximum_family_size:.0f}


INTERPRETATION:

{family_interpretation}

The visualization shows that survival was not necessarily
the same for every family size.

Family Size is derived from SibSp and Parch, so these
variables are mathematically related.


----------------------------------------------------------------
3.9 AGE GROUP ANALYSIS
----------------------------------------------------------------

INTERPRETATION:

Age was converted into five groups:

Child
Teenager
Young Adult
Adult
Senior

This makes it easier to compare groups instead of
looking at individual ages one by one.

{age_interpretation}


----------------------------------------------------------------
3.10 FARE GROUP ANALYSIS
----------------------------------------------------------------

INTERPRETATION:

Fare was divided into five groups:

Low
Medium-Low
Medium
High
Very High

This grouping makes the fare pattern easier to compare.

{fare_interpretation}

The result should be viewed together with Passenger
Class because Fare and Class are related.


----------------------------------------------------------------
3.11 MULTIVARIABLE VISUALIZATION
----------------------------------------------------------------

The project also used charts containing more than
one passenger characteristic.

These included:

- Sex + Class
- Class + Age Group
- Sex + Class + Age Group
- Family Size + Sex
- Class + Family Size Group
- Sex + Class + Family Group
- Fare Group + Class
- Sex + Fare Group
- Age Group + Sex
- Class + Sex + Fare Group


INTERPRETATION:

Multivariable charts provide more detail than a simple
one-variable comparison.

For example, survival can be compared across Sex and
Passenger Class at the same time.

This helps identify patterns that may not be visible
when variables are analyzed separately.


----------------------------------------------------------------
3.12 CORRELATION HEATMAP
----------------------------------------------------------------

INTERPRETATION:

The correlation heatmap provides an overall view of
relationships between numerical variables.

"""

if len(survival_correlations) > 0:
    for variable, correlation in survival_correlations.items():
        report += f"{variable:<15} : {correlation:.4f}\n"


report += """

A positive correlation means that two numerical variables
tend to increase together.

A negative correlation means that one tends to increase
while the other tends to decrease.

A value near zero indicates a weak linear relationship.

Correlation is useful for finding relationships, but
correlation alone does not prove causation.


----------------------------------------------------------------
3.13 VISUALIZATION SUMMARY
----------------------------------------------------------------

The visualization stage made the dataset easier to
understand by converting numerical information into
charts.

The major observations were:

1. Survival and non-survival were clearly different
   groups in the dataset.

2. Male and female passengers had different observed
   survival rates.

3. Passenger Class showed clear differences in
   survival.

4. Age groups showed different survival patterns.

5. Fare groups showed different survival patterns.

6. Embarkation groups showed different observed
   survival rates.

7. Family Size showed different survival patterns.

8. Travelling alone and travelling with family
   showed different patterns.

9. Combined charts helped compare several passenger
   characteristics together.

10. The correlation heatmap provided a quick view
    of numerical relationships.

The visualization stage therefore provided the first
clear picture of the important patterns in the data.

These patterns were then examined further using
statistical tests.


================================================================
4. STATISTICAL ANALYSIS / REGRESSION / ANOVA
================================================================

The statistical stage was used to check the relationships
observed during visualization.

The analysis included:

- Simple Linear Regression
- Multiple Linear Regression
- Logistic Regression
- One-Way ANOVA
- Two-Way ANOVA
- ANCOVA


----------------------------------------------------------------
4.1 SIMPLE LINEAR REGRESSION
----------------------------------------------------------------

Model:

survived ~ fare

R-Squared:
{safe_value(simple_linear_r2, 4)}

Fare p-value:
{pvalue_text(simple_linear_p)}


INTERPRETATION:

This model checks the linear relationship between Fare
and the numerical Survival outcome.

Because Survival is a binary variable, linear regression
has limitations here. Logistic Regression is more suitable
when Survival is treated as a binary outcome.


----------------------------------------------------------------
4.2 MULTIPLE LINEAR REGRESSION
----------------------------------------------------------------

Model:

survived ~ pclass + age + sibsp + parch + fare

R-Squared:
{safe_value(multiple_linear_r2, 4)}


INTERPRETATION:

This model examines several passenger variables together.

The result should be interpreted as a statistical
relationship within the analyzed dataset.

It does not establish that any particular variable
caused survival or non-survival.


----------------------------------------------------------------
4.3 LOGISTIC REGRESSION
----------------------------------------------------------------

Model:

survived ~ pclass + age + sibsp + parch + fare

Pseudo R-Squared:
{safe_value(logistic_pseudo_r2, 4)}


INTERPRETATION:

Logistic Regression is appropriate for a binary outcome
such as Survival.

It estimates how the included variables are related
to the probability of the observed survival outcome.


Logistic Regression P-values:

"""

if hasattr(logistic_pvalues, "items"):
    for variable, p in logistic_pvalues.items():
        if variable == "Intercept":
            continue

        report += f"{variable}: {pvalue_text(p)}\n"


report += f"""


----------------------------------------------------------------
4.4 ONE-WAY ANOVA
----------------------------------------------------------------

Test:

Fare among Passenger Classes

F-Statistic:
{safe_value(anova_fare_stat, 4)}

Result:

{pvalue_text(anova_fare_p)}


INTERPRETATION:

This test checks whether average Fare differed
between Passenger Classes.

The p-value helps determine whether the observed
difference is statistically significant.


----------------------------------------------------------------
4.5 TWO-WAY ANOVA
----------------------------------------------------------------

Dependent Variable:

Age

Factors:

Passenger Class
Sex

Passenger Class:

{pvalue_text(class_anova_p)}

Sex:

{pvalue_text(sex_anova_p)}

Class × Sex Interaction:

{pvalue_text(interaction_anova_p)}


INTERPRETATION:

Two-Way ANOVA examines Age while considering
Passenger Class and Sex together.

The interaction term checks whether the relationship
between Class and Age changes depending on Sex.


----------------------------------------------------------------
4.6 ANCOVA
----------------------------------------------------------------

Model:

fare ~ Passenger Class + Age

Passenger Class:

{pvalue_text(ancova_class_p)}

Age:

{pvalue_text(ancova_age_p)}


INTERPRETATION:

ANCOVA examines Fare differences between classes
while also taking Age into account.

This provides a more controlled comparison than
looking at Passenger Class alone.


================================================================
5. CORRELATION ANALYSIS
================================================================

Correlation with Survival:

"""


for variable, correlation in survival_correlations.items():
    report += f"{variable:<15} : {correlation:.4f}\n"


if len(survival_correlations) > 0:
    strongest_variable = survival_correlations.abs().idxmax()

    strongest_value = survival_correlations[strongest_variable]

    report += f"""

INTERPRETATION:

Among the numerical variables examined,
{strongest_variable} had the strongest absolute
linear correlation with Survival:

r = {strongest_value:.4f}

This only describes the strength and direction
of a linear relationship. It does not prove causation.
"""


report += f"""


================================================================
6. PARAMETRIC STATISTICAL TESTS
================================================================

6.1 WELCH T-TEST - FARE
----------------------------------------------------------------

T-Statistic:
{safe_value(fare_t_stat, 4)}

{pvalue_text(fare_t_p)}

Cohen's d:
{safe_value(fare_cohen_d, 4)}


INTERPRETATION:

The Welch T-Test compares the average Fare between
survivors and non-survivors.

The p-value indicates whether the observed difference
is statistically significant under the test assumptions.


----------------------------------------------------------------
6.2 WELCH T-TEST - AGE
----------------------------------------------------------------

T-Statistic:
{safe_value(age_t_stat, 4)}

{pvalue_text(age_t_p)}

Cohen's d:
{safe_value(age_cohen_d, 4)}


INTERPRETATION:

The test compares the average Age of survivors and
non-survivors.

Cohen's d provides an effect-size measure for the
difference between the two groups.


----------------------------------------------------------------
6.3 ONE-WAY ANOVA - AGE BY CLASS
----------------------------------------------------------------

F-Statistic:
{safe_value(age_anova_stat, 4)}

{pvalue_text(age_anova_p)}

Eta Squared:
{safe_value(eta_squared_age, 4)}


INTERPRETATION:

This test checks whether Age differed between
Passenger Classes.

Eta Squared describes the proportion of age variation
associated with the class grouping in this analysis.


================================================================
7. NON-PARAMETRIC STATISTICAL TESTS
================================================================

7.1 MANN-WHITNEY U - FARE
----------------------------------------------------------------

U-Statistic:
{safe_value(fare_u_stat, 4)}

{pvalue_text(fare_u_p)}


INTERPRETATION:

The Mann-Whitney U test compares Fare distributions
between survivors and non-survivors without requiring
the same assumptions as a standard independent
samples t-test.


----------------------------------------------------------------
7.2 MANN-WHITNEY U - AGE
----------------------------------------------------------------

U-Statistic:
{safe_value(age_u_stat, 4)}

{pvalue_text(age_u_p)}


INTERPRETATION:

This test compares the Age distributions of survivors
and non-survivors using a non-parametric approach.


----------------------------------------------------------------
7.3 KRUSKAL-WALLIS - AGE BY CLASS
----------------------------------------------------------------

H-Statistic:
{safe_value(age_kw_stat, 4)}

{pvalue_text(age_kw_p)}


INTERPRETATION:

The Kruskal-Wallis test checks whether Age distributions
were different across Passenger Classes.


----------------------------------------------------------------
7.4 KRUSKAL-WALLIS - FARE BY CLASS
----------------------------------------------------------------

H-Statistic:
{safe_value(fare_kw_stat, 4)}

{pvalue_text(fare_kw_p)}


INTERPRETATION:

This test examines whether Fare distributions differed
between Passenger Classes.


================================================================
8. CHI-SQUARE TESTS
================================================================

8.1 CLASS VS SURVIVAL
----------------------------------------------------------------

Chi-Square:
{safe_value(class_chi2, 4)}

Degrees of Freedom:
{class_dof}

{pvalue_text(class_chi_p)}

Cramer's V:
{safe_value(class_cramers_v, 4)}


INTERPRETATION:

The Chi-Square test checks whether Passenger Class
and Survival are statistically associated.

Cramer's V provides an effect-size measure for the
association.


----------------------------------------------------------------
8.2 EMBARKED VS SURVIVAL
----------------------------------------------------------------

Chi-Square:
{safe_value(embarked_chi2, 4)}

Degrees of Freedom:
{embarked_dof}

{pvalue_text(embarked_chi_p)}

Cramer's V:
{safe_value(embarked_cramers_v, 4)}


INTERPRETATION:

This test checks whether Embarkation Point and Survival
were statistically associated in the dataset.


================================================================
9. EDA SUMMARY
================================================================

Overall Survival Rate:
{safe_value(overall_survival_rate)}%


Female Survival:
{safe_value(female_survival)}%

Male Survival:
{safe_value(male_survival)}%


Passenger Class Survival:

"""


for class_value, rate in class_survival.items():
    report += f"Class {int(class_value)}: {rate:.2f}%\n"


report += f"""


Fare Difference:

Average Fare of Survivors:
{safe_value(average_fare_survived)}

Average Fare of Non-Survivors:
{safe_value(average_fare_died)}

Difference:
{safe_value(fare_survival_difference)}


INTERPRETATION:

The EDA stage provides a general understanding of
the passenger population before moving to more
advanced statistical analysis.

The main patterns involve Sex, Passenger Class,
Age, Fare, Embarkation and Family Size.


================================================================
10. ADVANCED ANALYSIS AND INSIGHTS
================================================================

Highest Observed Survival by Sex:

{str(highest_sex).title()} =
{safe_value(highest_sex_rate)}%


Highest Observed Class Survival:

Class {highest_class} =
{safe_value(highest_class_rate)}%


Lowest Observed Class Survival:

Class {lowest_class} =
{safe_value(lowest_class_rate)}%


Highest Observed Fare Group:

{highest_fare_group} =
{safe_value(highest_fare_group_rate)}%


Highest Observed Family Size:

{highest_family_size} =
{safe_value(highest_family_rate)}%


INTERPRETATION:

The advanced analysis combines the main variables
to understand the Titanic dataset from different
angles.

The strongest visible patterns were found by comparing
Sex, Class, Fare, Age and Family Size together.

These results are descriptive observations from the
dataset and should be considered alongside the formal
statistical tests.


----------------------------------------------------------------
MULTICOLLINEARITY
----------------------------------------------------------------

{family_size_relationship}

Important relationship:

family_size = sibsp + parch + 1

Because Family Size is calculated from SibSp and Parch,
these variables are mathematically related.

Including all of them in the same regression model can
create multicollinearity.

Therefore, variable selection is important when
building statistical or machine-learning models.


================================================================
11. INTERACTIVE TITANIC ANALYTICS DASHBOARD
================================================================

{dashboard_description}


================================================================
12. MAIN FINDINGS
================================================================

1. The dataset contained {total_passengers} passenger
   records used in the analysis.

2. The overall observed survival rate was
   {safe_value(overall_survival_rate)}%.

3. Male and female passengers showed different
   observed survival rates.

4. Passenger Class showed clear differences in
   observed survival rates.

5. Age groups showed different survival patterns.

6. Fare groups also showed different survival patterns.

7. Embarkation groups showed differences in observed
   survival rates.

8. Family Size showed different survival patterns.

9. Statistical tests were used to examine whether
   observed differences were statistically supported.

10. Logistic Regression was used because Survival
    is a binary outcome.

11. Correlation analysis helped identify relationships
    among numerical variables.

12. Family Size is mathematically related to SibSp
    and Parch, which is important when selecting
    variables for regression.

13. The Interactive Dashboard brought the major
    project results together in one visual interface.


================================================================
13. PROJECT LIMITATIONS
================================================================

The analysis has several limitations:

1. The Titanic dataset represents a historical
   passenger population.

2. The results describe this dataset and should not
   automatically be generalized to other populations.

3. Some variables contain missing values.

4. Some subgroups may contain fewer observations.

5. Passenger Class and Fare are related variables.

6. Family Size is derived from SibSp and Parch.

7. Statistical significance does not prove causation.

8. Regression results depend on the variables included
   in the model.

9. Observational data cannot by itself establish
   cause-and-effect relationships.


================================================================
14. COMPLETE PROJECT WORKFLOW
================================================================

Data Loading
      |
      v
Data Understanding
      |
      v
Data Cleaning
      |
      v
Data Preprocessing
      |
      v
Data Visualization
      |
      v
Statistical Analysis
      |
      v
Regression Analysis
      |
      v
ANOVA / ANCOVA
      |
      v
Correlation Analysis
      |
      v
Parametric Tests
      |
      v
Non-Parametric Tests
      |
      v
Advanced Analysis
      |
      v
Interactive Dashboard
      |
      v
Interpretation
      |
      v
Final Report


================================================================
15. FINAL CONCLUSION
================================================================

The Titanic Data Analysis Project demonstrates a complete
data analysis workflow using Python.

The project started with loading and understanding the
dataset. The data was then checked, cleaned and prepared
for analysis.

The Data Visualization stage converted the dataset into
different charts. These charts helped identify patterns
in Survival, Sex, Passenger Class, Age, Fare, Embarkation
and Family Size.

The visualization stage was important because it made
the data easier to understand before applying statistical
tests.

After visualization, statistical methods were applied
to examine the observed relationships.

The project included:

- Simple Linear Regression
- Multiple Linear Regression
- Logistic Regression
- ANOVA
- ANCOVA
- Correlation Analysis
- Welch T-Test
- Mann-Whitney U Test
- Kruskal-Wallis Test
- Chi-Square Test
- Cohen's d
- Eta Squared
- Cramer's V

The analysis also examined multicollinearity and showed
the mathematical relationship between Family Size,
SibSp and Parch.

The Interactive Titanic Analytics Dashboard then
presented the major results in a practical visual
interface.

The main lesson from the project is that a good data
analysis process should not depend on only one chart
or one statistical test.

Data visualization helps us see the pattern.

Statistical testing helps us examine the evidence.

Correlation helps us understand numerical relationships.

Regression helps us study several variables together.

The dashboard helps communicate the results.

Finally, interpretation connects all of these steps
and explains what the analysis actually shows.

Therefore, this project demonstrates practical skills
in:

- Python
- Pandas
- NumPy
- SciPy
- Statsmodels
- Matplotlib
- Tkinter
- Data Cleaning
- Data Preprocessing
- Exploratory Data Analysis
- Data Visualization
- Statistical Analysis
- Regression
- ANOVA
- ANCOVA
- Correlation
- Hypothesis Testing
- Effect Size Analysis
- Dashboard Development
- Data Interpretation


================================================================
DEVELOPER
================================================================

                    DEVELOPED BY

                     NAVEED AHMAD

                TITANIC DATA ANALYSIS
                     PROJECT


================================================================
FINAL REPORT COMPLETED
================================================================

Developer:
Naveed Ahmad

Project:
Titanic Data Analysis Project

Dataset:
titanic_cleaned.csv

Report:
Titanic_Final_Report.txt

================================================================
"""


# ============================================================
# PRINT FINAL REPORT
# ============================================================

print("\n")
print(report)


# ============================================================
# SAVE FINAL REPORT
# ============================================================

try:
    with open(OUTPUT_FILE, "w", encoding="utf-8") as file:
        file.write(report)

    print("\n")
    print("============================================================")
    print("             FINAL REPORT GENERATED")
    print("============================================================")
    print(f"File: {OUTPUT_FILE}")
    print("Developed by: Naveed Ahmad")
    print("============================================================")

except Exception as error:
    print("\nError while saving report:")
    print(error)


# ============================================================
# END OF STAGE 10
# ============================================================
