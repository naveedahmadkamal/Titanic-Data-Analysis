# 🚢 Titanic Data Analysis

A complete **Exploratory Data Analysis (EDA) and Statistical Analysis** project on the Titanic dataset using Python. This project follows a structured data analysis workflow, starting from data cleaning and preprocessing and progressing through descriptive statistics, visualization, regression, correlation, parametric tests, and non-parametric statistical tests.

---

## 📌 Project Overview

The purpose of this project is to analyze the factors associated with passenger survival on the Titanic using statistical and data visualization techniques.

The analysis investigates relationships between survival and variables such as:

* Passenger Sex
* Passenger Class
* Age
* Fare
* Family Size
* Siblings/Spouses
* Parents/Children
* Embarkation Point

The project demonstrates a complete practical workflow for analyzing a real-world dataset using Python.

---

## 🎯 Objectives

The main objectives of this project are:

1. Load and understand the Titanic dataset.
2. Clean and preprocess the data.
3. Handle missing values appropriately.
4. Investigate duplicate observations and potential outliers.
5. Perform descriptive statistical analysis.
6. Explore relationships between variables.
7. Create meaningful data visualizations.
8. Apply regression techniques.
9. Analyze correlations between numerical variables.
10. Perform parametric statistical tests.
11. Perform non-parametric statistical tests.
12. Interpret the statistical results.
13. Develop a final data-driven conclusion.

---

## 📊 Dataset

The project uses the standard **Titanic dataset** containing:

* **891 passenger records**
* **15 original variables**

### Original Variables

| Variable      | Description                             |
| ------------- | --------------------------------------- |
| `survived`    | Survival status (0 = No, 1 = Yes)       |
| `pclass`      | Passenger class                         |
| `sex`         | Passenger sex                           |
| `age`         | Passenger age                           |
| `sibsp`       | Number of siblings/spouses aboard       |
| `parch`       | Number of parents/children aboard       |
| `fare`        | Passenger fare                          |
| `embarked`    | Port of embarkation                     |
| `class`       | Passenger class as categorical variable |
| `who`         | Man, woman, or child                    |
| `adult_male`  | Whether passenger was an adult male     |
| `deck`        | Deck information                        |
| `embark_town` | Embarkation town                        |
| `alive`       | Survival status as text                 |
| `alone`       | Whether passenger traveled alone        |

---

# 🔄 Data Analysis Workflow

The project is divided into **7 major stages**.

### Stage 1 — Data Cleaning & Descriptive Statistics

* Dataset inspection
* Data types
* Missing-value analysis
* Missing-value treatment
* Duplicate investigation
* Outlier detection
* Mean
* Median
* Standard deviation
* Variance
* Minimum and maximum values
* Descriptive summary

### Stage 2 — Exploratory Data Analysis & Visualization

The following relationships were explored:

* Survival distribution
* Sex vs Survival
* Class vs Survival
* Age vs Survival
* Fare vs Survival
* Embarked vs Survival
* Alone vs Survival
* Family Size vs Survival
* Age Group vs Survival
* Sex + Class vs Survival
* Class + Age Group vs Survival
* Family Size + Sex vs Survival
* Fare Group vs Survival
* Embarked + Class vs Survival
* Age Group + Sex vs Survival
* Correlation heatmap

### Stage 3 — Regression & ANOVA

The project includes:

* Descriptive statistics
* Shapiro-Wilk normality test
* Simple linear regression
* Multiple linear regression
* Logistic regression
* One-way ANOVA
* Two-way ANOVA
* ANCOVA
* Chi-square analysis
* Independent-samples testing
* Effect-size analysis

Because `survived` is a binary variable, **logistic regression** was treated as the appropriate regression approach for modeling survival.

### Stage 4 — Correlation Analysis

Both:

* Pearson correlation
* Spearman correlation

were calculated.

Variance Inflation Factor (VIF) was also examined to identify potential multicollinearity among predictors.

### Stage 5 — Parametric Tests

The project applied:

* Levene's test
* Welch's t-test
* One-way ANOVA
* Two-way ANOVA
* ANCOVA
* Cohen's d
* Eta-squared

### Stage 6 — Non-Parametric Tests

The project applied:

* Mann–Whitney U test
* Kruskal–Wallis test
* Chi-square test

Non-parametric methods were included because several numerical variables did not satisfy normality assumptions.

### Stage 7 — Final Conclusion

The final stage combines the findings from all previous stages into a complete statistical interpretation, including:

* Major findings
* Statistical significance
* Effect sizes
* Data limitations
* Model limitations
* Overall conclusion

---

# 🧹 Data Cleaning

The following preprocessing steps were performed.

### Missing Values

Missing values were identified in:

* `age`
* `embarked`
* `deck`
* `embark_town`

The following approaches were used:

* `age` → Median imputation
* `embarked` → Mode imputation
* `embark_town` → Mode imputation
* `deck` → Removed because of a very high proportion of missing values

After cleaning, the dataset contained:

**891 rows × 14 columns**

and no missing values remained in the analyzed dataset.

### Duplicate Records

Duplicate-looking observations were investigated.

A total of **116 duplicated rows** were detected using pandas' duplicate check.

They were not automatically removed because the dataset did not contain a unique passenger ID in the analyzed variables. Identical observable characteristics can belong to different passengers.

### Outliers

The IQR method was used to identify potential outliers in numerical variables such as:

* Age
* Fare
* SibSp
* Parch

These observations were retained because an IQR outlier is not necessarily an incorrect observation. Some extreme values represent legitimate passengers or family structures.

---

# 📈 Key Findings

### Overall Survival

* Survival rate: **38.38%**
* Non-survival rate: **61.62%**

### Sex

* Female survival rate: **74.20%**
* Male survival rate: **18.89%**

### Passenger Class

* First Class: **62.96%**
* Second Class: **47.28%**
* Third Class: **24.24%**

### Fare

Mean fare:

* Non-survivors: **22.12**
* Survivors: **48.40**

### Age

Mean age:

* Non-survivors: **30.03**
* Survivors: **28.29**

The difference was not statistically significant in the Welch t-test.

### Embarkation

Observed survival rates:

* Cherbourg: **55.36%**
* Queenstown: **38.96%**
* Southampton: **33.90%**

---

# 📊 Statistical Results

## Logistic Regression

The logistic regression model included:

* Passenger class
* Age
* SibSp
* Parch
* Fare

The overall model was statistically significant.

Statistically significant predictors in the fitted model included:

* `pclass`
* `age`
* `sibsp`
* `parch`

`fare` was not statistically significant at the 5% level after adjustment for the other predictors.

---

## Correlation

Selected Pearson correlations with survival:

| Variable    | Correlation with Survival |
| ----------- | ------------------------: |
| Pclass      |                    -0.338 |
| Age         |                    -0.065 |
| SibSp       |                    -0.035 |
| Parch       |                     0.082 |
| Fare        |                     0.257 |
| Family Size |                     0.017 |

Passenger class showed a moderate negative relationship with survival, while fare showed a weak positive relationship.

---

## Parametric Testing

Important results included:

* Fare vs Survival → statistically significant
* Age vs Survival → not statistically significant
* Age vs Passenger Class → statistically significant
* Passenger Class effect in two-way ANOVA → statistically significant
* Sex effect in two-way ANOVA → statistically significant
* Class × Sex interaction → not statistically significant

---

## Non-Parametric Testing

Important results included:

* Fare vs Survival → significant Mann–Whitney U test
* Age vs Survival → not significant Mann–Whitney U test
* Age vs Class → significant Kruskal–Wallis test
* Fare vs Class → significant Kruskal–Wallis test
* Class vs Survival → significant chi-square test
* Embarked vs Survival → significant chi-square test
* Sex vs Survival → significant chi-square test

---

# ⚠️ Important Statistical Considerations

### 1. Association ≠ Causation

The Titanic dataset is observational. Therefore, the statistical relationships identified in this project should be interpreted as **associations**, not proof of causal relationships.

### 2. Age Imputation

Missing age values were replaced using the median. This may influence age-related statistical results.

### 3. Fare and Class

Fare and passenger class are strongly related. Therefore, their individual relationships with survival should be interpreted together.

### 4. Family Size and Multicollinearity

Family size was calculated as:

```text
Family Size = SibSp + Parch + 1
```

Therefore, including `sibsp`, `parch`, and `family_size` together in the same regression model creates an exact mathematical dependency and extremely high VIF values.

### 5. Small Subgroups

Some detailed combinations of sex, class, age group, fare group, and family size contained very few observations. Their percentages should therefore be interpreted cautiously.

---

# 🛠️ Technologies Used

* **Python**
* **Pandas**
* **NumPy**
* **Matplotlib**
* **SciPy**
* **Statsmodels**
* **Scikit-learn**

---

# 📁 Project Structure

```text
Titanic-Data-Analysis/
│
├── analysis.py
├── titanic.csv
├── titanic_cleaned.csv
└── README.md
```

---

# ⚙️ Installation

Clone or download the project and install the required Python libraries.

```bash
pip install pandas numpy matplotlib scipy statsmodels scikit-learn
```

---

# ▶️ How to Run

Open the project folder in VS Code.

Then run:

```bash
python analysis.py
```

Or:

```bash
python -u analysis.py
```

The script performs the complete analysis and prints the statistical results in the terminal.

---

# 🧪 Main Python Libraries

### Pandas

Used for:

* Data loading
* Data cleaning
* Data manipulation
* Grouping
* Aggregation

### NumPy

Used for:

* Numerical operations
* Statistical calculations

### Matplotlib

Used for:

* Bar charts
* Boxplots
* Histograms
* Heatmaps
* Statistical visualizations

### SciPy

Used for:

* Shapiro-Wilk test
* Levene's test
* t-tests
* Mann–Whitney U
* Kruskal–Wallis
* Chi-square
* Correlation tests

### Statsmodels

Used for:

* OLS regression
* Logistic regression
* ANOVA
* ANCOVA
* Statistical summaries

### Scikit-learn

Used for:

* Statistical/modeling utilities
* Machine-learning workflow support

---

# 📚 Learning Outcomes

Through this project, the following practical data-analysis skills were developed:

* Data cleaning
* Missing-value handling
* Duplicate investigation
* Outlier detection
* Feature engineering
* Exploratory Data Analysis
* Data visualization
* Descriptive statistics
* Regression analysis
* Logistic regression
* ANOVA
* ANCOVA
* Correlation analysis
* Multicollinearity analysis
* Hypothesis testing
* Parametric testing
* Non-parametric testing
* Statistical interpretation
* Report writing

---

# 📝 Final Conclusion

The Titanic analysis demonstrates a complete statistical data-analysis workflow from raw data preparation to final interpretation. The analysis identified substantial differences in observed survival across passenger sex and class, while fare also showed consistent statistical differences across survival groups.

The combination of descriptive statistics, visualization, regression, correlation, parametric tests, and non-parametric tests provided a broader understanding of the dataset than any single statistical technique alone.

The project also demonstrates the importance of checking assumptions, identifying multicollinearity, considering subgroup sample sizes, and distinguishing statistical association from causation.

Overall, this project provides practical experience in applying Python-based statistical analysis to a real-world dataset and converting statistical outputs into meaningful, evidence-based conclusions.

---

## 👨‍💻 Author

**Naveed Ahmad**

Data Science graduate
Government College University Faisalabad

### Skills Demonstrated

`Python` · `Pandas` · `NumPy` · `Matplotlib` · `SciPy` · `Statsmodels` · `Scikit-learn` · `EDA` · `Statistics` · `Data Visualization` · `Regression Analysis`

---

