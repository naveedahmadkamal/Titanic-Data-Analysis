# ============================================================
# TITANIC DATA ANALYSIS PROJECT
# DASHBOARD + FINAL REPORT
# Developed by Naveed Ahmad
# ============================================================

import os
import math
import html
from pathlib import Path

import numpy as np
import pandas as pd

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap

from reportlab.lib import colors
from reportlab.lib.enums import TA_LEFT, TA_CENTER, TA_RIGHT
from reportlab.lib.pagesizes import A4, landscape
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import mm
from reportlab.platypus import (
    SimpleDocTemplate,
    Paragraph,
    Spacer,
    Table,
    TableStyle,
    Image,
    PageBreak,
    KeepTogether,
)

# ============================================================
# PATHS
# ============================================================

BASE_DIR = Path(__file__).resolve().parent

DATA_FILE = BASE_DIR / "titanic_cleaned.csv"
CHART_DIR = BASE_DIR / "Titanic_PDF_Charts"

DASHBOARD_PDF = BASE_DIR / "Titanic_Dashboard.pdf"
FINAL_REPORT_PDF = BASE_DIR / "Titanic_Final_Report.pdf"

CHART_DIR.mkdir(exist_ok=True)

# ============================================================
# COLOR PALETTE
# ============================================================

BG = "#F4F7FC"
CARD = "#FFFFFF"
DARK = "#172554"
TEXT = "#1E293B"
MUTED = "#64748B"

BLUE = "#2563EB"
PURPLE = "#7C3AED"
CYAN = "#06B6D4"
PINK = "#EC4899"
GREEN = "#16A34A"
RED = "#DC2626"
ORANGE = "#F97316"

LIGHT_BORDER = "#E2E8F0"
LIGHT_BLUE = "#EFF6FF"
LIGHT_PURPLE = "#F5F3FF"
LIGHT_GREEN = "#F0FDF4"
LIGHT_RED = "#FEF2F2"
LIGHT_ORANGE = "#FFF7ED"

# ============================================================
# CATEGORY ORDERS
# ============================================================

AGE_ORDER = [
    "Child",
    "Teenager",
    "Young Adult",
    "Adult",
    "Senior",
]

FAMILY_GROUP_ORDER = [
    "Alone",
    "Small Family",
    "Medium Family",
    "Large Family",
]

FARE_GROUP_ORDER = [
    "Low",
    "Medium-Low",
    "Medium",
    "High",
    "Very High",
]

SEX_ORDER = [
    "female",
    "male",
]

WHO_ORDER = [
    "child",
    "woman",
    "man",
]

AGE_BIN_ORDER = [
    "0-5",
    "6-12",
    "13-18",
    "19-30",
    "31-45",
    "46-60",
    "61-80",
]

# ============================================================
# GLOBAL DATA
# ============================================================

df = None

# ============================================================
# GENERAL HELPERS
# ============================================================


def safe_filename(name):
    return str(name).replace("/", "_").replace("\\", "_").replace(":", "_")


def format_number(value, decimals=1):
    if pd.isna(value):
        return "N/A"
    return f"{value:,.{decimals}f}"


def clean_text(value):
    return html.escape(str(value))


def safe_rate(series):
    if series is None or len(series) == 0:
        return np.nan

    value = pd.to_numeric(series, errors="coerce").mean()

    if pd.isna(value):
        return np.nan

    return float(value * 100)


def rate_text(value):
    if value is None or pd.isna(value):
        return "N/A"
    return f"{value:.1f}%"


def style_ax(ax, title=None, xlabel=None, ylabel=None):
    ax.set_facecolor(CARD)

    if title:
        ax.set_title(
            title,
            fontsize=14,
            fontweight="bold",
            color=DARK,
        )

    if xlabel:
        ax.set_xlabel(
            xlabel,
            fontsize=10,
            color=TEXT,
            labelpad=8,
        )

    if ylabel:
        ax.set_ylabel(
            ylabel,
            fontsize=10,
            color=TEXT,
            labelpad=8,
        )

    ax.tick_params(
        colors=MUTED,
        labelsize=9,
    )

    ax.grid(
        axis="y",
        linestyle="--",
        linewidth=0.6,
        alpha=0.25,
    )

    for spine in ax.spines.values():
        spine.set_visible(False)


def save_figure(fig, filename):
    path = CHART_DIR / filename

    fig.patch.set_facecolor(BG)

    plt.tight_layout()

    fig.savefig(
        path,
        dpi=180,
        bbox_inches="tight",
        facecolor=BG,
    )

    plt.close(fig)

    return path


def add_value_labels(ax, bars, suffix=""):
    for bar in bars:
        height = bar.get_height()

        if np.isnan(height):
            continue

        ax.text(
            bar.get_x() + bar.get_width() / 2,
            height,
            f"{height:,.0f}{suffix}",
            ha="center",
            va="bottom",
            fontsize=8,
            color=TEXT,
            fontweight="bold",
        )


def normalize_survival(data):
    data = data.copy()

    data["survived"] = (
        pd.to_numeric(
            data["survived"],
            errors="coerce",
        )
        .fillna(0)
        .astype(int)
    )

    return data


# ============================================================
# DATA PREPARATION
# ============================================================


def prepare_data(data):
    data = data.copy()

    # --------------------------------------------------------
    # Normalize column names
    # --------------------------------------------------------

    data.columns = [str(col).strip().lower().replace(" ", "_") for col in data.columns]

    required = [
        "survived",
        "pclass",
        "sex",
        "age",
        "fare",
        "embarked",
    ]

    missing = [col for col in required if col not in data.columns]

    if missing:
        raise ValueError("Required columns missing: " + ", ".join(missing))

    # --------------------------------------------------------
    # Numeric columns
    # --------------------------------------------------------

    numeric_columns = [
        "survived",
        "pclass",
        "age",
        "fare",
        "sibsp",
        "parch",
    ]

    for col in numeric_columns:
        if col in data.columns:
            data[col] = pd.to_numeric(
                data[col],
                errors="coerce",
            )

    # --------------------------------------------------------
    # Survival
    # --------------------------------------------------------

    data["survived"] = data["survived"].fillna(0).astype(int)

    # --------------------------------------------------------
    # Sex
    # --------------------------------------------------------

    data["sex"] = data["sex"].astype(str).str.strip().str.lower()

    # --------------------------------------------------------
    # Embarked
    # --------------------------------------------------------

    data["embarked"] = data["embarked"].astype(str).str.strip().str.upper()

    data["embarked"] = data["embarked"].replace(
        {
            "NAN": "Unknown",
            "": "Unknown",
        }
    )

    # --------------------------------------------------------
    # SibSp / Parch
    # --------------------------------------------------------

    if "sibsp" not in data.columns:
        data["sibsp"] = 0

    if "parch" not in data.columns:
        data["parch"] = 0

    data["sibsp"] = data["sibsp"].fillna(0)
    data["parch"] = data["parch"].fillna(0)

    # --------------------------------------------------------
    # Family Size
    # --------------------------------------------------------

    data["family_size"] = data["sibsp"] + data["parch"] + 1

    # --------------------------------------------------------
    # Alone
    # --------------------------------------------------------

    data["alone"] = np.where(
        data["family_size"] == 1,
        "Alone",
        "With Family",
    )

    # --------------------------------------------------------
    # Age Group
    # --------------------------------------------------------

    def age_group(age):
        if pd.isna(age):
            return "Unknown"

        if age < 13:
            return "Child"

        if age < 18:
            return "Teenager"

        if age < 30:
            return "Young Adult"

        if age < 60:
            return "Adult"

        return "Senior"

    data["age_group"] = data["age"].apply(age_group)

    # --------------------------------------------------------
    # Age Bins
    # --------------------------------------------------------

    def age_bin(age):
        if pd.isna(age):
            return "Unknown"

        if age <= 5:
            return "0-5"

        if age <= 12:
            return "6-12"

        if age <= 18:
            return "13-18"

        if age <= 30:
            return "19-30"

        if age <= 45:
            return "31-45"

        if age <= 60:
            return "46-60"

        return "61-80"

    data["age_bin"] = data["age"].apply(age_bin)

    # --------------------------------------------------------
    # Family Group
    # --------------------------------------------------------

    def family_group(size):
        if size <= 1:
            return "Alone"

        if size <= 4:
            return "Small Family"

        if size <= 6:
            return "Medium Family"

        return "Large Family"

    data["family_group"] = data["family_size"].apply(family_group)

    # --------------------------------------------------------
    # Fare Group
    # --------------------------------------------------------

    def fare_group(fare):
        if pd.isna(fare):
            return "Unknown"

        if fare < 10:
            return "Low"

        if fare < 25:
            return "Medium-Low"

        if fare < 50:
            return "Medium"

        if fare < 100:
            return "High"

        return "Very High"

    data["fare_group"] = data["fare"].apply(fare_group)

    # --------------------------------------------------------
    # Who Category
    # --------------------------------------------------------

    if "who" not in data.columns:

        def who_category(row):
            age = row["age"]
            sex = row["sex"]

            if not pd.isna(age) and age < 16:
                return "child"

            if sex == "female":
                return "woman"

            return "man"

        data["who"] = data.apply(
            who_category,
            axis=1,
        )

    else:
        data["who"] = data["who"].astype(str).str.strip().str.lower()

    return data


# ============================================================
# CHART 1 — SURVIVAL DISTRIBUTION
# ============================================================


def chart_1(data):
    counts = data["survived"].value_counts().reindex([0, 1], fill_value=0)

    labels = [
        "Not Survived",
        "Survived",
    ]

    values = counts.values

    fig, ax = plt.subplots(figsize=(9, 5))

    bars = ax.bar(
        labels,
        values,
        color=[RED, GREEN],
        alpha=0.9,
        width=0.55,
    )

    add_value_labels(ax, bars)

    style_ax(
        ax,
        "Survival Distribution",
        "Survival Status",
        "Passengers",
    )

    return save_figure(
        fig,
        "01_Survival_Distribution.png",
    )


# ============================================================
# CHART 2 — SURVIVAL PERCENTAGE
# ============================================================


def chart_2(data):
    counts = data["survived"].value_counts().reindex([0, 1], fill_value=0)

    percentages = counts / counts.sum() * 100

    labels = [
        "Not Survived",
        "Survived",
    ]

    fig, ax = plt.subplots(figsize=(9, 5))

    bars = ax.bar(
        labels,
        percentages.values,
        color=[RED, GREEN],
        alpha=0.9,
        width=0.55,
    )

    add_value_labels(
        ax,
        bars,
        suffix="%",
    )

    style_ax(
        ax,
        "Survival Percentage",
        "Survival Status",
        "Percentage",
    )

    ax.set_ylim(
        0,
        max(percentages.values) * 1.18,
    )

    return save_figure(
        fig,
        "02_Survival_Percentage.png",
    )


# ============================================================
# GENERIC STACKED SURVIVAL CHART
# ============================================================


def create_stacked_survival_chart(
    data,
    category,
    title,
    filename,
    order=None,
    xlabel=None,
):
    temp = data.copy()

    temp["survived"] = temp["survived"].astype(int)

    table = pd.crosstab(
        temp[category],
        temp["survived"],
    )

    table = table.reindex(
        columns=[0, 1],
        fill_value=0,
    )

    if order:
        existing = [x for x in order if x in table.index]

        remaining = [x for x in table.index if x not in existing]

        table = table.reindex(existing + remaining)

    fig, ax = plt.subplots(figsize=(9, 5))

    x = np.arange(len(table))

    no_survival = table[0].values
    survival = table[1].values

    bars1 = ax.bar(
        x,
        no_survival,
        label="Not Survived",
        color=RED,
        alpha=0.88,
    )

    bars2 = ax.bar(
        x,
        survival,
        bottom=no_survival,
        label="Survived",
        color=GREEN,
        alpha=0.90,
    )

    ax.set_xticks(x)

    ax.set_xticklabels(
        [str(value) for value in table.index],
        rotation=0,
    )

    style_ax(
        ax,
        title,
        xlabel
        or category.replace(
            "_",
            " ",
        ).title(),
        "Passengers",
    )

    ax.legend(
        frameon=False,
        ncol=2,
    )

    totals = no_survival + survival

    max_total = max(totals) if len(totals) else 1

    for i, total in enumerate(totals):
        if total > 0:
            ax.text(
                i,
                total + max_total * 0.015,
                f"{int(total)}",
                ha="center",
                fontsize=8,
                fontweight="bold",
                color=TEXT,
            )

    return save_figure(
        fig,
        filename,
    )


# ============================================================
# CHART 3 — SEX AND SURVIVAL
# ============================================================


def chart_3(data):
    return create_stacked_survival_chart(
        data,
        "sex",
        "Sex and Survival",
        "03_Sex_and_Survival.png",
        order=SEX_ORDER,
        xlabel="Sex",
    )


# ============================================================
# CHART 4 — CLASS AND SURVIVAL
# ============================================================


def chart_4(data):
    return create_stacked_survival_chart(
        data,
        "pclass",
        "Class and Survival",
        "04_Class_and_Survival.png",
        order=[1, 2, 3],
        xlabel="Passenger Class",
    )


# ============================================================
# CHART 5 — AGE AND SURVIVAL
# ============================================================


def chart_5(data):
    temp = data.copy()

    table = pd.crosstab(
        temp["age_bin"],
        temp["survived"],
    )

    table = table.reindex(
        AGE_BIN_ORDER,
        fill_value=0,
    )

    table = table.reindex(
        columns=[0, 1],
        fill_value=0,
    )

    x = np.arange(len(table))

    width = 0.36

    fig, ax = plt.subplots(figsize=(10, 5))

    bars1 = ax.bar(
        x - width / 2,
        table[0],
        width,
        label="Not Survived",
        color=RED,
        alpha=0.85,
    )

    bars2 = ax.bar(
        x + width / 2,
        table[1],
        width,
        label="Survived",
        color=GREEN,
        alpha=0.9,
    )

    ax.set_xticks(x)

    ax.set_xticklabels(
        table.index,
    )

    style_ax(
        ax,
        "Age and Survival",
        "Age Group",
        "Passengers",
    )

    ax.legend(
        frameon=False,
        ncol=2,
    )

    return save_figure(
        fig,
        "05_Age_and_Survival.png",
    )


# ============================================================
# CHART 6 — FARE AND SURVIVAL
# ============================================================


def chart_6(data):
    values = data.groupby("survived")["fare"].mean().reindex([0, 1])

    labels = [
        "Not Survived",
        "Survived",
    ]

    fig, ax = plt.subplots(figsize=(9, 5))

    bars = ax.bar(
        labels,
        values.values,
        color=[RED, GREEN],
        alpha=0.9,
        width=0.55,
    )

    for bar, value in zip(
        bars,
        values.values,
    ):
        if not pd.isna(value):
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                value,
                f"${value:.1f}",
                ha="center",
                va="bottom",
                fontsize=8,
                fontweight="bold",
                color=TEXT,
            )

    style_ax(
        ax,
        "Fare and Survival",
        "Survival Status",
        "Average Fare",
    )

    return save_figure(
        fig,
        "06_Fare_and_Survival.png",
    )


# ============================================================
# CHART 7 — EMBARKED AND SURVIVAL
# ============================================================


def chart_7(data):
    return create_stacked_survival_chart(
        data,
        "embarked",
        "Embarked and Survival",
        "07_Embarked_and_Survival.png",
        order=["S", "C", "Q", "Unknown"],
        xlabel="Embarkation Port",
    )


# ============================================================
# CHART 8 — ALONE AND SURVIVAL
# ============================================================


def chart_8(data):
    return create_stacked_survival_chart(
        data,
        "alone",
        "Alone and Survival",
        "08_Alone_and_Survival.png",
        order=["Alone", "With Family"],
        xlabel="Family Status",
    )


# ============================================================
# CHART 9 — FAMILY SIZE AND SURVIVAL
# ============================================================


def chart_9(data):
    table = pd.crosstab(
        data["family_size"],
        data["survived"],
    )

    table = table.reindex(
        columns=[0, 1],
        fill_value=0,
    )

    x = np.arange(len(table))

    width = 0.38

    fig, ax = plt.subplots(figsize=(10, 5))

    ax.bar(
        x - width / 2,
        table[0],
        width,
        label="Not Survived",
        color=RED,
        alpha=0.85,
    )

    ax.bar(
        x + width / 2,
        table[1],
        width,
        label="Survived",
        color=GREEN,
        alpha=0.9,
    )

    ax.set_xticks(x)

    ax.set_xticklabels(table.index.astype(int))

    style_ax(
        ax,
        "Family Size and Survival",
        "Family Size",
        "Passengers",
    )

    ax.legend(
        frameon=False,
        ncol=2,
    )

    return save_figure(
        fig,
        "09_Family_Size_and_Survival.png",
    )


# ============================================================
# CHART 10 — AGE GROUP AND SURVIVAL
# ============================================================


def chart_10(data):
    return create_stacked_survival_chart(
        data,
        "age_group",
        "Age Group and Survival",
        "10_Age_Group_and_Survival.png",
        order=AGE_ORDER,
        xlabel="Age Group",
    )


# ============================================================
# CHART 11 — WHO CATEGORY AND SURVIVAL
# ============================================================


def chart_11(data):
    return create_stacked_survival_chart(
        data,
        "who",
        "Who Category and Survival",
        "11_Who_Category_and_Survival.png",
        order=WHO_ORDER,
        xlabel="Who Category",
    )


# ============================================================
# CHART 12 — SEX AND CLASS
# ============================================================


def chart_12(data):
    table = pd.crosstab(
        data["pclass"],
        data["sex"],
    )

    table = table.reindex(
        [1, 2, 3],
        fill_value=0,
    )

    x = np.arange(len(table))

    width = 0.36

    fig, ax = plt.subplots(figsize=(9, 5))

    ax.bar(
        x - width / 2,
        table.get("female", 0),
        width,
        label="Female",
        color=PINK,
    )

    ax.bar(
        x + width / 2,
        table.get("male", 0),
        width,
        label="Male",
        color=BLUE,
    )

    ax.set_xticks(x)

    ax.set_xticklabels(["Class 1", "Class 2", "Class 3"])

    style_ax(
        ax,
        "Sex and Class",
        "Passenger Class",
        "Passengers",
    )

    ax.legend(
        frameon=False,
        ncol=2,
    )

    return save_figure(
        fig,
        "12_Sex_and_Class.png",
    )


# ============================================================
# CHART 13 — CLASS AND AGE GROUP
# ============================================================


def chart_13(data):
    table = pd.crosstab(
        data["pclass"],
        data["age_group"],
    )

    table = table.reindex(
        [1, 2, 3],
        fill_value=0,
    )

    table = table.reindex(
        columns=AGE_ORDER,
        fill_value=0,
    )

    x = np.arange(len(table))

    width = 0.15

    fig, ax = plt.subplots(figsize=(11, 5))

    for i, category in enumerate(AGE_ORDER):
        ax.bar(
            x + (i - 2) * width,
            table[category],
            width,
            label=category,
        )

    ax.set_xticks(x)

    ax.set_xticklabels(
        [
            "Class 1",
            "Class 2",
            "Class 3",
        ]
    )

    style_ax(
        ax,
        "Class and Age Group",
        "Passenger Class",
        "Passengers",
    )

    ax.legend(
        frameon=False,
        ncol=3,
    )

    return save_figure(
        fig,
        "13_Class_and_Age_Group.png",
    )


# ============================================================
# CHART 14 — SEX / CLASS / AGE GROUP
# ============================================================


def chart_14(data):
    temp = data.copy()

    temp["segment"] = (
        "C"
        + temp["pclass"].astype(int).astype(str)
        + " • "
        + temp["sex"].str.title()
        + " • "
        + temp["age_group"]
    )

    grouped = (
        temp.groupby("segment")["survived"].mean().mul(100).sort_values(ascending=False)
    )

    grouped = grouped.head(15)

    fig, ax = plt.subplots(figsize=(12, 6))

    bars = ax.bar(
        np.arange(len(grouped)),
        grouped.values,
        color=PURPLE,
        alpha=0.9,
    )

    ax.set_xticks(np.arange(len(grouped)))

    ax.set_xticklabels(
        grouped.index,
        rotation=45,
        ha="right",
        fontsize=8,
    )

    style_ax(
        ax,
        "Sex / Class / Age Group",
        "Passenger Segment",
        "Survival Rate (%)",
    )

    add_value_labels(
        ax,
        bars,
        suffix="%",
    )

    return save_figure(
        fig,
        "14_Sex_Class_Age_Group.png",
    )


# ============================================================
# CHART 15 — FAMILY SIZE AND SEX
# ============================================================


def chart_15(data):
    table = pd.crosstab(
        data["family_size"],
        data["sex"],
    )

    table = table.reindex(
        columns=SEX_ORDER,
        fill_value=0,
    )

    x = np.arange(len(table))

    width = 0.36

    fig, ax = plt.subplots(figsize=(10, 5))

    ax.bar(
        x - width / 2,
        table["female"],
        width,
        label="Female",
        color=PINK,
    )

    ax.bar(
        x + width / 2,
        table["male"],
        width,
        label="Male",
        color=BLUE,
    )

    ax.set_xticks(x)

    ax.set_xticklabels(table.index.astype(int))

    style_ax(
        ax,
        "Family Size and Sex",
        "Family Size",
        "Passengers",
    )

    ax.legend(
        frameon=False,
        ncol=2,
    )

    return save_figure(
        fig,
        "15_Family_Size_and_Sex.png",
    )


# ============================================================
# CHART 16 — FAMILY SIZE GROUP
# ============================================================


def chart_16(data):
    counts = (
        data["family_group"]
        .value_counts()
        .reindex(
            FAMILY_GROUP_ORDER,
            fill_value=0,
        )
    )

    fig, ax = plt.subplots(figsize=(9, 5))

    bars = ax.bar(
        counts.index,
        counts.values,
        color=ORANGE,
        alpha=0.9,
    )

    add_value_labels(
        ax,
        bars,
    )

    style_ax(
        ax,
        "Family Size Group",
        "Family Group",
        "Passengers",
    )

    return save_figure(
        fig,
        "16_Family_Size_Group.png",
    )


# ============================================================
# CHART 17 — CLASS AND FAMILY SIZE GROUP
# ============================================================


def chart_17(data):
    table = pd.crosstab(
        data["pclass"],
        data["family_group"],
    )

    table = table.reindex(
        [1, 2, 3],
        fill_value=0,
    )

    table = table.reindex(
        columns=FAMILY_GROUP_ORDER,
        fill_value=0,
    )

    x = np.arange(len(table))

    width = 0.2

    fig, ax = plt.subplots(figsize=(11, 5))

    for i, group in enumerate(FAMILY_GROUP_ORDER):
        ax.bar(
            x + (i - 1.5) * width,
            table[group],
            width,
            label=group,
        )

    ax.set_xticks(x)

    ax.set_xticklabels(
        [
            "Class 1",
            "Class 2",
            "Class 3",
        ]
    )

    style_ax(
        ax,
        "Class and Family Size Group",
        "Passenger Class",
        "Passengers",
    )

    ax.legend(
        frameon=False,
        ncol=2,
    )

    return save_figure(
        fig,
        "17_Class_and_Family_Size_Group.png",
    )


# ============================================================
# CHART 18 — SEX / CLASS / FAMILY GROUP
# ============================================================


def chart_18(data):
    temp = data.copy()

    temp["segment"] = (
        "C"
        + temp["pclass"].astype(int).astype(str)
        + " • "
        + temp["sex"].str.title()
        + " • "
        + temp["family_group"]
    )

    rates = (
        temp.groupby("segment")["survived"]
        .mean()
        .mul(100)
        .sort_values(ascending=False)
        .head(15)
    )

    fig, ax = plt.subplots(figsize=(12, 6))

    bars = ax.bar(
        np.arange(len(rates)),
        rates.values,
        color=CYAN,
        alpha=0.9,
    )

    ax.set_xticks(np.arange(len(rates)))

    ax.set_xticklabels(
        rates.index,
        rotation=45,
        ha="right",
        fontsize=8,
    )

    style_ax(
        ax,
        "Sex / Class / Family Group",
        "Passenger Segment",
        "Survival Rate (%)",
    )

    add_value_labels(
        ax,
        bars,
        suffix="%",
    )

    return save_figure(
        fig,
        "18_Sex_Class_Family_Group.png",
    )


# ============================================================
# CHART 19 — FARE GROUP AND SURVIVAL
# ============================================================


def chart_19(data):
    return create_stacked_survival_chart(
        data,
        "fare_group",
        "Fare Group and Survival",
        "19_Fare_Group_and_Survival.png",
        order=FARE_GROUP_ORDER,
        xlabel="Fare Group",
    )


# ============================================================
# CHART 20 — FARE GROUP AND CLASS
# ============================================================


def chart_20(data):
    table = pd.crosstab(
        data["fare_group"],
        data["pclass"],
    )

    table = table.reindex(
        FARE_GROUP_ORDER,
        fill_value=0,
    )

    table = table.reindex(
        columns=[1, 2, 3],
        fill_value=0,
    )

    x = np.arange(len(table))

    width = 0.25

    fig, ax = plt.subplots(figsize=(11, 5))

    for i, cls in enumerate([1, 2, 3]):
        ax.bar(
            x + (i - 1) * width,
            table[cls],
            width,
            label=f"Class {cls}",
        )

    ax.set_xticks(x)

    ax.set_xticklabels(table.index)

    style_ax(
        ax,
        "Fare Group and Class",
        "Fare Group",
        "Passengers",
    )

    ax.legend(
        frameon=False,
        ncol=3,
    )

    return save_figure(
        fig,
        "20_Fare_Group_and_Class.png",
    )


# ============================================================
# CHART 21 — EMBARKED AND CLASS
# ============================================================


def chart_21(data):
    table = pd.crosstab(
        data["embarked"],
        data["pclass"],
    )

    table = table.reindex(
        ["S", "C", "Q", "Unknown"],
        fill_value=0,
    )

    table = table.reindex(
        columns=[1, 2, 3],
        fill_value=0,
    )

    x = np.arange(len(table))

    width = 0.25

    fig, ax = plt.subplots(figsize=(10, 5))

    for i, cls in enumerate([1, 2, 3]):
        ax.bar(
            x + (i - 1) * width,
            table[cls],
            width,
            label=f"Class {cls}",
        )

    ax.set_xticks(x)

    ax.set_xticklabels(table.index)

    style_ax(
        ax,
        "Embarked and Class",
        "Embarkation Port",
        "Passengers",
    )

    ax.legend(
        frameon=False,
        ncol=3,
    )

    return save_figure(
        fig,
        "21_Embarked_and_Class.png",
    )


# ============================================================
# CHART 22 — SEX AND FARE GROUP
# ============================================================


def chart_22(data):
    table = pd.crosstab(
        data["fare_group"],
        data["sex"],
    )

    table = table.reindex(
        FARE_GROUP_ORDER,
        fill_value=0,
    )

    table = table.reindex(
        columns=SEX_ORDER,
        fill_value=0,
    )

    x = np.arange(len(table))

    width = 0.36

    fig, ax = plt.subplots(figsize=(10, 5))

    ax.bar(
        x - width / 2,
        table["female"],
        width,
        label="Female",
        color=PINK,
    )

    ax.bar(
        x + width / 2,
        table["male"],
        width,
        label="Male",
        color=BLUE,
    )

    ax.set_xticks(x)

    ax.set_xticklabels(table.index)

    style_ax(
        ax,
        "Sex and Fare Group",
        "Fare Group",
        "Passengers",
    )

    ax.legend(
        frameon=False,
        ncol=2,
    )

    return save_figure(
        fig,
        "22_Sex_and_Fare_Group.png",
    )


# ============================================================
# CHART 23 — AGE GROUP AND SEX
# ============================================================


def chart_23(data):
    table = pd.crosstab(
        data["age_group"],
        data["sex"],
    )

    table = table.reindex(
        AGE_ORDER,
        fill_value=0,
    )

    table = table.reindex(
        columns=SEX_ORDER,
        fill_value=0,
    )

    x = np.arange(len(table))

    width = 0.36

    fig, ax = plt.subplots(figsize=(10, 5))

    ax.bar(
        x - width / 2,
        table["female"],
        width,
        label="Female",
        color=PINK,
    )

    ax.bar(
        x + width / 2,
        table["male"],
        width,
        label="Male",
        color=BLUE,
    )

    ax.set_xticks(x)

    ax.set_xticklabels(table.index)

    style_ax(
        ax,
        "Age Group and Sex",
        "Age Group",
        "Passengers",
    )

    ax.legend(
        frameon=False,
        ncol=2,
    )

    return save_figure(
        fig,
        "23_Age_Group_and_Sex.png",
    )


# ============================================================
# CHART 24 — CLASS / SEX / FARE GROUP
# ============================================================


def chart_24(data):
    temp = data.copy()

    temp["segment"] = (
        "C"
        + temp["pclass"].astype(int).astype(str)
        + " • "
        + temp["sex"].str.title()
        + " • "
        + temp["fare_group"]
    )

    rates = (
        temp.groupby("segment")["survived"]
        .mean()
        .mul(100)
        .sort_values(ascending=False)
        .head(15)
    )

    fig, ax = plt.subplots(figsize=(12, 6))

    bars = ax.bar(
        np.arange(len(rates)),
        rates.values,
        color=ORANGE,
        alpha=0.9,
    )

    ax.set_xticks(np.arange(len(rates)))

    ax.set_xticklabels(
        rates.index,
        rotation=45,
        ha="right",
        fontsize=8,
    )

    style_ax(
        ax,
        "Class / Sex / Fare Group",
        "Passenger Segment",
        "Survival Rate (%)",
    )

    add_value_labels(
        ax,
        bars,
        suffix="%",
    )

    return save_figure(
        fig,
        "24_Class_Sex_Fare_Group.png",
    )


# ============================================================
# CHART 25 — CORRELATION HEATMAP
# ============================================================


def chart_25(data):
    numeric_columns = [
        "survived",
        "pclass",
        "age",
        "sibsp",
        "parch",
        "fare",
        "family_size",
    ]

    numeric_columns = [col for col in numeric_columns if col in data.columns]

    corr = data[numeric_columns].corr()

    fig, ax = plt.subplots(figsize=(9, 7))

    cmap = LinearSegmentedColormap.from_list(
        "custom_blue",
        [
            "#DC2626",
            "#F8FAFC",
            "#2563EB",
        ],
    )

    image = ax.imshow(
        corr,
        cmap=cmap,
        vmin=-1,
        vmax=1,
    )

    ax.set_xticks(np.arange(len(corr.columns)))

    ax.set_yticks(np.arange(len(corr.columns)))

    ax.set_xticklabels(
        [col.replace("_", " ").title() for col in corr.columns],
        rotation=45,
        ha="right",
    )

    ax.set_yticklabels([col.replace("_", " ").title() for col in corr.columns])

    for i in range(len(corr)):
        for j in range(len(corr)):
            ax.text(
                j,
                i,
                f"{corr.iloc[i, j]:.2f}",
                ha="center",
                va="center",
                fontsize=8,
                color=TEXT,
            )

    ax.set_title(
        "Correlation Heatmap",
        fontsize=14,
        fontweight="bold",
        color=DARK,
    )

    fig.colorbar(
        image,
        ax=ax,
        fraction=0.046,
        pad=0.04,
    )

    return save_figure(
        fig,
        "25_Correlation_Heatmap.png",
    )


# ============================================================
# CHART 26 — VISUALIZATION SUMMARY
# ============================================================


def chart_26(data):
    total = len(data)

    survivors = int(data["survived"].sum())

    survival_rate = data["survived"].mean() * 100

    female_rate = safe_rate(
        data.loc[
            data["sex"] == "female",
            "survived",
        ]
    )

    male_rate = safe_rate(
        data.loc[
            data["sex"] == "male",
            "survived",
        ]
    )

    class_rates = data.groupby("pclass")["survived"].mean().mul(100)

    rows = [
        ["Metric", "Value"],
        ["Total Passengers", f"{total:,}"],
        ["Survivors", f"{survivors:,}"],
        ["Overall Survival", f"{survival_rate:.1f}%"],
        ["Female Survival", rate_text(female_rate)],
        ["Male Survival", rate_text(male_rate)],
        [
            "Class 1 Survival",
            rate_text(class_rates.get(1)),
        ],
        [
            "Class 2 Survival",
            rate_text(class_rates.get(2)),
        ],
        [
            "Class 3 Survival",
            rate_text(class_rates.get(3)),
        ],
    ]

    fig, ax = plt.subplots(figsize=(8, 6))

    ax.axis("off")

    table = ax.table(
        cellText=rows[1:],
        colLabels=rows[0],
        cellLoc="center",
        loc="center",
        colWidths=[0.55, 0.35],
    )

    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(
        1,
        1.55,
    )

    for (
        row,
        col,
    ), cell in table.get_celld().items():
        cell.set_edgecolor(LIGHT_BORDER)

        if row == 0:
            cell.set_facecolor(DARK)
            cell.set_text_props(
                color="white",
                fontweight="bold",
            )
        else:
            cell.set_facecolor("#F8FAFC" if row % 2 == 0 else "white")

    ax.set_title(
        "Visualization Summary",
        fontsize=14,
        fontweight="bold",
        color=DARK,
    )

    return save_figure(
        fig,
        "26_Visualization_Summary.png",
    )


# ============================================================
# CHART 27 — SURVIVAL PIE CHART
# ============================================================


def chart_survival_pie(data):
    counts = (
        data["survived"]
        .value_counts()
        .reindex(
            [0, 1],
            fill_value=0,
        )
    )

    labels = [
        "Not Survived",
        "Survived",
    ]

    fig, ax = plt.subplots(figsize=(7, 6))

    ax.pie(
        counts.values,
        labels=labels,
        autopct="%1.1f%%",
        startangle=90,
        colors=[RED, GREEN],
        textprops={
            "fontsize": 9,
            "color": TEXT,
        },
    )

    ax.set_title(
        "Survival Pie Chart",
        fontsize=14,
        fontweight="bold",
        color=DARK,
    )

    return save_figure(
        fig,
        "27_Survival_Pie.png",
    )


# ============================================================
# CHART 28 — SURVIVAL DONUT CHART
# ============================================================


def chart_survival_donut(data):
    counts = (
        data["survived"]
        .value_counts()
        .reindex(
            [0, 1],
            fill_value=0,
        )
    )

    labels = [
        "Not Survived",
        "Survived",
    ]

    fig, ax = plt.subplots(figsize=(7, 6))

    ax.pie(
        counts.values,
        labels=labels,
        autopct="%1.1f%%",
        startangle=90,
        colors=[RED, GREEN],
        wedgeprops={
            "width": 0.42,
            "edgecolor": "white",
        },
        textprops={
            "fontsize": 9,
            "color": TEXT,
        },
    )

    ax.set_title(
        "Survival Donut Chart",
        fontsize=14,
        fontweight="bold",
        color=DARK,
    )

    return save_figure(
        fig,
        "28_Survival_Donut.png",
    )


# ============================================================
# CHART REGISTRY — EXACTLY 28 UNIQUE CHARTS
# ============================================================


CHART_SPECS = [
    (
        1,
        "01_Survival_Distribution.png",
        "Survival Distribution",
        chart_1,
    ),
    (
        2,
        "02_Survival_Percentage.png",
        "Survival Percentage",
        chart_2,
    ),
    (
        3,
        "03_Sex_and_Survival.png",
        "Sex and Survival",
        chart_3,
    ),
    (
        4,
        "04_Class_and_Survival.png",
        "Class and Survival",
        chart_4,
    ),
    (
        5,
        "05_Age_and_Survival.png",
        "Age and Survival",
        chart_5,
    ),
    (
        6,
        "06_Fare_and_Survival.png",
        "Fare and Survival",
        chart_6,
    ),
    (
        7,
        "07_Embarked_and_Survival.png",
        "Embarked and Survival",
        chart_7,
    ),
    (
        8,
        "08_Alone_and_Survival.png",
        "Alone and Survival",
        chart_8,
    ),
    (
        9,
        "09_Family_Size_and_Survival.png",
        "Family Size and Survival",
        chart_9,
    ),
    (
        10,
        "10_Age_Group_and_Survival.png",
        "Age Group and Survival",
        chart_10,
    ),
    (
        11,
        "11_Who_Category_and_Survival.png",
        "Who Category and Survival",
        chart_11,
    ),
    (
        12,
        "12_Sex_and_Class.png",
        "Sex and Class",
        chart_12,
    ),
    (
        13,
        "13_Class_and_Age_Group.png",
        "Class and Age Group",
        chart_13,
    ),
    (
        14,
        "14_Sex_Class_Age_Group.png",
        "Sex / Class / Age Group",
        chart_14,
    ),
    (
        15,
        "15_Family_Size_and_Sex.png",
        "Family Size and Sex",
        chart_15,
    ),
    (
        16,
        "16_Family_Size_Group.png",
        "Family Size Group",
        chart_16,
    ),
    (
        17,
        "17_Class_and_Family_Size_Group.png",
        "Class and Family Size Group",
        chart_17,
    ),
    (
        18,
        "18_Sex_Class_Family_Group.png",
        "Sex / Class / Family Group",
        chart_18,
    ),
    (
        19,
        "19_Fare_Group_and_Survival.png",
        "Fare Group and Survival",
        chart_19,
    ),
    (
        20,
        "20_Fare_Group_and_Class.png",
        "Fare Group and Class",
        chart_20,
    ),
    (
        21,
        "21_Embarked_and_Class.png",
        "Embarked and Class",
        chart_21,
    ),
    (
        22,
        "22_Sex_and_Fare_Group.png",
        "Sex and Fare Group",
        chart_22,
    ),
    (
        23,
        "23_Age_Group_and_Sex.png",
        "Age Group and Sex",
        chart_23,
    ),
    (
        24,
        "24_Class_Sex_Fare_Group.png",
        "Class / Sex / Fare Group",
        chart_24,
    ),
    (
        25,
        "25_Correlation_Heatmap.png",
        "Correlation Heatmap",
        chart_25,
    ),
    (
        26,
        "26_Visualization_Summary.png",
        "Visualization Summary",
        chart_26,
    ),
    (
        27,
        "27_Survival_Pie.png",
        "Survival Pie Chart",
        chart_survival_pie,
    ),
    (
        28,
        "28_Survival_Donut.png",
        "Survival Donut Chart",
        chart_survival_donut,
    ),
]


# ============================================================
# GENERATE ALL CHARTS
# ============================================================


def generate_all_charts(data):
    print("\nGenerating chart images...")

    # Remove old PNGs
    for old_file in CHART_DIR.glob("*.png"):
        try:
            old_file.unlink()
        except Exception:
            pass

    generated = {}

    for number, filename, title, function in CHART_SPECS:
        try:
            path = function(data)

            generated[title] = path

            print(f"[OK] {number:02d}. {title}")

        except Exception as error:
            print(f"[ERROR] {number:02d}. {title}: {error}")
            raise

    print(f"\nTotal chart images generated: {len(generated)}")

    if len(generated) != 28:
        raise RuntimeError("Expected exactly 28 unique charts.")

    return generated


# ============================================================
# REPORTLAB STYLES
# ============================================================


styles = getSampleStyleSheet()

STYLE_TITLE = ParagraphStyle(
    "DashboardTitle",
    parent=styles["Title"],
    fontName="Helvetica-Bold",
    fontSize=20,
    leading=22,
    textColor=colors.white,
    alignment=TA_LEFT,
)

STYLE_SUBTITLE = ParagraphStyle(
    "DashboardSubtitle",
    parent=styles["Normal"],
    fontName="Helvetica",
    fontSize=8,
    leading=10,
    textColor=colors.HexColor("#CBD5E1"),
)

STYLE_AUTHOR = ParagraphStyle(
    "Author",
    parent=styles["Normal"],
    fontName="Helvetica-Bold",
    fontSize=10,
    leading=12,
    textColor=colors.white,
    alignment=TA_RIGHT,
)

STYLE_SECTION = ParagraphStyle(
    "Section",
    parent=styles["Heading2"],
    fontName="Helvetica-Bold",
    fontSize=14,
    leading=17,
    textColor=colors.HexColor(DARK),
    spaceAfter=7,
)

STYLE_BODY = ParagraphStyle(
    "Body",
    parent=styles["BodyText"],
    fontName="Helvetica",
    fontSize=9,
    leading=13,
    textColor=colors.HexColor(TEXT),
)

STYLE_SMALL = ParagraphStyle(
    "Small",
    parent=styles["BodyText"],
    fontName="Helvetica",
    fontSize=7,
    leading=9,
    textColor=colors.HexColor(MUTED),
)

STYLE_KPI_LABEL = ParagraphStyle(
    "KPILabel",
    parent=styles["Normal"],
    fontName="Helvetica",
    fontSize=7,
    leading=8,
    textColor=colors.HexColor(MUTED),
    alignment=TA_CENTER,
)

STYLE_KPI_VALUE = ParagraphStyle(
    "KPIValue",
    parent=styles["Normal"],
    fontName="Helvetica-Bold",
    fontSize=16,
    leading=18,
    textColor=colors.HexColor(DARK),
    alignment=TA_CENTER,
)

STYLE_FILTER = ParagraphStyle(
    "Filter",
    parent=styles["Normal"],
    fontName="Helvetica",
    fontSize=7,
    leading=8,
    textColor=colors.HexColor(TEXT),
)

STYLE_FILTER_LABEL = ParagraphStyle(
    "FilterLabel",
    parent=styles["Normal"],
    fontName="Helvetica-Bold",
    fontSize=6.5,
    leading=7,
    textColor=colors.HexColor(MUTED),
)

STYLE_FILTER_VALUE = ParagraphStyle(
    "FilterValue",
    parent=styles["Normal"],
    fontName="Helvetica",
    fontSize=7,
    leading=8,
    textColor=colors.HexColor(TEXT),
    alignment=TA_CENTER,
)

STYLE_CARD_TITLE = ParagraphStyle(
    "CardTitle",
    parent=styles["Normal"],
    fontName="Helvetica-Bold",
    fontSize=8,
    leading=10,
    textColor=colors.HexColor(DARK),
)

STYLE_REPORT_SUMMARY = ParagraphStyle(
    "ReportSummary",
    parent=styles["BodyText"],
    fontName="Helvetica",
    fontSize=7.5,
    leading=10,
    textColor=colors.HexColor(TEXT),
)

STYLE_REPORT_SMALL_TITLE = ParagraphStyle(
    "ReportSmallTitle",
    parent=styles["Normal"],
    fontName="Helvetica-Bold",
    fontSize=8,
    leading=10,
    textColor=colors.HexColor(DARK),
)


# ============================================================
# DASHBOARD HEADER
# ============================================================


def dashboard_header():

    left = [
        Paragraph(
            "Titanic Analytics Dashboard",
            STYLE_TITLE,
        ),
        Paragraph(
            "Data Analysis • Visualization • Statistical Insights",
            STYLE_SUBTITLE,
        ),
    ]

    right = [
        Paragraph(
            "Naveed Ahmad",
            STYLE_AUTHOR,
        ),
      
    ]

    table = Table(
        [[left, right]],
        colWidths=[
            205 * mm,
            72 * mm,
        ],
        rowHeights=[
            19 * mm,
        ],
    )

    table.setStyle(
        TableStyle(
            [
                (
                    "BACKGROUND",
                    (0, 0),
                    (-1, -1),
                    colors.HexColor(DARK),
                ),
                (
                    "VALIGN",
                    (0, 0),
                    (-1, -1),
                    "MIDDLE",
                ),
                (
                    "ALIGN",
                    (1, 0),
                    (1, 0),
                    "RIGHT",
                ),
                (
                    "LEFTPADDING",
                    (0, 0),
                    (0, 0),
                    8,
                ),
                (
                    "RIGHTPADDING",
                    (-1, 0),
                    (-1, 0),
                    8,
                ),
                (
                    "TOPPADDING",
                    (0, 0),
                    (-1, -1),
                    4,
                ),
                (
                    "BOTTOMPADDING",
                    (0, 0),
                    (-1, -1),
                    4,
                ),
            ]
        )
    )

    return table


# ============================================================
# KPI CARDS
# ============================================================


def make_kpi_card(
    label,
    value,
    accent,
    width=53 * mm,
):

    content = [
        Paragraph(
            str(value),
            STYLE_KPI_VALUE,
        ),
        Spacer(
            1,
            1.5 * mm,
        ),
        Paragraph(
            label,
            STYLE_KPI_LABEL,
        ),
    ]

    table = Table(
        [[content]],
        colWidths=[width],
        rowHeights=[18 * mm],
    )

    table.setStyle(
        TableStyle(
            [
                (
                    "BACKGROUND",
                    (0, 0),
                    (-1, -1),
                    colors.white,
                ),
                (
                    "BOX",
                    (0, 0),
                    (-1, -1),
                    0.7,
                    colors.HexColor(LIGHT_BORDER),
                ),
                (
                    "LINEABOVE",
                    (0, 0),
                    (-1, 0),
                    3,
                    colors.HexColor(accent),
                ),
                (
                    "VALIGN",
                    (0, 0),
                    (-1, -1),
                    "MIDDLE",
                ),
                (
                    "LEFTPADDING",
                    (0, 0),
                    (-1, -1),
                    3,
                ),
                (
                    "RIGHTPADDING",
                    (0, 0),
                    (-1, -1),
                    3,
                ),
            ]
        )
    )

    return table


def dashboard_kpis(data):

    total = len(data)

    survivors = int(data["survived"].sum())

    survival_rate = data["survived"].mean() * 100

    avg_age = data["age"].mean()
    avg_fare = data["fare"].mean()

    cards = [
        make_kpi_card(
            "TOTAL PASSENGERS",
            f"{total:,}",
            BLUE,
        ),
        make_kpi_card(
            "SURVIVORS",
            f"{survivors:,}",
            GREEN,
        ),
        make_kpi_card(
            "SURVIVAL RATE",
            f"{survival_rate:.1f}%",
            PURPLE,
        ),
        make_kpi_card(
            "AVERAGE AGE",
            f"{avg_age:.1f}",
            CYAN,
        ),
        make_kpi_card(
            "AVERAGE FARE",
            f"${avg_fare:.1f}",
            ORANGE,
        ),
    ]

    table = Table(
        [cards],
        colWidths=[
            54 * mm,
            54 * mm,
            54 * mm,
            54 * mm,
            54 * mm,
        ],
    )

    table.setStyle(
        TableStyle(
            [
                (
                    "VALIGN",
                    (0, 0),
                    (-1, -1),
                    "TOP",
                ),
                (
                    "LEFTPADDING",
                    (0, 0),
                    (-1, -1),
                    1.5,
                ),
                (
                    "RIGHTPADDING",
                    (0, 0),
                    (-1, -1),
                    1.5,
                ),
            ]
        )
    )

    return table


# ============================================================
# FILTER / SLICER PANEL
# ============================================================


def filter_chip(title, value, width=39 * mm):

    label = Paragraph(
        clean_text(title),
        STYLE_FILTER_LABEL,
    )

    value_box = Table(
        [
            [
                Paragraph(
                    clean_text(value),
                    STYLE_FILTER_VALUE,
                )
            ]
        ],
        colWidths=[width - 3 * mm],
        rowHeights=[6 * mm],
    )

    value_box.setStyle(
        TableStyle(
            [
                (
                    "BACKGROUND",
                    (0, 0),
                    (-1, -1),
                    colors.white,
                ),
                (
                    "BOX",
                    (0, 0),
                    (-1, -1),
                    0.6,
                    colors.HexColor(LIGHT_BORDER),
                ),
                (
                    "VALIGN",
                    (0, 0),
                    (-1, -1),
                    "MIDDLE",
                ),
                (
                    "ALIGN",
                    (0, 0),
                    (-1, -1),
                    "CENTER",
                ),
                (
                    "LEFTPADDING",
                    (0, 0),
                    (-1, -1),
                    2,
                ),
                (
                    "RIGHTPADDING",
                    (0, 0),
                    (-1, -1),
                    2,
                ),
            ]
        )
    )

    outer = Table(
        [
            [label],
            [value_box],
        ],
        colWidths=[width],
        rowHeights=[
            4.5 * mm,
            7 * mm,
        ],
    )

    outer.setStyle(
        TableStyle(
            [
                (
                    "VALIGN",
                    (0, 0),
                    (-1, -1),
                    "MIDDLE",
                ),
                (
                    "LEFTPADDING",
                    (0, 0),
                    (-1, -1),
                    1,
                ),
                (
                    "RIGHTPADDING",
                    (0, 0),
                    (-1, -1),
                    1,
                ),
                (
                    "TOPPADDING",
                    (0, 0),
                    (-1, -1),
                    0,
                ),
                (
                    "BOTTOMPADDING",
                    (0, 0),
                    (-1, -1),
                    0,
                ),
            ]
        )
    )

    return outer


def dashboard_filter_bar():

    filters = [
        filter_chip(
            "SEX",
            "All",
        ),
        filter_chip(
            "CLASS",
            "All Classes",
        ),
        filter_chip(
            "EMBARKED",
            "All Ports",
        ),
        filter_chip(
            "SURVIVAL",
            "All",
        ),
        filter_chip(
            "AGE GROUP",
            "All Groups",
        ),
        filter_chip(
            "FARE GROUP",
            "All Groups",
        ),
        filter_chip(
            "FAMILY GROUP",
            "All Groups",
        ),
    ]

    table = Table(
        [filters],
        colWidths=[
            39 * mm,
            39 * mm,
            39 * mm,
            39 * mm,
            39 * mm,
            39 * mm,
            39 * mm,
        ],
    )

    table.setStyle(
        TableStyle(
            [
                (
                    "BACKGROUND",
                    (0, 0),
                    (-1, -1),
                    colors.HexColor(LIGHT_BLUE),
                ),
                (
                    "VALIGN",
                    (0, 0),
                    (-1, -1),
                    "MIDDLE",
                ),
                (
                    "LEFTPADDING",
                    (0, 0),
                    (-1, -1),
                    1,
                ),
                (
                    "RIGHTPADDING",
                    (0, 0),
                    (-1, -1),
                    1,
                ),
                (
                    "TOPPADDING",
                    (0, 0),
                    (-1, -1),
                    2,
                ),
                (
                    "BOTTOMPADDING",
                    (0, 0),
                    (-1, -1),
                    2,
                ),
            ]
        )
    )

    return table


# ============================================================
# DASHBOARD CHART CARD
# ============================================================


def chart_card(
    chart_path,
    title,
    width=64 * mm,
    height=35 * mm,
):

    if not chart_path or not Path(chart_path).exists():
        content = Paragraph(
            f"<b>{clean_text(title)}</b><br/><br/>Chart unavailable",
            STYLE_BODY,
        )

        return Table(
            [[content]],
            colWidths=[69 * mm],
            rowHeights=[43 * mm],
        )

    image = Image(
        str(chart_path),
        width=width,
        height=height,
    )

    content = [
        image,
    ]

    table = Table(
        [[content]],
        colWidths=[69 * mm],
        rowHeights=[43 * mm],
    )

    table.setStyle(
        TableStyle(
            [
                (
                    "BACKGROUND",
                    (0, 0),
                    (-1, -1),
                    colors.white,
                ),
                (
                    "BOX",
                    (0, 0),
                    (-1, -1),
                    0.7,
                    colors.HexColor(LIGHT_BORDER),
                ),
                (
                    "VALIGN",
                    (0, 0),
                    (-1, -1),
                    "MIDDLE",
                ),
                (
                    "ALIGN",
                    (0, 0),
                    (-1, -1),
                    "CENTER",
                ),
                (
                    "LEFTPADDING",
                    (0, 0),
                    (-1, -1),
                    1,
                ),
                (
                    "RIGHTPADDING",
                    (0, 0),
                    (-1, -1),
                    1,
                ),
                (
                    "TOPPADDING",
                    (0, 0),
                    (-1, -1),
                    1,
                ),
                (
                    "BOTTOMPADDING",
                    (0, 0),
                    (-1, -1),
                    1,
                ),
            ]
        )
    )

    return table


# ============================================================
# DASHBOARD INSIGHTS
# ============================================================


def dashboard_insights(data):

    overall = data["survived"].mean() * 100

    female = safe_rate(
        data.loc[
            data["sex"] == "female",
            "survived",
        ]
    )

    male = safe_rate(
        data.loc[
            data["sex"] == "male",
            "survived",
        ]
    )

    class_rates = data.groupby("pclass")["survived"].mean().mul(100)

    insight_1 = f"<b>Overall:</b> {overall:.1f}% of passengers survived."

    insight_2 = (
        f"<b>Sex:</b> Female survival was "
        f"{rate_text(female)} compared with "
        f"{rate_text(male)} for male passengers."
    )

    insight_3 = (
        f"<b>Class:</b> Survival rates were "
        f"{rate_text(class_rates.get(1))} "
        f"(Class 1), "
        f"{rate_text(class_rates.get(2))} "
        f"(Class 2), and "
        f"{rate_text(class_rates.get(3))} "
        f"(Class 3)."
    )

    cells = [
        Paragraph(
            insight_1,
            STYLE_SMALL,
        ),
        Paragraph(
            insight_2,
            STYLE_SMALL,
        ),
        Paragraph(
            insight_3,
            STYLE_SMALL,
        ),
    ]

    table = Table(
        [cells],
        colWidths=[
            93 * mm,
            93 * mm,
            93 * mm,
        ],
        rowHeights=[
            11 * mm,
        ],
    )

    table.setStyle(
        TableStyle(
            [
                (
                    "BACKGROUND",
                    (0, 0),
                    (-1, -1),
                    colors.HexColor("#F8FAFC"),
                ),
                (
                    "BOX",
                    (0, 0),
                    (-1, -1),
                    0.6,
                    colors.HexColor(LIGHT_BORDER),
                ),
                (
                    "INNERGRID",
                    (0, 0),
                    (-1, -1),
                    0.5,
                    colors.HexColor(LIGHT_BORDER),
                ),
                (
                    "VALIGN",
                    (0, 0),
                    (-1, -1),
                    "MIDDLE",
                ),
                (
                    "LEFTPADDING",
                    (0, 0),
                    (-1, -1),
                    5,
                ),
                (
                    "RIGHTPADDING",
                    (0, 0),
                    (-1, -1),
                    5,
                ),
            ]
        )
    )

    return table


# ============================================================
# DASHBOARD FOOTER
# ============================================================


def dashboard_footer():

    footer = Table(
        [
            [
                Paragraph(
                    "Titanic Dataset Analysis",
                    STYLE_SMALL,
                ),
                Paragraph(
                    "Python • Pandas • Matplotlib • ReportLab",
                    ParagraphStyle(
                        "FooterCenter",
                        parent=STYLE_SMALL,
                        alignment=TA_CENTER,
                    ),
                ),
                Paragraph(
                    "Developed by Naveed Ahmad",
                    ParagraphStyle(
                        "FooterRight",
                        parent=STYLE_SMALL,
                        alignment=TA_RIGHT,
                    ),
                ),
            ]
        ],
        colWidths=[
            90 * mm,
            105 * mm,
            90 * mm,
        ],
    )

    return footer


# ============================================================
# PROFESSIONAL DASHBOARD PDF
# ============================================================


def create_dashboard_pdf(data, charts):

    print("\nCreating professional dashboard PDF...")

    doc = SimpleDocTemplate(
        str(DASHBOARD_PDF),
        pagesize=landscape(A4),
        rightMargin=5 * mm,
        leftMargin=5 * mm,
        topMargin=4 * mm,
        bottomMargin=3 * mm,
        title="Titanic Analytics Dashboard",
        author="Naveed Ahmad",
    )

    story = []

    # --------------------------------------------------------
    # HEADER
    # --------------------------------------------------------

    story.append(dashboard_header())

    story.append(
        Spacer(
            1,
            1.5 * mm,
        )
    )

    # --------------------------------------------------------
    # KPI ROW
    # --------------------------------------------------------

    story.append(dashboard_kpis(data))

    story.append(
        Spacer(
            1,
            1.5 * mm,
        )
    )

    # --------------------------------------------------------
    # FILTERS
    # --------------------------------------------------------

    story.append(dashboard_filter_bar())

    story.append(
        Spacer(
            1,
            1.5 * mm,
        )
    )

    # --------------------------------------------------------
    # SECTION TITLE
    # --------------------------------------------------------

    story.append(
        Paragraph(
            "Executive Overview",
            STYLE_SECTION,
        )
    )

    # --------------------------------------------------------
    # 8 MAIN DASHBOARD CHARTS
    # --------------------------------------------------------

    main_charts = [
        (
            "Survival Distribution",
            charts.get("Survival Distribution"),
        ),
        (
            "Sex and Survival",
            charts.get("Sex and Survival"),
        ),
        (
            "Class and Survival",
            charts.get("Class and Survival"),
        ),
        (
            "Age and Survival",
            charts.get("Age and Survival"),
        ),
        (
            "Fare and Survival",
            charts.get("Fare and Survival"),
        ),
        (
            "Survival Pie Chart",
            charts.get("Survival Pie Chart"),
        ),
        (
            "Survival Donut Chart",
            charts.get("Survival Donut Chart"),
        ),
        (
            "Correlation Heatmap",
            charts.get("Correlation Heatmap"),
        ),
    ]

    chart_rows = []

    for start in range(
        0,
        len(main_charts),
        4,
    ):
        row = []

        for title, path in main_charts[start : start + 4]:
            row.append(
                chart_card(
                    path,
                    title,
                )
            )

        chart_rows.append(row)

    charts_table = Table(
        chart_rows,
        colWidths=[
            69 * mm,
            69 * mm,
            69 * mm,
            69 * mm,
        ],
        rowHeights=[
            43 * mm,
            43 * mm,
        ],
    )

    charts_table.setStyle(
        TableStyle(
            [
                (
                    "VALIGN",
                    (0, 0),
                    (-1, -1),
                    "TOP",
                ),
                (
                    "LEFTPADDING",
                    (0, 0),
                    (-1, -1),
                    1,
                ),
                (
                    "RIGHTPADDING",
                    (0, 0),
                    (-1, -1),
                    1,
                ),
                (
                    "TOPPADDING",
                    (0, 0),
                    (-1, -1),
                    1,
                ),
                (
                    "BOTTOMPADDING",
                    (0, 0),
                    (-1, -1),
                    1,
                ),
            ]
        )
    )

    story.append(charts_table)

    story.append(
        Spacer(
            1,
            1.2 * mm,
        )
    )

    # --------------------------------------------------------
    # INSIGHTS
    # --------------------------------------------------------

    story.append(dashboard_insights(data))

    story.append(
        Spacer(
            1,
            1 * mm,
        )
    )

    story.append(dashboard_footer())

    doc.build(story)

    print(f"[OK] Dashboard PDF created: {DASHBOARD_PDF}")


# ============================================================
# REPORT HELPERS
# ============================================================


def report_page_number(canvas, doc):

    canvas.saveState()

    canvas.setFont(
        "Helvetica",
        7,
    )

    canvas.setFillColor(colors.HexColor(MUTED))

    canvas.drawString(
        16 * mm,
        7 * mm,
        "Titanic Data Analysis • Naveed Ahmad",
    )

    canvas.drawRightString(
        194 * mm,
        7 * mm,
        f"Page {doc.page}",
    )

    canvas.restoreState()


def report_chart_image(
    charts,
    title,
    width=86 * mm,
    height=49 * mm,
):

    path = charts.get(title)

    if path and Path(path).exists():
        return Image(
            str(path),
            width=width,
            height=height,
        )

    return Paragraph(
        f"<b>{clean_text(title)}</b><br/>Chart unavailable.",
        STYLE_BODY,
    )


def report_kpi_table(data):

    total = len(data)

    survivors = int(data["survived"].sum())

    rate = data["survived"].mean() * 100

    avg_age = data["age"].mean()
    avg_fare = data["fare"].mean()

    rows = [
        [
            "Total Passengers",
            "Survivors",
            "Survival Rate",
            "Average Age",
            "Average Fare",
        ],
        [
            f"{total:,}",
            f"{survivors:,}",
            f"{rate:.1f}%",
            f"{avg_age:.1f}",
            f"${avg_fare:.1f}",
        ],
    ]

    table = Table(
        rows,
        colWidths=[
            36 * mm,
            36 * mm,
            36 * mm,
            36 * mm,
            36 * mm,
        ],
    )

    table.setStyle(
        TableStyle(
            [
                (
                    "BACKGROUND",
                    (0, 0),
                    (-1, 0),
                    colors.HexColor(DARK),
                ),
                (
                    "TEXTCOLOR",
                    (0, 0),
                    (-1, 0),
                    colors.white,
                ),
                (
                    "FONTNAME",
                    (0, 0),
                    (-1, 0),
                    "Helvetica-Bold",
                ),
                (
                    "FONTNAME",
                    (0, 1),
                    (-1, 1),
                    "Helvetica-Bold",
                ),
                (
                    "FONTSIZE",
                    (0, 0),
                    (-1, -1),
                    8,
                ),
                (
                    "ALIGN",
                    (0, 0),
                    (-1, -1),
                    "CENTER",
                ),
                (
                    "VALIGN",
                    (0, 0),
                    (-1, -1),
                    "MIDDLE",
                ),
                (
                    "BACKGROUND",
                    (0, 1),
                    (-1, 1),
                    colors.HexColor("#F8FAFC"),
                ),
                (
                    "TEXTCOLOR",
                    (0, 1),
                    (-1, 1),
                    colors.HexColor(TEXT),
                ),
                (
                    "GRID",
                    (0, 0),
                    (-1, -1),
                    0.5,
                    colors.HexColor(LIGHT_BORDER),
                ),
                (
                    "TOPPADDING",
                    (0, 0),
                    (-1, -1),
                    7,
                ),
                (
                    "BOTTOMPADDING",
                    (0, 0),
                    (-1, -1),
                    7,
                ),
            ]
        )
    )

    return table


# ============================================================
# SUMMARY HELPERS FOR 28 CHARTS
# ============================================================


def highest_rate_segment(
    data,
    columns,
    min_count=5,
):

    grouped = data.groupby(columns).agg(
        rate=("survived", "mean"),
        count=("survived", "size"),
    )

    valid = grouped[grouped["count"] >= min_count]

    if valid.empty:
        valid = grouped

    if valid.empty:
        return None, np.nan, 0

    row = valid.sort_values(
        "rate",
        ascending=False,
    ).iloc[0]

    index_value = valid.sort_values(
        "rate",
        ascending=False,
    ).index[0]

    if not isinstance(
        index_value,
        tuple,
    ):
        index_value = (index_value,)

    return (
        " / ".join(str(x) for x in index_value),
        float(row["rate"] * 100),
        int(row["count"]),
    )


def lowest_rate_segment(
    data,
    columns,
    min_count=5,
):

    grouped = data.groupby(columns).agg(
        rate=("survived", "mean"),
        count=("survived", "size"),
    )

    valid = grouped[grouped["count"] >= min_count]

    if valid.empty:
        valid = grouped

    if valid.empty:
        return None, np.nan, 0

    row = valid.sort_values(
        "rate",
        ascending=True,
    ).iloc[0]

    index_value = valid.sort_values(
        "rate",
        ascending=True,
    ).index[0]

    if not isinstance(
        index_value,
        tuple,
    ):
        index_value = (index_value,)

    return (
        " / ".join(str(x) for x in index_value),
        float(row["rate"] * 100),
        int(row["count"]),
    )


def largest_combination(
    data,
    columns,
):

    grouped = data.groupby(columns).size().sort_values(ascending=False)

    if grouped.empty:
        return "N/A", 0

    index_value = grouped.index[0]

    if not isinstance(
        index_value,
        tuple,
    ):
        index_value = (index_value,)

    label = " / ".join(str(x) for x in index_value)

    return (
        label,
        int(grouped.iloc[0]),
    )


# ============================================================
# INDIVIDUAL CHART INTERPRETATIONS
# ============================================================


def chart_summary(number, data):

    # --------------------------------------------------------
    # 1
    # --------------------------------------------------------

    if number == 1:
        counts = (
            data["survived"]
            .value_counts()
            .reindex(
                [0, 1],
                fill_value=0,
            )
        )

        return (
            f"The chart shows {int(counts[1]):,} "
            f"survivors and {int(counts[0]):,} "
            f"passengers who did not survive."
        )

    # --------------------------------------------------------
    # 2
    # --------------------------------------------------------

    if number == 2:
        rate = data["survived"].mean() * 100

        return (
            f"The overall survival percentage "
            f"was {rate:.1f}%, while the remaining "
            f"{100 - rate:.1f}% of passengers did "
            f"not survive."
        )

    # --------------------------------------------------------
    # 3
    # --------------------------------------------------------

    if number == 3:
        female = safe_rate(
            data.loc[
                data["sex"] == "female",
                "survived",
            ]
        )

        male = safe_rate(
            data.loc[
                data["sex"] == "male",
                "survived",
            ]
        )

        return (
            f"Female passengers had a survival rate "
            f"of {rate_text(female)}, compared with "
            f"{rate_text(male)} for male passengers."
        )

    # --------------------------------------------------------
    # 4
    # --------------------------------------------------------

    if number == 4:
        rates = data.groupby("pclass")["survived"].mean().mul(100)

        return (
            f"Survival rates by class were "
            f"{rate_text(rates.get(1))} for Class 1, "
            f"{rate_text(rates.get(2))} for Class 2, "
            f"and {rate_text(rates.get(3))} for Class 3."
        )

    # --------------------------------------------------------
    # 5
    # --------------------------------------------------------

    if number == 5:
        return (
            "Survival varied across the defined age "
            "intervals. The chart allows passenger "
            "survival counts to be compared across "
            "children, teenagers, young adults, and older groups."
        )

    # --------------------------------------------------------
    # 6
    # --------------------------------------------------------

    if number == 6:
        rates = data.groupby("survived")["fare"].mean().reindex([0, 1])

        return (
            f"The average fare was "
            f"${rates.get(1, np.nan):.1f} among "
            f"survivors and "
            f"${rates.get(0, np.nan):.1f} among "
            f"passengers who did not survive."
        )

    # --------------------------------------------------------
    # 7
    # --------------------------------------------------------

    if number == 7:
        rates = data.groupby("embarked")["survived"].mean().mul(100)

        parts = []

        for port in ["S", "C", "Q"]:
            if port in rates.index:
                parts.append(f"{port}: {rates[port]:.1f}%")

        return "Survival differed across embarkation ports. " + ", ".join(parts) + "."

    # --------------------------------------------------------
    # 8
    # --------------------------------------------------------

    if number == 8:
        rates = data.groupby("alone")["survived"].mean().mul(100)

        return (
            f"Passengers travelling alone had a "
            f"survival rate of "
            f"{rate_text(rates.get('Alone'))}, "
            f"while passengers with family had "
            f"{rate_text(rates.get('With Family'))}."
        )

    # --------------------------------------------------------
    # 9
    # --------------------------------------------------------

    if number == 9:
        high_label, high_rate, high_count = highest_rate_segment(
            data,
            ["family_size"],
            min_count=5,
        )

        return (
            "Survival rates varied by family size. "
            f"Among family sizes represented by at least "
            f"five passengers, the highest observed rate "
            f"was {rate_text(high_rate)} for family size "
            f"{high_label} ({high_count} passengers)."
        )

    # --------------------------------------------------------
    # 10
    # --------------------------------------------------------

    if number == 10:
        rates = data.groupby("age_group")["survived"].mean().mul(100)

        available = [
            f"{group}: {rates[group]:.1f}%"
            for group in AGE_ORDER
            if group in rates.index
        ]

        return (
            "The age-group comparison shows variation "
            "in survival proportions across the passenger "
            "age categories. " + ", ".join(available) + "."
        )

    # --------------------------------------------------------
    # 11
    # --------------------------------------------------------

    if number == 11:
        rates = data.groupby("who")["survived"].mean().mul(100)

        available = [
            f"{group.title()}: {rates[group]:.1f}%"
            for group in WHO_ORDER
            if group in rates.index
        ]

        return (
            "The Who-category chart compares survival "
            "among children, women, and men. " + ", ".join(available) + "."
        )

    # --------------------------------------------------------
    # 12
    # --------------------------------------------------------

    if number == 12:
        label, count = largest_combination(
            data,
            ["pclass", "sex"],
        )

        return (
            f"The largest sex-and-class combination "
            f"was {label}, containing {count:,} passengers."
        )

    # --------------------------------------------------------
    # 13
    # --------------------------------------------------------

    if number == 13:
        label, count = largest_combination(
            data,
            ["pclass", "age_group"],
        )

        return (
            f"The most common class-and-age-group "
            f"combination was {label}, with "
            f"{count:,} passengers."
        )

    # --------------------------------------------------------
    # 14
    # --------------------------------------------------------

    if number == 14:
        high_label, high_rate, high_count = highest_rate_segment(
            data,
            ["pclass", "sex", "age_group"],
            min_count=5,
        )

        return (
            "The multivariate chart shows how survival "
            "changes when class, sex, and age group are "
            "considered together. Among segments with at "
            f"least five passengers, {high_label} recorded "
            f"{rate_text(high_rate)} survival "
            f"({high_count} passengers)."
        )

    # --------------------------------------------------------
    # 15
    # --------------------------------------------------------

    if number == 15:
        label, count = largest_combination(
            data,
            ["family_size", "sex"],
        )

        return (
            f"The largest family-size and sex combination "
            f"was {label}, with {count:,} passengers."
        )

    # --------------------------------------------------------
    # 16
    # --------------------------------------------------------

    if number == 16:
        counts = data["family_group"].value_counts()

        if not counts.empty:
            group = counts.index[0]
            count = counts.iloc[0]

            return (
                f"{group} was the most common family "
                f"category, containing {count:,} passengers."
            )

        return (
            "The chart displays the distribution of "
            "passengers across family-size groups."
        )

    # --------------------------------------------------------
    # 17
    # --------------------------------------------------------

    if number == 17:
        label, count = largest_combination(
            data,
            ["pclass", "family_group"],
        )

        return (
            f"The most common class-and-family-group "
            f"combination was {label}, with "
            f"{count:,} passengers."
        )

    # --------------------------------------------------------
    # 18
    # --------------------------------------------------------

    if number == 18:
        high_label, high_rate, high_count = highest_rate_segment(
            data,
            ["pclass", "sex", "family_group"],
            min_count=5,
        )

        return (
            "Combining class, sex, and family group "
            "reveals additional survival variation. "
            f"The highest rate among segments with at "
            f"least five passengers was {rate_text(high_rate)} "
            f"for {high_label} ({high_count} passengers)."
        )

    # --------------------------------------------------------
    # 19
    # --------------------------------------------------------

    if number == 19:
        rates = data.groupby("fare_group")["survived"].mean().mul(100)

        available = [
            f"{group}: {rates[group]:.1f}%"
            for group in FARE_GROUP_ORDER
            if group in rates.index
        ]

        return (
            "Survival proportions varied across fare "
            "groups. " + ", ".join(available) + "."
        )

    # --------------------------------------------------------
    # 20
    # --------------------------------------------------------

    if number == 20:
        label, count = largest_combination(
            data,
            ["fare_group", "pclass"],
        )

        return (
            f"The largest fare-group and class combination "
            f"was {label}, containing {count:,} passengers."
        )

    # --------------------------------------------------------
    # 21
    # --------------------------------------------------------

    if number == 21:
        label, count = largest_combination(
            data,
            ["embarked", "pclass"],
        )

        return (
            f"The most common embarkation-and-class "
            f"combination was {label}, with "
            f"{count:,} passengers."
        )

    # --------------------------------------------------------
    # 22
    # --------------------------------------------------------

    if number == 22:
        label, count = largest_combination(
            data,
            ["sex", "fare_group"],
        )

        return (
            f"The largest sex-and-fare-group segment "
            f"was {label}, with {count:,} passengers."
        )

    # --------------------------------------------------------
    # 23
    # --------------------------------------------------------

    if number == 23:
        label, count = largest_combination(
            data,
            ["age_group", "sex"],
        )

        return (
            f"The largest age-group and sex combination "
            f"was {label}, containing {count:,} passengers."
        )

    # --------------------------------------------------------
    # 24
    # --------------------------------------------------------

    if number == 24:
        high_label, high_rate, high_count = highest_rate_segment(
            data,
            ["pclass", "sex", "fare_group"],
            min_count=5,
        )

        return (
            "The combined class, sex, and fare analysis "
            "shows differences between passenger segments. "
            f"Among segments with at least five passengers, "
            f"{high_label} had a survival rate of "
            f"{rate_text(high_rate)} ({high_count} passengers)."
        )

    # --------------------------------------------------------
    # 25
    # --------------------------------------------------------

    if number == 25:
        numeric_columns = [
            "survived",
            "pclass",
            "age",
            "sibsp",
            "parch",
            "fare",
            "family_size",
        ]

        numeric_columns = [col for col in numeric_columns if col in data.columns]

        corr = data[numeric_columns].corr()

        if "survived" in corr.columns and len(corr.columns) > 1:
            values = corr["survived"].drop("survived").dropna()

            if not values.empty:
                strongest = values.abs().idxmax()
                value = values[strongest]

                return (
                    f"The strongest absolute linear "
                    f"correlation with survival among the "
                    f"included variables was with "
                    f"{strongest.replace('_', ' ')} "
                    f"(r = {value:.2f}). Correlation describes "
                    "linear association and does not by itself "
                    "establish causation."
                )

        return (
            "The heatmap presents pairwise correlations "
            "among the selected numeric variables."
        )

    # --------------------------------------------------------
    # 26
    # --------------------------------------------------------

    if number == 26:
        total = len(data)

        survivors = int(data["survived"].sum())

        rate = data["survived"].mean() * 100

        return (
            f"The summary table consolidates the main "
            f"descriptive results: {total:,} passengers, "
            f"{survivors:,} survivors, and an overall "
            f"survival rate of {rate:.1f}%."
        )

    # --------------------------------------------------------
    # 27
    # --------------------------------------------------------

    if number == 27:
        rate = data["survived"].mean() * 100

        return (
            f"The pie chart represents the overall "
            f"survival composition. Survivors account for "
            f"{rate:.1f}% of the dataset and non-survivors "
            f"account for {100 - rate:.1f}%."
        )

    # --------------------------------------------------------
    # 28
    # --------------------------------------------------------

    if number == 28:
        rate = data["survived"].mean() * 100

        return (
            f"The donut chart provides the same survival "
            f"composition in a compact form: {rate:.1f}% "
            f"survived and {100 - rate:.1f}% did not survive."
        )

    return (
        "This chart provides a visual summary of the "
        "corresponding Titanic passenger analysis."
    )


# ============================================================
# REPORT CHART CARD
# ============================================================


def report_chart_card(
    charts,
    number,
    title,
):

    path = charts.get(title)

    if path and Path(path).exists():
        image = report_chart_image(
            charts,
            title,
            width=84 * mm,
            height=45 * mm,
        )

    else:
        image = Paragraph(
            f"<b>{clean_text(title)}</b><br/>Chart unavailable.",
            STYLE_BODY,
        )

    summary = chart_summary(
        number,
        df,
    )

    content = [
        image,
        Spacer(
            1,
            1.5 * mm,
        ),
        Paragraph(
            f"<b>Interpretation:</b> {clean_text(summary)}",
            STYLE_REPORT_SUMMARY,
        ),
    ]

    table = Table(
        [[content]],
        colWidths=[89 * mm],
    )

    table.setStyle(
        TableStyle(
            [
                (
                    "BACKGROUND",
                    (0, 0),
                    (-1, -1),
                    colors.white,
                ),
                (
                    "BOX",
                    (0, 0),
                    (-1, -1),
                    0.6,
                    colors.HexColor(LIGHT_BORDER),
                ),
                (
                    "VALIGN",
                    (0, 0),
                    (-1, -1),
                    "TOP",
                ),
                (
                    "ALIGN",
                    (0, 0),
                    (-1, -1),
                    "CENTER",
                ),
                (
                    "LEFTPADDING",
                    (0, 0),
                    (-1, -1),
                    2,
                ),
                (
                    "RIGHTPADDING",
                    (0, 0),
                    (-1, -1),
                    2,
                ),
                (
                    "TOPPADDING",
                    (0, 0),
                    (-1, -1),
                    3,
                ),
                (
                    "BOTTOMPADDING",
                    (0, 0),
                    (-1, -1),
                    4,
                ),
            ]
        )
    )

    return table


# ============================================================
# FINAL REPORT
# ============================================================


def create_final_report(data, charts):

    print("\nCreating final analytical report PDF...")

    doc = SimpleDocTemplate(
        str(FINAL_REPORT_PDF),
        pagesize=A4,
        rightMargin=12 * mm,
        leftMargin=12 * mm,
        topMargin=13 * mm,
        bottomMargin=14 * mm,
        title="Titanic Final Analytical Report",
        author="Naveed Ahmad",
    )

    story = []

    # ========================================================
    # CALCULATIONS
    # ========================================================

    total = len(data)

    survivors = int(data["survived"].sum())

    non_survivors = total - survivors

    survival_rate = data["survived"].mean() * 100

    avg_age = data["age"].mean()
    avg_fare = data["fare"].mean()

    female_rate = safe_rate(
        data.loc[
            data["sex"] == "female",
            "survived",
        ]
    )

    male_rate = safe_rate(
        data.loc[
            data["sex"] == "male",
            "survived",
        ]
    )

    class_rates = data.groupby("pclass")["survived"].mean().mul(100)

    # ========================================================
    # TITLE PAGE
    # ========================================================

    story.append(
        Spacer(
            1,
            25 * mm,
        )
    )

    story.append(
        Paragraph(
            "TITANIC DATA ANALYSIS",
            ParagraphStyle(
                "MainReportTitle",
                parent=STYLE_TITLE,
                fontSize=27,
                leading=32,
                textColor=colors.HexColor(DARK),
                alignment=TA_CENTER,
            ),
        )
    )

    story.append(
        Spacer(
            1,
            5 * mm,
        )
    )

    story.append(
        Paragraph(
            "Comprehensive Statistical & Visualization Report",
            ParagraphStyle(
                "ReportSubtitle",
                parent=STYLE_BODY,
                fontSize=14,
                leading=18,
                textColor=colors.HexColor(MUTED),
                alignment=TA_CENTER,
            ),
        )
    )

    story.append(
        Spacer(
            1,
            18 * mm,
        )
    )

    story.append(report_kpi_table(data))

    story.append(
        Spacer(
            1,
            15 * mm,
        )
    )

    story.append(
        Paragraph(
            "<b>Prepared by:</b> Naveed Ahmad<br/>"
            "<b>Project:</b> Titanic Data Analysis Project<br/>"
            "<b>Charts:</b> 28 Unique Visualizations<br/>"
            "<b>Technology:</b> Python, Pandas, NumPy, "
            "Matplotlib, ReportLab",
            ParagraphStyle(
                "TitleInfo",
                parent=STYLE_BODY,
                fontSize=10,
                leading=16,
                alignment=TA_CENTER,
            ),
        )
    )

    story.append(PageBreak())

    # ========================================================
    # EXECUTIVE SUMMARY
    # ========================================================

    story.append(
        Paragraph(
            "1. Executive Summary",
            STYLE_SECTION,
        )
    )

    executive_text = (
        f"The Titanic dataset contains "
        f"<b>{total:,}</b> passenger records. "
        f"Among these passengers, <b>{survivors:,}</b> "
        f"survived while <b>{non_survivors:,}</b> "
        f"did not. The overall survival rate was "
        f"<b>{survival_rate:.1f}%</b>. The average "
        f"passenger age was <b>{avg_age:.1f}</b> years "
        f"and the average fare was "
        f"<b>${avg_fare:.1f}</b>."
    )

    story.append(
        Paragraph(
            executive_text,
            STYLE_BODY,
        )
    )

    story.append(
        Spacer(
            1,
            5 * mm,
        )
    )

    story.append(
        Paragraph(
            "The analysis examines survival through "
            "demographic, class, age, family, fare, "
            "embarkation, and multivariate perspectives. "
            "The report contains 28 unique visualizations "
            "with an interpretation provided below each chart.",
            STYLE_BODY,
        )
    )

    story.append(
        Spacer(
            1,
            6 * mm,
        )
    )

    # Executive charts 1 + 2
    story.append(
        Table(
            [
                [
                    report_chart_card(
                        charts,
                        1,
                        "Survival Distribution",
                    ),
                    report_chart_card(
                        charts,
                        2,
                        "Survival Percentage",
                    ),
                ]
            ],
            colWidths=[
                91 * mm,
                91 * mm,
            ],
            style=TableStyle(
                [
                    (
                        "VALIGN",
                        (0, 0),
                        (-1, -1),
                        "TOP",
                    ),
                    (
                        "LEFTPADDING",
                        (0, 0),
                        (-1, -1),
                        2,
                    ),
                    (
                        "RIGHTPADDING",
                        (0, 0),
                        (-1, -1),
                        2,
                    ),
                ]
            ),
        )
    )

    story.append(PageBreak())

    # ========================================================
    # ALL 28 CHARTS
    # ========================================================

    sections = [
        (
            "2. Survival Analysis",
            [3, 4, 27, 28],
        ),
        (
            "3. Demographic Analysis",
            [5, 10, 11, 12],
        ),
        (
            "4. Passenger Class and Age Analysis",
            [13, 14, 23, 15],
        ),
        (
            "5. Family Structure Analysis",
            [8, 9, 16, 17],
        ),
        (
            "6. Multivariate Family Analysis",
            [18, 24, 21, 22],
        ),
        (
            "7. Fare and Embarkation Analysis",
            [6, 7, 19, 20],
        ),
        (
            "8. Correlation and Statistical Overview",
            [25, 26],
        ),
    ]

    specs_by_number = {
        number: (
            filename,
            title,
            function,
        )
        for number, filename, title, function in CHART_SPECS
    }

    for section_index, (
        section_title,
        numbers,
    ) in enumerate(sections):
        story.append(
            Paragraph(
                section_title,
                STYLE_SECTION,
            )
        )

        # ----------------------------------------------------
        # Section introduction
        # ----------------------------------------------------

        if section_index == 0:
            intro = (
                "This section examines the survival outcome "
                "and compares survival patterns across key "
                "passenger characteristics."
            )

        elif section_index == 1:
            intro = (
                "Demographic variables provide a detailed "
                "view of passenger composition and survival "
                "patterns across age and sex categories."
            )

        elif section_index == 2:
            intro = (
                "Passenger class, age, and sex are combined "
                "to examine passenger composition and "
                "segment-level survival patterns."
            )

        elif section_index == 3:
            intro = (
                "Family-related variables are derived from "
                "SibSp and Parch and are used to study family "
                "size and survival."
            )

        elif section_index == 4:
            intro = (
                "Multivariate charts combine several categorical "
                "variables to show more detailed passenger "
                "segments and their survival proportions."
            )

        elif section_index == 5:
            intro = (
                "Fare groups and embarkation ports are used "
                "to examine passenger segmentation and survival."
            )

        else:
            intro = (
                "The final statistical section presents the "
                "correlation structure and a consolidated "
                "visualization summary."
            )

        story.append(
            Paragraph(
                intro,
                STYLE_BODY,
            )
        )

        story.append(
            Spacer(
                1,
                4 * mm,
            )
        )

        # ----------------------------------------------------
        # Four charts per page
        # ----------------------------------------------------

        for start in range(
            0,
            len(numbers),
            4,
        ):
            batch = numbers[start : start + 4]

            row_data = []

            for number in batch:
                (
                    filename,
                    title,
                    function,
                ) = specs_by_number[number]

                row_data.append(
                    report_chart_card(
                        charts,
                        number,
                        title,
                    )
                )

            while len(row_data) < 4:
                row_data.append(
                    Spacer(
                        1,
                        1,
                    )
                )

            grid = Table(
                [
                    row_data[:2],
                    row_data[2:4],
                ],
                colWidths=[
                    91 * mm,
                    91 * mm,
                ],
            )

            grid.setStyle(
                TableStyle(
                    [
                        (
                            "VALIGN",
                            (0, 0),
                            (-1, -1),
                            "TOP",
                        ),
                        (
                            "LEFTPADDING",
                            (0, 0),
                            (-1, -1),
                            2,
                        ),
                        (
                            "RIGHTPADDING",
                            (0, 0),
                            (-1, -1),
                            2,
                        ),
                        (
                            "TOPPADDING",
                            (0, 0),
                            (-1, -1),
                            2,
                        ),
                        (
                            "BOTTOMPADDING",
                            (0, 0),
                            (-1, -1),
                            2,
                        ),
                    ]
                )
            )

            story.append(grid)

            story.append(PageBreak())

    # ========================================================
    # ANALYTICAL OBSERVATIONS
    # ========================================================

    story.append(
        Paragraph(
            "9. Analytical Observations",
            STYLE_SECTION,
        )
    )

    observations = [
        (
            "Overall Survival",
            f"The overall survival rate was {survival_rate:.1f}%.",
        ),
        (
            "Sex",
            f"Female survival rate was "
            f"{rate_text(female_rate)}, while "
            f"male survival rate was "
            f"{rate_text(male_rate)}.",
        ),
        (
            "Passenger Class",
            f"Class 1: "
            f"{rate_text(class_rates.get(1))}; "
            f"Class 2: "
            f"{rate_text(class_rates.get(2))}; "
            f"Class 3: "
            f"{rate_text(class_rates.get(3))}.",
        ),
        (
            "Average Age",
            f"The average passenger age was {avg_age:.1f} years.",
        ),
        (
            "Average Fare",
            f"The average fare was ${avg_fare:.1f}.",
        ),
    ]

    observation_rows = [
        [
            Paragraph(
                "<b>Area</b>",
                STYLE_BODY,
            ),
            Paragraph(
                "<b>Observation</b>",
                STYLE_BODY,
            ),
        ]
    ]

    for title, observation in observations:
        observation_rows.append(
            [
                Paragraph(
                    f"<b>{clean_text(title)}</b>",
                    STYLE_BODY,
                ),
                Paragraph(
                    clean_text(observation),
                    STYLE_BODY,
                ),
            ]
        )

    observation_table = Table(
        observation_rows,
        colWidths=[
            45 * mm,
            130 * mm,
        ],
    )

    observation_table.setStyle(
        TableStyle(
            [
                (
                    "BACKGROUND",
                    (0, 0),
                    (-1, 0),
                    colors.HexColor(DARK),
                ),
                (
                    "TEXTCOLOR",
                    (0, 0),
                    (-1, 0),
                    colors.white,
                ),
                (
                    "GRID",
                    (0, 0),
                    (-1, -1),
                    0.5,
                    colors.HexColor(LIGHT_BORDER),
                ),
                (
                    "BACKGROUND",
                    (0, 1),
                    (-1, -1),
                    colors.HexColor("#F8FAFC"),
                ),
                (
                    "VALIGN",
                    (0, 0),
                    (-1, -1),
                    "TOP",
                ),
                (
                    "LEFTPADDING",
                    (0, 0),
                    (-1, -1),
                    7,
                ),
                (
                    "RIGHTPADDING",
                    (0, 0),
                    (-1, -1),
                    7,
                ),
                (
                    "TOPPADDING",
                    (0, 0),
                    (-1, -1),
                    7,
                ),
                (
                    "BOTTOMPADDING",
                    (0, 0),
                    (-1, -1),
                    7,
                ),
            ]
        )
    )

    story.append(observation_table)

    story.append(
        Spacer(
            1,
            10 * mm,
        )
    )

    # ========================================================
    # METHODOLOGY
    # ========================================================

    story.append(
        Paragraph(
            "10. Methodology and Tools",
            STYLE_SECTION,
        )
    )

    methodology = (
        "<b>Data Preparation:</b> Dataset cleaning, "
        "numeric conversion, missing-value handling, "
        "and derived-variable creation.<br/><br/>"
        "<b>Derived Variables:</b> Family Size, Alone, "
        "Age Group, Family Group, Fare Group, and "
        "Who Category.<br/><br/>"
        "<b>Analysis:</b> Descriptive statistics, group "
        "comparisons, survival-rate calculations, "
        "cross-tabulation, and correlation analysis.<br/><br/>"
        "<b>Visualization:</b> Bar charts, stacked charts, "
        "percentage charts, pie chart, donut chart, and "
        "correlation heatmap.<br/><br/>"
        "<b>Reporting:</b> Professional dashboard and "
        "analytical report generated using ReportLab."
    )

    story.append(
        Paragraph(
            methodology,
            STYLE_BODY,
        )
    )

    story.append(
        Spacer(
            1,
            10 * mm,
        )
    )

    # ========================================================
    # CONCLUSION
    # ========================================================

    story.append(
        Paragraph(
            "11. Conclusion",
            STYLE_SECTION,
        )
    )

    conclusion = (
        "The Titanic analysis demonstrates how passenger-level "
        "data can be transformed into meaningful statistical "
        "and visual insights. The project integrates data "
        "preparation, descriptive analysis, categorical "
        "comparisons, family and age segmentation, fare "
        "analysis, multivariate visualization, and correlation "
        "analysis. The resulting dashboard provides a compact "
        "executive view, while the final report documents the "
        "analytical process and presents interpretations for "
        "all 28 visualizations."
    )

    story.append(
        Paragraph(
            conclusion,
            STYLE_BODY,
        )
    )

    story.append(
        Spacer(
            1,
            20 * mm,
        )
    )

    story.append(
        Paragraph(
            "Developed by Naveed Ahmad",
            ParagraphStyle(
                "FinalSignature",
                parent=STYLE_BODY,
                fontSize=11,
                leading=14,
                textColor=colors.HexColor(DARK),
                alignment=TA_CENTER,
                fontName="Helvetica-Bold",
            ),
        )
    )

    # ========================================================
    # BUILD REPORT
    # ========================================================

    doc.build(
        story,
        onFirstPage=report_page_number,
        onLaterPages=report_page_number,
    )

    print(f"[OK] Final Report PDF created: {FINAL_REPORT_PDF}")


# ============================================================
# MAIN
# ============================================================


def main():

    global df

    print("=" * 70)
    print("TITANIC DATA ANALYSIS — STAGE 11")
    print("PROFESSIONAL PDF DASHBOARD + FINAL REPORT")
    print("=" * 70)

    # --------------------------------------------------------
    # Check dataset
    # --------------------------------------------------------

    if not DATA_FILE.exists():
        print(f"\n[ERROR] Dataset not found:\n{DATA_FILE}")

        print(
            "\nMake sure 'titanic_cleaned.csv' "
            "is in the same folder as this Python file."
        )

        return

    # --------------------------------------------------------
    # Load dataset
    # --------------------------------------------------------

    try:
        df = pd.read_csv(DATA_FILE)

        print(f"\nDataset loaded successfully: {len(df):,} rows")

    except Exception as error:
        print(f"\n[ERROR] Could not load dataset: {error}")

        return

    # --------------------------------------------------------
    # Prepare dataset
    # --------------------------------------------------------

    try:
        df = prepare_data(df)

    except Exception as error:
        print("\n[ERROR] Data preparation failed:")

        print(error)

        return

    print(f"Prepared dataset: {len(df):,} rows × {len(df.columns)} columns")

    print("\nStarting Stage 11 PDF generation...")

    # --------------------------------------------------------
    # Generate all charts
    # --------------------------------------------------------

    try:
        charts = generate_all_charts(df)

    except Exception as error:
        print("\n[ERROR] Chart generation failed:")

        print(error)

        return

    if len(charts) != 28:
        print("\n[ERROR] Exactly 28 charts were expected.")

        return

    # --------------------------------------------------------
    # Dashboard
    # --------------------------------------------------------

    try:
        create_dashboard_pdf(
            df,
            charts,
        )

    except Exception as error:
        print("\n[ERROR] Dashboard PDF failed:")

        print(
            type(error).__name__,
            ":",
            error,
        )

        return

    # --------------------------------------------------------
    # Final Report
    # --------------------------------------------------------

    try:
        create_final_report(
            df,
            charts,
        )

    except Exception as error:
        print("\n[ERROR] Final Report PDF failed:")

        print(
            type(error).__name__,
            ":",
            error,
        )

        return

    # --------------------------------------------------------
    # Completion
    # --------------------------------------------------------

    print("\n" + "=" * 70)

    print("STAGE 11 COMPLETED SUCCESSFULLY")

    print("=" * 70)

    print(f"\nDashboard PDF:\n{DASHBOARD_PDF}")

    print(f"\nFinal Report PDF:\n{FINAL_REPORT_PDF}")

    print(f"\nChart folder:\n{CHART_DIR}")

    print(f"\nUnique charts generated: {len(charts)}")

    print("\nDeveloped by Naveed Ahmad")


# ============================================================
# RUN
# ============================================================

if __name__ == "__main__":
    main()
