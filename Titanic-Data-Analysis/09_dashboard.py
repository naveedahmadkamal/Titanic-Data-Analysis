# ============================================================
# TITANIC ANALYTICS DASHBOARD
# ============================================================

import tkinter as tk
from tkinter import ttk, messagebox, filedialog
import os
import zipfile
from datetime import datetime
from matplotlib.backends.backend_pdf import PdfPages

import pandas as pd
import numpy as np

import matplotlib

matplotlib.use("TkAgg")

import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg


# ============================================================
# LOAD DATA
# ============================================================

try:
    df = pd.read_csv("titanic_cleaned.csv")
except FileNotFoundError:
    messagebox.showerror(
        "File Error",
        "titanic_cleaned.csv file not found.\n\n"
        "Please keep the CSV file in the same folder as this Python file.",
    )
    raise SystemExit


# ============================================================
# PREPROCESSING
# ============================================================

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


# ============================================================
# CREATE MISSING COLUMNS IF NEEDED
# ============================================================

if "alone" not in df.columns:
    df["alone"] = np.where(df["family_size"] == 1, "Alone", "With Family")

if "who" not in df.columns:

    def get_who(row):
        if row["age"] < 16:
            return "child"
        elif str(row["sex"]).lower() == "female":
            return "woman"
        else:
            return "man"

    df["who"] = df.apply(get_who, axis=1)


# ============================================================
# COLORS
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


# Different palette for different charts
chart_colors = [
    # Chart 1
    ["#2563EB", "#EF4444"],
    # Chart 2
    ["#16A34A", "#F59E0B"],
    # Chart 3
    ["#8B5CF6", "#EC4899"],
    # Chart 4
    ["#06B6D4", "#F97316"],
    # Chart 5
    ["#3B82F6", "#EF4444"],
    # Chart 6
    ["#10B981", "#F43F5E"],
    # Chart 7
    ["#6366F1", "#F59E0B"],
    # Chart 8
    ["#14B8A6", "#E11D48"],
    # Chart 9
    ["#7C3AED", "#0EA5E9", "#F97316", "#22C55E"],
    # Chart 10
    ["#EF4444", "#8B5CF6", "#06B6D4", "#F59E0B", "#16A34A"],
    # Chart 11
    ["#DB2777", "#2563EB", "#16A34A"],
    # Chart 12
    ["#F97316", "#8B5CF6", "#06B6D4"],
    # Chart 13
    ["#2563EB", "#F43F5E", "#10B981", "#F59E0B"],
    # Chart 14
    ["#7C3AED", "#06B6D4", "#EC4899", "#22C55E"],
    # Chart 15
    ["#F97316", "#3B82F6", "#16A34A"],
    # Chart 16
    ["#E11D48", "#7C3AED", "#0EA5E9", "#F59E0B"],
    # Chart 17
    ["#06B6D4", "#F97316", "#8B5CF6", "#16A34A"],
    # Chart 18
    ["#EC4899", "#2563EB", "#10B981", "#F59E0B"],
    # Chart 19
    ["#16A34A", "#F97316", "#8B5CF6", "#06B6D4", "#EF4444"],
    # Chart 20
    ["#2563EB", "#E11D48", "#10B981"],
    # Chart 21
    ["#7C3AED", "#F59E0B", "#06B6D4"],
    # Chart 22
    ["#F43F5E", "#16A34A", "#2563EB", "#F97316", "#8B5CF6"],
    # Chart 23
    ["#0EA5E9", "#EC4899"],
    # Chart 24
    ["#22C55E", "#EF4444", "#7C3AED", "#F59E0B"],
    # Chart 25
    ["#2563EB"],
    # Chart 26
    ["#16A34A"],
]


def get_colors(number, count):
    palette = chart_colors[number % len(chart_colors)]

    if count <= len(palette):
        return palette[:count]

    result = []

    for i in range(count):
        result.append(palette[i % len(palette)])

    return result


# ============================================================
# MATPLOTLIB SETTINGS
# ============================================================

plt.rcParams["font.family"] = "DejaVu Sans"
plt.rcParams["axes.titleweight"] = "bold"


def prepare_axis(ax):
    ax.set_facecolor("white")

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    ax.tick_params(axis="both", labelsize=9)

    ax.grid(axis="y", linestyle="--", alpha=0.25)


def create_figure():
    fig = plt.Figure(figsize=(7, 4), dpi=100)

    ax = fig.add_subplot(111)

    return fig, ax


# ============================================================
# CHART 1
# SURVIVAL DISTRIBUTION
# ============================================================


def chart_1(data, frame):

    fig, ax = create_figure()

    values = data["survived"].value_counts().sort_index()

    labels = ["Did Not Survive", "Survived"]

    y = [values.get(0, 0), values.get(1, 0)]

    colors = get_colors(0, 2)

    bars = ax.bar(labels, y, color=colors)

    ax.set_title("Survival Distribution")
    ax.set_ylabel("Passengers")

    prepare_axis(ax)

    for bar in bars:
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height(),
            f"{int(bar.get_height())}",
            ha="center",
            va="bottom",
            fontsize=10,
        )

    return fig


# ============================================================
# CHART 2
# SURVIVAL PERCENTAGE
# ============================================================


def chart_2(data, frame):

    fig, ax = create_figure()

    values = data["survived"].value_counts(normalize=True).sort_index() * 100

    labels = ["Did Not Survive", "Survived"]

    y = [values.get(0, 0), values.get(1, 0)]

    colors = get_colors(1, 2)

    bars = ax.bar(labels, y, color=colors)

    ax.set_title("Survival Percentage")
    ax.set_ylabel("Percentage (%)")

    prepare_axis(ax)

    for bar in bars:
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height(),
            f"{bar.get_height():.1f}%",
            ha="center",
            va="bottom",
        )

    return fig


# ============================================================
# CHART 3
# SEX AND SURVIVAL
# ============================================================


def chart_3(data, frame):

    fig, ax = create_figure()

    table = pd.crosstab(data["sex"], data["survived"])

    table.plot(kind="bar", ax=ax, color=get_colors(2, 2))

    ax.set_title("Sex and Survival")
    ax.set_xlabel("Sex")
    ax.set_ylabel("Passengers")

    ax.legend(["Did Not Survive", "Survived"])

    prepare_axis(ax)

    plt.setp(ax.get_xticklabels(), rotation=0)

    return fig


# ============================================================
# CHART 4
# CLASS AND SURVIVAL
# ============================================================


def chart_4(data, frame):

    fig, ax = create_figure()

    table = pd.crosstab(data["pclass"], data["survived"])

    table.plot(kind="bar", ax=ax, color=get_colors(3, 2))

    ax.set_title("Class and Survival")
    ax.set_xlabel("Passenger Class")
    ax.set_ylabel("Passengers")

    ax.legend(["Did Not Survive", "Survived"])

    prepare_axis(ax)

    plt.setp(ax.get_xticklabels(), rotation=0)

    return fig


# ============================================================
# CHART 5
# AGE AND SURVIVAL
# ============================================================


def chart_5(data, frame):

    fig, ax = create_figure()

    survived_0 = data.loc[data["survived"] == 0, "age"].dropna()

    survived_1 = data.loc[data["survived"] == 1, "age"].dropna()

    bp = ax.boxplot([survived_0, survived_1], patch_artist=True)

    colors = get_colors(4, 2)

    for patch, color in zip(bp["boxes"], colors):
        patch.set_facecolor(color)

    ax.set_xticklabels(["Did Not Survive", "Survived"])

    ax.set_title("Age and Survival")
    ax.set_ylabel("Age")

    prepare_axis(ax)

    return fig


# ============================================================
# CHART 6
# FARE AND SURVIVAL
# ============================================================


def chart_6(data, frame):

    fig, ax = create_figure()

    fare_0 = data.loc[data["survived"] == 0, "fare"].dropna()

    fare_1 = data.loc[data["survived"] == 1, "fare"].dropna()

    bp = ax.boxplot([fare_0, fare_1], patch_artist=True)

    colors = get_colors(5, 2)

    for patch, color in zip(bp["boxes"], colors):
        patch.set_facecolor(color)

    ax.set_xticklabels(["Did Not Survive", "Survived"])

    ax.set_title("Fare and Survival")
    ax.set_ylabel("Fare")

    prepare_axis(ax)

    return fig


# ============================================================
# CHART 7
# EMBARKED AND SURVIVAL
# ============================================================


def chart_7(data, frame):

    fig, ax = create_figure()

    table = pd.crosstab(data["embarked"], data["survived"])

    table.plot(kind="bar", ax=ax, color=get_colors(6, 2))

    ax.set_title("Embarked and Survival")
    ax.set_xlabel("Embarked")
    ax.set_ylabel("Passengers")

    ax.legend(["Did Not Survive", "Survived"])

    prepare_axis(ax)

    plt.setp(ax.get_xticklabels(), rotation=0)

    return fig


# ============================================================
# CHART 8
# ALONE AND SURVIVAL
# ============================================================


def chart_8(data, frame):

    fig, ax = create_figure()

    table = pd.crosstab(data["alone"], data["survived"])

    table.plot(kind="bar", ax=ax, color=get_colors(7, 2))

    ax.set_title("Alone and Survival")
    ax.set_xlabel("Travel Status")
    ax.set_ylabel("Passengers")

    ax.legend(["Did Not Survive", "Survived"])

    prepare_axis(ax)

    plt.setp(ax.get_xticklabels(), rotation=0)

    return fig


# ============================================================
# CHART 9
# FAMILY SIZE AND SURVIVAL
# ============================================================


def chart_9(data, frame):

    fig, ax = create_figure()

    result = data.groupby("family_size", observed=True)["survived"].mean() * 100

    colors = get_colors(8, len(result))

    bars = ax.bar(result.index.astype(str), result.values, color=colors)

    ax.set_title("Family Size and Survival")
    ax.set_xlabel("Family Size")
    ax.set_ylabel("Survival Rate (%)")

    prepare_axis(ax)

    for bar in bars:
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height(),
            f"{bar.get_height():.1f}%",
            ha="center",
            va="bottom",
            fontsize=8,
        )

    return fig


# ============================================================
# CHART 10
# AGE GROUP AND SURVIVAL
# ============================================================


def chart_10(data, frame):

    fig, ax = create_figure()

    result = data.groupby("age_group", observed=True)["survived"].mean() * 100

    colors = get_colors(9, len(result))

    bars = ax.bar(result.index.astype(str), result.values, color=colors)

    ax.set_title("Age Group and Survival")
    ax.set_xlabel("Age Group")
    ax.set_ylabel("Survival Rate (%)")

    prepare_axis(ax)

    plt.setp(ax.get_xticklabels(), rotation=25)

    for bar in bars:
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height(),
            f"{bar.get_height():.1f}%",
            ha="center",
            va="bottom",
            fontsize=8,
        )

    return fig


# ============================================================
# CHART 11
# WHO CATEGORY AND SURVIVAL
# ============================================================


def chart_11(data, frame):

    fig, ax = create_figure()

    table = pd.crosstab(data["who"], data["survived"])

    table.plot(kind="bar", ax=ax, color=get_colors(10, 2))

    ax.set_title("Who Category and Survival")
    ax.set_xlabel("Category")
    ax.set_ylabel("Passengers")

    ax.legend(["Did Not Survive", "Survived"])

    prepare_axis(ax)

    plt.setp(ax.get_xticklabels(), rotation=0)

    return fig


# ============================================================
# CHART 12
# SEX AND CLASS
# ============================================================


def chart_12(data, frame):

    fig, ax = create_figure()

    result = data.groupby(["sex", "pclass"], observed=True)["survived"].mean() * 100

    table = result.unstack()

    table.plot(kind="bar", ax=ax, color=get_colors(11, len(table.columns)))

    ax.set_title("Sex and Class")
    ax.set_xlabel("Sex")
    ax.set_ylabel("Survival Rate (%)")

    prepare_axis(ax)

    ax.legend(title="Class")

    plt.setp(ax.get_xticklabels(), rotation=0)

    return fig


# ============================================================
# CHART 13
# CLASS AND AGE GROUP
# ============================================================


def chart_13(data, frame):

    fig, ax = create_figure()

    result = (
        data.groupby(["pclass", "age_group"], observed=True)["survived"].mean() * 100
    )

    table = result.unstack()

    table.plot(kind="bar", ax=ax, color=get_colors(12, len(table.columns)))

    ax.set_title("Class and Age Group")
    ax.set_xlabel("Passenger Class")
    ax.set_ylabel("Survival Rate (%)")

    prepare_axis(ax)

    ax.legend(title="Age Group")

    plt.setp(ax.get_xticklabels(), rotation=0)

    return fig


# ============================================================
# CHART 14
# SEX, CLASS AND AGE GROUP
# ============================================================


def chart_14(data, frame):

    fig, ax = create_figure()

    result = (
        data.groupby(["sex", "pclass", "age_group"], observed=True)["survived"].mean()
        * 100
    )

    table = result.unstack(level=[1, 2])

    table.plot(kind="bar", ax=ax, color=get_colors(13, len(table.columns)))

    ax.set_title("Sex, Class and Age Group")
    ax.set_xlabel("Sex")
    ax.set_ylabel("Survival Rate (%)")

    prepare_axis(ax)

    ax.legend(title="Class / Age Group", fontsize=7)

    plt.setp(ax.get_xticklabels(), rotation=0)

    return fig


# ============================================================
# CHART 15
# FAMILY SIZE AND SEX
# ============================================================


def chart_15(data, frame):

    fig, ax = create_figure()

    result = (
        data.groupby(["family_size", "sex"], observed=True)["survived"].mean() * 100
    )

    table = result.unstack()

    table.plot(kind="bar", ax=ax, color=get_colors(14, len(table.columns)))

    ax.set_title("Family Size and Sex")
    ax.set_xlabel("Family Size")
    ax.set_ylabel("Survival Rate (%)")

    prepare_axis(ax)

    ax.legend(title="Sex")

    plt.setp(ax.get_xticklabels(), rotation=0)

    return fig


# ============================================================
# CHART 16
# FAMILY SIZE GROUP
# ============================================================


def chart_16(data, frame):

    fig, ax = create_figure()

    result = data.groupby("family_size_group", observed=True)["survived"].mean() * 100

    colors = get_colors(15, len(result))

    bars = ax.bar(result.index.astype(str), result.values, color=colors)

    ax.set_title("Family Size Group")
    ax.set_xlabel("Family Group")
    ax.set_ylabel("Survival Rate (%)")

    prepare_axis(ax)

    plt.setp(ax.get_xticklabels(), rotation=20)

    for bar in bars:
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height(),
            f"{bar.get_height():.1f}%",
            ha="center",
            va="bottom",
            fontsize=8,
        )

    return fig


# ============================================================
# CHART 17
# CLASS AND FAMILY SIZE GROUP
# ============================================================


def chart_17(data, frame):

    fig, ax = create_figure()

    result = (
        data.groupby(["pclass", "family_size_group"], observed=True)["survived"].mean()
        * 100
    )

    table = result.unstack()

    table.plot(kind="bar", ax=ax, color=get_colors(16, len(table.columns)))

    ax.set_title("Class and Family Size Group")
    ax.set_xlabel("Passenger Class")
    ax.set_ylabel("Survival Rate (%)")

    prepare_axis(ax)

    ax.legend(title="Family Group", fontsize=8)

    plt.setp(ax.get_xticklabels(), rotation=0)

    return fig


# ============================================================
# CHART 18
# SEX, CLASS AND FAMILY GROUP
# ============================================================


def chart_18(data, frame):

    fig, ax = create_figure()

    result = (
        data.groupby(["sex", "pclass", "family_size_group"], observed=True)[
            "survived"
        ].mean()
        * 100
    )

    table = result.unstack(level=[1, 2])

    table.plot(kind="bar", ax=ax, color=get_colors(17, len(table.columns)))

    ax.set_title("Sex, Class and Family Group")
    ax.set_xlabel("Sex")
    ax.set_ylabel("Survival Rate (%)")

    prepare_axis(ax)

    ax.legend(title="Class / Family", fontsize=7)

    plt.setp(ax.get_xticklabels(), rotation=0)

    return fig


# ============================================================
# CHART 19
# FARE GROUP AND SURVIVAL
# ============================================================


def chart_19(data, frame):

    fig, ax = create_figure()

    result = data.groupby("fare_group", observed=True)["survived"].mean() * 100

    colors = get_colors(18, len(result))

    bars = ax.bar(result.index.astype(str), result.values, color=colors)

    ax.set_title("Fare Group and Survival")
    ax.set_xlabel("Fare Group")
    ax.set_ylabel("Survival Rate (%)")

    prepare_axis(ax)

    plt.setp(ax.get_xticklabels(), rotation=20)

    for bar in bars:
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height(),
            f"{bar.get_height():.1f}%",
            ha="center",
            va="bottom",
            fontsize=8,
        )

    return fig


# ============================================================
# CHART 20
# FARE GROUP AND CLASS
# ============================================================


def chart_20(data, frame):

    fig, ax = create_figure()

    result = (
        data.groupby(["fare_group", "pclass"], observed=True)["survived"].mean() * 100
    )

    table = result.unstack()

    table.plot(kind="bar", ax=ax, color=get_colors(19, len(table.columns)))

    ax.set_title("Fare Group and Class")
    ax.set_xlabel("Fare Group")
    ax.set_ylabel("Survival Rate (%)")

    prepare_axis(ax)

    ax.legend(title="Class")

    plt.setp(ax.get_xticklabels(), rotation=20)

    return fig


# ============================================================
# CHART 21
# EMBARKED AND CLASS
# ============================================================


def chart_21(data, frame):

    fig, ax = create_figure()

    result = (
        data.groupby(["embarked", "pclass"], observed=True)["survived"].mean() * 100
    )

    table = result.unstack()

    table.plot(kind="bar", ax=ax, color=get_colors(20, len(table.columns)))

    ax.set_title("Embarked and Class")
    ax.set_xlabel("Embarked")
    ax.set_ylabel("Survival Rate (%)")

    prepare_axis(ax)

    ax.legend(title="Class")

    plt.setp(ax.get_xticklabels(), rotation=0)

    return fig


# ============================================================
# CHART 22
# SEX AND FARE GROUP
# ============================================================


def chart_22(data, frame):

    fig, ax = create_figure()

    result = data.groupby(["sex", "fare_group"], observed=True)["survived"].mean() * 100

    table = result.unstack()

    table.plot(kind="bar", ax=ax, color=get_colors(21, len(table.columns)))

    ax.set_title("Sex and Fare Group")
    ax.set_xlabel("Sex")
    ax.set_ylabel("Survival Rate (%)")

    prepare_axis(ax)

    ax.legend(title="Fare Group", fontsize=7)

    plt.setp(ax.get_xticklabels(), rotation=0)

    return fig


# ============================================================
# CHART 23
# AGE GROUP AND SEX
# ============================================================


def chart_23(data, frame):

    fig, ax = create_figure()

    result = data.groupby(["age_group", "sex"], observed=True)["survived"].mean() * 100

    table = result.unstack()

    table.plot(kind="bar", ax=ax, color=get_colors(22, len(table.columns)))

    ax.set_title("Age Group and Sex")
    ax.set_xlabel("Age Group")
    ax.set_ylabel("Survival Rate (%)")

    prepare_axis(ax)

    ax.legend(title="Sex")

    plt.setp(ax.get_xticklabels(), rotation=20)

    return fig


# ============================================================
# CHART 24
# CLASS, SEX AND FARE GROUP
# ============================================================


def chart_24(data, frame):

    fig, ax = create_figure()

    result = (
        data.groupby(["pclass", "sex", "fare_group"], observed=True)["survived"].mean()
        * 100
    )

    table = result.unstack(level=[1, 2])

    table.plot(kind="bar", ax=ax, color=get_colors(23, len(table.columns)))

    ax.set_title("Class, Sex and Fare Group")
    ax.set_xlabel("Passenger Class")
    ax.set_ylabel("Survival Rate (%)")

    prepare_axis(ax)

    ax.legend(title="Sex / Fare Group", fontsize=7)

    plt.setp(ax.get_xticklabels(), rotation=0)

    return fig


# ============================================================
# CHART 25
# CORRELATION HEATMAP
# ============================================================


def chart_25(data, frame):

    fig = plt.Figure(figsize=(7, 5), dpi=100)

    ax = fig.add_subplot(111)

    columns = ["survived", "pclass", "age", "sibsp", "parch", "fare", "family_size"]

    corr = data[columns].corr()

    image = ax.imshow(corr, cmap="coolwarm", vmin=-1, vmax=1)

    ax.set_title("Correlation Heatmap")

    ax.set_xticks(range(len(columns)))

    ax.set_yticks(range(len(columns)))

    ax.set_xticklabels(columns, rotation=45, ha="right")

    ax.set_yticklabels(columns)

    for i in range(len(columns)):
        for j in range(len(columns)):
            ax.text(
                j, i, f"{corr.iloc[i, j]:.2f}", ha="center", va="center", fontsize=8
            )

    fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04)

    return fig


# ============================================================
# CHART 26
# VISUALIZATION SUMMARY
# ============================================================


def chart_26(data, frame):

    fig = plt.Figure(figsize=(7, 4), dpi=100)

    ax = fig.add_subplot(111)

    ax.axis("off")

    text = (
        "Visualization Summary\n\n"
        "• Survival was higher among female passengers.\n\n"
        "• First class showed higher survival than third class.\n\n"
        "• Higher fare groups generally showed higher survival.\n\n"
        "• Family size showed different survival patterns.\n\n"
        "• Age groups also showed differences in survival.\n\n"
        "• The correlation plot shows relationships between "
        "numerical variables.\n\n"
        "• Visual results should be interpreted together with "
        "statistical analysis."
    )

    ax.text(0.05, 0.95, text, transform=ax.transAxes, va="top", fontsize=11, color=TEXT)

    return fig


# ============================================================
# EXECUTIVE PIE CHART
# ============================================================


def chart_survival_pie(data, frame):

    fig, ax = create_figure()

    values = data["survived"].value_counts()

    labels = ["Did Not Survive", "Survived"]

    sizes = [values.get(0, 0), values.get(1, 0)]

    colors = ["#EF4444", "#22C55E"]

    ax.pie(
        sizes,
        labels=labels,
        autopct="%1.1f%%",
        startangle=90,
        colors=colors,
        wedgeprops={"edgecolor": "white", "linewidth": 2},
    )

    ax.set_title("Survival Pie Chart")

    return fig


# ============================================================
# EXECUTIVE DONUT CHART
# ============================================================


def chart_survival_donut(data, frame):

    fig, ax = create_figure()

    values = data["survived"].value_counts()

    labels = ["Did Not Survive", "Survived"]

    sizes = [values.get(0, 0), values.get(1, 0)]

    colors = ["#7C3AED", "#06B6D4"]

    ax.pie(
        sizes,
        labels=labels,
        autopct="%1.1f%%",
        startangle=90,
        colors=colors,
        wedgeprops={"width": 0.42, "edgecolor": "white", "linewidth": 2},
    )

    ax.set_title("Survival Donut Chart")

    return fig


# ============================================================
# MAIN WINDOW
# ============================================================

root = tk.Tk()

root.title("Titanic Analytics Dashboard | Naveed Ahmad")

root.geometry("1600x1000")

root.minsize(1250, 800)

root.configure(bg=BG)


# ============================================================
# STYLE
# ============================================================

style = ttk.Style()

try:
    style.theme_use("clam")
except:
    pass

style.configure("TNotebook", background=BG, borderwidth=0)

style.configure("TNotebook.Tab", padding=[18, 10], font=("Segoe UI", 10, "bold"))

style.map(
    "TNotebook.Tab", background=[("selected", BLUE)], foreground=[("selected", "white")]
)


# ============================================================
# HEADER
# ============================================================

header = tk.Frame(root, bg=DARK, height=85)

header.pack(fill="x")

header.pack_propagate(False)


title_frame = tk.Frame(header, bg=DARK)

title_frame.pack(side="left", padx=25)


tk.Label(
    title_frame,
    text="Titanic Analytics Dashboard",
    font=("Segoe UI", 23, "bold"),
    fg="white",
    bg=DARK,
).pack(anchor="w")


tk.Label(
    title_frame,
    text="Data Analysis • Visualization • Statistical Insights",
    font=("Segoe UI", 10),
    fg="#CBD5E1",
    bg=DARK,
).pack(anchor="w")


tk.Label(
    header, text="Naveed Ahmad", font=("Segoe UI", 13, "bold"), fg="white", bg=DARK
).pack(side="right", padx=30)


# ============================================================
# FILTER SECTION
# ============================================================

filter_frame = tk.Frame(root, bg=BG)

filter_frame.pack(fill="x", padx=18, pady=(12, 5))


def get_unique_values(column):

    values = df[column].dropna().astype(str).unique().tolist()

    return sorted(values)


# Filter variables

sex_var = tk.StringVar(value="All")
class_var = tk.StringVar(value="All")
embarked_var = tk.StringVar(value="All")
survival_var = tk.StringVar(value="All")
age_group_var = tk.StringVar(value="All")
fare_group_var = tk.StringVar(value="All")
family_group_var = tk.StringVar(value="All")


def make_filter(parent, label, variable, values, column):

    box = tk.Frame(parent, bg=CARD, bd=1, relief="solid")

    box.pack(side="left", padx=5, pady=3)

    tk.Label(box, text=label, font=("Segoe UI", 8, "bold"), bg=CARD, fg=MUTED).pack(
        padx=8, pady=(5, 0)
    )

    combo = ttk.Combobox(
        box, textvariable=variable, values=["All"] + values, state="readonly", width=14
    )

    combo.pack(padx=8, pady=(2, 6))

    return combo


make_filter(filter_frame, "Sex", sex_var, get_unique_values("sex"), "sex")

make_filter(
    filter_frame,
    "Class",
    class_var,
    [str(x) for x in sorted(df["pclass"].dropna().unique())],
    "pclass",
)

make_filter(
    filter_frame, "Embarked", embarked_var, get_unique_values("embarked"), "embarked"
)

make_filter(
    filter_frame,
    "Survival",
    survival_var,
    ["0 - Did Not Survive", "1 - Survived"],
    "survived",
)

make_filter(
    filter_frame, "Age Group", age_group_var, [str(x) for x in age_labels], "age_group"
)

make_filter(
    filter_frame,
    "Fare Group",
    fare_group_var,
    [str(x) for x in fare_labels],
    "fare_group",
)

make_filter(
    filter_frame,
    "Family Group",
    family_group_var,
    [str(x) for x in family_labels],
    "family_size_group",
)


# ============================================================
# FILTER DATA
# ============================================================


def get_filtered_data():

    data = df.copy()

    if sex_var.get() != "All":
        data = data[data["sex"].astype(str) == sex_var.get()]

    if class_var.get() != "All":
        data = data[data["pclass"].astype(str) == class_var.get()]

    if embarked_var.get() != "All":
        data = data[data["embarked"].astype(str) == embarked_var.get()]

    if survival_var.get() != "All":
        selected = survival_var.get().split(" ")[0]

        data = data[data["survived"].astype(str) == selected]

    if age_group_var.get() != "All":
        data = data[data["age_group"].astype(str) == age_group_var.get()]

    if fare_group_var.get() != "All":
        data = data[data["fare_group"].astype(str) == fare_group_var.get()]

    if family_group_var.get() != "All":
        data = data[data["family_size_group"].astype(str) == family_group_var.get()]

    return data


# ============================================================
# KPI FRAME
# ============================================================

kpi_frame = tk.Frame(root, bg=BG)

kpi_frame.pack(fill="x", padx=18, pady=8)


kpi_labels = {}


def create_kpi(parent, title, color, column):

    card = tk.Frame(parent, bg=CARD, bd=1, relief="solid")

    card.grid(row=0, column=column, sticky="nsew", padx=5)

    parent.grid_columnconfigure(column, weight=1)

    tk.Frame(card, bg=color, height=5).pack(fill="x")

    tk.Label(card, text=title, font=("Segoe UI", 9, "bold"), bg=CARD, fg=MUTED).pack(
        pady=(9, 0)
    )

    value_label = tk.Label(
        card, text="0", font=("Segoe UI", 19, "bold"), bg=CARD, fg=TEXT
    )

    value_label.pack(pady=(2, 10))

    kpi_labels[title] = value_label


create_kpi(kpi_frame, "Total Passengers", BLUE, 0)

create_kpi(kpi_frame, "Survivors", GREEN, 1)

create_kpi(kpi_frame, "Survival Rate", PURPLE, 2)

create_kpi(kpi_frame, "Average Age", ORANGE, 3)

create_kpi(kpi_frame, "Average Fare", CYAN, 4)


# ============================================================
# NOTEBOOK
# ============================================================

notebook = ttk.Notebook(root)

notebook.pack(fill="both", expand=True, padx=18, pady=(4, 12))


# ============================================================
# CREATE SCROLLABLE TAB
# ============================================================


def create_scrollable_tab(title):

    tab = tk.Frame(notebook, bg=BG)

    notebook.add(tab, text=title)

    canvas = tk.Canvas(tab, bg=BG, highlightthickness=0)

    scrollbar = ttk.Scrollbar(tab, orient="vertical", command=canvas.yview)

    content = tk.Frame(canvas, bg=BG)

    content_window = canvas.create_window((0, 0), window=content, anchor="nw")

    def configure_scroll(event):

        canvas.configure(scrollregion=canvas.bbox("all"))

    def configure_width(event):

        canvas.itemconfig(content_window, width=event.width)

    content.bind("<Configure>", configure_scroll)

    canvas.bind("<Configure>", configure_width)

    canvas.configure(yscrollcommand=scrollbar.set)

    canvas.pack(side="left", fill="both", expand=True)

    scrollbar.pack(side="right", fill="y")

    return content


# ============================================================
# CREATE CHART CARD
# ============================================================


def create_chart_card(parent, title, chart_function, height=430):

    card = tk.Frame(parent, bg=CARD, bd=1, relief="solid", height=height)

    card.grid_propagate(False)

    title_label = tk.Label(
        card, text=title, font=("Segoe UI", 12, "bold"), bg=CARD, fg=DARK
    )

    title_label.pack(anchor="w", padx=15, pady=(12, 3))

    chart_area = tk.Frame(card, bg=CARD)

    chart_area.pack(fill="both", expand=True, padx=8, pady=5)

    card.chart_function = chart_function
    card.chart_title = title
    card.chart_area = chart_area

    return card


# ============================================================
# ADD CHART GRID
# ============================================================


def add_chart_grid(parent, charts):

    cards = []

    for index, item in enumerate(charts):
        title, function, height = item

        row = index // 2
        column = index % 2

        parent.grid_columnconfigure(column, weight=1)

        card = create_chart_card(parent, title, function, height)

        card.grid(row=row, column=column, sticky="nsew", padx=8, pady=8)

        cards.append(card)

    return cards


# ============================================================
# TABS
# ============================================================

executive_tab = create_scrollable_tab("Executive Overview")

demographics_tab = create_scrollable_tab("Demographics")

fare_tab = create_scrollable_tab("Fare & Embarkation")

family_tab = create_scrollable_tab("Family & Age")

statistics_tab = create_scrollable_tab("Correlation & Statistics")

insights_tab = create_scrollable_tab("Analyst Insights")


# ============================================================
# CHART LISTS
# ============================================================

executive_charts = [
    ("Survival Distribution", chart_1, 430),
    ("Survival Percentage", chart_2, 430),
    ("Survival Pie Chart", chart_survival_pie, 430),
    ("Survival Donut Chart", chart_survival_donut, 430),
]


demographics_charts = [
    ("Sex and Survival", chart_3, 430),
    ("Class and Survival", chart_4, 430),
    ("Age and Survival", chart_5, 430),
    ("Who Category and Survival", chart_11, 430),
    ("Sex and Class", chart_12, 430),
    ("Class and Age Group", chart_13, 430),
    ("Sex, Class and Age Group", chart_14, 470),
    ("Age Group and Sex", chart_23, 430),
]


fare_charts = [
    ("Fare and Survival", chart_6, 430),
    ("Embarked and Survival", chart_7, 430),
    ("Fare Group and Survival", chart_19, 430),
    ("Fare Group and Class", chart_20, 430),
    ("Embarked and Class", chart_21, 430),
    ("Sex and Fare Group", chart_22, 430),
    ("Class, Sex and Fare Group", chart_24, 470),
]


family_charts = [
    ("Alone and Survival", chart_8, 430),
    ("Family Size and Survival", chart_9, 430),
    ("Age Group and Survival", chart_10, 430),
    ("Family Size and Sex", chart_15, 430),
    ("Family Size Group", chart_16, 430),
    ("Class and Family Size Group", chart_17, 430),
    ("Sex, Class and Family Group", chart_18, 470),
]


statistics_charts = [
    ("Correlation Heatmap", chart_25, 550),
    ("Visualization Summary", chart_26, 500),
]


# ============================================================
# ADD CHARTS
# ============================================================

all_chart_cards = []

all_chart_cards.extend(add_chart_grid(executive_tab, executive_charts))

all_chart_cards.extend(add_chart_grid(demographics_tab, demographics_charts))

all_chart_cards.extend(add_chart_grid(fare_tab, fare_charts))

all_chart_cards.extend(add_chart_grid(family_tab, family_charts))

all_chart_cards.extend(add_chart_grid(statistics_tab, statistics_charts))


# ============================================================
# ANALYST INSIGHTS
# ============================================================

insight_card = tk.Frame(insights_tab, bg=CARD, bd=1, relief="solid")

insight_card.pack(fill="x", padx=15, pady=15)


tk.Label(
    insight_card,
    text="Analyst Insights",
    font=("Segoe UI", 18, "bold"),
    bg=CARD,
    fg=DARK,
).pack(anchor="w", padx=20, pady=(18, 8))


insight_text = tk.Label(
    insight_card,
    text="",
    font=("Segoe UI", 11),
    bg=CARD,
    fg=TEXT,
    justify="left",
    anchor="w",
)

insight_text.pack(fill="x", padx=20, pady=(0, 20))


# ============================================================
# UPDATE INSIGHTS
# ============================================================


def update_insights(data):

    if len(data) == 0:
        insight_text.config(text="No passengers match the selected filters.")

        return

    total = len(data)

    survival_rate = data["survived"].mean() * 100

    female_rate = None
    male_rate = None

    if "female" in data["sex"].astype(str).unique():
        female_data = data[data["sex"].astype(str) == "female"]

        if len(female_data) > 0:
            female_rate = female_data["survived"].mean() * 100

    if "male" in data["sex"].astype(str).unique():
        male_data = data[data["sex"].astype(str) == "male"]

        if len(male_data) > 0:
            male_rate = male_data["survived"].mean() * 100

    class_rates = data.groupby("pclass")["survived"].mean() * 100

    lines = []

    lines.append(f"• Filtered passenger count: {total:,}")

    lines.append(f"• Overall survival rate: {survival_rate:.1f}%")

    if female_rate is not None:
        lines.append(f"• Female survival rate: {female_rate:.1f}%")

    if male_rate is not None:
        lines.append(f"• Male survival rate: {male_rate:.1f}%")

    if 1 in class_rates.index:
        lines.append(f"• First class survival rate: {class_rates[1]:.1f}%")

    if 2 in class_rates.index:
        lines.append(f"• Second class survival rate: {class_rates[2]:.1f}%")

    if 3 in class_rates.index:
        lines.append(f"• Third class survival rate: {class_rates[3]:.1f}%")

    lines.append(
        "• These observations describe the filtered data "
        "and do not by themselves establish causation."
    )

    insight_text.config(text="\n\n".join(lines))


# ============================================================
# DRAW CHART
# ============================================================


def draw_chart(card, data):

    for widget in card.chart_area.winfo_children():
        widget.destroy()

    try:
        if data.empty:
            tk.Label(
                card.chart_area,
                text="No data available for selected filters.",
                font=("Segoe UI", 12, "bold"),
                bg=CARD,
                fg=MUTED,
            ).pack(expand=True)

            return

        fig = card.chart_function(data, card.chart_area)

        canvas = FigureCanvasTkAgg(fig, master=card.chart_area)

        canvas.draw()

        canvas.get_tk_widget().pack(fill="both", expand=True)

        plt.close(fig)

    except Exception as error:
        tk.Label(
            card.chart_area,
            text=f"Chart could not be displayed.\n\n{error}",
            font=("Segoe UI", 10),
            bg=CARD,
            fg=RED,
            justify="left",
        ).pack(expand=True, padx=15)


# ============================================================
# UPDATE KPIs
# ============================================================


def update_kpis(data):

    total = len(data)

    survivors = int(data["survived"].sum()) if total > 0 else 0

    survival_rate = data["survived"].mean() * 100 if total > 0 else 0

    avg_age = data["age"].mean() if total > 0 else 0

    avg_fare = data["fare"].mean() if total > 0 else 0

    kpi_labels["Total Passengers"].config(text=f"{total:,}")

    kpi_labels["Survivors"].config(text=f"{survivors:,}")

    kpi_labels["Survival Rate"].config(text=f"{survival_rate:.1f}%")

    kpi_labels["Average Age"].config(text=f"{avg_age:.1f}")

    kpi_labels["Average Fare"].config(text=f"{avg_fare:.2f}")


# ============================================================
# REFRESH DASHBOARD
# ============================================================


def refresh_dashboard():

    data = get_filtered_data()

    update_kpis(data)

    update_insights(data)

    for card in all_chart_cards:
        draw_chart(card, data)


# ============================================================
# RESET FILTERS
# ============================================================


def reset_filters():

    sex_var.set("All")
    class_var.set("All")
    embarked_var.set("All")
    survival_var.set("All")
    age_group_var.set("All")
    fare_group_var.set("All")
    family_group_var.set("All")

    refresh_dashboard()


# ============================================================
# FILTER BUTTONS
# ============================================================

button_frame = tk.Frame(filter_frame, bg=BG)

button_frame.pack(side="left", padx=10)


refresh_button = tk.Button(
    button_frame,
    text="Apply Filters",
    command=refresh_dashboard,
    bg=BLUE,
    fg="white",
    font=("Segoe UI", 9, "bold"),
    relief="flat",
    padx=15,
    pady=8,
    cursor="hand2",
)

refresh_button.pack(side="left", padx=4)


reset_button = tk.Button(
    button_frame,
    text="Reset",
    command=reset_filters,
    bg=DARK,
    fg="white",
    font=("Segoe UI", 9, "bold"),
    relief="flat",
    padx=15,
    pady=8,
    cursor="hand2",
)

reset_button.pack(side="left", padx=4)


# ============================================================
# SAVE / EXPORT / SHARE DASHBOARD
# ============================================================


def export_all_charts():
    """Export every dashboard chart as individual high-resolution PNG files."""
    data = get_filtered_data()

    if data.empty:
        messagebox.showwarning("Export", "No data available for the selected filters.")
        return

    folder = filedialog.askdirectory(title="Select folder to save dashboard charts")
    if not folder:
        return

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    export_folder = os.path.join(folder, f"Titanic_Dashboard_Charts_{timestamp}")
    os.makedirs(export_folder, exist_ok=True)

    exported = 0

    for index, card in enumerate(all_chart_cards, start=1):
        try:
            fig = card.chart_function(data, card.chart_area)
            filename = f"{index:02d}_{card.chart_title.replace('/', '-').replace(' ', '_')}.png"
            filepath = os.path.join(export_folder, filename)
            fig.savefig(filepath, dpi=300, bbox_inches="tight", facecolor="white")
            plt.close(fig)
            exported += 1
        except Exception:
            try:
                plt.close(fig)
            except Exception:
                pass

    messagebox.showinfo(
        "Export Complete",
        f"{exported} chart(s) exported successfully.\n\nSaved to:\n{export_folder}",
    )


def export_dashboard_pdf():
    """Export all dashboard charts into one PDF file."""
    data = get_filtered_data()

    if data.empty:
        messagebox.showwarning(
            "PDF Export", "No data available for the selected filters."
        )
        return

    filepath = filedialog.asksaveasfilename(
        title="Save Dashboard PDF",
        defaultextension=".pdf",
        filetypes=[("PDF files", "*.pdf")],
    )

    if not filepath:
        return

    exported = 0

    try:
        with PdfPages(filepath) as pdf:
            for index, card in enumerate(all_chart_cards, start=1):
                try:
                    fig = card.chart_function(data, card.chart_area)
                    fig.suptitle(
                        f"Titanic Analytics Dashboard — {card.chart_title}",
                        fontsize=14,
                        fontweight="bold",
                        y=0.98,
                    )
                    pdf.savefig(fig, bbox_inches="tight")
                    plt.close(fig)
                    exported += 1
                except Exception:
                    try:
                        plt.close(fig)
                    except Exception:
                        pass

        messagebox.showinfo(
            "PDF Export Complete",
            f"{exported} chart(s) exported successfully.\n\nPDF saved to:\n{filepath}",
        )

    except Exception as error:
        messagebox.showerror("PDF Export Error", str(error))


def save_dashboard_screenshot():
    """Save a screenshot of the currently visible dashboard window."""
    filepath = filedialog.asksaveasfilename(
        title="Save Dashboard Screenshot",
        defaultextension=".png",
        filetypes=[("PNG image", "*.png"), ("JPEG image", "*.jpg")],
    )

    if not filepath:
        return

    try:
        # Windows screenshot of the complete visible application window.
        root.update_idletasks()
        root.update()

        try:
            from PIL import ImageGrab
        except ImportError:
            messagebox.showerror(
                "Screenshot Error",
                "Pillow is required for screenshots.\n\n"
                "Install it with:\npip install pillow",
            )
            return

        x = root.winfo_rootx()
        y = root.winfo_rooty()
        width = root.winfo_width()
        height = root.winfo_height()

        image = ImageGrab.grab(bbox=(x, y, x + width, y + height))
        image.save(filepath)

        messagebox.showinfo(
            "Screenshot Saved",
            f"Dashboard screenshot saved successfully:\n\n{filepath}",
        )

    except Exception as error:
        messagebox.showerror("Screenshot Error", str(error))


def create_shareable_zip():
    """Create a ZIP containing the dashboard Python file and CSV dataset."""
    py_file = os.path.abspath(__file__)
    csv_file = os.path.join(os.path.dirname(py_file), "titanic_cleaned.csv")

    if not os.path.exists(csv_file):
        messagebox.showwarning(
            "Share Project",
            "titanic_cleaned.csv was not found in the same folder as the Python file.",
        )
        return

    filepath = filedialog.asksaveasfilename(
        title="Save Shareable Dashboard ZIP",
        defaultextension=".zip",
        filetypes=[("ZIP archive", "*.zip")],
    )

    if not filepath:
        return

    try:
        with zipfile.ZipFile(filepath, "w", zipfile.ZIP_DEFLATED) as z:
            z.write(py_file, os.path.basename(py_file))
            z.write(csv_file, "titanic_cleaned.csv")

        messagebox.showinfo(
            "Shareable ZIP Created",
            "Your dashboard package is ready.\n\n"
            f"Saved to:\n{filepath}\n\n"
            "You can upload this ZIP to Google Drive, OneDrive, GitHub, "
            "or send it to another person.",
        )

    except Exception as error:
        messagebox.showerror("ZIP Error", str(error))


# ============================================================
# EXPORT BUTTONS
# ============================================================

export_frame = tk.Frame(filter_frame, bg=BG)
export_frame.pack(side="left", padx=8)

export_png_button = tk.Button(
    export_frame,
    text="Save Screenshot",
    command=save_dashboard_screenshot,
    bg=GREEN,
    fg="white",
    font=("Segoe UI", 9, "bold"),
    relief="flat",
    padx=12,
    pady=8,
    cursor="hand2",
)
export_png_button.pack(side="left", padx=3)

export_pdf_button = tk.Button(
    export_frame,
    text="Export PDF",
    command=export_dashboard_pdf,
    bg=PURPLE,
    fg="white",
    font=("Segoe UI", 9, "bold"),
    relief="flat",
    padx=12,
    pady=8,
    cursor="hand2",
)
export_pdf_button.pack(side="left", padx=3)

export_charts_button = tk.Button(
    export_frame,
    text="Export Charts",
    command=export_all_charts,
    bg=ORANGE,
    fg="white",
    font=("Segoe UI", 9, "bold"),
    relief="flat",
    padx=12,
    pady=8,
    cursor="hand2",
)
export_charts_button.pack(side="left", padx=3)

share_zip_button = tk.Button(
    export_frame,
    text="Create Share ZIP",
    command=create_shareable_zip,
    bg=DARK,
    fg="white",
    font=("Segoe UI", 9, "bold"),
    relief="flat",
    padx=12,
    pady=8,
    cursor="hand2",
)
share_zip_button.pack(side="left", padx=3)


# ============================================================
# FOOTER
# ============================================================

footer = tk.Frame(root, bg=DARK, height=28)

footer.pack(fill="x")

footer.pack_propagate(False)


tk.Label(
    footer,
    text="Titanic Dataset Analysis | Python • Pandas • Matplotlib • Tkinter",
    font=("Segoe UI", 8),
    bg=DARK,
    fg="#CBD5E1",
).pack(pady=5)


# ============================================================
# INITIAL DASHBOARD
# ============================================================

refresh_dashboard()


# ============================================================
# START APPLICATION
# ============================================================

root.mainloop()
