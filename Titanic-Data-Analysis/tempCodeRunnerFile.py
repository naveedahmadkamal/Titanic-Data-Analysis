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
