import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from matplotlib.ticker import FuncFormatter

import matplotlib.pyplot as plt
import requests
from sidrapy import get_table

data = get_table(
    table_code="4093",
    territorial_level="1",
    ibge_territorial_code="all",
    period="all",
    category="39426"        # Total category
)
data.query("D3C=='4099'")

4099


# Step 1: Fetch Data from OECD API
url = "https://stats.oecd.org/SDMX-JSON/data/DP_LIVE/.UNEMP.TOT.PC.LF.SA/OECD?contentType=csv&detail=code&separator=comma&csv-lang=en"
response = requests.get(url)
data = response.content.decode('utf-8')

# Step 2: Process the Data
df = pd.read_csv(pd.compat.StringIO(data))


def set_global_seaborn_config():
    sns.set_context("talk", font_scale=1.2)
    sns.set_style("whitegrid")
    sns.set_palette("Set2")
    plt.rcParams["figure.figsize"] = (14, 8)
    plt.rcParams["xtick.labelsize"] = 14
    plt.rcParams["ytick.labelsize"] = 14
    plt.rcParams["axes.titlesize"] = 16
    plt.rcParams["axes.titleweight"] = "bold"
    plt.rcParams["axes.labelsize"] = 14
    plt.rcParams["grid.linestyle"] = "--"
    plt.rcParams["grid.linewidth"] = 0.7
    plt.rcParams["grid.color"] = "gray"
    plt.rcParams["axes.grid"] = True
    plt.rcParams["savefig.dpi"] = 300


set_global_seaborn_config()


def add_source_mark(data_source: str):
    plt.figtext(0.99, 0.01, f"Data Source: {data_source}", horizontalalignment="right", fontsize=10, style="italic")


def get_line_plot_fig(df_plot: pd.DataFrame, data_source: str):
    """Plot Prive per Pax over time.

    Args:
        df_plot: Table with price data
        data_source: Source of data

    Returns:
        plt.Figure
    """
    fig, ax = plt.subplots(figsize=(14, 8))
    plt.figure(figsize=(14, 8))
    destination_alias = get_destination_alias(df_plot)

    sns.lineplot(data=df_plot, x="Date", y="PrecoPorPax", hue=destination_alias, marker="o", ax=ax)

    ax.set_title("Price per Pax Over Time")
    ax.set_xlabel("Trip Date")
    ax.set_ylabel("Price per Pax")
    ax.legend(title=destination_alias)

    add_source_mark(data_source)
    ax.set_xticklabels(ax.get_xticklabels(), rotation=45)
    plt.tight_layout()

    return fig


def get_nulls_histogram(nulls_proportion, data_source, destination_alias):

    fig, ax = plt.subplots(figsize=(14, 8))  # Create a figure and axis

    # Plot the histogram
    sns.histplot(nulls_proportion["null_perc"], bins=30, kde=True, ax=ax)

    # Set the title and labels
    ax.set_title(f"Histograma do percentual de valores nulos por {destination_alias}")
    ax.set_xlabel("% de nulos")
    ax.set_ylabel(f"Contagem de {destination_alias}s")
    ax.xaxis.set_major_formatter(FuncFormatter(lambda x, _: f"{x*100:.0f}%"))
    plt.figtext(
        0.99,
        0.03,
        f"Total de {destination_alias}s: {nulls_proportion.shape[0]} ",
        ha="right",
        fontsize=10,
        style="italic",
    )
    add_source_mark(data_source)
    ax.set_xticklabels(ax.get_xticklabels(), rotation=45)
    plt.tight_layout()

    return fig


def get_scatter_plot_fig(df_plot: pd.DataFrame, data_source: str):
    """Plot Prive per Pax over time.

    Args:
        df_plot: Table with price data
        data_source: Source of data

    Returns:
        plt.Figure
    """
    fig, ax = plt.subplots(figsize=(14, 8))
    plt.figure(figsize=(14, 8))
    destination_alias = get_destination_alias(df_plot)

    sns.scatterplot(data=df_plot, x="Date", y="PrecoPorPax", hue=destination_alias, marker="o", ax=ax)

    ax.set_title("Price per Pax Over Time")
    ax.set_xlabel("Trip Date")
    ax.set_ylabel("Price per Pax")
    ax.legend(title=destination_alias)

    add_source_mark(data_source)
    ax.set_xticklabels(ax.get_xticklabels(), rotation=45)
    plt.tight_layout()

    return fig
