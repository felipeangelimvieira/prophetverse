import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from matplotlib.ticker import FuncFormatter

import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from prophetverse.examples.repository.repositories import load_dataset

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


def add_source_mark(msg: str):
    plt.figtext(0.99, 0.01, msg, horizontalalignment="right", fontsize=10, style="italic")


def unemployment(change_point:bool=False, linear_trend:bool=False, piece_wise_linear_trend:bool=False, subset:Literal["first_peak", None]):
    df = load_dataset("brazilian_unemployment_ibge")

    if subset=='first_peak':
        df = df[3:-23:3].copy(deep=True)
    
    fig, ax = plt.subplots(figsize=(14, 8))
    plt.figure(figsize=(14, 8))

    sns.lineplot(data=df, x="Date", y="Unemployment Rate", marker="o", ax=ax)

    ax.set_title("Brazilian Unemployment Rate over Time")
    ax.set_xlabel("Date")
    ax.set_ylabel("Unemployment Rate")

    add_source_mark("Data source: IBGE")
    ax.set_xticklabels(ax.get_xticklabels(), rotation=45)
    if linear_trend:
        X = df.reset_index()[['index']]
        y = df['Unemployment Rate']

        
        model = LinearRegression()
        model.fit(X, y)

        df['linear_reg_y_hat'] = model.predict(X)
        sns.lineplot(data=df, x="Date", y="linear_reg_y_hat", ax=ax)




    if change_point:
        highlight_x = '2014-02-01'
        highlight_y = 6.9
        ax.scatter(highlight_x, highlight_y, color='red', s=200, label='Change Point', zorder=5, alpha=0.7)
        ax.legend()

    plt.tight_layout()

    return fig

