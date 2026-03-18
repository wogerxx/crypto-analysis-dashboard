"""
step_4_visualize.py
===================
Creates 6 publication-quality Matplotlib charts:

  1. Price history (line chart)
  2. Daily returns (bar chart)
  3. 7-day rolling average overlay
  4. Correlation heatmap
  5. Market dominance (pie chart)
  6. Volatility comparison (horizontal bar)
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.ticker import FuncFormatter
import seaborn as sns


# ─── STYLE SETUP ──────────────────────────────────────────────────────────────

PALETTE = {
    "bitcoin":  "#F7931A",
    "ethereum": "#627EEA",
    "solana":   "#9945FF",
    "cardano":  "#0D47A1",
    "ripple":   "#00AAE4",
}

def style_setup():
    plt.style.use("dark_background")
    plt.rcParams.update({
        "font.family":      "DejaVu Sans",
        "axes.facecolor":   "#0d1117",
        "figure.facecolor": "#0d1117",
        "axes.edgecolor":   "#30363d",
        "grid.color":       "#21262d",
        "grid.linestyle":   "--",
        "grid.alpha":       0.6,
        "axes.labelcolor":  "#c9d1d9",
        "xtick.color":      "#8b949e",
        "ytick.color":      "#8b949e",
        "text.color":       "#c9d1d9",
    })

def usd_formatter(x, _):
    if x >= 1_000:
        return f"${x:,.0f}"
    return f"${x:.4f}"


# ─── CHART 1: Price History ───────────────────────────────────────────────────

def plot_price_history(historical: pd.DataFrame, ax: plt.Axes):
    """Line chart showing raw price for each coin over the full period."""
    ax.set_title("📈  30-Day Price History", fontsize=13, fontweight="bold",
                 color="#f0f6fc", pad=10)

    for coin in historical.columns:
        color = PALETTE.get(coin, "#ffffff")
        ax.plot(historical.index, historical[coin],
                label=coin.capitalize(), color=color,
                linewidth=2, alpha=0.9)
        # shade area under line
        ax.fill_between(historical.index, historical[coin],
                        alpha=0.07, color=color)

    ax.yaxis.set_major_formatter(FuncFormatter(usd_formatter))
    ax.legend(fontsize=8, loc="upper left",
              facecolor="#161b22", edgecolor="#30363d")
    ax.set_xlabel("Date", fontsize=9)
    ax.set_ylabel("Price (USD)", fontsize=9)
    ax.grid(True)
    plt.setp(ax.xaxis.get_majorticklabels(), rotation=30, ha="right", fontsize=7)


# ─── CHART 2: Daily Returns ───────────────────────────────────────────────────

def plot_daily_returns(returns: pd.DataFrame, ax: plt.Axes):
    """Bar chart of average daily return per coin."""
    ax.set_title("📊  Avg Daily Return (%)", fontsize=13, fontweight="bold",
                 color="#f0f6fc", pad=10)

    mean_returns = returns.mean().sort_values()
    colors = [PALETTE.get(c, "#8b949e") for c in mean_returns.index]
    bars = ax.barh(mean_returns.index, mean_returns.values,
                   color=colors, edgecolor="none", height=0.5)

    # label each bar
    for bar, val in zip(bars, mean_returns.values):
        ax.text(val + (0.002 if val >= 0 else -0.002),
                bar.get_y() + bar.get_height() / 2,
                f"{val:+.3f}%", va="center", ha="left" if val >= 0 else "right",
                fontsize=8, color="#c9d1d9")

    ax.axvline(0, color="#f0f6fc", linewidth=0.8, alpha=0.5)
    ax.set_xlabel("Average Daily Return (%)", fontsize=9)
    ax.grid(True, axis="x")
    ax.set_yticklabels([c.capitalize() for c in mean_returns.index], fontsize=9)


# ─── CHART 3: Rolling Average Overlay ────────────────────────────────────────

def plot_rolling_average(historical: pd.DataFrame, window: int = 7, ax: plt.Axes = None):
    """
    Pick Bitcoin (the most liquid/recognisable coin) and overlay
    actual price with 7-day rolling average to demonstrate trend smoothing.
    """
    coin = "bitcoin"
    rolling = historical[coin].rolling(window=window).mean()

    ax.set_title(f"📉  Bitcoin: Price vs {window}-Day MA",
                 fontsize=13, fontweight="bold", color="#f0f6fc", pad=10)
    ax.plot(historical.index, historical[coin],
            color="#F7931A", linewidth=1.5, alpha=0.5, label="Daily Price")
    ax.plot(historical.index, rolling,
            color="#ffffff", linewidth=2.2, label=f"{window}-Day MA")
    ax.fill_between(historical.index, historical[coin], rolling,
                    where=historical[coin] >= rolling,
                    alpha=0.15, color="#00d46a", label="Price > MA")
    ax.fill_between(historical.index, historical[coin], rolling,
                    where=historical[coin] < rolling,
                    alpha=0.15, color="#ff6b6b", label="Price < MA")

    ax.yaxis.set_major_formatter(FuncFormatter(usd_formatter))
    ax.legend(fontsize=8, facecolor="#161b22", edgecolor="#30363d")
    ax.set_xlabel("Date", fontsize=9)
    ax.set_ylabel("Price (USD)", fontsize=9)
    ax.grid(True)
    plt.setp(ax.xaxis.get_majorticklabels(), rotation=30, ha="right", fontsize=7)


# ─── CHART 4: Correlation Heatmap ────────────────────────────────────────────

def plot_correlation(returns: pd.DataFrame, ax: plt.Axes):
    """Seaborn heatmap of return correlations between coins."""
    corr = returns.corr()
    ax.set_title("🔗  Return Correlation Matrix", fontsize=13,
                 fontweight="bold", color="#f0f6fc", pad=10)

    mask = np.triu(np.ones_like(corr, dtype=bool), k=1)   # hide upper triangle
    sns.heatmap(
        corr, ax=ax, mask=mask,
        cmap="coolwarm", vmin=-1, vmax=1,
        annot=True, fmt=".2f", annot_kws={"size": 10, "color": "white"},
        linewidths=0.5, linecolor="#30363d",
        cbar_kws={"shrink": 0.8},
    )
    # fix tick labels
    labels = [c.capitalize() for c in corr.columns]
    ax.set_xticklabels(labels, rotation=30, ha="right", fontsize=8)
    ax.set_yticklabels(labels, rotation=0, fontsize=8)


# ─── CHART 5: Market Dominance ────────────────────────────────────────────────

def plot_market_dominance(current: pd.DataFrame, ax: plt.Axes):
    """Pie chart showing each coin's share of total market cap."""
    ax.set_title("🥧  Market Cap Dominance", fontsize=13,
                 fontweight="bold", color="#f0f6fc", pad=10)

    names  = [current.loc[c, "name"] for c in current.index]
    caps   = current["market_cap"].values
    colors = [PALETTE.get(c, "#8b949e") for c in current.index]

    wedges, texts, autotexts = ax.pie(
        caps, labels=names, colors=colors,
        autopct="%1.1f%%", startangle=140,
        pctdistance=0.75,
        wedgeprops=dict(edgecolor="#0d1117", linewidth=1.5),
    )
    for t in texts:
        t.set_fontsize(8)
        t.set_color("#c9d1d9")
    for at in autotexts:
        at.set_fontsize(8)
        at.set_color("white")
        at.set_fontweight("bold")


# ─── CHART 6: Volatility Comparison ──────────────────────────────────────────

def plot_volatility(returns: pd.DataFrame, ax: plt.Axes):
    """Horizontal bar chart of annualised volatility per coin."""
    vol = (returns.std() * np.sqrt(365)).sort_values()
    ax.set_title("⚡  Annualised Volatility (%)", fontsize=13,
                 fontweight="bold", color="#f0f6fc", pad=10)

    colors = [PALETTE.get(c, "#8b949e") for c in vol.index]
    bars = ax.barh(vol.index, vol.values, color=colors, edgecolor="none", height=0.5)

    for bar, val in zip(bars, vol.values):
        ax.text(val + 0.3, bar.get_y() + bar.get_height() / 2,
                f"{val:.1f}%", va="center", fontsize=9, color="#c9d1d9")

    ax.set_xlabel("Annualised Volatility (%)", fontsize=9)
    ax.set_yticklabels([c.capitalize() for c in vol.index], fontsize=9)
    ax.grid(True, axis="x")


# ─── ASSEMBLE DASHBOARD ───────────────────────────────────────────────────────

def build_dashboard(historical: pd.DataFrame, current: pd.DataFrame,
                    save_path: str = "crypto_dashboard.png"):
    style_setup()
    returns = historical.pct_change() * 100

    fig = plt.figure(figsize=(18, 14))
    fig.suptitle("🪙  Cryptocurrency Market Analysis Dashboard",
                 fontsize=18, fontweight="bold", color="#f0f6fc", y=0.98)

    gs = gridspec.GridSpec(3, 3, figure=fig, hspace=0.45, wspace=0.35)

    ax1 = fig.add_subplot(gs[0, :2])     # wide — price history
    ax2 = fig.add_subplot(gs[0, 2])      # daily returns
    ax3 = fig.add_subplot(gs[1, :2])     # rolling average
    ax4 = fig.add_subplot(gs[1, 2])      # correlation heatmap
    ax5 = fig.add_subplot(gs[2, 0])      # market dominance
    ax6 = fig.add_subplot(gs[2, 1:])     # volatility

    plot_price_history(historical, ax1)
    plot_daily_returns(returns, ax2)
    plot_rolling_average(historical, window=7, ax=ax3)
    plot_correlation(returns, ax4)
    plot_market_dominance(current, ax5)
    plot_volatility(returns, ax6)

    plt.savefig(save_path, dpi=150, bbox_inches="tight",
                facecolor="#0d1117")
    print(f"✓ Dashboard saved to: {save_path}")
    return fig


# ─── MAIN ────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    from step_3_analysis import load_data
    current, historical = load_data()
    build_dashboard(historical, current)
