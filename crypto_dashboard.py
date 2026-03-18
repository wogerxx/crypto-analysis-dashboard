import requests
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.ticker import FuncFormatter
import seaborn as sns
from datetime import datetime, timedelta
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.metrics import r2_score
import time
import random
 
 
# ══════════════════════════════════════════════════════════════════════════════
#  SECTION 1 — CONFIGURATION
# ══════════════════════════════════════════════════════════════════════════════
 
COINS         = ["bitcoin", "ethereum", "solana", "cardano", "ripple"]
VS_CURRENCY   = "usd"
HISTORY_DAYS  = 30
BASE_URL      = "https://api.coingecko.com/api/v3"
 
COIN_COLORS = {
    "bitcoin":  "#F7931A",
    "ethereum": "#627EEA",
    "solana":   "#9945FF",
    "cardano":  "#0D47A1",
    "ripple":   "#00AAE4",
}
 
# Realistic starting prices (used for simulation fallback)
START_PRICES = {
    "bitcoin":  65000,
    "ethereum": 3300,
    "solana":   160,
    "cardano":  0.48,
    "ripple":   0.62,
}
 
MARKET_CAPS = {
    "bitcoin":  1_280_000_000_000,
    "ethereum":   390_000_000_000,
    "solana":      73_000_000_000,
    "cardano":     16_800_000_000,
    "ripple":      34_000_000_000,
}
 
 
# ══════════════════════════════════════════════════════════════════════════════
#  SECTION 2 — DATA FETCHING
# ══════════════════════════════════════════════════════════════════════════════
 
def fetch_current_prices() -> pd.DataFrame:
    """
    Attempt to fetch live data from CoinGecko.
    Falls back to realistic simulated data if the network is unavailable.
    """
    try:
        print("  → Fetching current prices from CoinGecko API ...")
        url = f"{BASE_URL}/coins/markets"
        params = {
            "vs_currency": VS_CURRENCY,
            "ids": ",".join(COINS),
            "order": "market_cap_desc",
            "price_change_percentage": "24h,7d",
        }
        resp = requests.get(url, params=params, timeout=8)
        resp.raise_for_status()
        data = resp.json()
        df = pd.DataFrame(data)[[
            "id", "symbol", "name",
            "current_price", "market_cap", "total_volume",
            "price_change_percentage_24h",
            "price_change_percentage_7d_in_currency",
            "high_24h", "low_24h",
        ]]
        df.columns = [
            "id", "symbol", "name",
            "price_usd", "market_cap", "volume_24h",
            "change_24h_pct", "change_7d_pct",
            "high_24h", "low_24h",
        ]
        df.set_index("id", inplace=True)
        print("  ✓ Live data fetched successfully.\n")
        return df
 
    except Exception as e:
        print(f"  ⚠ API unavailable ({e}). Using simulated data.\n")
        return _simulate_current_prices()
 
 
def fetch_historical_prices() -> pd.DataFrame:
    """
    Attempt to fetch 30-day price history from CoinGecko.
    Falls back to realistic simulated data if unavailable.
    """
    try:
        print("  → Fetching historical prices ...")
        all_prices = {}
        for coin in COINS:
            url = f"{BASE_URL}/coins/{coin}/market_chart"
            params = {"vs_currency": VS_CURRENCY, "days": HISTORY_DAYS,
                      "interval": "daily"}
            resp = requests.get(url, params=params, timeout=8)
            resp.raise_for_status()
            prices = resp.json()["prices"]
            dates  = [datetime.fromtimestamp(p[0] / 1000).date() for p in prices]
            values = [p[1] for p in prices]
            all_prices[coin] = pd.Series(values, index=pd.to_datetime(dates))
            print(f"    ✓ {coin}: {len(values)} data points")
            time.sleep(0.5)
 
        df = pd.DataFrame(all_prices).sort_index()
        print("  ✓ Historical data fetched.\n")
        return df
 
    except Exception:
        print("  ⚠ API unavailable. Using simulated historical data.\n")
        return _simulate_historical_prices()
 
 
# ══════════════════════════════════════════════════════════════════════════════
#  SECTION 3 — DATA SIMULATION (fallback)
# ══════════════════════════════════════════════════════════════════════════════
 
def _simulate_historical_prices() -> pd.DataFrame:
    """
    Generate realistic crypto price history using Geometric Brownian Motion —
    the same model used in quantitative finance.
    """
    np.random.seed(42)
    dates = pd.date_range(end=datetime.today(), periods=HISTORY_DAYS + 1, freq="D")
 
    # Coin-specific volatility and drift
    params = {
        "bitcoin":  (0.0008, 0.028),
        "ethereum": (0.0006, 0.035),
        "solana":   (0.0010, 0.055),
        "cardano":  (0.0005, 0.045),
        "ripple":   (0.0004, 0.042),
    }
 
    all_prices = {}
    for coin in COINS:
        mu, sigma = params[coin]
        S0 = START_PRICES[coin]
        # GBM: S_t = S_{t-1} * exp((μ - σ²/2) + σ·Z)
        shocks = np.exp(
            (mu - 0.5 * sigma ** 2) + sigma * np.random.randn(HISTORY_DAYS)
        )
        prices = [S0]
        for shock in shocks:
            prices.append(prices[-1] * shock)
        all_prices[coin] = prices
 
    return pd.DataFrame(all_prices, index=dates)
 
 
def _simulate_current_prices() -> pd.DataFrame:
    """Build a current-prices DataFrame from the last simulated day."""
    hist = _simulate_historical_prices()
    rows = []
    for coin in COINS:
        price = hist[coin].iloc[-1]
        prev  = hist[coin].iloc[-2]
        prev7 = hist[coin].iloc[-8]
        rows.append({
            "id":           coin,
            "symbol":       coin[:3].upper(),
            "name":         coin.capitalize(),
            "price_usd":    round(price, 6),
            "market_cap":   MARKET_CAPS[coin],
            "volume_24h":   MARKET_CAPS[coin] * 0.035,
            "change_24h_pct": round((price / prev - 1) * 100, 2),
            "change_7d_pct":  round((price / prev7 - 1) * 100, 2),
            "high_24h":     round(price * 1.02, 6),
            "low_24h":      round(price * 0.98, 6),
        })
    df = pd.DataFrame(rows).set_index("id")
    return df
 
 
# ══════════════════════════════════════════════════════════════════════════════
#  SECTION 4 — PANDAS ANALYSIS
# ══════════════════════════════════════════════════════════════════════════════
 
def run_analysis(historical: pd.DataFrame, current: pd.DataFrame) -> dict:
    """
    Run all statistical analyses and return results as a dictionary.
    """
    print("─" * 55)
    print("  SECTION 4 — PANDAS ANALYSIS")
    print("─" * 55)
 
    # 4a. Descriptive statistics
    stats = historical.describe().T
    stats["range"] = stats["max"] - stats["min"]
    stats["cv_%"]  = (stats["std"] / stats["mean"] * 100).round(2)
    print("\n[A] Descriptive Statistics:")
    print(stats[["mean", "std", "min", "max", "cv_%"]].round(2).to_string())
 
    # 4b. Daily percentage returns
    returns = historical.pct_change() * 100
    print("\n[B] Mean Daily Returns (%):")
    print(returns.mean().round(4).to_string())
 
    # 4c. 7-day rolling average
    rolling_7d = historical.rolling(window=7).mean()
    print("\n[C] 7-Day Rolling Average (latest):")
    print(rolling_7d.iloc[-1].round(4).to_string())
 
    # 4d. Annualised volatility
    vol = returns.std() * np.sqrt(365)
    print("\n[D] Annualised Volatility (%):")
    print(vol.round(2).to_string())
 
    # 4e. Correlation matrix
    corr = returns.corr()
    print("\n[E] Correlation Matrix:")
    print(corr.round(3).to_string())
 
    # 4f. Cumulative return
    cum_ret = ((1 + returns / 100).cumprod() - 1) * 100
    print("\n[F] Total Cumulative Return over Period (%):")
    print(cum_ret.iloc[-1].round(2).to_string())
 
    # 4g. Market dominance
    total_cap = current["market_cap"].sum()
    dominance  = (current["market_cap"] / total_cap * 100).round(2)
    print("\n[G] Market Dominance (%):")
    print(dominance.sort_values(ascending=False).to_string())
 
    print()
    return {
        "stats":      stats,
        "returns":    returns,
        "rolling_7d": rolling_7d,
        "vol":        vol,
        "corr":       corr,
        "cum_ret":    cum_ret,
        "dominance":  dominance,
    }
 
 
# ══════════════════════════════════════════════════════════════════════════════
#  SECTION 5 — MATPLOTLIB VISUALISATIONS
# ══════════════════════════════════════════════════════════════════════════════
 
def _apply_style():
    plt.style.use("dark_background")
    plt.rcParams.update({
        "font.family":       "DejaVu Sans",
        "axes.facecolor":    "#0d1117",
        "figure.facecolor":  "#0d1117",
        "axes.edgecolor":    "#30363d",
        "grid.color":        "#21262d",
        "grid.linestyle":    "--",
        "grid.alpha":        0.6,
        "axes.labelcolor":   "#c9d1d9",
        "xtick.color":       "#8b949e",
        "ytick.color":       "#8b949e",
        "text.color":        "#c9d1d9",
    })
 
def _usd_fmt(x, _):
    return f"${x:,.0f}" if x >= 1 else f"${x:.4f}"
 
 
def _chart1_price_history(ax, historical):
    """Line chart: 30-day price for all coins."""
    ax.set_title("📈  30-Day Price History", fontsize=12, fontweight="bold",
                 color="#f0f6fc", pad=8)
    for coin in historical.columns:
        c = COIN_COLORS.get(coin, "#ffffff")
        ax.plot(historical.index, historical[coin],
                label=coin.capitalize(), color=c, linewidth=2, alpha=0.9)
        ax.fill_between(historical.index, historical[coin], alpha=0.06, color=c)
    ax.yaxis.set_major_formatter(FuncFormatter(_usd_fmt))
    ax.legend(fontsize=8, loc="upper left",
              facecolor="#161b22", edgecolor="#30363d")
    ax.set_ylabel("Price (USD)", fontsize=8)
    ax.grid(True)
    plt.setp(ax.get_xticklabels(), rotation=30, ha="right", fontsize=6)
 
 
def _chart2_daily_returns(ax, returns):
    """Horizontal bar: average daily return per coin."""
    ax.set_title("📊  Avg Daily Return (%)", fontsize=12, fontweight="bold",
                 color="#f0f6fc", pad=8)
    mean_r = returns.mean().sort_values()
    colors = [COIN_COLORS.get(c, "#8b949e") for c in mean_r.index]
    bars = ax.barh(mean_r.index, mean_r.values, color=colors, height=0.5)
    for bar, val in zip(bars, mean_r.values):
        ax.text(val + (0.001 if val >= 0 else -0.001),
                bar.get_y() + bar.get_height() / 2,
                f"{val:+.3f}%", va="center",
                ha="left" if val >= 0 else "right", fontsize=8)
    ax.axvline(0, color="#f0f6fc", linewidth=0.7, alpha=0.5)
    ax.set_yticklabels([c.capitalize() for c in mean_r.index], fontsize=8)
    ax.grid(True, axis="x")
 
 
def _chart3_rolling_ma(ax, historical):
    """Bitcoin price vs 7-day moving average."""
    coin    = "bitcoin"
    price   = historical[coin]
    ma7     = price.rolling(7).mean()
    ax.set_title("📉  Bitcoin: Price vs 7-Day MA", fontsize=12,
                 fontweight="bold", color="#f0f6fc", pad=8)
    ax.plot(price.index, price,  color="#F7931A", lw=1.5, alpha=0.55, label="Price")
    ax.plot(ma7.index,   ma7,    color="#ffffff", lw=2.2, label="7-Day MA")
    ax.fill_between(price.index, price, ma7,
                    where=price >= ma7, alpha=0.12, color="#00d46a")
    ax.fill_between(price.index, price, ma7,
                    where=price < ma7,  alpha=0.12, color="#ff6b6b")
    ax.yaxis.set_major_formatter(FuncFormatter(_usd_fmt))
    ax.legend(fontsize=8, facecolor="#161b22", edgecolor="#30363d")
    ax.set_ylabel("Price (USD)", fontsize=8)
    ax.grid(True)
    plt.setp(ax.get_xticklabels(), rotation=30, ha="right", fontsize=6)
 
 
def _chart4_heatmap(ax, corr):
    """Seaborn correlation heatmap (lower triangle)."""
    ax.set_title("🔗  Return Correlations", fontsize=12, fontweight="bold",
                 color="#f0f6fc", pad=8)
    mask = np.triu(np.ones_like(corr, dtype=bool), k=1)
    sns.heatmap(corr, ax=ax, mask=mask, cmap="coolwarm",
                vmin=-1, vmax=1, annot=True, fmt=".2f",
                annot_kws={"size": 9, "color": "white"},
                linewidths=0.5, linecolor="#30363d",
                cbar_kws={"shrink": 0.75})
    labels = [c.capitalize() for c in corr.columns]
    ax.set_xticklabels(labels, rotation=30, ha="right", fontsize=8)
    ax.set_yticklabels(labels, rotation=0, fontsize=8)
 
 
def _chart5_dominance(ax, current):
    """Pie chart of market-cap dominance."""
    ax.set_title("🥧  Market Cap Dominance", fontsize=12, fontweight="bold",
                 color="#f0f6fc", pad=8)
    names  = [current.loc[c, "name"] for c in current.index]
    caps   = current["market_cap"].values
    colors = [COIN_COLORS.get(c, "#8b949e") for c in current.index]
    wedges, texts, autotexts = ax.pie(
        caps, labels=names, colors=colors, autopct="%1.1f%%",
        startangle=140, pctdistance=0.78,
        wedgeprops=dict(edgecolor="#0d1117", linewidth=1.5),
    )
    for t in texts:      t.set_fontsize(8);  t.set_color("#c9d1d9")
    for at in autotexts: at.set_fontsize(8); at.set_color("white"); at.set_fontweight("bold")
 
 
def _chart6_volatility(ax, returns):
    """Horizontal bar chart of annualised volatility."""
    ax.set_title("⚡  Annualised Volatility (%)", fontsize=12, fontweight="bold",
                 color="#f0f6fc", pad=8)
    vol    = (returns.std() * np.sqrt(365)).sort_values()
    colors = [COIN_COLORS.get(c, "#8b949e") for c in vol.index]
    bars   = ax.barh(vol.index, vol.values, color=colors, height=0.5)
    for bar, val in zip(bars, vol.values):
        ax.text(val + 0.3, bar.get_y() + bar.get_height() / 2,
                f"{val:.1f}%", va="center", fontsize=9)
    ax.set_xlabel("Annualised Volatility (%)", fontsize=8)
    ax.set_yticklabels([c.capitalize() for c in vol.index], fontsize=9)
    ax.grid(True, axis="x")
 
 
# ══════════════════════════════════════════════════════════════════════════════
#  SECTION 5b — ML PRICE PREDICTION
# ══════════════════════════════════════════════════════════════════════════════
 
def predict_prices(historical: pd.DataFrame, forecast_days: int = 7) -> dict:
    """
    Polynomial Regression (degree=3) price forecast for each coin.
 
    Steps:
      1. Encode dates as integer day indices (X)
      2. Fit a degree-3 polynomial on historical prices (y)
      3. Predict the next `forecast_days` days
      4. Compute confidence interval using residual std
      5. Report R² score (how well the model fits history)
 
    Returns a dict per coin:
      {
        "future_dates":  [...],
        "predicted":     [...],
        "ci_upper":      [...],
        "ci_lower":      [...],
        "r2":            float,
        "last_price":    float,
        "pred_price":    float,   # price at end of forecast
        "pct_change":    float,
      }
    """
    print("\n─" * 28)
    print("  SECTION 5b — ML PRICE PREDICTION")
    print("─" * 28)
 
    results = {}
    degree  = 3   # cubic polynomial — captures curves without wild overfitting
 
    for coin in historical.columns:
        prices = historical[coin].dropna().values
        n      = len(prices)
 
        # X = day index (0, 1, 2, ..., n-1)
        X = np.arange(n).reshape(-1, 1)
        y = prices
 
        # Build polynomial features: [1, x, x², x³]
        poly    = PolynomialFeatures(degree=degree, include_bias=False)
        X_poly  = poly.fit_transform(X)
 
        # Fit Linear Regression on the polynomial features
        model   = LinearRegression()
        model.fit(X_poly, y)
 
        # R² on training data
        y_pred_train = model.predict(X_poly)
        r2           = r2_score(y, y_pred_train)
 
        # Residual standard deviation → confidence interval
        residuals    = y - y_pred_train
        residual_std = np.std(residuals)
 
        # Forecast X values: n, n+1, ..., n+forecast_days
        X_future      = np.arange(n, n + forecast_days + 1).reshape(-1, 1)
        X_future_poly = poly.transform(X_future)
        y_future      = model.predict(X_future_poly)
 
        # 95% confidence interval  ≈  ±1.96 × σ  (assumes normal residuals)
        ci = 1.96 * residual_std
        ci_upper = y_future + ci
        ci_lower = y_future - ci
 
        # Future dates
        last_date    = historical.index[-1]
        future_dates = pd.date_range(start=last_date, periods=forecast_days + 1, freq="D")
 
        pct_change   = (y_future[-1] / prices[-1] - 1) * 100
 
        results[coin] = {
            "future_dates": future_dates,
            "predicted":    y_future,
            "ci_upper":     ci_upper,
            "ci_lower":     ci_lower,
            "r2":           r2,
            "last_price":   prices[-1],
            "pred_price":   y_future[-1],
            "pct_change":   pct_change,
            "hist_prices":  prices,
            "hist_dates":   historical.index,
        }
 
        arrow  = "▲" if pct_change > 0 else "▼"
        print(f"  {coin:<12}  R²={r2:.3f}  |  "
              f"now=${prices[-1]:>10,.2f}  →  "
              f"7d=${y_future[-1]:>10,.2f}  {arrow}{abs(pct_change):.1f}%")
 
    print()
    return results
 
 
def _chart7_prediction(fig, historical: pd.DataFrame, predictions: dict):
    """
    Full-width prediction chart spanning the bottom of the dashboard.
    Shows history + 7-day forecast with confidence bands for all coins.
    """
    # Add a new row below the existing 3×3 grid
    ax = fig.add_axes([0.05, -0.32, 0.90, 0.28])   # [left, bottom, width, height]
    ax.set_facecolor("#0d1117")
 
    ax.set_title(
        "  7-Day Price Forecast   —   Polynomial Regression (degree 3)   |   "
        "Shaded area = 95% Confidence Interval",
        fontsize=11, fontweight="bold", color="#f0f6fc", pad=10, loc="left"
    )
 
    for coin, res in predictions.items():
        color        = COIN_COLORS.get(coin, "#ffffff")
        hist_prices  = res["hist_prices"]
        hist_dates   = res["hist_dates"]
        fut_dates    = res["future_dates"]
        predicted    = res["predicted"]
        ci_upper     = res["ci_upper"]
        ci_lower     = res["ci_lower"]
        r2           = res["r2"]
        pct          = res["pct_change"]
 
        # Normalise to % change from last price (so all coins fit on one axis)
        base         = hist_prices[-1]
        hist_norm    = (hist_prices / base - 1) * 100
        pred_norm    = (predicted   / base - 1) * 100
        upper_norm   = (ci_upper    / base - 1) * 100
        lower_norm   = (ci_lower    / base - 1) * 100
 
        # Historical line (faded)
        ax.plot(hist_dates, hist_norm,
                color=color, linewidth=1.4, alpha=0.45, linestyle="--")
 
        # Forecast line (solid, bright)
        ax.plot(fut_dates, pred_norm,
                color=color, linewidth=2.5, alpha=0.95,
                label=f"{coin.capitalize()}  R²={r2:.2f}  ({pct:+.1f}%)")
 
        # Confidence band
        ax.fill_between(fut_dates, lower_norm, upper_norm,
                        alpha=0.10, color=color)
 
        # Dot at forecast endpoint
        ax.scatter(fut_dates[-1], pred_norm[-1],
                   color=color, s=55, zorder=5, edgecolors="#0d1117", linewidths=1)
 
    # Vertical line separating history from forecast
    ax.axvline(historical.index[-1], color="#f0f6fc",
               linewidth=1.2, linestyle=":", alpha=0.6)
    ax.text(historical.index[-1], ax.get_ylim()[1] if ax.get_ylim()[1] != 1.0 else 5,
            "  TODAY", fontsize=8, color="#8b949e", va="top")
 
    ax.axhline(0, color="#f0f6fc", linewidth=0.7, alpha=0.3)
    ax.yaxis.set_major_formatter(FuncFormatter(lambda x, _: f"{x:+.1f}%"))
    ax.set_ylabel("Price Change from Today (%)", fontsize=9)
    ax.set_xlabel("Date", fontsize=9)
    ax.legend(fontsize=8.5, loc="upper left",
              facecolor="#161b22", edgecolor="#30363d", ncol=5)
    ax.grid(True, alpha=0.4)
    plt.setp(ax.get_xticklabels(), rotation=20, ha="right", fontsize=7)
 
    return ax
 
 
def build_dashboard(historical: pd.DataFrame, current: pd.DataFrame,
                    analysis: dict, predictions: dict,
                    save_path: str = "crypto_dashboard.png"):
    """Assemble 6 analysis charts + full-width ML prediction chart."""
    _apply_style()
    returns = analysis["returns"]
    corr    = analysis["corr"]
 
    fig = plt.figure(figsize=(20, 20))
    fig.patch.set_facecolor("#0d1117")
    fig.suptitle(
        "Cryptocurrency Market Analysis Dashboard",
        fontsize=20, fontweight="bold", color="#f0f6fc", y=0.99
    )
 
    # Top 3 rows: analysis charts — leave bottom 30% for prediction
    gs = gridspec.GridSpec(3, 3, figure=fig,
                           hspace=0.50, wspace=0.38,
                           top=0.96, bottom=0.36)
 
    ax1 = fig.add_subplot(gs[0, :2])
    ax2 = fig.add_subplot(gs[0, 2])
    ax3 = fig.add_subplot(gs[1, :2])
    ax4 = fig.add_subplot(gs[1, 2])
    ax5 = fig.add_subplot(gs[2, 0])
    ax6 = fig.add_subplot(gs[2, 1:])
 
    _chart1_price_history(ax1, historical)
    _chart2_daily_returns(ax2, returns)
    _chart3_rolling_ma(ax3, historical)
    _chart4_heatmap(ax4, corr)
    _chart5_dominance(ax5, current)
    _chart6_volatility(ax6, returns)
 
    # Prediction chart — full width at the bottom
    _chart7_prediction(fig, historical, predictions)
 
    fig.text(0.5, 0.01,
             f"Data: CoinGecko API  |  ML: Polynomial Regression degree-3 (sklearn)  |  "
             f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M')}",
             ha="center", fontsize=8, color="#484f58")
 
    plt.savefig(save_path, dpi=150, bbox_inches="tight", facecolor="#0d1117")
    print(f"\n  ✓ Dashboard saved → {save_path}")
    return fig
 
 
# ══════════════════════════════════════════════════════════════════════════════
#  SECTION 6 — MAIN
# ══════════════════════════════════════════════════════════════════════════════
 
def main():
    print("\n" + "═" * 55)
    print("   CRYPTO MARKET ANALYSIS DASHBOARD")
    print("═" * 55 + "\n")
 
    print("─" * 55)
    print("  SECTION 2 — DATA FETCHING")
    print("─" * 55)
    current    = fetch_current_prices()
    historical = fetch_historical_prices()
 
    analysis    = run_analysis(historical, current)
    predictions = predict_prices(historical, forecast_days=7)
 
    print("─" * 55)
    print("  SECTION 5 — BUILDING DASHBOARD")
    print("─" * 55)
    build_dashboard(historical, current, analysis, predictions)
 
    print("\n" + "═" * 55)
    print("   DONE!  Open crypto_dashboard.png to view.")
    print("═" * 55 + "\n")
 
 
if __name__ == "__main__":
    main()