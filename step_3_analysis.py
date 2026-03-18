import pandas as pd
import numpy as np


# ─── LOAD DATA ────────────────────────────────────────────────────────────────

def load_data(current_csv: str = "current_prices.csv",
              historical_csv: str = "historical_prices.csv"):
    """Load the CSVs saved by step_2_fetch_data.py"""
    current    = pd.read_csv(current_csv, index_col="id")
    historical = pd.read_csv(historical_csv, index_col=0, parse_dates=True)
    return current, historical


# ─── ANALYSIS FUNCTIONS ───────────────────────────────────────────────────────

def descriptive_stats(historical: pd.DataFrame) -> pd.DataFrame:
    """
    Basic statistics for each coin over the period.
    Returns a DataFrame with mean, std, min, max, and range.
    """
    stats = historical.describe().T                        # shape: (coins × stats)
    stats["range"]     = stats["max"] - stats["min"]
    stats["cv_%"]      = (stats["std"] / stats["mean"] * 100).round(2)  # coeff. of variation
    print("=== Descriptive Statistics ===")
    print(stats[["mean", "std", "min", "max", "range", "cv_%"]].to_string())
    print()
    return stats


def daily_returns(historical: pd.DataFrame) -> pd.DataFrame:
    """
    Compute daily percentage returns:  (price_today / price_yesterday) - 1
    First row will be NaN — that is expected and correct.
    """
    returns = historical.pct_change() * 100    # in %
    print("=== Daily Returns (last 5 rows, %) ===")
    print(returns.tail().round(3).to_string())
    print()
    return returns


def rolling_average(historical: pd.DataFrame, window: int = 7) -> pd.DataFrame:
    """
    Compute a rolling N-day moving average for each coin.
    Useful for spotting trends while smoothing out noise.
    """
    rolling = historical.rolling(window=window).mean()
    print(f"=== {window}-Day Rolling Average (last 5 rows) ===")
    print(rolling.tail().round(2).to_string())
    print()
    return rolling


def volatility(returns: pd.DataFrame) -> pd.Series:
    """
    Annualised volatility = std(daily_returns) × sqrt(365)
    Higher value → riskier coin.
    """
    vol = returns.std() * np.sqrt(365)
    vol.name = "annualised_volatility_%"
    print("=== Annualised Volatility (%) ===")
    print(vol.round(2).to_string())
    print()
    return vol


def correlation_matrix(returns: pd.DataFrame) -> pd.DataFrame:
    """
    Pearson correlation of daily returns between coins.
    Values close to 1 → coins move together.
    Values close to 0 → coins move independently.
    """
    corr = returns.corr()
    print("=== Correlation Matrix ===")
    print(corr.round(3).to_string())
    print()
    return corr


def market_dominance(current: pd.DataFrame) -> pd.Series:
    """
    Each coin's share of total market cap (among the selected coins).
    """
    total = current["market_cap"].sum()
    dominance = (current["market_cap"] / total * 100).round(2)
    dominance.name = "market_dominance_%"
    print("=== Market Dominance (%) ===")
    print(dominance.sort_values(ascending=False).to_string())
    print()
    return dominance


def cumulative_returns(returns: pd.DataFrame) -> pd.DataFrame:
    """
    Cumulative return from the start of the period.
    Shows how $1 invested on day 1 grows (or shrinks).
    """
    cum = (1 + returns / 100).cumprod() - 1
    cum *= 100   # back to %
    print("=== Cumulative Return over Period (%) ===")
    print(cum.tail(1).round(2).to_string())
    print()
    return cum


# ─── MAIN ────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    current, historical = load_data()

    stats   = descriptive_stats(historical)
    returns = daily_returns(historical)
    rolling = rolling_average(historical, window=7)
    vol     = volatility(returns)
    corr    = correlation_matrix(returns)
    dom     = market_dominance(current)
    cum_ret = cumulative_returns(returns)

    print("✓ All analyses complete — ready for visualisation (step 4).")
