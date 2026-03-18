"""
step_2_fetch_data.py
====================
Fetches cryptocurrency market data from the CoinGecko public API.
No API key required — completely free to use.

What we fetch:
  - Current prices, market cap, volume, 24h change
  - 30-day historical price data for each coin
"""

import requests
import pandas as pd
import json
import time
from datetime import datetime


# ─── CONFIG ──────────────────────────────────────────────────────────────────

COINS = ["bitcoin", "ethereum", "solana", "cardano", "ripple"]
VS_CURRENCY = "usd"
HISTORY_DAYS = 30

BASE_URL = "https://api.coingecko.com/api/v3"


# ─── FUNCTIONS ────────────────────────────────────────────────────────────────

def fetch_current_prices(coins: list, currency: str = "usd") -> pd.DataFrame:
    """
    Fetch current market data for a list of coins.
    Returns a DataFrame with: price, market_cap, volume, price_change_24h, etc.
    """
    url = f"{BASE_URL}/coins/markets"
    params = {
        "vs_currency": currency,
        "ids": ",".join(coins),
        "order": "market_cap_desc",
        "price_change_percentage": "24h,7d",
    }

    print(f"[1/2] Fetching current prices for: {coins} ...")
    response = requests.get(url, params=params, timeout=10)
    response.raise_for_status()

    data = response.json()

    df = pd.DataFrame(data)[[
        "id", "symbol", "name",
        "current_price", "market_cap", "total_volume",
        "price_change_percentage_24h",
        "price_change_percentage_7d_in_currency",
        "high_24h", "low_24h",
        "circulating_supply",
    ]]
    df.columns = [
        "id", "symbol", "name",
        "price_usd", "market_cap", "volume_24h",
        "change_24h_pct", "change_7d_pct",
        "high_24h", "low_24h",
        "circulating_supply",
    ]
    df.set_index("id", inplace=True)

    print(f"    ✓ Got data for {len(df)} coins.\n")
    return df


def fetch_historical_prices(coins: list, days: int = 30,
                             currency: str = "usd") -> pd.DataFrame:
    """
    Fetch daily closing prices for the past `days` days for each coin.
    Returns a wide DataFrame: index=date, columns=coin names.
    """
    print(f"[2/2] Fetching {days}-day historical prices ...")
    all_prices = {}

    for coin in coins:
        url = f"{BASE_URL}/coins/{coin}/market_chart"
        params = {"vs_currency": currency, "days": days, "interval": "daily"}

        resp = requests.get(url, params=params, timeout=10)
        resp.raise_for_status()

        prices = resp.json()["prices"]          # [[timestamp_ms, price], ...]
        dates  = [datetime.fromtimestamp(p[0] / 1000).date() for p in prices]
        values = [p[1] for p in prices]

        all_prices[coin] = pd.Series(values, index=dates)
        print(f"    ✓ {coin}: {len(values)} days fetched.")
        time.sleep(0.5)                          # be polite to the free API

    df = pd.DataFrame(all_prices)
    df.index = pd.to_datetime(df.index)
    df.sort_index(inplace=True)
    print()
    return df


# ─── MAIN ────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    prices_df    = fetch_current_prices(COINS)
    historical   = fetch_historical_prices(COINS, days=HISTORY_DAYS)

    print("=== Current Prices ===")
    print(prices_df[["name", "price_usd", "change_24h_pct"]].to_string())
    print("\n=== Historical Price Table (last 5 rows) ===")
    print(historical.tail())

    # Save to CSV so the next steps can load without re-fetching
    prices_df.to_csv("current_prices.csv")
    historical.to_csv("historical_prices.csv")
    print("\n✓ Data saved to current_prices.csv and historical_prices.csv")
