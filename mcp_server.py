#!/usr/bin/env python3
"""
TradingView MCP Server
Exposes TradingView market data to Claude via the Model Context Protocol.
Run via: python mcp_server.py
"""

import os
import math
import logging
import json
import itertools
from datetime import datetime, timedelta
from typing import Optional, List

import requests

from mcp.server.fastmcp import FastMCP

logging.basicConfig(level=logging.WARNING)

# ---------------------------------------------------------------------------
# Optional dependency guards
# ---------------------------------------------------------------------------

try:
    import numpy as np
    import pandas as pd
    NUMPY_OK = True
except ImportError:
    NUMPY_OK = False

try:
    from tradingview_screener import Query, Column
    SCREENER_OK = True
except ImportError:
    SCREENER_OK = False

try:
    from tradingview_ta import TA_Handler, Interval
    TA_OK = True
except ImportError:
    TA_OK = False

try:
    from tradingview_historical_fetcher import TradingViewHistoricalFetcher
    HISTORICAL_OK = True
except Exception:
    HISTORICAL_OK = False

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

INTERVAL_MAP = {
    "1m":  Interval.INTERVAL_1_MINUTE   if TA_OK else None,
    "5m":  Interval.INTERVAL_5_MINUTES  if TA_OK else None,
    "15m": Interval.INTERVAL_15_MINUTES if TA_OK else None,
    "30m": Interval.INTERVAL_30_MINUTES if TA_OK else None,
    "1h":  Interval.INTERVAL_1_HOUR     if TA_OK else None,
    "2h":  Interval.INTERVAL_2_HOURS    if TA_OK else None,
    "4h":  Interval.INTERVAL_4_HOURS    if TA_OK else None,
    "1d":  Interval.INTERVAL_1_DAY      if TA_OK else None,
    "1w":  Interval.INTERVAL_1_WEEK     if TA_OK else None,
    "1M":  Interval.INTERVAL_1_MONTH    if TA_OK else None,
}

def _clean(value):
    """Convert NaN / inf to None so results are JSON-serialisable."""
    if value is None:
        return None
    try:
        if math.isnan(value) or math.isinf(value):
            return None
    except TypeError:
        pass
    return value

def _clean_dict(d: dict) -> dict:
    return {k: _clean(v) for k, v in d.items()}

# ---------------------------------------------------------------------------
# Scanner / REST API helper
# ---------------------------------------------------------------------------



_SCANNER_HEADERS = {
    "accept": "application/json",
    "content-type": "text/plain;charset=UTF-8",
    "origin": "https://www.tradingview.com",
    "referer": "https://www.tradingview.com/",
    "user-agent": "Mozilla/5.0 (compatible; TradingBot/1.0)",
    "dnt": "1",
}

def _scanner_post(url: str, payload: dict, timeout: int = 20) -> dict:
    resp = requests.post(url, headers=_SCANNER_HEADERS, json=payload, timeout=timeout)
    resp.raise_for_status()
    return resp.json()

def _scanner_get(url: str, params: dict | None = None, timeout: int = 20) -> dict:
    resp = requests.get(url, headers=_SCANNER_HEADERS, params=params, timeout=timeout)
    resp.raise_for_status()
    return resp.json()

# ---------------------------------------------------------------------------
# MCP server
# ---------------------------------------------------------------------------

mcp = FastMCP(
    "tradingview",
    instructions=(
        "Provides real-time and historical market data from TradingView. "
        "Use screen_stocks to scan markets in bulk, get_technical_analysis for "
        "indicators on a single symbol, get_historical_ohlcv for candlestick data, "
        "top_gainers/top_losers for movers, bollinger_scan for breakout setups, "
        "get_earnings_calendar and get_ipo_calendar for events, "
        "get_fibonacci_retracement, get_support_resistance, get_pivot_points "
        "for price levels, get_volume_profile for market structure, "
        "and calculate_correlation for pair analysis."
    ),
)

# ---------------------------------------------------------------------------
# Tool 1: Bulk screener
# ---------------------------------------------------------------------------

@mcp.tool()
def screen_stocks(
    exchanges: str = "NASDAQ,NYSE,AMEX",
    sector: Optional[str] = None,
    min_price: Optional[float] = None,
    max_price: Optional[float] = None,
    min_market_cap: Optional[float] = None,
    sort_by: str = "market_cap_basic",
    sort_desc: bool = True,
    limit: int = 50,
) -> list[dict]:
    """
    Screen stocks from TradingView with fundamental and price data.

    Args:
        exchanges: Comma-separated list of exchanges (e.g. "NASDAQ,NYSE,AMEX").
        sector: Optional sector filter (e.g. "Technology", "Healthcare").
        min_price: Minimum stock price filter.
        max_price: Maximum stock price filter.
        min_market_cap: Minimum market cap in USD (e.g. 1e9 for $1B).
        sort_by: Field to sort by. Common values: "market_cap_basic", "close",
                 "volume", "price_earnings_ttm", "Perf.Y", "relative_volume_10d_calc".
        sort_desc: True = descending (largest first).
        limit: Max number of results (1–500).

    Returns:
        List of dicts with symbol, name, exchange, price, market_cap, pe_ratio,
        sector, industry, volume, beta, performance_1y, and more.
    """
    if not SCREENER_OK:
        raise RuntimeError("tradingview-screener package not installed.")

    exchange_list = [e.strip().upper() for e in exchanges.split(",") if e.strip()]
    limit = max(1, min(limit, 500))

    q = (
        Query()
        .select(
            "name", "close", "volume", "market_cap_basic",
            "sector", "industry", "exchange", "type",
            "price_earnings_ttm", "price_book_fq",
            "debt_to_equity", "current_ratio",
            "gross_margin", "operating_margin", "net_margin",
            "return_on_equity", "return_on_assets",
            "dividends_yield",
            "beta_1_year", "Volatility.D",
            "Perf.W", "Perf.1M", "Perf.3M", "Perf.6M", "Perf.Y", "Perf.YTD",
            "price_52_week_high", "price_52_week_low",
            "relative_volume_10d_calc", "average_volume_10d_calc",
            "Recommend.All",
        )
        .where(
            Column("type").isin(["stock", "dr"]),
            Column("exchange").isin(exchange_list),
        )
        .limit(limit)
    )

    if sector:
        q = q.where(Column("sector").isin([sector]))
    if min_price is not None:
        q = q.where(Column("close") >= min_price)
    if max_price is not None:
        q = q.where(Column("close") <= max_price)
    if min_market_cap is not None:
        q = q.where(Column("market_cap_basic") >= min_market_cap)

    if sort_desc:
        q = q.order_by(sort_by, ascending=False)
    else:
        q = q.order_by(sort_by, ascending=True)

    _, df = q.get_scanner_data()

    if df.empty:
        return []

    # Split "EXCHANGE:SYMBOL" ticker column
    df[["exch", "sym"]] = df["ticker"].str.split(":", expand=True)

    def _rec_to_map(record: dict) -> dict:
        recommend_raw = record.get("Recommend.All")
        if recommend_raw is not None and recommend_raw == recommend_raw:
            if recommend_raw >= 0.5:
                rec = "Strong Buy"
            elif recommend_raw >= 0.1:
                rec = "Buy"
            elif recommend_raw >= -0.1:
                rec = "Neutral"
            elif recommend_raw >= -0.5:
                rec = "Sell"
            else:
                rec = "Strong Sell"
        else:
            rec = None

        div = record.get("dividends_yield")
        return _clean_dict({
            "symbol":          record.get("sym"),
            "name":            record.get("name"),
            "exchange":        record.get("exch"),
            "sector":          record.get("sector"),
            "industry":        record.get("industry"),
            "price":           record.get("close"),
            "volume":          record.get("volume"),
            "avg_volume_10d":  record.get("average_volume_10d_calc"),
            "rel_volume":      record.get("relative_volume_10d_calc"),
            "market_cap":      record.get("market_cap_basic"),
            "pe_ratio":        record.get("price_earnings_ttm"),
            "pb_ratio":        record.get("price_book_fq"),
            "debt_to_equity":  record.get("debt_to_equity"),
            "current_ratio":   record.get("current_ratio"),
            "gross_margin_pct":    record.get("gross_margin"),
            "operating_margin_pct":record.get("operating_margin"),
            "net_margin_pct":      record.get("net_margin"),
            "roe_pct":         record.get("return_on_equity"),
            "roa_pct":         record.get("return_on_assets"),
            "dividend_yield_pct": div * 100 if div and div == div else None,
            "beta":            record.get("beta_1_year"),
            "volatility_d":    record.get("Volatility.D"),
            "perf_1w_pct":     record.get("Perf.W"),
            "perf_1m_pct":     record.get("Perf.1M"),
            "perf_3m_pct":     record.get("Perf.3M"),
            "perf_6m_pct":     record.get("Perf.6M"),
            "perf_ytd_pct":    record.get("Perf.YTD"),
            "perf_1y_pct":     record.get("Perf.Y"),
            "high_52w":        record.get("price_52_week_high"),
            "low_52w":         record.get("price_52_week_low"),
            "analyst_rating":  rec,
        })

    records = df.to_dict("records")
    return [_rec_to_map(r) for r in records]


# ---------------------------------------------------------------------------
# Tool 2: Technical analysis for a single symbol
# ---------------------------------------------------------------------------

@mcp.tool()
def get_technical_analysis(
    symbol: str,
    exchange: str = "NASDAQ",
    screener: str = "america",
    interval: str = "1d",
) -> dict:
    """
    Get technical analysis and indicators for a single stock from TradingView.

    Args:
        symbol: Ticker symbol (e.g. "AAPL", "MSFT").
        exchange: Exchange the symbol is listed on (e.g. "NASDAQ", "NYSE",
                  "LSE", "XETR", "EURONEXT", "TSX").
        screener: TradingView screener region. Common values: "america",
                  "europe", "uk", "canada", "australia", "india".
        interval: Candle interval. One of: "1m","5m","15m","30m","1h","2h",
                  "4h","1d","1w","1M". Default "1d".

    Returns:
        Dict with price, RSI, MACD, Bollinger Bands, SMAs, EMAs, ADX, and
        an overall BUY / SELL / NEUTRAL recommendation with signal counts.
    """
    if not TA_OK:
        raise RuntimeError("tradingview-ta package not installed.")

    iv = INTERVAL_MAP.get(interval)
    if iv is None:
        raise ValueError(
            f"Unknown interval '{interval}'. "
            f"Use one of: {list(INTERVAL_MAP.keys())}"
        )

    handler = TA_Handler(
        symbol=symbol.upper(),
        exchange=exchange.upper(),
        screener=screener.lower(),
        interval=iv,
        timeout=15,
    )

    analysis = handler.get_analysis()
    ind = analysis.indicators
    summ = analysis.summary
    osc = analysis.oscillators
    ma = analysis.moving_averages

    return _clean_dict({
        "symbol":       symbol.upper(),
        "exchange":     exchange.upper(),
        "interval":     interval,
        # Price
        "price":        ind.get("close"),
        "open":         ind.get("open"),
        "high":         ind.get("high"),
        "low":          ind.get("low"),
        "volume":       ind.get("volume"),
        "change_pct":   ind.get("change"),
        # Overall recommendation
        "recommendation": summ.get("RECOMMENDATION"),
        "buy_signals":    summ.get("BUY"),
        "sell_signals":   summ.get("SELL"),
        "neutral_signals":summ.get("NEUTRAL"),
        # Oscillators
        "rsi":              ind.get("RSI"),
        "rsi_14":           ind.get("RSI[1]"),
        "stoch_k":          ind.get("Stoch.K"),
        "stoch_d":          ind.get("Stoch.D"),
        "cci_20":           ind.get("CCI20"),
        "adx":              ind.get("ADX"),
        "ao":               ind.get("AO"),
        "momentum_10":      ind.get("Mom"),
        "macd_macd":        ind.get("MACD.macd"),
        "macd_signal":      ind.get("MACD.signal"),
        "williams_r":       ind.get("W.R"),
        "bull_bear_power":  ind.get("BBPower"),
        "ultimate_osc":     ind.get("UO"),
        # Moving averages
        "ema_10":  ind.get("EMA10"),
        "ema_20":  ind.get("EMA20"),
        "ema_30":  ind.get("EMA30"),
        "ema_50":  ind.get("EMA50"),
        "ema_100": ind.get("EMA100"),
        "ema_200": ind.get("EMA200"),
        "sma_10":  ind.get("SMA10"),
        "sma_20":  ind.get("SMA20"),
        "sma_30":  ind.get("SMA30"),
        "sma_50":  ind.get("SMA50"),
        "sma_100": ind.get("SMA100"),
        "sma_200": ind.get("SMA200"),
        # Bollinger Bands
        "bb_upper":  ind.get("BB.upper"),
        "bb_lower":  ind.get("BB.lower"),
        # 52-week
        "high_52w": ind.get("High.All"),
        "low_52w":  ind.get("Low.All"),
        # MA summary
        "ma_buy_signals":     ma.get("BUY"),
        "ma_sell_signals":    ma.get("SELL"),
        "ma_neutral_signals": ma.get("NEUTRAL"),
        # Oscillator summary
        "osc_buy_signals":     osc.get("BUY"),
        "osc_sell_signals":    osc.get("SELL"),
        "osc_neutral_signals": osc.get("NEUTRAL"),
    })


# ---------------------------------------------------------------------------
# Tool 3: Historical OHLCV via TradingView WebSocket
# ---------------------------------------------------------------------------

@mcp.tool()
def get_historical_ohlcv(
    symbol: str,
    exchange: str = "NASDAQ",
    freq: str = "1D",
    bars: int = 365,
) -> list[dict]:
    """
    Fetch historical OHLCV (candlestick) data from TradingView via WebSocket.
    No API key required. Does not hit Yahoo Finance.

    Args:
        symbol: Ticker symbol (e.g. "AAPL").
        exchange: Exchange (e.g. "NASDAQ", "NYSE", "LSE", "XETR").
        freq: Bar frequency. Common values: "1D" (daily), "1W" (weekly),
              "1M" (monthly), "240" (4-hour), "60" (1-hour).
        bars: Number of bars to return (max ~5000).

    Returns:
        List of dicts with date (ISO string), open, high, low, close, volume.
    """
    if not HISTORICAL_OK:
        raise RuntimeError(
            "TradingViewHistoricalFetcher unavailable — "
            "check websocket-client is installed."
        )

    fetcher = TradingViewHistoricalFetcher()
    df = fetcher.get_historical_data(
        symbol=symbol.upper(),
        exchange=exchange.upper(),
        freq=freq,
        number_of_values=bars,
    )

    if df is None or df.empty:
        return []

    rows = []
    for _, row in df.iterrows():
        rows.append({
            "date":   row["date"].isoformat() if hasattr(row["date"], "isoformat") else str(row["date"]),
            "open":   _clean(row.get("open")),
            "high":   _clean(row.get("high")),
            "low":    _clean(row.get("low")),
            "close":  _clean(row.get("close")),
            "volume": _clean(row.get("volume")),
        })

    return rows


# ---------------------------------------------------------------------------
# Tool 4: Symbol search
# ---------------------------------------------------------------------------

@mcp.tool()
def search_symbol(
    query: str,
    exchanges: str = "NASDAQ,NYSE,AMEX",
    limit: int = 20,
) -> list[dict]:
    """
    Search TradingView for stock symbols matching a company name or ticker.

    Args:
        query: Ticker or partial company name to search for.
        exchanges: Comma-separated exchanges to search within.
        limit: Max results (1–100).

    Returns:
        List of dicts with symbol, name, exchange, sector, price, market_cap.
    """
    if not SCREENER_OK:
        raise RuntimeError("tradingview-screener package not installed.")

    exchange_list = [e.strip().upper() for e in exchanges.split(",") if e.strip()]
    limit = max(1, min(limit, 100))
    q = upper = query.strip().upper()

    _, df = (
        Query()
        .select("name", "close", "market_cap_basic", "sector", "exchange", "type")
        .where(
            Column("type").isin(["stock", "dr"]),
            Column("exchange").isin(exchange_list),
        )
        .limit(500)
        .get_scanner_data()
    )

    if df.empty:
        return []

    df[["exch", "sym"]] = df["ticker"].str.split(":", expand=True)

    # Filter client-side by symbol prefix or name substring
    mask = df["sym"].str.upper().str.startswith(upper) | df["name"].str.upper().str.contains(upper, na=False)
    matched = df[mask].head(limit)

    results = []
    for _, row in matched.iterrows():
        results.append(_clean_dict({
            "symbol":     row.get("sym"),
            "name":       row.get("name"),
            "exchange":   row.get("exch"),
            "sector":     row.get("sector"),
            "price":      row.get("close"),
            "market_cap": row.get("market_cap_basic"),
        }))
    return results


# ---------------------------------------------------------------------------
# Tool 6: Top gainers
# ---------------------------------------------------------------------------

@mcp.tool()
def top_gainers(
    exchanges: str = "NASDAQ,NYSE,AMEX",
    market_type: str = "stock",
    min_price: Optional[float] = None,
    min_volume: Optional[float] = None,
    limit: int = 20,
) -> list[dict]:
    """
    Return the top gaining assets for the current session sorted by daily % change.

    Args:
        exchanges: Comma-separated exchanges (e.g. "NASDAQ,NYSE,AMEX").
                   For crypto use "BINANCE,COINBASE"; for forex use "FX,FX_IDC".
        market_type: Asset type: "stock", "crypto", "forex", "fund". Default "stock".
        min_price: Minimum price filter.
        min_volume: Minimum volume filter.
        limit: Max results (1–200).

    Returns:
        List of dicts sorted by daily change descending.
    """
    if not SCREENER_OK:
        raise RuntimeError("tradingview-screener package not installed.")

    exchange_list = [e.strip().upper() for e in exchanges.split(",") if e.strip()]
    limit = max(1, min(limit, 200))

    type_map = {"stock": ["stock", "dr"], "crypto": ["crypto"], "forex": ["forex"], "fund": ["fund", "etf"]}
    types = type_map.get(market_type.lower(), ["stock", "dr"])

    q = (
        Query()
        .select("name", "close", "change", "change_abs", "volume", "market_cap_basic",
                "sector", "exchange", "type", "Perf.D", "Perf.W", "Recommend.All")
        .where(
            Column("type").isin(types),
            Column("exchange").isin(exchange_list),
            Column("change") > 0,
        )
        .order_by("change", ascending=False)
        .limit(limit)
    )
    if min_price is not None:
        q = q.where(Column("close") >= min_price)
    if min_volume is not None:
        q = q.where(Column("volume") >= min_volume)

    _, df = q.get_scanner_data()
    if df.empty:
        return []

    df[["exch", "sym"]] = df["ticker"].str.split(":", expand=True)
    results = []
    for _, row in df.iterrows():
        results.append(_clean_dict({
            "symbol":      row.get("sym"),
            "name":        row.get("name"),
            "exchange":    row.get("exch"),
            "sector":      row.get("sector"),
            "price":       row.get("close"),
            "change_pct":  row.get("change"),
            "change_abs":  row.get("change_abs"),
            "volume":      row.get("volume"),
            "market_cap":  row.get("market_cap_basic"),
            "perf_1d_pct": row.get("Perf.D"),
            "perf_1w_pct": row.get("Perf.W"),
            "analyst_rating": row.get("Recommend.All"),
        }))
    return results


# ---------------------------------------------------------------------------
# Tool 7: Top losers
# ---------------------------------------------------------------------------

@mcp.tool()
def top_losers(
    exchanges: str = "NASDAQ,NYSE,AMEX",
    market_type: str = "stock",
    min_price: Optional[float] = None,
    min_volume: Optional[float] = None,
    limit: int = 20,
) -> list[dict]:
    """
    Return the biggest declining assets for the current session sorted by daily % change ascending.

    Args:
        exchanges: Comma-separated exchanges (e.g. "NASDAQ,NYSE,AMEX").
        market_type: Asset type: "stock", "crypto", "forex", "fund". Default "stock".
        min_price: Minimum price filter.
        min_volume: Minimum volume filter.
        limit: Max results (1–200).

    Returns:
        List of dicts sorted by daily change ascending (most negative first).
    """
    if not SCREENER_OK:
        raise RuntimeError("tradingview-screener package not installed.")

    exchange_list = [e.strip().upper() for e in exchanges.split(",") if e.strip()]
    limit = max(1, min(limit, 200))

    type_map = {"stock": ["stock", "dr"], "crypto": ["crypto"], "forex": ["forex"], "fund": ["fund", "etf"]}
    types = type_map.get(market_type.lower(), ["stock", "dr"])

    q = (
        Query()
        .select("name", "close", "change", "change_abs", "volume", "market_cap_basic",
                "sector", "exchange", "type", "Perf.D", "Perf.W", "Recommend.All")
        .where(
            Column("type").isin(types),
            Column("exchange").isin(exchange_list),
            Column("change") < 0,
        )
        .order_by("change", ascending=True)
        .limit(limit)
    )
    if min_price is not None:
        q = q.where(Column("close") >= min_price)
    if min_volume is not None:
        q = q.where(Column("volume") >= min_volume)

    _, df = q.get_scanner_data()
    if df.empty:
        return []

    df[["exch", "sym"]] = df["ticker"].str.split(":", expand=True)
    results = []
    for _, row in df.iterrows():
        results.append(_clean_dict({
            "symbol":      row.get("sym"),
            "name":        row.get("name"),
            "exchange":    row.get("exch"),
            "sector":      row.get("sector"),
            "price":       row.get("close"),
            "change_pct":  row.get("change"),
            "change_abs":  row.get("change_abs"),
            "volume":      row.get("volume"),
            "market_cap":  row.get("market_cap_basic"),
            "perf_1d_pct": row.get("Perf.D"),
            "perf_1w_pct": row.get("Perf.W"),
            "analyst_rating": row.get("Recommend.All"),
        }))
    return results


# ---------------------------------------------------------------------------
# Tool 8: Bollinger Band scan
# ---------------------------------------------------------------------------

@mcp.tool()
def bollinger_scan(
    exchanges: str = "NASDAQ,NYSE,AMEX",
    mode: str = "squeeze",
    market_type: str = "stock",
    limit: int = 30,
) -> list[dict]:
    """
    Scan for assets with notable Bollinger Band conditions.

    Args:
        exchanges: Comma-separated exchanges.
        mode: One of "squeeze" (tight bands, low volatility breakout setup),
              "upper_touch" (price near/above upper BB — overbought or breakout),
              "lower_touch" (price near/below lower BB — oversold or breakdown).
        market_type: "stock", "crypto", "forex", "fund".
        limit: Max results (1–200).

    Returns:
        List of dicts with symbol, price, BB upper/middle/lower, BB width, and position.
    """
    if not SCREENER_OK:
        raise RuntimeError("tradingview-screener package not installed.")

    exchange_list = [e.strip().upper() for e in exchanges.split(",") if e.strip()]
    limit = max(1, min(limit, 200))

    type_map = {"stock": ["stock", "dr"], "crypto": ["crypto"], "forex": ["forex"], "fund": ["fund", "etf"]}
    types = type_map.get(market_type.lower(), ["stock", "dr"])

    q = (
        Query()
        .select("name", "close", "BB.upper", "BB.lower", "BB.basis",
                "volume", "market_cap_basic", "sector", "exchange", "Recommend.All")
        .where(
            Column("type").isin(types),
            Column("exchange").isin(exchange_list),
        )
        .limit(500)
    )

    _, df = q.get_scanner_data()
    if df.empty:
        return []

    df[["exch", "sym"]] = df["ticker"].str.split(":", expand=True)
    df = df.dropna(subset=["BB.upper", "BB.lower", "BB.basis", "close"])
    df["bb_width"] = (df["BB.upper"] - df["BB.lower"]) / df["BB.basis"]
    df["bb_position"] = (df["close"] - df["BB.lower"]) / (df["BB.upper"] - df["BB.lower"])

    if mode == "squeeze":
        df = df.nsmallest(limit, "bb_width")
    elif mode == "upper_touch":
        df = df[df["bb_position"] >= 0.9].nlargest(limit, "bb_position")
    elif mode == "lower_touch":
        df = df[df["bb_position"] <= 0.1].nsmallest(limit, "bb_position")
    else:
        df = df.nsmallest(limit, "bb_width")

    results = []
    for _, row in df.iterrows():
        results.append(_clean_dict({
            "symbol":      row.get("sym"),
            "name":        row.get("name"),
            "exchange":    row.get("exch"),
            "sector":      row.get("sector"),
            "price":       row.get("close"),
            "bb_upper":    row.get("BB.upper"),
            "bb_middle":   row.get("BB.basis"),
            "bb_lower":    row.get("BB.lower"),
            "bb_width_pct": row.get("bb_width") * 100 if row.get("bb_width") else None,
            "bb_position_pct": row.get("bb_position") * 100 if row.get("bb_position") else None,
            "volume":      row.get("volume"),
            "market_cap":  row.get("market_cap_basic"),
            "analyst_rating": row.get("Recommend.All"),
        }))
    return results


# ---------------------------------------------------------------------------
# Tool 9: Rating filter (TV oscillator/MA consensus)
# ---------------------------------------------------------------------------

@mcp.tool()
def rating_filter(
    exchanges: str = "NASDAQ,NYSE,AMEX",
    min_rating: float = 0.1,
    max_rating: float = 1.0,
    market_type: str = "stock",
    min_market_cap: Optional[float] = None,
    sort_by: str = "market_cap_basic",
    limit: int = 30,
) -> list[dict]:
    """
    Filter assets by TradingView's combined oscillator+MA rating.

    The rating is a float from –1 (Strong Sell) to +1 (Strong Buy):
      -1.0 to -0.5  → Strong Sell
      -0.5 to -0.1  → Sell
      -0.1 to  0.1  → Neutral
       0.1 to  0.5  → Buy
       0.5 to  1.0  → Strong Buy

    Args:
        exchanges: Comma-separated exchanges.
        min_rating: Lower bound of Recommend.All (inclusive), default 0.1 (Buy+).
        max_rating: Upper bound of Recommend.All (inclusive), default 1.0.
        market_type: "stock", "crypto", "forex", "fund".
        min_market_cap: Minimum market cap in USD.
        sort_by: Field to sort by. Default "market_cap_basic".
        limit: Max results (1–200).

    Returns:
        List of dicts with symbol, price, rating value, rating label, and fundamentals.
    """
    if not SCREENER_OK:
        raise RuntimeError("tradingview-screener package not installed.")

    exchange_list = [e.strip().upper() for e in exchanges.split(",") if e.strip()]
    limit = max(1, min(limit, 200))

    type_map = {"stock": ["stock", "dr"], "crypto": ["crypto"], "forex": ["forex"], "fund": ["fund", "etf"]}
    types = type_map.get(market_type.lower(), ["stock", "dr"])

    q = (
        Query()
        .select("name", "close", "volume", "market_cap_basic", "sector", "exchange",
                "Recommend.All", "Recommend.MA", "Recommend.Other",
                "Perf.W", "Perf.1M", "Perf.Y")
        .where(
            Column("type").isin(types),
            Column("exchange").isin(exchange_list),
            Column("Recommend.All") >= min_rating,
            Column("Recommend.All") <= max_rating,
        )
        .order_by(sort_by, ascending=False)
        .limit(limit)
    )
    if min_market_cap is not None:
        q = q.where(Column("market_cap_basic") >= min_market_cap)

    _, df = q.get_scanner_data()
    if df.empty:
        return []

    df[["exch", "sym"]] = df["ticker"].str.split(":", expand=True)

    def _label(v):
        if v is None:
            return None
        if v >= 0.5:   return "Strong Buy"
        if v >= 0.1:   return "Buy"
        if v >= -0.1:  return "Neutral"
        if v >= -0.5:  return "Sell"
        return "Strong Sell"

    results = []
    for _, row in df.iterrows():
        rv = row.get("Recommend.All")
        results.append(_clean_dict({
            "symbol":           row.get("sym"),
            "name":             row.get("name"),
            "exchange":         row.get("exch"),
            "sector":           row.get("sector"),
            "price":            row.get("close"),
            "volume":           row.get("volume"),
            "market_cap":       row.get("market_cap_basic"),
            "rating_value":     rv,
            "rating_label":     _label(rv),
            "ma_rating":        row.get("Recommend.MA"),
            "oscillator_rating":row.get("Recommend.Other"),
            "perf_1w_pct":      row.get("Perf.W"),
            "perf_1m_pct":      row.get("Perf.1M"),
            "perf_1y_pct":      row.get("Perf.Y"),
        }))
    return results


# ---------------------------------------------------------------------------
# Tool 10: Price for a single symbol (lightweight)
# ---------------------------------------------------------------------------

@mcp.tool()
def get_price(
    symbol: str,
    exchange: str = "NASDAQ",
    screener: str = "america",
) -> dict:
    """
    Get the current price and basic stats for a single symbol.

    Args:
        symbol: Ticker symbol (e.g. "AAPL", "BTC").
        exchange: Exchange (e.g. "NASDAQ", "BINANCE", "NYSE").
        screener: TradingView screener region ("america", "crypto", "europe", etc.).

    Returns:
        Dict with price, open, high, low, volume, change_pct, and recommendation.
    """
    if not TA_OK:
        raise RuntimeError("tradingview-ta package not installed.")

    iv = INTERVAL_MAP.get("1d")
    handler = TA_Handler(
        symbol=symbol.upper(), exchange=exchange.upper(),
        screener=screener.lower(), interval=iv, timeout=15,
    )
    analysis = handler.get_analysis()
    ind = analysis.indicators
    return _clean_dict({
        "symbol":     symbol.upper(),
        "exchange":   exchange.upper(),
        "price":      ind.get("close"),
        "open":       ind.get("open"),
        "high":       ind.get("high"),
        "low":        ind.get("low"),
        "volume":     ind.get("volume"),
        "change_pct": ind.get("change"),
        "recommendation": analysis.summary.get("RECOMMENDATION"),
    })


# ---------------------------------------------------------------------------
# Tool 11: Batch prices for multiple symbols
# ---------------------------------------------------------------------------

@mcp.tool()
def get_multiple_prices(
    tickers: str,
    screener: str = "america",
    interval: str = "1d",
) -> list[dict]:
    """
    Get current prices for multiple symbols at once via the TradingView scanner.

    Args:
        tickers: Comma-separated list in "EXCHANGE:SYMBOL" format,
                 e.g. "NASDAQ:AAPL,NYSE:TSLA,BINANCE:BTCUSD".
        screener: Region hint (used for fallback TA; not used in scanner path).
        interval: Candle interval for change computation. Default "1d".

    Returns:
        List of dicts with symbol, exchange, price, change_pct, volume, market_cap.
    """
    ticker_list = [t.strip().upper() for t in tickers.split(",") if t.strip()]
    if not ticker_list:
        return []

    payload = {
        "columns": ["close", "change", "change_abs", "volume", "market_cap_basic",
                    "name", "Recommend.All"],
        "symbols": {"tickers": ticker_list},
    }
    data = _scanner_post(
        "https://scanner.tradingview.com/global/scan?label-product=popup-watchlists",
        payload,
    )
    results = []
    for item in data.get("data", []):
        ticker = item.get("s", "")
        vals = item.get("d", [])
        parts = ticker.split(":", 1)
        exch = parts[0] if len(parts) == 2 else ""
        sym  = parts[1] if len(parts) == 2 else ticker
        results.append(_clean_dict({
            "symbol":        sym,
            "exchange":      exch,
            "price":         vals[0] if len(vals) > 0 else None,
            "change_pct":    vals[1] if len(vals) > 1 else None,
            "change_abs":    vals[2] if len(vals) > 2 else None,
            "volume":        vals[3] if len(vals) > 3 else None,
            "market_cap":    vals[4] if len(vals) > 4 else None,
            "name":          vals[5] if len(vals) > 5 else None,
            "analyst_rating":vals[6] if len(vals) > 6 else None,
        }))
    return results


# ---------------------------------------------------------------------------
# Tool 12: Crypto coin analysis
# ---------------------------------------------------------------------------

@mcp.tool()
def coin_analysis(
    symbol: str,
    exchange: str = "BINANCE",
    interval: str = "1h",
) -> dict:
    """
    Complete technical analysis for a cryptocurrency coin.

    Args:
        symbol: Crypto pair (e.g. "BTCUSD", "ETHUSD", "SOLUSDT").
        exchange: Exchange (e.g. "BINANCE", "COINBASE", "KRAKEN").
        interval: Candle interval: "1m","5m","15m","30m","1h","2h","4h","1d","1w","1M".

    Returns:
        Full TA dict identical to get_technical_analysis, optimised for crypto screener.
    """
    return get_technical_analysis(
        symbol=symbol,
        exchange=exchange,
        screener="crypto",
        interval=interval,
    )


# ---------------------------------------------------------------------------
# Tool 13: Bollinger Bands for a single symbol
# ---------------------------------------------------------------------------

@mcp.tool()
def get_bollinger_bands(
    symbol: str,
    exchange: str = "NASDAQ",
    screener: str = "america",
    interval: str = "1d",
    period: int = 20,
) -> dict:
    """
    Get Bollinger Bands for a symbol from TradingView.

    Args:
        symbol: Ticker (e.g. "AAPL").
        exchange: Exchange (e.g. "NASDAQ").
        screener: Region ("america", "crypto", "europe", etc.).
        interval: Candle interval. Default "1d".
        period: BB period (used as context — TV uses 20 internally). Default 20.

    Returns:
        Dict with price, bb_upper, bb_middle, bb_lower, bb_width, bb_percent_b,
        and squeeze flag.
    """
    if not TA_OK:
        raise RuntimeError("tradingview-ta package not installed.")
    iv = INTERVAL_MAP.get(interval)
    if iv is None:
        raise ValueError(f"Unknown interval '{interval}'.")
    handler = TA_Handler(symbol=symbol.upper(), exchange=exchange.upper(),
                         screener=screener.lower(), interval=iv, timeout=15)
    ind = handler.get_analysis().indicators
    price  = ind.get("close")
    upper  = ind.get("BB.upper")
    lower  = ind.get("BB.lower")
    middle = (upper + lower) / 2 if upper and lower else None
    width  = (upper - lower) / middle * 100 if upper and lower and middle else None
    pct_b  = (price - lower) / (upper - lower) * 100 if price and upper and lower and upper != lower else None
    squeeze = width is not None and width < 5.0
    return _clean_dict({
        "symbol":     symbol.upper(),
        "exchange":   exchange.upper(),
        "interval":   interval,
        "price":      price,
        "bb_upper":   upper,
        "bb_middle":  middle,
        "bb_lower":   lower,
        "bb_width_pct":   width,
        "bb_percent_b":   pct_b,
        "squeeze":    squeeze,
    })


# ---------------------------------------------------------------------------
# Tool 14: MACD for a single symbol
# ---------------------------------------------------------------------------

@mcp.tool()
def get_macd(
    symbol: str,
    exchange: str = "NASDAQ",
    screener: str = "america",
    interval: str = "1d",
) -> dict:
    """
    Get MACD indicator for a symbol from TradingView (12/26/9 settings).

    Args:
        symbol: Ticker (e.g. "AAPL").
        exchange: Exchange (e.g. "NASDAQ").
        screener: Region ("america", "crypto", "europe", etc.).
        interval: Candle interval. Default "1d".

    Returns:
        Dict with macd_line, signal_line, histogram, and crossover direction.
    """
    if not TA_OK:
        raise RuntimeError("tradingview-ta package not installed.")
    iv = INTERVAL_MAP.get(interval)
    if iv is None:
        raise ValueError(f"Unknown interval '{interval}'.")
    handler = TA_Handler(symbol=symbol.upper(), exchange=exchange.upper(),
                         screener=screener.lower(), interval=iv, timeout=15)
    ind = handler.get_analysis().indicators
    macd   = ind.get("MACD.macd")
    signal = ind.get("MACD.signal")
    hist   = (macd - signal) if macd is not None and signal is not None else None
    crossover = None
    if hist is not None:
        crossover = "bullish" if hist > 0 else ("bearish" if hist < 0 else "neutral")
    return _clean_dict({
        "symbol":     symbol.upper(),
        "exchange":   exchange.upper(),
        "interval":   interval,
        "price":      ind.get("close"),
        "macd_line":  macd,
        "signal_line":signal,
        "histogram":  hist,
        "crossover":  crossover,
    })


# ---------------------------------------------------------------------------
# Tool 15: Moving averages for a single symbol
# ---------------------------------------------------------------------------

@mcp.tool()
def get_moving_averages(
    symbol: str,
    exchange: str = "NASDAQ",
    screener: str = "america",
    interval: str = "1d",
) -> dict:
    """
    Get all major SMAs and EMAs for a symbol from TradingView.

    Args:
        symbol: Ticker (e.g. "AAPL").
        exchange: Exchange.
        screener: Region.
        interval: Candle interval. Default "1d".

    Returns:
        Dict with price, all SMA/EMA levels (10/20/30/50/100/200), price position
        relative to each MA, and MA buy/sell/neutral signal counts.
    """
    if not TA_OK:
        raise RuntimeError("tradingview-ta package not installed.")
    iv = INTERVAL_MAP.get(interval)
    if iv is None:
        raise ValueError(f"Unknown interval '{interval}'.")
    handler = TA_Handler(symbol=symbol.upper(), exchange=exchange.upper(),
                         screener=screener.lower(), interval=iv, timeout=15)
    analysis = handler.get_analysis()
    ind = analysis.indicators
    ma  = analysis.moving_averages
    price = ind.get("close")
    result = _clean_dict({
        "symbol":   symbol.upper(),
        "exchange": exchange.upper(),
        "interval": interval,
        "price":    price,
        "sma_10":   ind.get("SMA10"),
        "sma_20":   ind.get("SMA20"),
        "sma_30":   ind.get("SMA30"),
        "sma_50":   ind.get("SMA50"),
        "sma_100":  ind.get("SMA100"),
        "sma_200":  ind.get("SMA200"),
        "ema_10":   ind.get("EMA10"),
        "ema_20":   ind.get("EMA20"),
        "ema_30":   ind.get("EMA30"),
        "ema_50":   ind.get("EMA50"),
        "ema_100":  ind.get("EMA100"),
        "ema_200":  ind.get("EMA200"),
        "ma_buy_signals":     ma.get("BUY"),
        "ma_sell_signals":    ma.get("SELL"),
        "ma_neutral_signals": ma.get("NEUTRAL"),
        "ma_recommendation":  ma.get("RECOMMENDATION"),
    })
    # Annotate price vs MA
    if price:
        for k in ["sma_20", "sma_50", "sma_200", "ema_20", "ema_50", "ema_200"]:
            v = result.get(k)
            if v:
                result[f"price_above_{k}"] = price > v
    return result


# ---------------------------------------------------------------------------
# Tool 16: Stochastic oscillator
# ---------------------------------------------------------------------------

@mcp.tool()
def get_stochastic(
    symbol: str,
    exchange: str = "NASDAQ",
    screener: str = "america",
    interval: str = "1d",
) -> dict:
    """
    Get the Stochastic oscillator (%K and %D) for a symbol.

    Args:
        symbol: Ticker (e.g. "AAPL").
        exchange: Exchange.
        screener: Region.
        interval: Candle interval. Default "1d".

    Returns:
        Dict with stoch_k, stoch_d, zone (overbought/oversold/neutral), and crossover signal.
    """
    if not TA_OK:
        raise RuntimeError("tradingview-ta package not installed.")
    iv = INTERVAL_MAP.get(interval)
    if iv is None:
        raise ValueError(f"Unknown interval '{interval}'.")
    handler = TA_Handler(symbol=symbol.upper(), exchange=exchange.upper(),
                         screener=screener.lower(), interval=iv, timeout=15)
    ind = handler.get_analysis().indicators
    k = ind.get("Stoch.K")
    d = ind.get("Stoch.D")
    zone = None
    if k is not None:
        if k > 80:
            zone = "overbought"
        elif k < 20:
            zone = "oversold"
        else:
            zone = "neutral"
    crossover = None
    if k is not None and d is not None:
        if k > d:
            crossover = "bullish"
        elif k < d:
            crossover = "bearish"
        else:
            crossover = "neutral"
    return _clean_dict({
        "symbol":     symbol.upper(),
        "exchange":   exchange.upper(),
        "interval":   interval,
        "price":      ind.get("close"),
        "stoch_k":    k,
        "stoch_d":    d,
        "zone":       zone,
        "crossover":  crossover,
    })


# ---------------------------------------------------------------------------
# Tool 17: ADX — trend strength
# ---------------------------------------------------------------------------

@mcp.tool()
def get_adx(
    symbol: str,
    exchange: str = "NASDAQ",
    screener: str = "america",
    interval: str = "1d",
) -> dict:
    """
    Get the ADX (Average Directional Index) and directional indicators for a symbol.

    Args:
        symbol: Ticker (e.g. "AAPL").
        exchange: Exchange.
        screener: Region.
        interval: Candle interval. Default "1d".

    Returns:
        Dict with adx value, trend_strength label (weak/moderate/strong/very_strong),
        plus DI+ and DI- where available.
    """
    if not TA_OK:
        raise RuntimeError("tradingview-ta package not installed.")
    iv = INTERVAL_MAP.get(interval)
    if iv is None:
        raise ValueError(f"Unknown interval '{interval}'.")
    handler = TA_Handler(symbol=symbol.upper(), exchange=exchange.upper(),
                         screener=screener.lower(), interval=iv, timeout=15)
    ind = handler.get_analysis().indicators
    adx = ind.get("ADX")
    strength = None
    if adx is not None:
        if adx < 20:    strength = "weak"
        elif adx < 40:  strength = "moderate"
        elif adx < 60:  strength = "strong"
        else:           strength = "very_strong"
    di_plus  = ind.get("ADX+DI")
    di_minus = ind.get("ADX-DI")
    direction = None
    if di_plus is not None and di_minus is not None:
        direction = "bullish" if di_plus > di_minus else "bearish"
    return _clean_dict({
        "symbol":         symbol.upper(),
        "exchange":       exchange.upper(),
        "interval":       interval,
        "price":          ind.get("close"),
        "adx":            adx,
        "trend_strength": strength,
        "di_plus":        di_plus,
        "di_minus":       di_minus,
        "direction":      direction,
    })


# ---------------------------------------------------------------------------
# Tool 18: Ichimoku Cloud
# ---------------------------------------------------------------------------

@mcp.tool()
def get_ichimoku_cloud(
    symbol: str,
    exchange: str = "NASDAQ",
    screener: str = "america",
    interval: str = "1d",
) -> dict:
    """
    Get Ichimoku Cloud components for a symbol.

    Args:
        symbol: Ticker (e.g. "AAPL").
        exchange: Exchange.
        screener: Region.
        interval: Candle interval. Default "1d".

    Returns:
        Dict with tenkan_sen (conversion), kijun_sen (base), senkou_span_a/b,
        cloud position, and overall Ichimoku signal.
    """
    if not TA_OK:
        raise RuntimeError("tradingview-ta package not installed.")
    iv = INTERVAL_MAP.get(interval)
    if iv is None:
        raise ValueError(f"Unknown interval '{interval}'.")
    handler = TA_Handler(symbol=symbol.upper(), exchange=exchange.upper(),
                         screener=screener.lower(), interval=iv, timeout=15)
    ind = handler.get_analysis().indicators
    price       = ind.get("close")
    tenkan      = ind.get("Ichimoku.BLine")   # Kijun-sen in some builds
    kijun       = ind.get("Ichimoku.CLine")   # Tenkan-sen
    span_a      = ind.get("Ichimoku.Lead1")
    span_b      = ind.get("Ichimoku.Lead2")

    above_cloud = None
    if price and span_a is not None and span_b is not None:
        cloud_top = max(span_a, span_b)
        cloud_bot = min(span_a, span_b)
        if price > cloud_top:
            above_cloud = "above"
        elif price < cloud_bot:
            above_cloud = "below"
        else:
            above_cloud = "inside"

    signal = None
    if above_cloud == "above":
        signal = "bullish"
    elif above_cloud == "below":
        signal = "bearish"
    elif above_cloud == "inside":
        signal = "neutral"

    return _clean_dict({
        "symbol":       symbol.upper(),
        "exchange":     exchange.upper(),
        "interval":     interval,
        "price":        price,
        "tenkan_sen":   tenkan,
        "kijun_sen":    kijun,
        "senkou_span_a":span_a,
        "senkou_span_b":span_b,
        "price_vs_cloud": above_cloud,
        "signal":       signal,
    })


# ---------------------------------------------------------------------------
# Tool 19: Consecutive candles scan
# ---------------------------------------------------------------------------

@mcp.tool()
def consecutive_candles_scan(
    exchanges: str = "NASDAQ,NYSE,AMEX",
    direction: str = "bullish",
    market_type: str = "stock",
    min_price: Optional[float] = 5.0,
    min_volume: Optional[float] = 500000,
    limit: int = 30,
) -> list[dict]:
    """
    Scan for assets showing strong consecutive directional candle momentum.

    Uses recent performance across daily, weekly and monthly timeframes as a proxy
    for consecutive green (bullish) or red (bearish) candles.

    Args:
        exchanges: Comma-separated exchanges.
        direction: "bullish" (consecutive green candles) or "bearish" (red candles).
        market_type: "stock", "crypto", "forex", "fund".
        min_price: Minimum price filter. Default 5.0.
        min_volume: Minimum daily volume filter. Default 500000.
        limit: Max results (1–100).

    Returns:
        List of dicts with symbol, price, daily/weekly/monthly performance.
    """
    if not SCREENER_OK:
        raise RuntimeError("tradingview-screener package not installed.")

    exchange_list = [e.strip().upper() for e in exchanges.split(",") if e.strip()]
    limit = max(1, min(limit, 100))

    type_map = {"stock": ["stock", "dr"], "crypto": ["crypto"], "forex": ["forex"], "fund": ["fund", "etf"]}
    types = type_map.get(market_type.lower(), ["stock", "dr"])

    q = (
        Query()
        .select("name", "close", "change", "Perf.W", "Perf.1M", "volume",
                "market_cap_basic", "sector", "exchange", "Recommend.All",
                "RSI", "AO")
        .where(
            Column("type").isin(types),
            Column("exchange").isin(exchange_list),
        )
        .limit(500)
    )
    if min_price is not None:
        q = q.where(Column("close") >= min_price)
    if min_volume is not None:
        q = q.where(Column("volume") >= min_volume)

    _, df = q.get_scanner_data()
    if df.empty:
        return []

    df[["exch", "sym"]] = df["ticker"].str.split(":", expand=True)
    df = df.dropna(subset=["change", "Perf.W", "Perf.1M"])

    if direction == "bullish":
        # All three timeframes positive
        df = df[(df["change"] > 0) & (df["Perf.W"] > 0) & (df["Perf.1M"] > 0)]
        df = df.sort_values("Perf.1M", ascending=False)
    else:
        df = df[(df["change"] < 0) & (df["Perf.W"] < 0) & (df["Perf.1M"] < 0)]
        df = df.sort_values("Perf.1M", ascending=True)

    df = df.head(limit)
    results = []
    for _, row in df.iterrows():
        results.append(_clean_dict({
            "symbol":      row.get("sym"),
            "name":        row.get("name"),
            "exchange":    row.get("exch"),
            "sector":      row.get("sector"),
            "price":       row.get("close"),
            "change_d_pct":row.get("change"),
            "change_w_pct":row.get("Perf.W"),
            "change_m_pct":row.get("Perf.1M"),
            "volume":      row.get("volume"),
            "market_cap":  row.get("market_cap_basic"),
            "rsi":         row.get("RSI"),
            "analyst_rating": row.get("Recommend.All"),
        }))
    return results


# ---------------------------------------------------------------------------
# Tool 20: Advanced candle pattern (multi-timeframe TA)
# ---------------------------------------------------------------------------

@mcp.tool()
def advanced_candle_pattern(
    symbol: str,
    exchange: str = "NASDAQ",
    screener: str = "america",
    intervals: str = "15m,1h,4h,1d",
) -> list[dict]:
    """
    Run full technical analysis across multiple timeframes for a single symbol
    to identify multi-timeframe candle pattern alignment.

    Args:
        symbol: Ticker (e.g. "AAPL", "BTCUSD").
        exchange: Exchange (e.g. "NASDAQ", "BINANCE").
        screener: Region ("america", "crypto", "europe", etc.).
        intervals: Comma-separated list of intervals to analyse.
                   Options: "1m","5m","15m","30m","1h","2h","4h","1d","1w","1M".

    Returns:
        List of analysis dicts — one per interval — each with recommendation,
        signal counts, RSI, MACD, BB, and key MAs.
    """
    if not TA_OK:
        raise RuntimeError("tradingview-ta package not installed.")

    interval_list = [i.strip() for i in intervals.split(",") if i.strip()]
    results = []
    for ivl in interval_list:
        iv = INTERVAL_MAP.get(ivl)
        if iv is None:
            results.append({"interval": ivl, "error": f"Unknown interval '{ivl}'"})
            continue
        try:
            handler = TA_Handler(
                symbol=symbol.upper(), exchange=exchange.upper(),
                screener=screener.lower(), interval=iv, timeout=15,
            )
            analysis = handler.get_analysis()
            ind  = analysis.indicators
            summ = analysis.summary
            results.append(_clean_dict({
                "interval":       ivl,
                "price":          ind.get("close"),
                "change_pct":     ind.get("change"),
                "recommendation": summ.get("RECOMMENDATION"),
                "buy_signals":    summ.get("BUY"),
                "sell_signals":   summ.get("SELL"),
                "neutral_signals":summ.get("NEUTRAL"),
                "rsi":            ind.get("RSI"),
                "macd_line":      ind.get("MACD.macd"),
                "macd_signal":    ind.get("MACD.signal"),
                "bb_upper":       ind.get("BB.upper"),
                "bb_lower":       ind.get("BB.lower"),
                "adx":            ind.get("ADX"),
                "stoch_k":        ind.get("Stoch.K"),
                "ema_20":         ind.get("EMA20"),
                "ema_50":         ind.get("EMA50"),
                "ema_200":        ind.get("EMA200"),
                "sma_200":        ind.get("SMA200"),
            }))
        except Exception as e:
            results.append({"interval": ivl, "error": str(e)})
    return results


# ---------------------------------------------------------------------------
# Computed-from-historical helpers
# ---------------------------------------------------------------------------

def _get_ohlcv_df(symbol: str, exchange: str, freq: str, bars: int):
    """Fetch historical OHLCV as a DataFrame; raises if unavailable."""
    if not HISTORICAL_OK:
        raise RuntimeError("TradingViewHistoricalFetcher unavailable.")
    fetcher = TradingViewHistoricalFetcher()
    df = fetcher.get_historical_data(
        symbol=symbol.upper(), exchange=exchange.upper(),
        freq=freq, number_of_values=bars,
    )
    if df is None or df.empty:
        raise ValueError(f"No historical data returned for {exchange}:{symbol}.")
    return df


# ---------------------------------------------------------------------------
# Tool 21: Fibonacci retracement levels
# ---------------------------------------------------------------------------

@mcp.tool()
def get_fibonacci_retracement(
    symbol: str,
    exchange: str = "NASDAQ",
    freq: str = "1D",
    bars: int = 200,
) -> dict:
    """
    Calculate Fibonacci retracement levels from the highest high and lowest low
    over the specified historical period.

    Args:
        symbol: Ticker (e.g. "AAPL").
        exchange: Exchange (e.g. "NASDAQ").
        freq: Bar frequency: "1D" (daily), "1W" (weekly), "60" (1-hour), "240" (4-hour).
        bars: Number of historical bars to scan for the swing high/low.

    Returns:
        Dict with swing_high, swing_low, and Fibonacci levels (0%, 23.6%, 38.2%,
        50%, 61.8%, 78.6%, 100%) with both retracement and extension values.
    """
    df = _get_ohlcv_df(symbol, exchange, freq, bars)
    swing_high = float(df["high"].max())
    swing_low  = float(df["low"].min())
    price      = float(df["close"].iloc[-1])
    diff = swing_high - swing_low

    ratios = [0.0, 0.236, 0.382, 0.5, 0.618, 0.786, 1.0]
    levels = {}
    for r in ratios:
        key = f"fib_{int(r*1000):04d}"   # e.g. "fib_0000", "fib_0236"
        levels[key] = round(swing_high - diff * r, 4)

    extensions = {}
    for r in [1.272, 1.414, 1.618, 2.0, 2.618]:
        key = f"ext_{int(r*1000):04d}"
        extensions[key] = round(swing_low - diff * (r - 1), 4)

    return {
        "symbol":     symbol.upper(),
        "exchange":   exchange.upper(),
        "freq":       freq,
        "bars_used":  len(df),
        "current_price": round(price, 4),
        "swing_high": round(swing_high, 4),
        "swing_low":  round(swing_low, 4),
        "retracement_levels": levels,
        "extension_levels":   extensions,
    }


# ---------------------------------------------------------------------------
# Tool 22: Support & Resistance levels
# ---------------------------------------------------------------------------

@mcp.tool()
def get_support_resistance(
    symbol: str,
    exchange: str = "NASDAQ",
    freq: str = "1D",
    bars: int = 200,
    window: int = 10,
    cluster_pct: float = 1.0,
) -> dict:
    """
    Auto-detect support and resistance levels from historical price pivots.

    Args:
        symbol: Ticker (e.g. "AAPL").
        exchange: Exchange (e.g. "NASDAQ").
        freq: Bar frequency: "1D" (daily), "1W", "60" (1-hour), "240" (4-hour).
        bars: Number of historical bars.
        window: Rolling window to detect local highs/lows (default 10 bars each side).
        cluster_pct: Merge levels within this % of each other into one zone.

    Returns:
        Dict with current price, support_levels (ascending) and resistance_levels
        (ascending) as lists, each with price and strength (touch count).
    """
    if not NUMPY_OK:
        raise RuntimeError("numpy/pandas not available.")
    df = _get_ohlcv_df(symbol, exchange, freq, bars)
    price = float(df["close"].iloc[-1])
    highs = df["high"].values
    lows  = df["low"].values

    # Find local extrema
    def local_extrema(arr, w, find_max=True):
        pts = []
        for i in range(w, len(arr) - w):
            segment = arr[i - w : i + w + 1]
            if find_max and arr[i] == segment.max():
                pts.append(float(arr[i]))
            elif not find_max and arr[i] == segment.min():
                pts.append(float(arr[i]))
        return pts

    resistance_pts = local_extrema(highs, window, find_max=True)
    support_pts    = local_extrema(lows,  window, find_max=False)

    def cluster(pts, pct):
        if not pts:
            return []
        pts = sorted(pts)
        clusters = []
        group = [pts[0]]
        for p in pts[1:]:
            if (p - group[0]) / group[0] * 100 <= pct:
                group.append(p)
            else:
                clusters.append({"price": round(sum(group)/len(group), 4), "strength": len(group)})
                group = [p]
        clusters.append({"price": round(sum(group)/len(group), 4), "strength": len(group)})
        return clusters

    supports    = [l for l in cluster(support_pts, cluster_pct) if l["price"] < price]
    resistances = [l for l in cluster(resistance_pts, cluster_pct) if l["price"] > price]

    return {
        "symbol":           symbol.upper(),
        "exchange":         exchange.upper(),
        "freq":             freq,
        "current_price":    round(price, 4),
        "support_levels":   sorted(supports, key=lambda x: x["price"]),
        "resistance_levels":sorted(resistances, key=lambda x: x["price"]),
    }


# ---------------------------------------------------------------------------
# Tool 23: Pivot points
# ---------------------------------------------------------------------------

@mcp.tool()
def get_pivot_points(
    symbol: str,
    exchange: str = "NASDAQ",
    freq: str = "1D",
) -> dict:
    """
    Calculate classic, Fibonacci, and Camarilla pivot points from the last
    completed candle's OHLC.

    Args:
        symbol: Ticker (e.g. "AAPL").
        exchange: Exchange (e.g. "NASDAQ").
        freq: Reference period: "1D" (daily pivots), "1W" (weekly), "1M" (monthly).

    Returns:
        Dict with pivot, classic R1-R3/S1-S3, Fibonacci pivots, and Camarilla levels.
    """
    df = _get_ohlcv_df(symbol, exchange, freq, 5)
    # Use the second-to-last bar (last completed candle)
    ref = df.iloc[-2] if len(df) >= 2 else df.iloc[-1]
    H = float(ref["high"])
    L = float(ref["low"])
    C = float(ref["close"])
    O = float(ref["open"]) if "open" in ref else (H + L + C) / 3

    P = (H + L + C) / 3
    r = round

    classic = {
        "pivot": r(P, 4),
        "r1": r(2*P - L, 4), "r2": r(P + (H - L), 4), "r3": r(H + 2*(P - L), 4),
        "s1": r(2*P - H, 4), "s2": r(P - (H - L), 4), "s3": r(L - 2*(H - P), 4),
    }
    fib = {
        "pivot": r(P, 4),
        "r1": r(P + 0.382*(H-L), 4), "r2": r(P + 0.618*(H-L), 4), "r3": r(P + 1.0*(H-L), 4),
        "s1": r(P - 0.382*(H-L), 4), "s2": r(P - 0.618*(H-L), 4), "s3": r(P - 1.0*(H-L), 4),
    }
    cam_range = H - L
    camarilla = {
        "r4": r(C + cam_range * 1.1/2, 4), "r3": r(C + cam_range * 1.1/4, 4),
        "r2": r(C + cam_range * 1.1/6, 4), "r1": r(C + cam_range * 1.1/12, 4),
        "s1": r(C - cam_range * 1.1/12, 4), "s2": r(C - cam_range * 1.1/6, 4),
        "s3": r(C - cam_range * 1.1/4, 4), "s4": r(C - cam_range * 1.1/2, 4),
    }

    return {
        "symbol":     symbol.upper(),
        "exchange":   exchange.upper(),
        "ref_freq":   freq,
        "ref_high":   r(H, 4),
        "ref_low":    r(L, 4),
        "ref_close":  r(C, 4),
        "current_price": round(float(df["close"].iloc[-1]), 4),
        "classic":    classic,
        "fibonacci":  fib,
        "camarilla":  camarilla,
    }


# ---------------------------------------------------------------------------
# Tool 24: ATR — Average True Range
# ---------------------------------------------------------------------------

@mcp.tool()
def get_atr(
    symbol: str,
    exchange: str = "NASDAQ",
    freq: str = "1D",
    bars: int = 100,
    period: int = 14,
) -> dict:
    """
    Calculate ATR (Average True Range) for a symbol.

    Args:
        symbol: Ticker (e.g. "AAPL").
        exchange: Exchange.
        freq: Bar frequency: "1D", "1W", "60" (1-hour), "240" (4-hour).
        bars: Number of historical bars to use (min period+1).
        period: ATR period. Default 14.

    Returns:
        Dict with current ATR value, ATR as % of price, and volatility label.
    """
    if not NUMPY_OK:
        raise RuntimeError("numpy/pandas not available.")
    df = _get_ohlcv_df(symbol, exchange, freq, max(bars, period + 10))
    df = df.dropna(subset=["high", "low", "close"])
    price = float(df["close"].iloc[-1])
    high  = df["high"].values.astype(float)
    low   = df["low"].values.astype(float)
    close = df["close"].values.astype(float)

    tr_list = [high[0] - low[0]]
    for i in range(1, len(df)):
        tr = max(high[i] - low[i], abs(high[i] - close[i-1]), abs(low[i] - close[i-1]))
        tr_list.append(tr)
    tr_arr = pd.Series(tr_list)
    atr_series = tr_arr.ewm(span=period, adjust=False).mean()
    atr = float(atr_series.iloc[-1])
    atr_pct = atr / price * 100 if price else None

    if atr_pct is None:
        volatility = None
    elif atr_pct < 1:
        volatility = "very_low"
    elif atr_pct < 2:
        volatility = "low"
    elif atr_pct < 4:
        volatility = "moderate"
    elif atr_pct < 8:
        volatility = "high"
    else:
        volatility = "very_high"

    return _clean_dict({
        "symbol":         symbol.upper(),
        "exchange":       exchange.upper(),
        "freq":           freq,
        "period":         period,
        "current_price":  round(price, 4),
        "atr":            round(atr, 4),
        "atr_pct":        round(atr_pct, 4) if atr_pct else None,
        "volatility":     volatility,
    })


# ---------------------------------------------------------------------------
# Tool 25: VWAP
# ---------------------------------------------------------------------------

@mcp.tool()
def get_vwap(
    symbol: str,
    exchange: str = "NASDAQ",
    freq: str = "60",
    bars: int = 390,
) -> dict:
    """
    Calculate VWAP (Volume Weighted Average Price) for a symbol.

    Best used with intraday bars (e.g. freq="60" for hourly or "5" for 5-minute).

    Args:
        symbol: Ticker (e.g. "AAPL").
        exchange: Exchange.
        freq: Bar frequency in minutes as string ("1","5","15","30","60","240")
              or "1D" for daily VWAP.
        bars: Number of bars to include in the VWAP calculation.

    Returns:
        Dict with vwap value, current price, and price position vs VWAP.
    """
    if not NUMPY_OK:
        raise RuntimeError("numpy/pandas not available.")
    df = _get_ohlcv_df(symbol, exchange, freq, bars)
    df = df.dropna(subset=["high", "low", "close", "volume"])
    typical_price = (df["high"] + df["low"] + df["close"]) / 3
    cum_tp_vol = (typical_price * df["volume"]).cumsum()
    cum_vol    = df["volume"].cumsum()
    vwap_series = cum_tp_vol / cum_vol
    vwap  = float(vwap_series.iloc[-1])
    price = float(df["close"].iloc[-1])
    diff_pct = (price - vwap) / vwap * 100 if vwap else None
    position = None
    if diff_pct is not None:
        position = "above" if diff_pct > 0 else "below"

    return _clean_dict({
        "symbol":        symbol.upper(),
        "exchange":      exchange.upper(),
        "freq":          freq,
        "bars_used":     len(df),
        "current_price": round(price, 4),
        "vwap":          round(vwap, 4),
        "diff_pct":      round(diff_pct, 4) if diff_pct else None,
        "position":      position,
    })


# ---------------------------------------------------------------------------
# Tool 26: Volume profile
# ---------------------------------------------------------------------------

@mcp.tool()
def get_volume_profile(
    symbol: str,
    exchange: str = "NASDAQ",
    freq: str = "1D",
    bars: int = 100,
    bins: int = 20,
) -> dict:
    """
    Build a volume profile (volume at price) from historical OHLCV data.

    Args:
        symbol: Ticker (e.g. "AAPL").
        exchange: Exchange.
        freq: Bar frequency. Default "1D".
        bars: Number of historical bars.
        bins: Number of price buckets. Default 20.

    Returns:
        Dict with current price, POC (Point of Control), value area high/low,
        and a list of {price_low, price_high, volume} bins sorted by price.
    """
    if not NUMPY_OK:
        raise RuntimeError("numpy/pandas not available.")
    df = _get_ohlcv_df(symbol, exchange, freq, bars)
    df = df.dropna(subset=["high", "low", "close", "volume"])
    price = float(df["close"].iloc[-1])
    price_min = float(df["low"].min())
    price_max = float(df["high"].max())

    bin_edges = [price_min + i * (price_max - price_min) / bins for i in range(bins + 1)]
    bucket_vols = [0.0] * bins

    for _, row in df.iterrows():
        bar_vol = float(row["volume"])
        bar_low  = float(row["low"])
        bar_high = float(row["high"])
        bar_range = bar_high - bar_low or 1e-9
        for b in range(bins):
            overlap_low  = max(bar_low, bin_edges[b])
            overlap_high = min(bar_high, bin_edges[b + 1])
            if overlap_high > overlap_low:
                fraction = (overlap_high - overlap_low) / bar_range
                bucket_vols[b] += bar_vol * fraction

    poc_idx = bucket_vols.index(max(bucket_vols))
    poc_price = (bin_edges[poc_idx] + bin_edges[poc_idx + 1]) / 2

    total_vol = sum(bucket_vols)
    target_vol = total_vol * 0.7
    sorted_by_vol = sorted(range(bins), key=lambda i: bucket_vols[i], reverse=True)
    va_indices = set()
    accum = 0.0
    for idx in sorted_by_vol:
        va_indices.add(idx)
        accum += bucket_vols[idx]
        if accum >= target_vol:
            break
    va_high = bin_edges[max(va_indices) + 1]
    va_low  = bin_edges[min(va_indices)]

    profile = [
        {
            "price_low":  round(bin_edges[b], 4),
            "price_high": round(bin_edges[b + 1], 4),
            "volume":     round(bucket_vols[b], 2),
        }
        for b in range(bins)
    ]

    return {
        "symbol":        symbol.upper(),
        "exchange":      exchange.upper(),
        "freq":          freq,
        "bars_used":     len(df),
        "current_price": round(price, 4),
        "poc_price":     round(poc_price, 4),
        "value_area_high": round(va_high, 4),
        "value_area_low":  round(va_low, 4),
        "total_volume":    round(total_vol, 2),
        "profile":         profile,
    }


# ---------------------------------------------------------------------------
# Tool 27: Detect unfilled gaps
# ---------------------------------------------------------------------------

@mcp.tool()
def detect_unfilled_gaps(
    symbol: str,
    exchange: str = "NASDAQ",
    freq: str = "1D",
    bars: int = 200,
    min_gap_pct: float = 0.5,
) -> dict:
    """
    Detect unfilled price gaps (fair value gaps) in historical data.

    Args:
        symbol: Ticker (e.g. "AAPL").
        exchange: Exchange.
        freq: Bar frequency. Default "1D".
        bars: Historical bars to scan.
        min_gap_pct: Minimum gap size as % of price to report. Default 0.5%.

    Returns:
        Dict with current price and lists of bullish_gaps (upward) and
        bearish_gaps (downward), each with date, gap_low, gap_high, gap_pct,
        and filled status.
    """
    df = _get_ohlcv_df(symbol, exchange, freq, bars)
    df = df.reset_index(drop=True)
    price = float(df["close"].iloc[-1])

    bull_gaps = []
    bear_gaps = []

    for i in range(1, len(df)):
        prev_high = float(df.loc[i-1, "high"])
        prev_low  = float(df.loc[i-1, "low"])
        cur_high  = float(df.loc[i, "high"])
        cur_low   = float(df.loc[i, "low"])
        date_str  = df.loc[i, "date"].isoformat() if hasattr(df.loc[i, "date"], "isoformat") else str(df.loc[i, "date"])

        # Bullish gap: current low > previous high
        if cur_low > prev_high:
            gap_pct = (cur_low - prev_high) / prev_high * 100
            if gap_pct >= min_gap_pct:
                filled = price <= cur_low   # gap filled if price returned below gap bottom
                bull_gaps.append({
                    "date": date_str,
                    "gap_low":  round(prev_high, 4),
                    "gap_high": round(cur_low, 4),
                    "gap_pct":  round(gap_pct, 4),
                    "filled":   filled,
                })

        # Bearish gap: current high < previous low
        if cur_high < prev_low:
            gap_pct = (prev_low - cur_high) / prev_low * 100
            if gap_pct >= min_gap_pct:
                filled = price >= cur_high
                bear_gaps.append({
                    "date": date_str,
                    "gap_low":  round(cur_high, 4),
                    "gap_high": round(prev_low, 4),
                    "gap_pct":  round(gap_pct, 4),
                    "filled":   filled,
                })

    return {
        "symbol":        symbol.upper(),
        "exchange":      exchange.upper(),
        "freq":          freq,
        "current_price": round(price, 4),
        "bullish_gaps":  [g for g in bull_gaps if not g["filled"]],
        "bearish_gaps":  [g for g in bear_gaps if not g["filled"]],
        "all_bullish_gaps": bull_gaps,
        "all_bearish_gaps": bear_gaps,
    }


# ---------------------------------------------------------------------------
# Tool 28: Correlation between two assets
# ---------------------------------------------------------------------------

@mcp.tool()
def calculate_correlation(
    symbol1: str,
    symbol2: str,
    exchange1: str = "NASDAQ",
    exchange2: str = "NASDAQ",
    freq: str = "1D",
    bars: int = 252,
) -> dict:
    """
    Calculate the price return correlation between two assets.

    Args:
        symbol1: First ticker (e.g. "AAPL").
        symbol2: Second ticker (e.g. "MSFT").
        exchange1: Exchange for symbol1.
        exchange2: Exchange for symbol2.
        freq: Bar frequency. Default "1D".
        bars: Number of bars for the calculation (default 252 ≈ 1 trading year).

    Returns:
        Dict with pearson_correlation, spearman_correlation, rolling_30d_correlation,
        and interpretation.
    """
    if not NUMPY_OK:
        raise RuntimeError("numpy/pandas not available.")

    df1 = _get_ohlcv_df(symbol1, exchange1, freq, bars)
    df2 = _get_ohlcv_df(symbol2, exchange2, freq, bars)

    df1 = df1.set_index("date")["close"].rename("s1")
    df2 = df2.set_index("date")["close"].rename("s2")
    combined = pd.concat([df1, df2], axis=1).dropna()

    if len(combined) < 10:
        raise ValueError("Not enough overlapping data points for correlation.")

    r1 = combined["s1"].pct_change().dropna()
    r2 = combined["s2"].pct_change().dropna()
    both = pd.concat([r1, r2], axis=1).dropna()

    pearson   = float(both.corr().iloc[0, 1])
    from scipy.stats import spearmanr
    spearman, _ = spearmanr(both.iloc[:, 0], both.iloc[:, 1])

    rolling_corr = both.rolling(30).corr().unstack()["s2"]["s1"].dropna()
    rolling_30 = float(rolling_corr.iloc[-1]) if len(rolling_corr) else None

    if abs(pearson) >= 0.8:
        interp = "very_strong"
    elif abs(pearson) >= 0.6:
        interp = "strong"
    elif abs(pearson) >= 0.4:
        interp = "moderate"
    elif abs(pearson) >= 0.2:
        interp = "weak"
    else:
        interp = "very_weak"
    direction = "positive" if pearson >= 0 else "negative"

    return _clean_dict({
        "symbol1":               symbol1.upper(),
        "symbol2":               symbol2.upper(),
        "freq":                  freq,
        "bars_used":             len(both),
        "pearson_correlation":   round(pearson, 4),
        "spearman_correlation":  round(float(spearman), 4),
        "rolling_30_correlation":round(rolling_30, 4) if rolling_30 else None,
        "strength":              interp,
        "direction":             direction,
    })


# ---------------------------------------------------------------------------
# Tool 29: Symbol info (scanner REST API)
# ---------------------------------------------------------------------------

@mcp.tool()
def get_symbol_info(
    symbol: str,
    exchange: str = "NASDAQ",
) -> dict:
    """
    Get extended symbol information from TradingView including 52-week range,
    performance across multiple periods, sector/country, and options Greeks.

    Args:
        symbol: Ticker (e.g. "AAPL").
        exchange: Exchange prefix (e.g. "NASDAQ", "NYSE", "BINANCE").

    Returns:
        Dict with 52w high/low, monthly high/low, performance (W/1M/3M/6M/Y/YTD),
        sector, country, market, average volumes, and options data if available.
    """
    full_symbol = f"{exchange.upper()}:{symbol.upper()}"
    fields = (
        "price_52_week_high,price_52_week_low,sector,country,market,"
        "Low.1M,High.1M,Perf.W,Perf.1M,Perf.3M,Perf.6M,Perf.Y,Perf.YTD,"
        "Recommend.All,average_volume_10d_calc,average_volume_30d_calc,"
        "close,change,market_cap_basic,description,exchange,type,"
        "price_earnings_ttm,price_book_fq,dividends_yield,beta_1_year"
    )
    data = _scanner_get(
        "https://scanner.tradingview.com/symbol",
        params={"symbol": full_symbol, "fields": fields, "no_404": "true"},
    )
    if not data:
        return {"error": f"No data returned for {full_symbol}"}

    def _label(v):
        if v is None: return None
        if v >= 0.5:  return "Strong Buy"
        if v >= 0.1:  return "Buy"
        if v >= -0.1: return "Neutral"
        if v >= -0.5: return "Sell"
        return "Strong Sell"

    rec = data.get("Recommend.All")
    return _clean_dict({
        "symbol":           symbol.upper(),
        "exchange":         data.get("exchange", exchange.upper()),
        "name":             data.get("description"),
        "type":             data.get("type"),
        "market":           data.get("market"),
        "sector":           data.get("sector"),
        "country":          data.get("country"),
        "price":            data.get("close"),
        "change_pct":       data.get("change"),
        "market_cap":       data.get("market_cap_basic"),
        "pe_ratio":         data.get("price_earnings_ttm"),
        "pb_ratio":         data.get("price_book_fq"),
        "dividend_yield":   data.get("dividends_yield"),
        "beta_1y":          data.get("beta_1_year"),
        "high_52w":         data.get("price_52_week_high"),
        "low_52w":          data.get("price_52_week_low"),
        "high_1m":          data.get("High.1M"),
        "low_1m":           data.get("Low.1M"),
        "perf_1w_pct":      data.get("Perf.W"),
        "perf_1m_pct":      data.get("Perf.1M"),
        "perf_3m_pct":      data.get("Perf.3M"),
        "perf_6m_pct":      data.get("Perf.6M"),
        "perf_1y_pct":      data.get("Perf.Y"),
        "perf_ytd_pct":     data.get("Perf.YTD"),
        "avg_volume_10d":   data.get("average_volume_10d_calc"),
        "avg_volume_30d":   data.get("average_volume_30d_calc"),
        "analyst_rating":   _label(rec),
        "analyst_rating_value": rec,
    })


# ---------------------------------------------------------------------------
# Tool 30: Earnings calendar
# ---------------------------------------------------------------------------

@mcp.tool()
def get_earnings_calendar(
    days_ahead: int = 7,
    days_behind: int = 0,
    markets: str = "america",
    limit: int = 50,
) -> list[dict]:
    """
    Fetch upcoming (and recent) earnings announcements from TradingView.

    Args:
        days_ahead: How many calendar days ahead to look (default 7).
        days_behind: How many calendar days behind to include (default 0).
        markets: Comma-separated market regions, e.g. "america" or
                 "america,europe,uk,canada,india". Default "america".
        limit: Max results (1–200).

    Returns:
        List of dicts with company name, symbol, earnings date/time, EPS
        (actual and estimate), revenue (actual and estimate), and surprise data.
    """
    now = datetime.utcnow()
    start_ts = int((now - timedelta(days=days_behind)).timestamp())
    end_ts   = int((now + timedelta(days=days_ahead)).timestamp())

    market_list = [m.strip() for m in markets.split(",") if m.strip()]

    payload = {
        "markets": market_list,
        "filter": [{"left": "earnings_release_date,earnings_release_next_date",
                    "operation": "in_range", "right": [start_ts, end_ts]}],
        "columns": [
            "earnings_release_next_date", "earnings_release_date",
            "name", "description", "exchange", "market",
            "earnings_per_share_fq", "earnings_per_share_forecast_next_fq",
            "eps_surprise_fq", "eps_surprise_percent_fq",
            "revenue_fq", "revenue_forecast_next_fq",
            "market_cap_basic", "earnings_release_time",
            "earnings_release_next_time", "fundamental_currency_code",
            "earnings_publication_type_next_fq",
        ],
        "options": {"lang": "en"},
        "range": [0, min(limit, 200)],
    }

    data = _scanner_post(
        "https://scanner.tradingview.com/global/scan?label-product=calendar-earnings",
        payload,
    )

    results = []
    columns = payload["columns"]
    for item in data.get("data", []):
        ticker = item.get("s", "")
        vals   = item.get("d", [])
        d = dict(zip(columns, vals))
        parts = ticker.split(":", 1)
        exch = parts[0] if len(parts) == 2 else ""
        sym  = parts[1] if len(parts) == 2 else ticker

        release_ts = d.get("earnings_release_next_date") or d.get("earnings_release_date")
        release_dt = datetime.utcfromtimestamp(release_ts).strftime("%Y-%m-%d") if release_ts else None

        results.append(_clean_dict({
            "symbol":              sym,
            "exchange":            exch,
            "name":                d.get("description"),
            "market":              d.get("market"),
            "earnings_date":       release_dt,
            "release_time":        d.get("earnings_release_next_time") or d.get("earnings_release_time"),
            "eps_actual":          d.get("earnings_per_share_fq"),
            "eps_estimate":        d.get("earnings_per_share_forecast_next_fq"),
            "eps_surprise":        d.get("eps_surprise_fq"),
            "eps_surprise_pct":    d.get("eps_surprise_percent_fq"),
            "revenue_actual":      d.get("revenue_fq"),
            "revenue_estimate":    d.get("revenue_forecast_next_fq"),
            "market_cap":          d.get("market_cap_basic"),
            "currency":            d.get("fundamental_currency_code"),
            "report_type":         d.get("earnings_publication_type_next_fq"),
        }))
    return results


# ---------------------------------------------------------------------------
# Tool 31: IPO calendar
# ---------------------------------------------------------------------------

@mcp.tool()
def get_ipo_calendar(
    days_ahead: int = 14,
    days_behind: int = 7,
    markets: str = "america",
    limit: int = 30,
) -> list[dict]:
    """
    Fetch upcoming and recent IPOs from TradingView.

    Args:
        days_ahead: Calendar days ahead to look. Default 14.
        days_behind: Calendar days behind to include. Default 7.
        markets: Comma-separated market regions. Default "america".
        limit: Max results (1–100).

    Returns:
        List of dicts with company name, IPO date, offer price, status,
        shares offered, deal amount, and market cap.
    """
    now = datetime.utcnow()
    start_ts = int((now - timedelta(days=days_behind)).timestamp())
    end_ts   = int((now + timedelta(days=days_ahead)).timestamp())

    market_list = [m.strip() for m in markets.split(",") if m.strip()]

    payload = {
        "columns": [
            "name", "description", "typespecs", "type", "exchange", "market",
            "ipo_offer_time", "ipo_offer_price_usd", "ipo_offer_status",
            "ipo_offered_shares", "ipo_deal_amount_usd", "ipo_market_cap_usd",
            "ipo_price_range_usd",
        ],
        "filter": [{"left": "ipo_offer_time", "operation": "in_range",
                    "right": [start_ts, end_ts]}],
        "ignore_unknown_fields": False,
        "options": {"lang": "en"},
        "sort": {"sortBy": "ipo_offer_time", "sortOrder": "asc"},
        "markets": market_list,
        "preset": "ipo_calendar",
        "range": [0, min(limit, 100)],
    }

    data = _scanner_post(
        "https://scanner.tradingview.com/global/scan?label-product=calendar-ipo",
        payload,
    )

    columns = payload["columns"]
    results = []
    for item in data.get("data", []):
        ticker = item.get("s", "")
        vals   = item.get("d", [])
        d = dict(zip(columns, vals))
        parts = ticker.split(":", 1)
        exch = parts[0] if len(parts) == 2 else ""
        sym  = parts[1] if len(parts) == 2 else ticker

        offer_ts  = d.get("ipo_offer_time")
        offer_dt  = datetime.utcfromtimestamp(offer_ts).strftime("%Y-%m-%d") if offer_ts else None

        results.append(_clean_dict({
            "symbol":           sym,
            "exchange":         exch,
            "name":             d.get("description"),
            "market":           d.get("market"),
            "ipo_date":         offer_dt,
            "offer_price_usd":  d.get("ipo_offer_price_usd"),
            "price_range_usd":  d.get("ipo_price_range_usd"),
            "status":           d.get("ipo_offer_status"),
            "shares_offered":   d.get("ipo_offered_shares"),
            "deal_amount_usd":  d.get("ipo_deal_amount_usd"),
            "market_cap_usd":   d.get("ipo_market_cap_usd"),
        }))
    return results


# ---------------------------------------------------------------------------
# Tool 32: Watchlist bulk data
# ---------------------------------------------------------------------------

@mcp.tool()
def get_watchlist_data(
    tickers: str,
    include_technicals: bool = True,
) -> list[dict]:
    """
    Fetch comprehensive performance, volatility, and technical data for a
    custom list of tickers in a single API call.

    Args:
        tickers: Comma-separated tickers in "EXCHANGE:SYMBOL" format,
                 e.g. "NASDAQ:AAPL,NYSE:TSLA,BINANCE:BTCUSD,TVC:GOLD".
                 Up to 100 tickers.
        include_technicals: If True, also return RSI, AO, CCI20, and
                            momentum alongside performance metrics.

    Returns:
        List of dicts with weekly/monthly/YTD/1Y/3Y/5Y performance,
        betas, volatility, and TV recommendations.
    """
    ticker_list = [t.strip().upper() for t in tickers.split(",") if t.strip()][:100]
    if not ticker_list:
        return []

    columns = [
        "Perf.W", "Perf.1M", "Perf.3M", "Perf.YTD", "Perf.Y",
        "Perf.5Y", "Perf.All",
        "beta_1_year", "beta_3_year", "beta_5_year",
        "Volatility.D", "Volatility.W", "Volatility.M",
        "Recommend.All", "Recommend.MA", "Recommend.Other",
        "close", "change", "volume", "market_cap_basic", "name",
    ]
    if include_technicals:
        columns += ["RSI", "Mom", "AO", "CCI20"]

    payload = {
        "columns": columns,
        "symbols": {"tickers": ticker_list},
    }
    data = _scanner_post(
        "https://scanner.tradingview.com/global/scan?label-product=popup-watchlists",
        payload,
    )

    def _label(v):
        if v is None: return None
        if v >= 0.5:  return "Strong Buy"
        if v >= 0.1:  return "Buy"
        if v >= -0.1: return "Neutral"
        if v >= -0.5: return "Sell"
        return "Strong Sell"

    results = []
    for item in data.get("data", []):
        ticker = item.get("s", "")
        vals   = item.get("d", [])
        d = dict(zip(columns, vals))
        parts = ticker.split(":", 1)
        exch = parts[0] if len(parts) == 2 else ""
        sym  = parts[1] if len(parts) == 2 else ticker
        rec = d.get("Recommend.All")
        row = _clean_dict({
            "symbol":        sym,
            "exchange":      exch,
            "name":          d.get("name"),
            "price":         d.get("close"),
            "change_pct":    d.get("change"),
            "volume":        d.get("volume"),
            "market_cap":    d.get("market_cap_basic"),
            "perf_1w_pct":   d.get("Perf.W"),
            "perf_1m_pct":   d.get("Perf.1M"),
            "perf_3m_pct":   d.get("Perf.3M"),
            "perf_ytd_pct":  d.get("Perf.YTD"),
            "perf_1y_pct":   d.get("Perf.Y"),
            "perf_5y_pct":   d.get("Perf.5Y"),
            "perf_all_pct":  d.get("Perf.All"),
            "beta_1y":       d.get("beta_1_year"),
            "beta_3y":       d.get("beta_3_year"),
            "beta_5y":       d.get("beta_5_year"),
            "volatility_d":  d.get("Volatility.D"),
            "volatility_w":  d.get("Volatility.W"),
            "volatility_m":  d.get("Volatility.M"),
            "rating_value":  rec,
            "rating_label":  _label(rec),
            "ma_rating":     d.get("Recommend.MA"),
            "osc_rating":    d.get("Recommend.Other"),
        })
        if include_technicals:
            row.update(_clean_dict({
                "rsi":      d.get("RSI"),
                "momentum": d.get("Mom"),
                "ao":       d.get("AO"),
                "cci20":    d.get("CCI20"),
            }))
        results.append(row)
    return results


# ---------------------------------------------------------------------------
# Tool 33: Bond info for a company
# ---------------------------------------------------------------------------

@mcp.tool()
def get_bond_info(
    symbol: str,
    exchange: str = "NASDAQ",
    limit: int = 10,
) -> list[dict]:
    """
    Get bonds related to a publicly listed company from TradingView.

    Args:
        symbol: Parent stock ticker (e.g. "AAPL").
        exchange: Exchange of the parent stock (e.g. "NASDAQ").
        limit: Max bonds to return (1–50).

    Returns:
        List of bond dicts with yield_to_maturity and maturity_date.
    """
    full_symbol = f"{exchange.upper()}:{symbol.upper()}"
    payload = {
        "columns": ["yield_to_maturity", "maturity_date", "name", "description"],
        "ignore_unknown_fields": False,
        "range": [0, min(limit, 50)],
        "index_filters": [{"name": "bond_issuer_cr_parent_stock_symbol",
                           "values": [full_symbol]}],
        "preset": "stocks_related_bonds",
    }
    data = _scanner_post(
        "https://scanner.tradingview.com/bond/scan?label-product=right-details",
        payload,
    )
    columns = payload["columns"]
    results = []
    for item in data.get("data", []):
        ticker = item.get("s", "")
        vals   = item.get("d", [])
        d = dict(zip(columns, vals))
        mat_ts = d.get("maturity_date")
        mat_dt = datetime.utcfromtimestamp(mat_ts).strftime("%Y-%m-%d") if mat_ts else None
        results.append(_clean_dict({
            "bond_id":          ticker,
            "name":             d.get("name"),
            "description":      d.get("description"),
            "yield_to_maturity":d.get("yield_to_maturity"),
            "maturity_date":    mat_dt,
        }))
    return results


# ---------------------------------------------------------------------------
# Tool 34: List available forex pairs
# ---------------------------------------------------------------------------

@mcp.tool()
def list_available_pairs(
    base_currency: Optional[str] = None,
    quote_currency: Optional[str] = None,
    exchanges: str = "FX,FX_IDC,OANDA",
    limit: int = 50,
) -> list[dict]:
    """
    List available forex pairs from TradingView.

    Args:
        base_currency: Optional base currency filter (e.g. "EUR", "USD").
        quote_currency: Optional quote currency filter (e.g. "USD", "JPY").
        exchanges: Comma-separated forex exchanges. Default "FX,FX_IDC,OANDA".
        limit: Max results (1–200).

    Returns:
        List of dicts with symbol, name, exchange, price, and daily change.
    """
    if not SCREENER_OK:
        raise RuntimeError("tradingview-screener package not installed.")

    exchange_list = [e.strip().upper() for e in exchanges.split(",") if e.strip()]
    limit = max(1, min(limit, 200))

    q = (
        Query()
        .select("name", "close", "change", "volume", "exchange")
        .where(
            Column("type").isin(["forex"]),
            Column("exchange").isin(exchange_list),
        )
        .order_by("volume", ascending=False)
        .limit(limit)
    )
    if base_currency:
        q = q.where(Column("base_currency") == base_currency.upper())
    if quote_currency:
        q = q.where(Column("currency_code") == quote_currency.upper())

    _, df = q.get_scanner_data()
    if df.empty:
        return []

    df[["exch", "sym"]] = df["ticker"].str.split(":", expand=True)
    results = []
    for _, row in df.iterrows():
        results.append(_clean_dict({
            "symbol":     row.get("sym"),
            "name":       row.get("name"),
            "exchange":   row.get("exch"),
            "price":      row.get("close"),
            "change_pct": row.get("change"),
            "volume":     row.get("volume"),
        }))
    return results


# ---------------------------------------------------------------------------
# MCP Resource: exchanges list
# ---------------------------------------------------------------------------

@mcp.resource("exchanges://list")
def list_exchanges() -> str:
    """
    List all major exchanges and markets supported by the TradingView screener,
    grouped by region and asset class.
    """
    exchanges = {
        "americas_stocks": ["NASDAQ", "NYSE", "AMEX", "TSX", "TSXV", "BMV", "BOVESPA"],
        "europe_stocks": [
            "LSE", "EURONEXT", "XETR", "EPA", "BIT", "BME", "AMS", "SWX",
            "OMX", "WSE", "MOEX", "ATHEX", "PSE", "BVB", "BIST",
        ],
        "asia_pacific_stocks": [
            "TSE", "HKEX", "SSE", "SZSE", "NSE", "BSE", "ASX",
            "KRX", "SGX", "SET", "IDX", "BURSA",
        ],
        "middle_east_africa": ["TADAWUL", "DFSA", "ADX", "EGX", "JSE"],
        "crypto": [
            "BINANCE", "COINBASE", "KRAKEN", "BYBIT", "OKX",
            "BITFINEX", "HUOBI", "KUCOIN", "GEMINI",
        ],
        "forex": ["FX", "FX_IDC", "OANDA", "FXCM"],
        "commodities_futures": ["NYMEX", "COMEX", "CME", "CBOT", "ICE", "MCX"],
        "indices": ["SP500", "DJ", "NASDAQ100", "TVC"],
        "bonds": ["CBOT", "ICE"],
        "funds_etfs": ["NASDAQ", "NYSE", "AMEX", "LSE", "EURONEXT"],
    }
    lines = ["# TradingView Supported Exchanges\n"]
    for category, exch_list in exchanges.items():
        lines.append(f"## {category.replace('_', ' ').title()}")
        lines.append(", ".join(exch_list))
        lines.append("")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Backtesting engine — private helpers
# ---------------------------------------------------------------------------

def _bt_annualization(freq: str) -> float:
    freq = freq.upper()
    if freq in ("1D", "D"):   return math.sqrt(252)
    if freq in ("1W", "W"):   return math.sqrt(52)
    if freq in ("1M", "M"):   return math.sqrt(12)
    if freq == "240":          return math.sqrt(252 * 6.5 / 4)
    if freq == "60":           return math.sqrt(252 * 6.5)
    if freq == "30":           return math.sqrt(252 * 6.5 * 2)
    if freq == "15":           return math.sqrt(252 * 6.5 * 4)
    if freq == "5":            return math.sqrt(252 * 6.5 * 12)
    return math.sqrt(252)


def _bt_rsi(close, period=14):
    import pandas as _pd, numpy as _np
    delta = _np.diff(close, prepend=close[0])
    gain  = _np.where(delta > 0, delta, 0.0)
    loss  = _np.where(delta < 0, -delta, 0.0)
    avg_g = _pd.Series(gain).ewm(alpha=1.0/period, adjust=False).mean().values
    avg_l = _pd.Series(loss).ewm(alpha=1.0/period, adjust=False).mean().values
    rs    = _np.where(avg_l == 0, _np.inf, avg_g / avg_l)
    return 100.0 - (100.0 / (1.0 + rs))


def _bt_macd(close, fast=12, slow=26, signal=9):
    import pandas as _pd
    s = _pd.Series(close)
    macd = (s.ewm(span=fast, adjust=False).mean()
           - s.ewm(span=slow, adjust=False).mean()).values
    sig  = _pd.Series(macd).ewm(span=signal, adjust=False).mean().values
    return macd, sig


def _bt_bb(close, period=20, std_dev=2.0):
    import pandas as _pd
    s   = _pd.Series(close)
    mid = s.rolling(period).mean()
    std = s.rolling(period).std()
    return (mid + std_dev*std).values, mid.values, (mid - std_dev*std).values


def _bt_sma(close, period):
    import pandas as _pd
    return _pd.Series(close).rolling(period).mean().values


def _bt_ema(close, period):
    import pandas as _pd
    return _pd.Series(close).ewm(span=period, adjust=False).mean().values


def _generate_signals(df, indicator: str, params: dict):
    import numpy as _np
    close = df["close"].values.astype(float)
    n = len(close)
    sig = _np.zeros(n, dtype=int)
    ind = indicator.lower().replace("-", "_")

    if ind == "rsi":
        period     = int(params.get("period", 14))
        oversold   = float(params.get("oversold", 30))
        overbought = float(params.get("overbought", 70))
        rsi = _bt_rsi(close, period)
        for i in range(1, n):
            if rsi[i-1] <= oversold  and rsi[i] > oversold:   sig[i] =  1
            if rsi[i-1] >= overbought and rsi[i] < overbought: sig[i] = -1

    elif ind == "macd":
        fast   = int(params.get("fast", 12))
        slow   = int(params.get("slow", 26))
        signal = int(params.get("signal", 9))
        macd, sig_line = _bt_macd(close, fast, slow, signal)
        for i in range(1, n):
            if macd[i-1] <= sig_line[i-1] and macd[i] > sig_line[i]: sig[i] =  1
            if macd[i-1] >= sig_line[i-1] and macd[i] < sig_line[i]: sig[i] = -1

    elif ind in ("bb", "bollinger", "bollinger_bands"):
        period  = int(params.get("period", 20))
        std_dev = float(params.get("std_dev", 2.0))
        upper, mid, lower = _bt_bb(close, period, std_dev)
        for i in range(1, n):
            if _np.isnan(lower[i]) or _np.isnan(upper[i]): continue
            if close[i-1] >= lower[i-1] and close[i] < lower[i]: sig[i] =  1
            if close[i-1] <= upper[i-1] and close[i] > upper[i]: sig[i] = -1

    elif ind in ("ma_cross", "sma_cross"):
        fast_p = int(params.get("fast", 20))
        slow_p = int(params.get("slow", 50))
        fma = _bt_sma(close, fast_p)
        sma = _bt_sma(close, slow_p)
        for i in range(1, n):
            if any(_np.isnan(v) for v in [fma[i-1], fma[i], sma[i-1], sma[i]]): continue
            if fma[i-1] <= sma[i-1] and fma[i] > sma[i]: sig[i] =  1
            if fma[i-1] >= sma[i-1] and fma[i] < sma[i]: sig[i] = -1

    elif ind == "ema_cross":
        fast_p = int(params.get("fast", 9))
        slow_p = int(params.get("slow", 21))
        fma = _bt_ema(close, fast_p)
        sma = _bt_ema(close, slow_p)
        for i in range(1, n):
            if fma[i-1] <= sma[i-1] and fma[i] > sma[i]: sig[i] =  1
            if fma[i-1] >= sma[i-1] and fma[i] < sma[i]: sig[i] = -1

    else:
        raise ValueError(
            f"Unknown indicator '{indicator}'. "
            "Supported: rsi, macd, bb, ma_cross, ema_cross"
        )
    return sig


def _run_backtest(df, signals, freq="1D", initial_capital=10000.0, commission_pct=0.1):
    import numpy as _np
    close  = df["close"].values.astype(float)
    opens  = df["open"].values.astype(float) if "open" in df.columns else close.copy()
    dates  = df["date"].values
    n      = len(close)
    ann    = _bt_annualization(freq)

    capital = initial_capital
    shares  = 0.0
    entry_price = entry_date = None
    trades  = []
    equity  = _np.zeros(n)
    equity[0] = capital

    for i in range(1, n):
        exec_price = opens[i]
        prev_sig   = signals[i-1]

        if prev_sig == 1 and shares == 0.0 and exec_price > 0:
            commission  = capital * commission_pct / 100.0
            shares      = (capital - commission) / exec_price
            entry_price = exec_price
            entry_date  = dates[i]
            capital     = 0.0

        elif prev_sig == -1 and shares > 0.0:
            proceeds   = shares * exec_price
            commission = proceeds * commission_pct / 100.0
            capital    = proceeds - commission
            ret_pct    = (exec_price - entry_price) / entry_price * 100.0
            trades.append({
                "entry_date":  str(entry_date)[:10],
                "exit_date":   str(dates[i])[:10],
                "entry_price": round(float(entry_price), 4),
                "exit_price":  round(float(exec_price), 4),
                "return_pct":  round(ret_pct, 3),
                "outcome":     "win" if ret_pct > 0 else "loss",
            })
            shares = 0.0

        equity[i] = capital + shares * close[i]

    if shares > 0.0:
        exec_price = close[-1]
        proceeds   = shares * exec_price
        commission = proceeds * commission_pct / 100.0
        capital    = proceeds - commission
        ret_pct    = (exec_price - entry_price) / entry_price * 100.0
        trades.append({
            "entry_date":  str(entry_date)[:10],
            "exit_date":   str(dates[-1])[:10],
            "entry_price": round(float(entry_price), 4),
            "exit_price":  round(float(close[-1]), 4),
            "return_pct":  round(ret_pct, 3),
            "outcome":     "win" if ret_pct > 0 else "loss",
            "note":        "open position closed at last bar",
        })
        equity[-1] = capital

    final_capital = float(equity[-1]) if equity[-1] > 0 else capital
    strategy_return = (final_capital - initial_capital) / initial_capital * 100.0
    bh_return       = (close[-1] - close[0]) / close[0] * 100.0
    num_trades = len(trades)
    wins   = [t for t in trades if t["outcome"] == "win"]
    losses = [t for t in trades if t["outcome"] == "loss"]
    win_rate     = len(wins) / num_trades * 100.0 if num_trades else 0.0
    avg_win      = float(_np.mean([t["return_pct"] for t in wins]))   if wins   else 0.0
    avg_loss     = float(_np.mean([t["return_pct"] for t in losses])) if losses else 0.0
    gross_profit = sum(t["return_pct"] for t in wins)
    gross_loss   = abs(sum(t["return_pct"] for t in losses))
    profit_factor = round(gross_profit / gross_loss, 3) if gross_loss > 0 else None

    peak   = _np.maximum.accumulate(equity)
    dd     = _np.where(peak > 0, (equity - peak) / peak * 100.0, 0.0)
    max_dd = float(_np.min(dd))

    eq_ret = _np.diff(equity) / _np.where(equity[:-1] > 0, equity[:-1], 1.0)
    eq_ret = eq_ret[~_np.isnan(eq_ret) & ~_np.isinf(eq_ret)]
    sharpe = (float(_np.mean(eq_ret) / _np.std(eq_ret)) * ann
              if len(eq_ret) > 1 and _np.std(eq_ret) > 0 else None)
    calmar = (strategy_return / abs(max_dd) if max_dd < 0 else None)

    return {
        "strategy_return_pct":    round(strategy_return, 2),
        "buy_and_hold_return_pct":round(bh_return, 2),
        "outperformance_pct":     round(strategy_return - bh_return, 2),
        "num_trades":             num_trades,
        "win_rate_pct":           round(win_rate, 2),
        "avg_win_pct":            round(avg_win, 3),
        "avg_loss_pct":           round(avg_loss, 3),
        "profit_factor":          profit_factor,
        "max_drawdown_pct":       round(max_dd, 2),
        "sharpe_ratio":           round(sharpe, 3) if sharpe is not None else None,
        "calmar_ratio":           round(calmar, 3) if calmar is not None else None,
        "initial_capital":        initial_capital,
        "final_capital":          round(final_capital, 2),
        "trades":                 trades,
    }


# ---------------------------------------------------------------------------
# Tool 35: backtest_indicator
# ---------------------------------------------------------------------------

@mcp.tool()
def backtest_indicator(
    symbol: str,
    exchange: str = "NASDAQ",
    freq: str = "1D",
    bars: int = 500,
    indicator: str = "rsi",
    params: str = "{}",
    initial_capital: float = 10000.0,
    commission_pct: float = 0.1,
) -> dict:
    """
    Backtest a single indicator strategy on historical OHLCV data and compare
    it against a buy-and-hold baseline.

    Args:
        symbol: Ticker (e.g. "AAPL", "BTCUSD").
        exchange: Exchange (e.g. "NASDAQ", "BINANCE").
        freq: Bar frequency — "1D" (daily), "1W" (weekly), "60" (1-hour),
              "240" (4-hour). Default "1D".
        bars: Number of historical bars (min ~50, recommended 300-500).
        indicator: Strategy to test. One of:
                   "rsi"       — RSI mean-reversion
                   "macd"      — MACD line crossover
                   "bb"        — Bollinger Band mean-reversion
                   "ma_cross"  — SMA golden/death cross
                   "ema_cross" — EMA crossover
        params: JSON string of indicator parameters. Defaults per indicator:
                rsi       → {"period": 14, "oversold": 30, "overbought": 70}
                macd      → {"fast": 12, "slow": 26, "signal": 9}
                bb        → {"period": 20, "std_dev": 2.0}
                ma_cross  → {"fast": 20, "slow": 50}
                ema_cross → {"fast": 9, "slow": 21}
        initial_capital: Starting capital in USD. Default 10000.
        commission_pct: Commission as % of trade value. Default 0.1%.

    Returns:
        Dict with strategy_return_pct, buy_and_hold_return_pct,
        outperformance_pct, num_trades, win_rate_pct, avg_win_pct,
        avg_loss_pct, profit_factor, max_drawdown_pct, sharpe_ratio,
        calmar_ratio, final_capital, and a full trades list.
    """
    if not NUMPY_OK:
        raise RuntimeError("numpy/pandas not available.")
    if not HISTORICAL_OK:
        raise RuntimeError("TradingViewHistoricalFetcher unavailable.")

    p  = json.loads(params) if params.strip() not in ("{}", "") else {}
    df = _get_ohlcv_df(symbol, exchange, freq, bars)
    df = df.reset_index(drop=True)
    if "open" not in df.columns:
        df["open"] = df["close"]

    signals = _generate_signals(df, indicator, p)
    result  = _run_backtest(df, signals, freq, initial_capital, commission_pct)
    result.update({"symbol": symbol.upper(), "exchange": exchange.upper(),
                   "freq": freq, "bars_used": len(df),
                   "indicator": indicator, "params": p})
    return result


# ---------------------------------------------------------------------------
# Tool 36: backtest_optimize
# ---------------------------------------------------------------------------

@mcp.tool()
def backtest_optimize(
    symbol: str,
    exchange: str = "NASDAQ",
    freq: str = "1D",
    bars: int = 500,
    indicator: str = "rsi",
    param_grid: str = "{}",
    rank_by: str = "sharpe_ratio",
    top_n: int = 10,
    initial_capital: float = 10000.0,
    commission_pct: float = 0.1,
) -> list[dict]:
    """
    Sweep a grid of indicator parameters and rank all combinations to find
    the best settings for a given symbol and timeframe.

    Args:
        symbol: Ticker (e.g. "AAPL").
        exchange: Exchange (e.g. "NASDAQ").
        freq: Bar frequency. Default "1D".
        bars: Historical bars. Default 500.
        indicator: One of "rsi", "macd", "bb", "ma_cross", "ema_cross".
        param_grid: JSON string mapping each parameter name to a list of values.
                    RSI   → '{"period":[7,14,21],"oversold":[25,30,35],"overbought":[65,70,75]}'
                    MACD  → '{"fast":[8,12,16],"slow":[21,26,30],"signal":[7,9,11]}'
                    BB    → '{"period":[10,15,20,30],"std_dev":[1.5,2.0,2.5]}'
                    MA    → '{"fast":[10,20,50],"slow":[50,100,200]}'
                    EMA   → '{"fast":[5,9,12],"slow":[21,26,50]}'
        rank_by: Metric to sort by: "sharpe_ratio", "strategy_return_pct",
                 "profit_factor", "calmar_ratio", "win_rate_pct". Default "sharpe_ratio".
        top_n: How many top combinations to return. Default 10.
        initial_capital: Starting capital. Default 10000.
        commission_pct: Commission % per trade. Default 0.1.

    Returns:
        List of top_n dicts with params + all performance metrics, best-first.
    """
    if not NUMPY_OK:
        raise RuntimeError("numpy/pandas not available.")
    if not HISTORICAL_OK:
        raise RuntimeError("TradingViewHistoricalFetcher unavailable.")

    grid = json.loads(param_grid) if param_grid.strip() not in ("{}", "") else {}
    if not grid:
        raise ValueError(
            "param_grid must contain at least one parameter list. "
            'Example: \'{"period": [7, 14, 21], "oversold": [25, 30]}\''
        )

    df = _get_ohlcv_df(symbol, exchange, freq, bars)
    df = df.reset_index(drop=True)
    if "open" not in df.columns:
        df["open"] = df["close"]

    param_names  = list(grid.keys())
    combinations = list(itertools.product(*grid.values()))

    results = []
    for combo in combinations:
        p = dict(zip(param_names, combo))
        try:
            signals = _generate_signals(df, indicator, p)
            metrics = _run_backtest(df, signals, freq, initial_capital, commission_pct)
            row = {k: v for k, v in metrics.items() if k != "trades"}
            row["params"] = p
            results.append(row)
        except Exception:
            pass

    results.sort(key=lambda r: (r.get(rank_by) is None, -(r.get(rank_by) or 0)))
    top = results[:top_n]
    for i, r in enumerate(top):
        r["rank"] = i + 1
    return top


# ---------------------------------------------------------------------------
# Tool 37: backtest_compare
# ---------------------------------------------------------------------------

@mcp.tool()
def backtest_compare(
    symbols: str,
    exchange: str = "NASDAQ",
    freq: str = "1D",
    bars: int = 500,
    indicator: str = "rsi",
    params: str = "{}",
    initial_capital: float = 10000.0,
    commission_pct: float = 0.1,
) -> list[dict]:
    """
    Run the same indicator strategy across multiple symbols side-by-side
    to find which assets respond best to a given strategy.

    Args:
        symbols: Comma-separated tickers, e.g. "AAPL,MSFT,NVDA,GOOGL".
                 For crypto: "BTCUSD,ETHUSD,SOLUSD". Max 20 symbols.
        exchange: Exchange applied to all symbols. Default "NASDAQ".
        freq: Bar frequency. Default "1D".
        bars: Historical bars. Default 500.
        indicator: One of "rsi", "macd", "bb", "ma_cross", "ema_cross".
        params: JSON string of indicator parameters (applied to all symbols).
        initial_capital: Starting capital per symbol. Default 10000.
        commission_pct: Commission % per trade. Default 0.1.

    Returns:
        List sorted by strategy_return_pct descending, each entry has
        symbol, all performance metrics, and the full trades list.
    """
    if not NUMPY_OK:
        raise RuntimeError("numpy/pandas not available.")
    if not HISTORICAL_OK:
        raise RuntimeError("TradingViewHistoricalFetcher unavailable.")

    symbol_list = [s.strip().upper() for s in symbols.split(",") if s.strip()][:20]
    if not symbol_list:
        raise ValueError("Provide at least one symbol.")

    p = json.loads(params) if params.strip() not in ("{}", "") else {}
    results = []
    for sym in symbol_list:
        try:
            df = _get_ohlcv_df(sym, exchange, freq, bars)
            df = df.reset_index(drop=True)
            if "open" not in df.columns:
                df["open"] = df["close"]
            signals = _generate_signals(df, indicator, p)
            metrics = _run_backtest(df, signals, freq, initial_capital, commission_pct)
            metrics.update({"symbol": sym, "exchange": exchange.upper(),
                            "freq": freq, "bars_used": len(df),
                            "indicator": indicator, "params": p})
            results.append(metrics)
        except Exception as e:
            results.append({"symbol": sym, "exchange": exchange.upper(), "error": str(e)})

    results.sort(key=lambda r: ("error" in r, -(r.get("strategy_return_pct") or float("-inf"))))
    return results


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    mcp.run(transport="stdio")
