# TradingView MCP Server

A [Model Context Protocol](https://modelcontextprotocol.io) server that exposes **41 TradingView market-data tools** to Claude and other AI agents — no rate limiting, no API key required.

## Project structure

```
tradingview-mcp/
├── mcp_server.py                      # MCP server — 41 tools
├── tradingview_historical_fetcher.py  # Historical OHLCV helper
├── requirements.txt
└── README.md
```

## Quick start

```bash
pip install -r requirements.txt
python mcp_server.py
```

Add it to your Claude Code config (`.mcp.json`):

```json
{
  "tradingview": {
    "type": "stdio",
    "command": "python",
    "args": ["/path/to/mcp_server.py"]
  }
}
```

## Environment variables

Create a `.env` file for optional features:

```bash
# Required only for get_watchlist_tickers
# Copy from browser cookies while logged into tradingview.com
TV_SESSION_ID=your_sessionid_cookie
TV_SESSION_SIGN=your_sessionid_sign_cookie
```

## Available tools (41)

#### Screening & prices
| Tool | Description |
|------|-------------|
| `screen_stocks` | Bulk screener — filter by exchange, sector, price, market cap |
| `get_price` | Current price for a single ticker |
| `get_multiple_prices` | Prices for up to 100 tickers in one call |
| `top_gainers` / `top_losers` | Biggest movers for the day |
| `rating_filter` | Screen by TV recommendation (Buy/Sell/Neutral) |
| `search_symbol` | Search for a ticker by name or keyword |
| `get_symbol_info` | Company description, sector, country, exchange |

#### Technical indicators
| Tool | Description |
|------|-------------|
| `get_technical_analysis` | Full TA summary (RSI, MACD, MAs, oscillators, signal) |
| `coin_analysis` | Same as above, optimised for crypto |
| `get_bollinger_bands` | BB upper/middle/lower, width, %B, squeeze flag |
| `get_macd` | MACD line, signal, histogram |
| `get_moving_averages` | SMA/EMA across 5, 10, 20, 50, 100, 200 periods |
| `get_stochastic` | %K, %D, crossover signal |
| `get_adx` | ADX, +DI, -DI, trend strength |
| `get_ichimoku_cloud` | Tenkan, Kijun, Senkou A/B, Chikou |
| `get_atr` | Average True Range |
| `get_vwap` | Volume-Weighted Average Price |

#### Chart patterns & structure
| Tool | Description |
|------|-------------|
| `get_fibonacci_retracement` | Fib levels from swing high/low |
| `get_support_resistance` | Key S/R levels via pivot clustering |
| `get_pivot_points` | Classic, Camarilla, Woodie pivots |
| `get_volume_profile` | Volume by price bucket (POC, VAH, VAL) |
| `detect_unfilled_gaps` | Open price gaps on the chart |
| `consecutive_candles_scan` | Scan for N consecutive bullish/bearish candles |
| `advanced_candle_pattern` | Doji, engulfing, hammer, shooting star, etc. |
| `bollinger_scan` | Screen stocks near BB extremes |

#### Historical data & correlation
| Tool | Description |
|------|-------------|
| `get_historical_ohlcv` | OHLCV candles (1m → 1M) |
| `calculate_correlation` | Pearson correlation between two symbols |
| `list_available_pairs` | Browse forex pairs by base/quote currency |

#### Options & volatility
| Tool | Description |
|------|-------------|
| `get_iv_smile` | Implied-volatility smile by strike or delta for a specific expiry |
| `get_iv_term_structure` | IV at standard tenors (1w/2w/1m/2m/3m/6m/9m/1y) and real expiries |

#### Watchlists
| Tool | Description |
|------|-------------|
| `get_watchlist_data` | Performance, volatility, RSI, rating for an explicit ticker list |
| `get_watchlist_tickers` | Fetch tickers from a TradingView watchlist by its numeric ID |

#### Calendars & bonds
| Tool | Description |
|------|-------------|
| `get_earnings_calendar` | Upcoming earnings dates and EPS estimates |
| `get_ipo_calendar` | Scheduled IPOs |
| `get_bond_info` | Corporate bonds related to a stock (yield, maturity) |

#### News
| Tool | Description |
|------|-------------|
| `get_symbol_news` | Latest news articles for a ticker with cursor pagination |

#### Backtesting
| Tool | Description |
|------|-------------|
| `backtest_indicator` | Run a single strategy on a symbol and get P&L metrics |
| `backtest_optimize` | Grid-search indicator parameters for best return |
| `backtest_compare` | Compare strategies across multiple symbols |

#### Portfolio
| Tool | Description |
|------|-------------|
| `get_portfolio_stocks` | Read a local SQLite portfolio database |

## Tech stack

- Python 3.9+, [FastMCP](https://github.com/jlowin/fastmcp)
- TradingView public APIs (scanner, options-charting, news-mediator)
- Pandas, NumPy (backtesting engine)

## Disclaimer

For educational and informational purposes only. Not financial advice.
