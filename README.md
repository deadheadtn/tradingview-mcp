# TradingView MCP Server

A Model Context Protocol (MCP) server that exposes TradingView market data to Claude and other MCP-compatible AI clients. No API key required.

## Features

### Market Screening
| Tool | Description |
|------|-------------|
| `screen_stocks` | Bulk screener with fundamentals, sector, price, and market cap filters |
| `top_gainers` | Top performing assets for the current session |
| `top_losers` | Biggest declining assets for the current session |
| `bollinger_scan` | Find assets in BB squeeze, upper-touch, or lower-touch conditions |
| `rating_filter` | Filter by TradingView's combined oscillator + MA rating |
| `consecutive_candles_scan` | Scan for momentum across daily/weekly/monthly timeframes |

### Technical Analysis
| Tool | Description |
|------|-------------|
| `get_technical_analysis` | Full TA for a single symbol (RSI, MACD, BB, MAs, ADX, Stoch, Ichimoku) |
| `coin_analysis` | Full TA optimised for crypto pairs |
| `advanced_candle_pattern` | Multi-timeframe alignment analysis across custom intervals |
| `get_bollinger_bands` | BB upper/lower/middle, width %, %B, squeeze flag |
| `get_macd` | MACD line, signal, histogram, crossover direction |
| `get_moving_averages` | SMA/EMA 10–200, price-above-MA flags, signal counts |
| `get_stochastic` | %K, %D, overbought/oversold zone, crossover |
| `get_adx` | ADX trend strength + DI+/DI- directional indicators |
| `get_ichimoku_cloud` | Tenkan/Kijun/Span A/Span B, price vs cloud position |

### Price & Market Data
| Tool | Description |
|------|-------------|
| `get_price` | Current price and basic stats for a single symbol |
| `get_multiple_prices` | Batch price quotes for arbitrary ticker lists |
| `get_watchlist_data` | Bulk performance, volatility, beta, and technicals for any tickers |
| `get_symbol_info` | Extended symbol info: 52w range, all performance periods, fundamentals |
| `search_symbol` | Search for symbols by ticker or company name |

### Price Levels & Computed Indicators
| Tool | Description |
|------|-------------|
| `get_fibonacci_retracement` | Swing high/low → all fib levels + extensions |
| `get_support_resistance` | Auto-detected S/R zones with touch-count strength |
| `get_pivot_points` | Classic, Fibonacci, and Camarilla pivot points |
| `get_atr` | ATR with volatility label (very_low → very_high) |
| `get_vwap` | VWAP with price position vs average |
| `get_volume_profile` | POC, value area, full price-bin volume distribution |
| `detect_unfilled_gaps` | Bullish/bearish price gaps with fill status |
| `calculate_correlation` | Pearson + Spearman + 30-bar rolling correlation |

### Calendar & Events
| Tool | Description |
|------|-------------|
| `get_earnings_calendar` | Upcoming earnings with EPS/revenue actuals, estimates, and surprises |
| `get_ipo_calendar` | Upcoming and recent IPOs with offer price, deal size, market cap |
| `get_bond_info` | Bonds related to a listed company (yield, maturity) |

### Historical Data
| Tool | Description |
|------|-------------|
| `get_historical_ohlcv` | OHLCV candlestick data via TradingView WebSocket (no API key) |

### Other
| Tool | Description |
|------|-------------|
| `list_available_pairs` | List forex pairs with base/quote currency filters |
| `exchanges://list` | MCP resource listing all supported exchanges by region/asset class |

## Installation

```bash
pip install -r requirements.txt
```

## Usage

### Standalone

```bash
python mcp_server.py
```

### With Claude Desktop

Add to your `claude_desktop_config.json`:

```json
{
  "mcpServers": {
    "tradingview": {
      "command": "python",
      "args": ["/path/to/tradingview-mcp/mcp_server.py"]
    }
  }
}
```

### With Claude Code (`.mcp.json`)

```json
{
  "mcpServers": {
    "tradingview": {
      "type": "stdio",
      "command": "python",
      "args": ["/path/to/tradingview-mcp/mcp_server.py"]
    }
  }
}
```

## Data Sources

- **Screener data** — `tradingview-screener` Python package (scanner.tradingview.com)
- **Technical analysis** — `tradingview-ta` Python package
- **Historical OHLCV** — TradingView WebSocket API (`wss://data.tradingview.com`)
- **Events & symbol info** — TradingView Scanner REST API (public endpoints)

All endpoints are public and require no authentication.

## Dependencies

- `mcp` — Model Context Protocol SDK
- `tradingview-screener` — TradingView screener wrapper
- `tradingview-ta` — TradingView technical analysis
- `websocket-client` — WebSocket for historical data
- `requests`, `pandas`, `numpy`, `scipy`
