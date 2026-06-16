"""
TradingView Historical Data Fetcher
Uses TradingView's WebSocket API to fetch historical OHLCV data
This bypasses Yahoo Finance rate limiting issues
"""
import json
import time
import random
import string
import logging
import pandas as pd
from typing import List, Optional, Tuple
from datetime import datetime
from websocket import create_connection

logger = logging.getLogger(__name__)


class TradingViewHistoricalFetcher:
    """
    Fetch historical price data from TradingView using WebSocket API
    """
    
    WEBSOCKET = True
    CHART = False
    
    headers = json.dumps({
        "Origin": "https://data.tradingview.com",
        "user-agent": "Mozilla/5.0 (compatible; TradingBot/1.0)"
    })
    
    skip_messages = ["series_loading", "series_completed", "quote_completed", "du"]
    
    def __init__(self):
        self.request_version = 1
        self.websocket_session = self.generate_session(self.WEBSOCKET)
        self.chart_session = self.generate_session(self.CHART)
        self.ws = None
    
    def generate_session(self, session: bool) -> str:
        """
        Generate a random session ID
        
        Args:
            session: True for websocket, False for chart
        
        Returns:
            Session ID string
        """
        string_length = 12
        letters = string.ascii_lowercase
        random_string = "".join(
            random.choice(letters) for i in range(string_length)
        )
        prefix = "qs" if session else "cs"
        return f"{prefix}_{random_string}"
    
    def prepend_header(self, sentences: str) -> str:
        """
        Format data into websocket message format
        
        Args:
            sentences: Message content
        
        Returns:
            Formatted message with header
        """
        return f"~m~{len(sentences)}~m~{sentences}"
    
    def construct_message(self, function_name: str, parameters: List[str]) -> str:
        """
        Construct a message in JSON format
        
        Args:
            function_name: Function to call
            parameters: List of parameters
        
        Returns:
            JSON formatted message
        """
        return json.dumps(
            {"m": function_name, "p": parameters}, separators=(",", ":")
        )
    
    def create_message(self, function_name: str, parameters: List[str]) -> str:
        """
        Create a complete websocket message
        
        Args:
            function_name: Function to call
            parameters: List of parameters
        
        Returns:
            Complete formatted message
        """
        return self.prepend_header(
            self.construct_message(function_name, parameters)
        )
    
    def send_message(self, func: str, args: List[str]) -> int:
        """
        Send a message through the websocket
        
        Args:
            func: Function name
            args: Arguments list
        
        Returns:
            Number of bytes sent
        """
        if not self.ws:
            raise Exception("WebSocket not connected")
        
        return self.ws.send(self.create_message(func, args))
    
    def receive_response(self) -> Tuple[Optional[dict], Optional[dict], Optional[pd.DataFrame]]:
        """
        Receive and parse response from websocket
        
        Returns:
            Tuple of (session_info, series_description, data)
        """
        qsd_dicts = []
        loading = True
        session_info = None
        series_description = None
        data = None
        
        while loading:
            try:
                res = self.ws.recv()
                
                for r in res.split("~m~"):
                    try:
                        r = json.loads(r)
                    except json.JSONDecodeError:
                        continue
                    
                    if not isinstance(r, dict):
                        continue
                    
                    if "session_id" in r:
                        session_info = r
                        continue
                    elif "m" in r:
                        message = r.get("m")
                        
                        # Break the loop if qsd starts to repeat
                        if message == "qsd":
                            if r in qsd_dicts:
                                loading = False
                                break
                            else:
                                qsd_dicts.append(r)
                        elif message in self.skip_messages:
                            continue
                        elif message == "symbol_resolved":
                            series_description = r['p'][2]
                        elif message == "timescale_update":
                            # Extract OHLCV data
                            data = pd.DataFrame(r['p'][1]['sds_1']['s'])['v'].apply(pd.Series)
                            data.columns = ['date', 'open', 'high', 'low', 'close', 'volume']
                            loading = False
                        elif (message == "symbol_error") or (message == "series_error"):
                            logger.warning(f"Symbol error or series error received")
                            loading = False
                            break
                        else:
                            logger.debug(f"Unhandled message: {message}")
            
            except Exception as e:
                logger.error(f"Error receiving response: {str(e)}")
                loading = False
                break
        
        return session_info, series_description, data
    
    def get_historical_data(
        self,
        symbol: str,
        exchange: str = "NASDAQ",
        freq: str = "1D",
        number_of_values: int = 365
    ) -> Optional[pd.DataFrame]:
        """
        Get historical data for a single symbol
        
        Args:
            symbol: Stock symbol
            exchange: Exchange name (NASDAQ, NYSE, etc.)
            freq: Frequency (1D for daily, 1W for weekly, 1M for monthly)
            number_of_values: Number of data points to fetch
        
        Returns:
            DataFrame with OHLCV data or None if failed
        """
        try:
            # Create websocket connection
            self.ws = create_connection(
                "wss://data.tradingview.com/socket.io/websocket",
                headers=self.headers,
                timeout=30
            )
            
            # Login and create sessions
            self.send_message("set_auth_token", ["unauthorized_user_token"])
            self.send_message("set_locale", ["en", "US"])
            self.send_message("chart_create_session", [self.chart_session, ""])
            self.send_message("quote_create_session", [self.websocket_session])
            
            # Format symbol with exchange
            full_symbol = f"{exchange}:{symbol}"
            resolve_symbol = json.dumps({"symbol": full_symbol, "adjustment": "splits"})
            
            # Request symbol data
            self.send_message("quote_add_symbols", [self.websocket_session, f"={resolve_symbol}"])
            self.send_message("resolve_symbol", [self.chart_session, "sds_sym_1", f"={resolve_symbol}"])
            self.send_message("create_series", [
                self.chart_session,
                "sds_1",
                f"s{self.request_version}",
                "sds_sym_1",
                freq,
                number_of_values,
                ""
            ])
            
            self.request_version += 1
            
            # Receive response
            session_info, series_description, data = self.receive_response()
            
            # Close websocket
            if self.ws:
                self.ws.close()
                self.ws = None
            
            if data is not None and not data.empty:
                # Convert timestamp to datetime
                data['date'] = data['date'].apply(lambda x: datetime.utcfromtimestamp(x))
                data['symbol'] = symbol
                
                # Add metadata if available
                if series_description:
                    if 'country' in series_description:
                        data['country'] = series_description['country']
                    if 'type' in series_description:
                        data['type'] = series_description['type']
                
                logger.info(f"Successfully fetched {len(data)} records for {symbol}")
                return data
            else:
                logger.warning(f"No data received for {symbol}")
                return None
        
        except Exception as e:
            logger.error(f"Error fetching historical data for {symbol}: {str(e)}")
            if self.ws:
                try:
                    self.ws.close()
                except:
                    pass
                self.ws = None
            return None
    
    def get_historical_data_batch(
        self,
        symbols: List[str],
        exchange: str = "NASDAQ",
        freq: str = "1D",
        number_of_values: int = 365,
        sleep_time: float = 2.0
    ) -> pd.DataFrame:
        """
        Get historical data for multiple symbols
        
        Args:
            symbols: List of stock symbols
            exchange: Exchange name
            freq: Frequency
            number_of_values: Number of data points
            sleep_time: Delay between requests (seconds)
        
        Returns:
            Combined DataFrame with all symbols' data
        """
        dataset = []
        
        for symbol in symbols:
            logger.info(f"Fetching historical data for {symbol}...")
            
            data = self.get_historical_data(
                symbol=symbol,
                exchange=exchange,
                freq=freq,
                number_of_values=number_of_values
            )
            
            if data is not None:
                dataset.append(data)
            
            # Sleep to avoid rate limiting
            time.sleep(sleep_time)
        
        if len(dataset) > 0:
            combined_data = pd.concat(dataset, ignore_index=True)
            logger.info(f"Successfully fetched data for {len(dataset)}/{len(symbols)} symbols")
            return combined_data
        else:
            logger.warning("No data fetched for any symbols")
            return pd.DataFrame()
    
    def infer_exchange(self, symbol: str) -> str:
        """
        Infer the exchange for a symbol based on common patterns
        
        Args:
            symbol: Stock symbol
        
        Returns:
            Exchange name
        """
        # Common exchange mappings
        if symbol.endswith('.L'):
            return 'LSE'
        elif symbol.endswith('.DE'):
            return 'XETRA'
        elif symbol.endswith('.PA'):
            return 'EURONEXT'
        elif symbol.endswith('.HK'):
            return 'HKEX'
        elif symbol.endswith('.T'):
            return 'TSE'
        else:
            # Default to NASDAQ for US stocks
            return 'NASDAQ'
