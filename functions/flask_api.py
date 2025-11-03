"""
COMPLETE Flask API v7.0 - PYTHONANYWHERE PRODUCTION READY
✅ Full Logging & Monitoring
✅ Error Tracking
✅ Performance Metrics
✅ Database Monitoring
"""

from flask import Flask, jsonify, request, g
from flask_cors import CORS
import requests
import numpy as np
from datetime import datetime, timedelta
import logging
from logging.handlers import RotatingFileHandler, TimedRotatingFileHandler
import sqlite3
from typing import Dict
from scipy.ndimage import gaussian_filter1d
import yfinance as yf
import sys
import os
import traceback
import time
from functools import wraps
import json


# ============= ENHANCED LOGGING SYSTEM =============

def setup_logging():
    """Setup comprehensive logging system"""
    
    # Create logs directory if not exists
    log_dir = 'logs'
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
    
    # Main application logger
    logger = logging.getLogger('flask_api')
    logger.setLevel(logging.DEBUG)
    
    # Remove existing handlers
    logger.handlers = []
    
    # Format for log messages
    detailed_format = logging.Formatter(
        '%(asctime)s | %(levelname)-8s | %(name)s | %(funcName)s:%(lineno)d | %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    
    simple_format = logging.Formatter(
        '%(asctime)s | %(levelname)s | %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    
    # 1. Console Handler (INFO and above)
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(simple_format)
    logger.addHandler(console_handler)
    
    # 2. Main Application Log (Rotating by size - 10MB)
    app_handler = RotatingFileHandler(
        f'{log_dir}/flask_api.log',
        maxBytes=10*1024*1024,  # 10MB
        backupCount=5
    )
    app_handler.setLevel(logging.INFO)
    app_handler.setFormatter(detailed_format)
    logger.addHandler(app_handler)
    
    # 3. Error Log (Only errors - Rotating by size)
    error_handler = RotatingFileHandler(
        f'{log_dir}/errors.log',
        maxBytes=10*1024*1024,
        backupCount=5
    )
    error_handler.setLevel(logging.ERROR)
    error_handler.setFormatter(detailed_format)
    logger.addHandler(error_handler)
    
    # 4. Daily Activity Log (Rotating daily)
    daily_handler = TimedRotatingFileHandler(
        f'{log_dir}/daily_activity.log',
        when='midnight',
        interval=1,
        backupCount=30  # Keep 30 days
    )
    daily_handler.setLevel(logging.INFO)
    daily_handler.setFormatter(simple_format)
    logger.addHandler(daily_handler)
    
    # 5. Performance Log
    perf_handler = RotatingFileHandler(
        f'{log_dir}/performance.log',
        maxBytes=10*1024*1024,
        backupCount=3
    )
    perf_handler.setLevel(logging.DEBUG)
    perf_format = logging.Formatter(
        '%(asctime)s | %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    perf_handler.setFormatter(perf_format)
    
    perf_logger = logging.getLogger('performance')
    perf_logger.setLevel(logging.DEBUG)
    perf_logger.addHandler(perf_handler)
    
    # 6. API Request Log
    request_handler = RotatingFileHandler(
        f'{log_dir}/api_requests.log',
        maxBytes=10*1024*1024,
        backupCount=5
    )
    request_handler.setLevel(logging.INFO)
    request_handler.setFormatter(simple_format)
    
    request_logger = logging.getLogger('api_requests')
    request_logger.setLevel(logging.INFO)
    request_logger.addHandler(request_handler)
    
    return logger


# Initialize logging
logger = setup_logging()
perf_logger = logging.getLogger('performance')
request_logger = logging.getLogger('api_requests')


# ============= MONITORING DECORATORS =============

def log_performance(func):
    """Decorator to log function performance"""
    @wraps(func)
    def wrapper(*args, **kwargs):
        start_time = time.time()
        try:
            result = func(*args, **kwargs)
            execution_time = time.time() - start_time
            perf_logger.info(f"{func.__name__} | SUCCESS | {execution_time:.3f}s")
            return result
        except Exception as e:
            execution_time = time.time() - start_time
            perf_logger.error(f"{func.__name__} | ERROR | {execution_time:.3f}s | {str(e)}")
            raise
    return wrapper


def log_api_request(func):
    """Decorator to log API requests"""
    @wraps(func)
    def wrapper(*args, **kwargs):
        start_time = time.time()
        
        # Log request details
        request_logger.info(
            f"REQUEST | {request.method} | {request.path} | "
            f"IP: {request.remote_addr} | "
            f"Args: {dict(request.args)}"
        )
        
        try:
            result = func(*args, **kwargs)
            execution_time = time.time() - start_time
            
            # Log response
            status_code = result[1] if isinstance(result, tuple) else 200
            request_logger.info(
                f"RESPONSE | {request.method} | {request.path} | "
                f"Status: {status_code} | Time: {execution_time:.3f}s"
            )
            
            return result
            
        except Exception as e:
            execution_time = time.time() - start_time
            request_logger.error(
                f"ERROR | {request.method} | {request.path} | "
                f"Time: {execution_time:.3f}s | Error: {str(e)}"
            )
            raise
    
    return wrapper


# ============= FLASK APP INITIALIZATION =============

app = Flask(__name__)
CORS(app)

# Log app startup
logger.info("="*80)
logger.info("🚀 Flask API v7.0 - PYTHONANYWHERE PRODUCTION")
logger.info("="*80)


# ============= REQUEST/RESPONSE LOGGING =============

@app.before_request
def before_request():
    """Log before each request"""
    g.start_time = time.time()
    logger.debug(f"→ {request.method} {request.path} from {request.remote_addr}")


@app.after_request
def after_request(response):
    """Log after each request"""
    if hasattr(g, 'start_time'):
        execution_time = time.time() - g.start_time
        logger.debug(
            f"← {request.method} {request.path} | "
            f"Status: {response.status_code} | "
            f"Time: {execution_time:.3f}s"
        )
    
    # Add custom headers for monitoring
    response.headers['X-Response-Time'] = f"{execution_time:.3f}s" if hasattr(g, 'start_time') else "N/A"
    response.headers['X-API-Version'] = "7.0"
    
    return response


@app.errorhandler(Exception)
def handle_exception(e):
    """Global exception handler"""
    logger.error(
        f"UNHANDLED EXCEPTION | {request.method} {request.path} | "
        f"Error: {str(e)}\n"
        f"Traceback: {traceback.format_exc()}"
    )
    
    return jsonify({
        'error': 'Internal server error',
        'message': str(e),
        'timestamp': datetime.now().isoformat()
    }), 500


# ============= API CONFIGURATION =============

BINANCE_API = "https://api.binance.com/api/v3/klines"
BINANCE_FUNDING = "https://fapi.binance.com/fapi/v1/fundingRate"
BINANCE_OI = "https://fapi.binance.com/fapi/v1/openInterest"
BINANCE_ORDERBOOK = "https://api.binance.com/api/v3/depth"
FGI_API = "https://api.alternative.me/fng/?limit=1"
COINGECKO_API = "https://api.coingecko.com/api/v3/global"
USDT_API = "https://api.coingecko.com/api/v3/coins/tether"


# ============= ENHANCED DATABASE WITH MONITORING =============

class TradeDatabase:
    def __init__(self):
        self.db_path = 'trades.db'
        self.conn = sqlite3.connect(self.db_path, check_same_thread=False)
        self.init_db()
        logger.info(f"✅ Database initialized: {self.db_path}")
    
    def init_db(self):
        """Initialize database with monitoring"""
        try:
            c = self.conn.cursor()
            c.execute('''CREATE TABLE IF NOT EXISTS trades
                         (id INTEGER PRIMARY KEY AUTOINCREMENT,
                          timestamp TEXT, symbol TEXT, timeframe TEXT,
                          signal_type TEXT, entry_price REAL, stop_loss REAL,
                          targets TEXT, confidence REAL, bullish_score REAL,
                          bearish_score REAL, status TEXT DEFAULT 'ACTIVE',
                          exit_price REAL, pnl REAL, notes TEXT)''')
            self.conn.commit()
            logger.info("✅ Database tables initialized")
        except Exception as e:
            logger.error(f"Database initialization error: {e}")
            raise
    
    @log_performance
    def save_trade(self, data: Dict):
        """Save trade with logging"""
        try:
            c = self.conn.cursor()
            c.execute('''INSERT INTO trades 
                         (timestamp, symbol, timeframe, signal_type, entry_price, 
                          stop_loss, targets, confidence, bullish_score, bearish_score, notes)
                         VALUES (?,?,?,?,?,?,?,?,?,?,?)''',
                      (data['timestamp'], data['symbol'], data['timeframe'],
                       data['signal_type'], data['entry'], data['sl'],
                       str(data['targets']), data['confidence'],
                       data.get('bullish_score', 0), data.get('bearish_score', 0),
                       data.get('notes', '')))
            self.conn.commit()
            trade_id = c.lastrowid
            logger.info(f"✅ Trade saved: ID={trade_id} | {data['symbol']} | {data['signal_type']}")
            return trade_id
        except Exception as e:
            logger.error(f"Error saving trade: {e}")
            raise
    
    @log_performance
    def get_all_trades(self, limit=100, status=None):
        """Get trades with logging"""
        try:
            c = self.conn.cursor()
            if status:
                c.execute('SELECT * FROM trades WHERE status=? ORDER BY id DESC LIMIT ?', (status, limit))
            else:
                c.execute('SELECT * FROM trades ORDER BY id DESC LIMIT ?', (limit,))
            
            columns = ['id', 'timestamp', 'symbol', 'timeframe', 'signal_type',
                       'entry_price', 'stop_loss', 'targets', 'confidence',
                       'bullish_score', 'bearish_score', 'status', 'exit_price', 'pnl', 'notes']
            trades = [dict(zip(columns, row)) for row in c.fetchall()]
            logger.debug(f"Retrieved {len(trades)} trades from database")
            return trades
        except Exception as e:
            logger.error(f"Error retrieving trades: {e}")
            return []
    
    @log_performance
    def update_trade(self, trade_id: int, exit_price: float, pnl: float, status: str):
        """Update trade with logging"""
        try:
            c = self.conn.cursor()
            c.execute('UPDATE trades SET exit_price=?, pnl=?, status=? WHERE id=?',
                      (exit_price, pnl, status, trade_id))
            self.conn.commit()
            logger.info(f"✅ Trade updated: ID={trade_id} | Status={status} | PnL={pnl:.2f}")
        except Exception as e:
            logger.error(f"Error updating trade {trade_id}: {e}")
            raise


db = TradeDatabase()


# ============= TECHNICAL INDICATORS =============

def calc_rsi(prices, period=14):
    """Calculate RSI"""
    try:
        deltas = np.diff(prices)
        gains = np.where(deltas > 0, deltas, 0)
        losses = np.where(deltas < 0, -deltas, 0)
        avg_gain = np.mean(gains[-period:])
        avg_loss = np.mean(losses[-period:])
        if avg_loss == 0:
            return 100
        rs = avg_gain / avg_loss
        return round(100 - (100 / (1 + rs)), 2)
    except Exception as e:
        logger.error(f"RSI calculation error: {e}")
        return 50


def calc_ema(prices, period):
    """Calculate EMA"""
    try:
        multiplier = 2 / (period + 1)
        ema = [prices[0]]
        for price in prices[1:]:
            ema.append((price - ema[-1]) * multiplier + ema[-1])
        return ema
    except Exception as e:
        logger.error(f"EMA calculation error: {e}")
        return prices


def calc_sma(prices, period):
    """Calculate SMA"""
    try:
        if len(prices) < period:
            return prices[-1]
        return sum(prices[-period:]) / period
    except Exception as e:
        logger.error(f"SMA calculation error: {e}")
        return prices[-1]


def calc_macd(prices):
    """Calculate MACD"""
    try:
        ema_12 = calc_ema(prices, 12)
        ema_26 = calc_ema(prices, 26)
        macd_line = [ema_12[i] - ema_26[i] for i in range(len(prices))]
        signal_line = calc_ema(macd_line, 9)
        return macd_line[-1], signal_line[-1]
    except Exception as e:
        logger.error(f"MACD calculation error: {e}")
        return 0, 0


def calc_atr(highs, lows, closes, period=14):
    """Calculate ATR"""
    try:
        tr_list = []
        for i in range(1, len(closes)):
            hl = highs[i] - lows[i]
            hc = abs(highs[i] - closes[i-1])
            lc = abs(lows[i] - closes[i-1])
            tr = max(hl, hc, lc)
            tr_list.append(tr)
        return sum(tr_list[-period:]) / period if tr_list else 0
    except Exception as e:
        logger.error(f"ATR calculation error: {e}")
        return 0


def calc_supertrend(highs, lows, closes, period=10, multiplier=3.0):
    """Calculate Supertrend"""
    try:
        atr = calc_atr(highs, lows, closes, period)
        hl_avg = [(highs[i] + lows[i]) / 2 for i in range(len(closes))]
        upper_band = [hl_avg[i] + (multiplier * atr) for i in range(len(closes))]
        lower_band = [hl_avg[i] - (multiplier * atr) for i in range(len(closes))]
        
        supertrend = [upper_band[0]]
        direction = [1]
        
        for i in range(1, len(closes)):
            if closes[i] > supertrend[i-1]:
                direction.append(1)
                supertrend.append(lower_band[i])
            elif closes[i] < supertrend[i-1]:
                direction.append(-1)
                supertrend.append(upper_band[i])
            else:
                direction.append(direction[i-1])
                if direction[i] == 1:
                    supertrend.append(max(lower_band[i], supertrend[i-1]))
                else:
                    supertrend.append(min(upper_band[i], supertrend[i-1]))
        
        return supertrend[-1], direction[-1]
    except Exception as e:
        logger.error(f"Supertrend calculation error: {e}")
        return closes[-1], 1


def calc_sar(highs, lows):
    """Calculate Parabolic SAR"""
    try:
        if len(highs) < 2:
            return lows[0]
        trend = 1 if highs[-1] > highs[-2] else -1
        sar = lows[-2] if trend == 1 else highs[-2]
        return round(sar, 4)
    except Exception as e:
        logger.error(f"SAR calculation error: {e}")
        return lows[-1]


def calc_gaussian_channel(prices, length=100, std_dev=2.0):
    """Calculate Gaussian Channel"""
    try:
        if len(prices) < length:
            return prices[-1], prices[-1]
        smoothed = gaussian_filter1d(prices[-length:], sigma=length/10)
        mean = smoothed[-1]
        std = np.std(prices[-length:])
        upper = mean + (std_dev * std)
        lower = mean - (std_dev * std)
        return round(upper, 4), round(lower, 4)
    except Exception as e:
        logger.error(f"Gaussian channel calculation error: {e}")
        return prices[-1], prices[-1]


def calc_trm(highs, lows, closes, length=20):
    """Calculate Trend Momentum"""
    try:
        if len(closes) < length:
            return 0
        momentum = closes[-1] - closes[-length]
        volatility = max(highs[-length:]) - min(lows[-length:])
        if volatility == 0:
            return 0
        trm = (momentum / volatility) * 100
        return round(trm, 2)
    except Exception as e:
        logger.error(f"TRM calculation error: {e}")
        return 0


# ============= MARKET DATA FUNCTIONS WITH LOGGING =============

@log_performance
def get_orderbook_liquidity(symbol: str) -> Dict:
    """Get orderbook liquidity with monitoring"""
    try:
        logger.debug(f"Fetching orderbook for {symbol}")
        response = requests.get(
            BINANCE_ORDERBOOK,
            params={'symbol': symbol.upper(), 'limit': 50},
            timeout=5
        )
        
        if response.status_code != 200:
            logger.warning(f"Orderbook API returned status {response.status_code}")
            return {'imbalance': 0, 'bid_liq': 0, 'ask_liq': 0}
        
        data = response.json()
        
        bids = np.array([[float(p), float(q)] for p, q in data['bids'][:50]])
        asks = np.array([[float(p), float(q)] for p, q in data['asks'][:50]])
        
        bid_liq = np.sum(bids[:, 1])
        ask_liq = np.sum(asks[:, 1])
        
        imbalance = (bid_liq - ask_liq) / (bid_liq + ask_liq) if (bid_liq + ask_liq) > 0 else 0
        
        result = {
            'imbalance': round(imbalance, 4),
            'bid_liq': round(bid_liq, 2),
            'ask_liq': round(ask_liq, 2)
        }
        logger.debug(f"Orderbook {symbol}: Imbalance={result['imbalance']}")
        return result
        
    except Exception as e:
        logger.error(f"Orderbook error for {symbol}: {e}")
        return {'imbalance': 0, 'bid_liq': 0, 'ask_liq': 0}


@log_performance
def get_btc_usdt_dominance() -> Dict:
    """Get BTC/USDT dominance with monitoring"""
    try:
        logger.debug("Fetching market dominance data")
        
        cg_response = requests.get(COINGECKO_API, timeout=5)
        cg_data = cg_response.json()
        total_mc = cg_data.get('data', {}).get('total_market_cap', {}).get('usd', 0)
        
        usdt_response = requests.get(USDT_API, timeout=5)
        usdt_data = usdt_response.json()
        usdt_mc = usdt_data.get('market_data', {}).get('market_cap', {}).get('usd', 0)
        
        usdt_dominance = (usdt_mc / total_mc * 100) if total_mc > 0 else 0
        btc_dominance = cg_data.get('data', {}).get('market_cap_percentage', {}).get('btc', 48.5)
        
        result = {
            'btc_dominance': round(btc_dominance, 2),
            'usdt_dominance': round(usdt_dominance, 2),
            'total_market_cap': total_mc
        }
        logger.debug(f"Dominance: BTC={result['btc_dominance']}%, USDT={result['usdt_dominance']}%")
        return result
        
    except Exception as e:
        logger.error(f"Dominance error: {e}")
        return {'btc_dominance': 48.5, 'usdt_dominance': 5.2, 'total_market_cap': 0}


@log_performance
def get_spx_trend() -> str:
    """Get SPX trend with monitoring"""
    try:
        logger.debug("Fetching SPX trend")
        spx = yf.Ticker('^GSPC')
        hist = spx.history(period='10d')
        if len(hist) >= 5:
            trend = 'Uptrend' if hist['Close'].iloc[-1] > hist['Close'].iloc[-5] else 'Downtrend'
            logger.debug(f"SPX Trend: {trend}")
            return trend
        return 'Unknown'
    except Exception as e:
        logger.error(f"SPX error: {e}")
        return 'Unknown'


# ============= MAIN ANALYSIS FUNCTION WITH COMPREHENSIVE LOGGING =============

@log_performance
def analyze_symbol_full(symbol: str, timeframe: str = '15m'):
    """Full symbol analysis with comprehensive logging"""
    try:
        logger.info(f"Starting analysis: {symbol} | Timeframe: {timeframe}")
        
        # Fetch data
        response = requests.get(
            BINANCE_API,
            params={'symbol': symbol.upper(), 'interval': timeframe, 'limit': 200},
            timeout=10
        )
        
        if response.status_code != 200:
            logger.error(f"Binance API error: Status {response.status_code}")
            return {'error': 'Failed to fetch data'}
        
        klines = response.json()
        if not klines:
            logger.warning(f"No data received for {symbol}")
            return {'error': 'No data'}
        
        logger.debug(f"Received {len(klines)} candles for {symbol}")
        
        # Extract OHLCV
        opens = [float(k[1]) for k in klines]
        highs = [float(k[2]) for k in klines]
        lows = [float(k[3]) for k in klines]
        closes = [float(k[4]) for k in klines]
        volumes = [float(k[5]) for k in klines]
        
        # Calculate indicators
        logger.debug("Calculating technical indicators")
        rsi = calc_rsi(closes)
        ema_10 = calc_ema(closes, 10)[-1]
        ema_20 = calc_ema(closes, 20)[-1]
        ema_50 = calc_ema(closes, 50)[-1]
        ema_200 = calc_ema(closes, 200)[-1] if len(closes) >= 200 else ema_50
        sma_20 = calc_sma(closes, 20)
        
        macd, macd_signal = calc_macd(closes)
        atr = calc_atr(highs, lows, closes)
        
        std = np.std(closes[-20:])
        bb_upper = sma_20 + (2 * std)
        bb_lower = sma_20 - (2 * std)
        
        typical_price = [(highs[i] + lows[i] + closes[i]) / 3 for i in range(len(closes))]
        vwap = sum([typical_price[i] * volumes[i] for i in range(-20, 0)]) / sum(volumes[-20:])
        
        supertrend_val, st_direction = calc_supertrend(highs, lows, closes)
        sar = calc_sar(highs, lows)
        gaussian_upper, gaussian_lower = calc_gaussian_channel(closes)
        trm = calc_trm(highs, lows, closes)
        
        current_price = closes[-1]
        prev_price = closes[-2]
        price_change = current_price - prev_price
        price_change_percent = (price_change / prev_price) * 100
        
        # Signal scoring
        logger.debug("Calculating signal scores")
        bullish_score = 0
        bearish_score = 0
        
        # RSI scoring
        if rsi < 30:
            bullish_score += 3
        elif rsi > 70:
            bearish_score += 3
        elif rsi < 40:
            bullish_score += 1.5
        elif rsi > 60:
            bearish_score += 1.5
        
        # EMA scoring
        if current_price > ema_50:
            bullish_score += 1.5
        else:
            bearish_score += 1.5
        
        if current_price > ema_200:
            bullish_score += 1
        else:
            bearish_score += 1
        
        if ema_10 > ema_20:
            bullish_score += 2
        else:
            bearish_score += 2
        
        # MACD scoring
        if macd > macd_signal:
            bullish_score += 2
        else:
            bearish_score += 2
        
        # Bollinger Bands scoring
        if current_price <= bb_lower:
            bullish_score += 2
        elif current_price >= bb_upper:
            bearish_score += 2
        
        # VWAP scoring
        if current_price > vwap:
            bullish_score += 1.5
        else:
            bearish_score += 1.5
        
        # Supertrend scoring
        if st_direction == 1:
            bullish_score += 2.5
        else:
            bearish_score += 2.5
        
        # SAR scoring
        if current_price > sar:
            bullish_score += 1.5
        else:
            bearish_score += 1.5
        
        # Gaussian Channel scoring
        if current_price < gaussian_lower:
            bullish_score += 1
        elif current_price > gaussian_upper:
            bearish_score += 1
        
        # TRM scoring
        if trm > 20:
            bullish_score += 1
        elif trm < -20:
            bearish_score += 1
        
        min_score = 7
        
        # Generate signal
        if bullish_score >= min_score and bullish_score > bearish_score:
            signal_type = 'LONG'
            entry = current_price * 0.999
            stop_loss = entry - (atr * 1.8)
            targets = [entry + (atr * 2), entry + (atr * 4), entry + (atr * 6)]
            logger.info(f"✅ LONG SIGNAL: {symbol} | Confidence: {bullish_score}/{bearish_score}")
        elif bearish_score >= min_score and bearish_score > bullish_score:
            signal_type = 'SHORT'
            entry = current_price * 1.001
            stop_loss = entry + (atr * 1.8)
            targets = [entry - (atr * 2), entry - (atr * 4), entry - (atr * 6)]
            logger.info(f"✅ SHORT SIGNAL: {symbol} | Confidence: {bearish_score}/{bullish_score}")
        else:
            signal_type = 'NO_TRADE'
            entry = current_price
            stop_loss = current_price
            targets = []
            logger.info(f"⚠️ NO TRADE: {symbol} | Scores: {bullish_score}/{bearish_score}")
        
        confidence = round(max(bullish_score, bearish_score) / 20 * 100, 1)
        
        result = {
            'symbol': symbol.upper(),
            'timeframe': timeframe,
            'timestamp': datetime.now().isoformat(),
            'signal': {
                'type': signal_type,
                'entry': round(entry, 4),
                'stop_loss': round(stop_loss, 4),
                'targets': [round(t, 4) for t in targets],
                'bullish_score': round(bullish_score, 1),
                'bearish_score': round(bearish_score, 1),
                'confidence': confidence
            },
            'indicators': {
                'price': round(current_price, 4),
                'price_change': round(price_change, 4),
                'price_change_percent': round(price_change_percent, 2),
                'rsi': round(rsi, 2),
                'macd': round(macd, 4),
                'macd_signal': round(macd_signal, 4),
                'ema_10': round(ema_10, 4),
                'ema_20': round(ema_20, 4),
                'ema_50': round(ema_50, 4),
                'ema_200': round(ema_200, 4),
                'vwap': round(vwap, 4),
                'atr': round(atr, 4),
                'bb_upper': round(bb_upper, 4),
                'bb_mid': round(sma_20, 4),
                'bb_lower': round(bb_lower, 4),
                'supertrend': round(supertrend_val, 4),
                'st_direction': st_direction,
                'sar': sar,
                'gaussian_upper': gaussian_upper,
                'gaussian_lower': gaussian_lower,
                'trm': trm
            }
        }
        
        logger.debug(f"Analysis completed for {symbol}")
        return result
        
    except Exception as e:
        logger.error(f"Analysis error for {symbol}: {e}\n{traceback.format_exc()}")
        return {'error': str(e)}


# ============= FLASK ROUTES WITH LOGGING =============

@app.route('/')
@log_api_request
def home():
    """Home endpoint with API documentation"""
    logger.info("Home endpoint accessed")
    return jsonify({
        'status': 'online',
        'version': '7.0 PRODUCTION',
        'server_time': datetime.now().isoformat(),
        'endpoints': {
            'analysis': '/api/analysis/<symbol>?timeframe=15m',
            'timeframes': '/api/timeframes/<symbol>',
            'symbols': '/api/symbols?search=BTC',
            'market_context': '/api/market-context/<symbol>',
            'orderbook': '/api/orderbook/<symbol>',
            'dominance': '/api/dominance',
            'spx_trend': '/api/spx-trend',
            'funding': '/api/funding/<symbol>',
            'open_interest': '/api/open-interest/<symbol>',
            'fear_greed': '/api/fear-greed',
            'trades_get': '/api/trades [GET]',
            'trades_post': '/api/trades [POST]',
            'trades_update': '/api/trades/<id> [PUT]',
            'health': '/api/health',
            'stats': '/api/stats',
            'logs': '/api/logs?lines=50'
        }
    })


@app.route('/api/analysis/<symbol>')
@log_api_request
def get_analysis(symbol):
    """Get trading analysis for symbol"""
    timeframe = request.args.get('timeframe', '15m')
    logger.info(f"Analysis requested: {symbol} | {timeframe}")
    return jsonify(analyze_symbol_full(symbol, timeframe))


@app.route('/api/timeframes/<symbol>')
@log_api_request
def get_timeframes(symbol):
    """Get analysis for multiple timeframes"""
    try:
        logger.info(f"Multi-timeframe analysis: {symbol}")
        results = []
        for tf in ['15m', '1h', '2h', '4h']:
            analysis = analyze_symbol_full(symbol, tf)
            if 'error' not in analysis:
                results.append({
                    'timeframe': tf,
                    'direction': analysis['signal']['type'],
                    'bullish_score': analysis['signal']['bullish_score'],
                    'bearish_score': analysis['signal']['bearish_score'],
                    'confidence': analysis['signal']['confidence']
                })
        return jsonify({'timeframes': results})
    except Exception as e:
        logger.error(f"Timeframes error: {e}")
        return jsonify({'error': str(e)}), 500


@app.route('/api/symbols')
@log_api_request
def search_symbols():
    """Search available symbols"""
    query = request.args.get('search', '').upper()
    all_symbols = [
        'BTCUSDT', 'ETHUSDT', 'BNBUSDT', 'SOLUSDT', 'XRPUSDT',
        'ADAUSDT', 'DOGEUSDT', 'DOTUSDT', 'MATICUSDT', 'AVAXUSDT',
        'LINKUSDT', 'UNIUSDT', 'ATOMUSDT', 'LTCUSDT', 'SHIBUSDT'
    ]
    if query:
        all_symbols = [s for s in all_symbols if query in s]
    logger.debug(f"Symbols search: '{query}' | Results: {len(all_symbols)}")
    return jsonify({'symbols': all_symbols})


@app.route('/api/market-context/<symbol>')
@log_api_request
def get_market_context(symbol):
    """Get market context data"""
    try:
        logger.info(f"Market context requested: {symbol}")
        context = {}
        
        # Fear & Greed Index
        try:
            fgi = requests.get(FGI_API, timeout=5).json()
            context['fear_greed_index'] = int(fgi['data'][0]['value'])
            context['fgi_classification'] = fgi['data'][0]['value_classification']
        except Exception as e:
            logger.warning(f"FGI API error: {e}")
            context['fear_greed_index'] = 50
            context['fgi_classification'] = 'Neutral'
        
        # Funding Rate
        try:
            funding = requests.get(BINANCE_FUNDING, params={'symbol': symbol.upper(), 'limit': 1}, timeout=5).json()
            context['funding_rate'] = float(funding[0]['fundingRate'])
        except Exception as e:
            logger.warning(f"Funding rate error: {e}")
            context['funding_rate'] = 0.0001
        
        # Open Interest
        try:
            oi = requests.get(BINANCE_OI, params={'symbol': symbol.upper()}, timeout=5).json()
            context['open_interest'] = float(oi['openInterest'])
        except Exception as e:
            logger.warning(f"Open interest error: {e}")
            context['open_interest'] = 15000000000
        
        # Orderbook Liquidity
        liquidity = get_orderbook_liquidity(symbol)
        context['liquidity_imbalance'] = liquidity['imbalance']
        context['bid_liquidity'] = liquidity['bid_liq']
        context['ask_liquidity'] = liquidity['ask_liq']
        
        # Dominance
        dominance = get_btc_usdt_dominance()
        context['btc_dominance'] = dominance['btc_dominance']
        context['usdt_dominance'] = dominance['usdt_dominance']
        context['total_market_cap'] = dominance['total_market_cap']
        
        # SPX Trend
        context['spx_trend'] = get_spx_trend()
        context['oi_change'] = 0
        
        logger.info(f"Market context completed: {symbol}")
        return jsonify(context)
    except Exception as e:
        logger.error(f"Market context error: {e}")
        return jsonify({'error': str(e)}), 500


@app.route('/api/orderbook/<symbol>')
@log_api_request
def get_orderbook(symbol):
    """Get orderbook liquidity"""
    logger.info(f"Orderbook requested: {symbol}")
    return jsonify(get_orderbook_liquidity(symbol))


@app.route('/api/dominance')
@log_api_request
def get_dominance():
    """Get market dominance"""
    logger.info("Dominance data requested")
    return jsonify(get_btc_usdt_dominance())


@app.route('/api/spx-trend')
@log_api_request
def get_spx():
    """Get SPX trend"""
    logger.info("SPX trend requested")
    return jsonify({'spx_trend': get_spx_trend()})


@app.route('/api/funding/<symbol>')
@log_api_request
def get_funding(symbol):
    """Get funding rate"""
    try:
        logger.info(f"Funding rate requested: {symbol}")
        funding = requests.get(BINANCE_FUNDING, params={'symbol': symbol.upper(), 'limit': 1}, timeout=5).json()
        return jsonify({'funding_rate': float(funding[0]['fundingRate'])})
    except Exception as e:
        logger.error(f"Funding rate error: {e}")
        return jsonify({'funding_rate': 0.0001})


@app.route('/api/open-interest/<symbol>')
@log_api_request
def get_open_interest(symbol):
    """Get open interest"""
    try:
        logger.info(f"Open interest requested: {symbol}")
        oi = requests.get(BINANCE_OI, params={'symbol': symbol.upper()}, timeout=5).json()
        return jsonify({'open_interest': float(oi['openInterest'])})
    except Exception as e:
        logger.error(f"Open interest error: {e}")
        return jsonify({'open_interest': 15000000000})


@app.route('/api/fear-greed')
@log_api_request
def get_fear_greed():
    """Get fear & greed index"""
    try:
        logger.info("Fear & Greed index requested")
        fgi = requests.get(FGI_API, timeout=5).json()
        return jsonify({
            'value': int(fgi['data'][0]['value']),
            'classification': fgi['data'][0]['value_classification']
        })
    except Exception as e:
        logger.error(f"Fear & Greed error: {e}")
        return jsonify({'value': 50, 'classification': 'Neutral'})


@app.route('/api/trades', methods=['GET', 'POST'])
@log_api_request
def handle_trades():
    """Handle trade operations"""
    if request.method == 'POST':
        try:
            data = request.json
            logger.info(f"Saving trade: {data.get('symbol')} | {data.get('signal_type')}")
            trade_id = db.save_trade(data)
            return jsonify({'success': True, 'trade_id': trade_id})
        except Exception as e:
            logger.error(f"Trade save error: {e}")
            return jsonify({'success': False, 'error': str(e)}), 500
    else:
        status = request.args.get('status')
        limit = int(request.args.get('limit', 100))
        logger.info(f"Retrieving trades: limit={limit}, status={status}")
        trades = db.get_all_trades(limit, status)
        return jsonify({'trades': trades})


@app.route('/api/trades/<int:trade_id>', methods=['PUT'])
@log_api_request
def update_trade(trade_id):
    """Update trade"""
    try:
        data = request.json
        logger.info(f"Updating trade {trade_id}: {data.get('status')}")
        db.update_trade(trade_id, data['exit_price'], data['pnl'], data['status'])
        return jsonify({'success': True})
    except Exception as e:
        logger.error(f"Trade update error: {e}")
        return jsonify({'success': False, 'error': str(e)}), 500


@app.route('/api/health')
def health_check():
    """Health check endpoint"""
    uptime = datetime.now() - app.config.get('start_time', datetime.now())
    return jsonify({
        'status': 'healthy',
        'version': '7.0',
        'timestamp': datetime.now().isoformat(),
        'uptime_seconds': int(uptime.total_seconds())
    })


@app.route('/api/stats')
@log_api_request
def get_stats():
    """Get API statistics"""
    try:
        # Database stats
        trades = db.get_all_trades(1000)
        active_trades = [t for t in trades if t['status'] == 'ACTIVE']
        closed_trades = [t for t in trades if t['status'] in ['CLOSED', 'STOPPED']]
        
        # Calculate PnL
        total_pnl = sum([t.get('pnl', 0) for t in closed_trades if t.get('pnl')])
        
        stats = {
            'database': {
                'total_trades': len(trades),
                'active_trades': len(active_trades),
                'closed_trades': len(closed_trades),
                'total_pnl': round(total_pnl, 2)
            },
            'server': {
                'version': '7.0',
                'start_time': app.config.get('start_time', datetime.now()).isoformat(),
                'uptime_seconds': int((datetime.now() - app.config.get('start_time', datetime.now())).total_seconds())
            }
        }
        
        logger.info("Stats retrieved successfully")
        return jsonify(stats)
        
    except Exception as e:
        logger.error(f"Stats error: {e}")
        return jsonify({'error': str(e)}), 500


@app.route('/api/logs')
@log_api_request
def get_logs():
    """Get recent logs"""
    try:
        lines = int(request.args.get('lines', 50))
        log_type = request.args.get('type', 'main')  # main, error, performance, requests
        
        log_files = {
            'main': 'logs/flask_api.log',
            'error': 'logs/errors.log',
            'performance': 'logs/performance.log',
            'requests': 'logs/api_requests.log'
        }
        
        log_file = log_files.get(log_type, 'logs/flask_api.log')
        
        if os.path.exists(log_file):
            with open(log_file, 'r') as f:
                all_lines = f.readlines()
                recent_lines = all_lines[-lines:]
                return jsonify({
                    'log_type': log_type,
                    'lines': recent_lines,
                    'total_lines': len(all_lines)
                })
        else:
            return jsonify({'error': f'Log file not found: {log_file}'}), 404
            
    except Exception as e:
        logger.error(f"Logs retrieval error: {e}")
        return jsonify({'error': str(e)}), 500


# ============= STARTUP =============

if __name__ == '__main__':
    # Store start time
    app.config['start_time'] = datetime.now()
    
    logger.info("="*80)
    logger.info("🚀 Flask API v7.0 - PYTHONANYWHERE PRODUCTION READY")
    logger.info("="*80)
    logger.info("✅ Logging: Comprehensive multi-level logging enabled")
    logger.info("✅ Monitoring: Performance tracking active")
    logger.info("✅ Database: SQLite trades database initialized")
    logger.info("✅ Server: http://0.0.0.0:5000")
    logger.info("✅ Endpoints: 15+ API endpoints available")
    logger.info("="*80)
    
    # Run server
    app.run(host='0.0.0.0', port=5000, debug=False, threaded=True, use_reloader=False)
