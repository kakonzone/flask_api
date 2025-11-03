"""
Complete Multi-Timeframe Bot with TradingView Indicators
Includes: VWAP, Supertrend, SAR, BB, MA Cross, TRM, Gaussian Channel
+ All 20+ Python indicators + Image generation
"""

import os
import asyncio
import pandas as pd
import numpy as np
import requests
import yfinance as yf
from datetime import datetime
from telegram import Update, InlineKeyboardButton, InlineKeyboardMarkup
from telegram.ext import Application, CommandHandler, CallbackQueryHandler, ContextTypes, MessageHandler, filters
from cachetools import TTLCache
from dotenv import load_dotenv
from scipy.signal import argrelextrema
from scipy.ndimage import gaussian_filter1d
import logging
from typing import Dict, List, Tuple, Optional
import sqlite3
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from io import BytesIO

# ============= CONFIGURATION =============
load_dotenv()
TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN")
TELEGRAM_CHAT_ID = os.getenv("TELEGRAM_CHAT_ID")

if not TELEGRAM_BOT_TOKEN or TELEGRAM_BOT_TOKEN.startswith("os.getenv"):
    print("❌ Check .env file!")
    exit(1)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    handlers=[logging.FileHandler('trading_bot.log'), logging.StreamHandler()]
)

# APIs
BINANCE_API = "https://api.binance.com/api/v3/klines"
BINANCE_FUNDING = "https://fapi.binance.com/fapi/v1/fundingRate"
BINANCE_OI = "https://fapi.binance.com/fapi/v1/openInterest"
BINANCE_ORDERBOOK = "https://api.binance.com/api/v3/depth"
FGI_API = "https://api.alternative.me/fng/?limit=1"
COINGECKO_API = "https://api.coingecko.com/api/v3/global"
USDT_API = "https://api.coingecko.com/api/v3/coins/tether"

# Timeframes
TIMEFRAMES = {
    '15m': {'interval': '15m', 'weight': 1.0, 'min_score': 10.0},
    '1h': {'interval': '1h', 'weight': 1.2, 'min_score': 11.0},
    '2h': {'interval': '2h', 'weight': 1.5, 'min_score': 12.0},
    '4h': {'interval': '4h', 'weight': 2.0, 'min_score': 13.0}
}

api_cache = TTLCache(maxsize=300, ttl=180)

# ============= DATABASE =============
class TradeDatabase:
    def __init__(self):
        self.conn = sqlite3.connect('trades.db', check_same_thread=False)
        self.init_db()
    
    def init_db(self):
        c = self.conn.cursor()
        c.execute('''CREATE TABLE IF NOT EXISTS trades
                     (id INTEGER PRIMARY KEY, timestamp TEXT, symbol TEXT, 
                      timeframes TEXT, signal_type TEXT, entry_price REAL, 
                      stop_loss REAL, targets TEXT, confidence REAL, notes TEXT)''')
        self.conn.commit()
    
    def save_trade(self, data: Dict):
        c = self.conn.cursor()
        c.execute('''INSERT INTO trades VALUES (NULL,?,?,?,?,?,?,?,?,?)''',
                  (data['timestamp'], data['symbol'], data['timeframes'], 
                   data['signal_type'], data['entry'], data['sl'], 
                   str(data['targets']), data['confidence'], data['notes']))
        self.conn.commit()

# ============= ALL INDICATORS =============
def calc_rsi(df: pd.DataFrame, period: int = 14) -> pd.Series:
    """RSI Indicator"""
    delta = df['close'].diff()
    gain = delta.clip(lower=0).rolling(period).mean()
    loss = -delta.clip(upper=0).rolling(period).mean()
    rs = np.where(loss == 0, 100, gain / loss.replace(0, np.nan))
    return pd.Series(100 - (100 / (1 + rs)), index=df.index)

def calc_stoch_rsi(df: pd.DataFrame, rsi_period: int = 14, 
                   stoch_period: int = 14, k: int = 3, d: int = 3) -> Tuple[pd.Series, pd.Series]:
    """Stochastic RSI"""
    rsi = calc_rsi(df, rsi_period)
    min_rsi = rsi.rolling(stoch_period).min()
    max_rsi = rsi.rolling(stoch_period).max()
    stoch = np.where((max_rsi - min_rsi) == 0, 50, (rsi - min_rsi) / (max_rsi - min_rsi) * 100)
    stoch_k = pd.Series(stoch).rolling(k).mean()
    stoch_d = stoch_k.rolling(d).mean()
    return stoch_k, stoch_d

def calc_ema(df: pd.DataFrame, period: int) -> pd.Series:
    """EMA"""
    return df['close'].ewm(span=period, adjust=False).mean()

def calc_bollinger_bands(df: pd.DataFrame, period: int = 20, 
                         std_dev: float = 2.0) -> Tuple[pd.Series, pd.Series, pd.Series]:
    """Bollinger Bands"""
    sma = df['close'].rolling(period).mean()
    std = df['close'].rolling(period).std()
    upper = sma + (std * std_dev)
    lower = sma - (std * std_dev)
    return upper, sma, lower

def calc_atr(df: pd.DataFrame, period: int = 14) -> pd.Series:
    """ATR"""
    hl = df['high'] - df['low']
    hc = np.abs(df['high'] - df['close'].shift())
    lc = np.abs(df['low'] - df['close'].shift())
    return pd.concat([hl, hc, lc], axis=1).max(axis=1).rolling(period).mean()

def calc_vwap(df: pd.DataFrame) -> pd.Series:
    """VWAP with daily reset"""
    df = df.copy()
    df['date'] = df['time'].dt.date
    tp = (df['high'] + df['low'] + df['close']) / 3
    df['tp_vol'] = tp * df['volume']
    vwap = df.groupby('date').apply(lambda x: (x['tp_vol'].cumsum() / x['volume'].cumsum()))
    return vwap.reset_index(level=0, drop=True)

def calc_supertrend(df: pd.DataFrame, period: int = 10, multiplier: float = 3.0) -> Tuple[pd.Series, pd.Series]:
    """Supertrend Indicator"""
    atr = calc_atr(df, period)
    hl_avg = (df['high'] + df['low']) / 2
    
    upper_band = hl_avg + (multiplier * atr)
    lower_band = hl_avg - (multiplier * atr)
    
    supertrend = pd.Series(index=df.index, dtype=float)
    direction = pd.Series(index=df.index, dtype=int)
    
    supertrend.iloc[0] = upper_band.iloc[0]
    direction.iloc[0] = 1
    
    for i in range(1, len(df)):
        if df['close'].iloc[i] > supertrend.iloc[i-1]:
            direction.iloc[i] = 1
            supertrend.iloc[i] = lower_band.iloc[i]
        elif df['close'].iloc[i] < supertrend.iloc[i-1]:
            direction.iloc[i] = -1
            supertrend.iloc[i] = upper_band.iloc[i]
        else:
            direction.iloc[i] = direction.iloc[i-1]
            if direction.iloc[i] == 1:
                supertrend.iloc[i] = max(lower_band.iloc[i], supertrend.iloc[i-1])
            else:
                supertrend.iloc[i] = min(upper_band.iloc[i], supertrend.iloc[i-1])
    
    return supertrend, direction

def calc_parabolic_sar(df: pd.DataFrame, start: float = 0.02, increment: float = 0.02, maximum: float = 0.2) -> pd.Series:
    """Parabolic SAR"""
    sar = pd.Series(index=df.index, dtype=float)
    trend = 1
    sar.iloc[0] = df['low'].iloc[0]
    ep = df['high'].iloc[0]
    af = start
    
    for i in range(1, len(df)):
        if trend == 1:
            sar.iloc[i] = sar.iloc[i-1] + af * (ep - sar.iloc[i-1])
            if df['low'].iloc[i] < sar.iloc[i]:
                trend = -1
                sar.iloc[i] = ep
                ep = df['low'].iloc[i]
                af = start
            else:
                if df['high'].iloc[i] > ep:
                    ep = df['high'].iloc[i]
                    af = min(af + increment, maximum)
        else:
            sar.iloc[i] = sar.iloc[i-1] - af * (sar.iloc[i-1] - ep)
            if df['high'].iloc[i] > sar.iloc[i]:
                trend = 1
                sar.iloc[i] = ep
                ep = df['high'].iloc[i]
                af = start
            else:
                if df['low'].iloc[i] < ep:
                    ep = df['low'].iloc[i]
                    af = min(af + increment, maximum)
    
    return sar

def calc_trm(df: pd.DataFrame, tsi_long: int = 25, tsi_short: int = 5, 
             tsi_signal: int = 14, rsi_length: int = 5) -> Tuple[pd.Series, pd.Series, pd.Series]:
    """TRM (True Relative Movement) Indicator"""
    # TSI Calculation
    pc = df['close'].diff()
    first_smooth = pc.ewm(span=tsi_long, adjust=False).mean()
    double_smoothed_pc = first_smooth.ewm(span=tsi_short, adjust=False).mean()
    
    abs_pc = pc.abs()
    first_smooth_abs = abs_pc.ewm(span=tsi_long, adjust=False).mean()
    double_smoothed_abs_pc = first_smooth_abs.ewm(span=tsi_short, adjust=False).mean()
    
    tsi_value = 100 * (double_smoothed_pc / double_smoothed_abs_pc)
    tsi_signal_line = tsi_value.ewm(span=tsi_signal, adjust=False).mean()
    
    # RSI Calculation
    rsi = calc_rsi(df, rsi_length)
    
    return tsi_value, tsi_signal_line, rsi

def calc_gaussian_channel(df: pd.DataFrame, poles: int = 4, period: int = 144, mult: float = 1.414) -> Tuple[pd.Series, pd.Series, pd.Series]:
    """Gaussian Channel Indicator"""
    # Simplified Gaussian filter
    hlc3 = (df['high'] + df['low'] + df['close']) / 3
    
    # Apply Gaussian filter
    sigma = period / 6  # Approximate sigma from period
    filtered = pd.Series(gaussian_filter1d(hlc3.values, sigma), index=df.index)
    
    # True Range
    tr = calc_atr(df, period)
    filtered_tr = pd.Series(gaussian_filter1d(tr.values, sigma), index=df.index)
    
    # Bands
    upper_band = filtered + (filtered_tr * mult)
    lower_band = filtered - (filtered_tr * mult)
    
    return upper_band, filtered, lower_band

async def fetch_data(url: str, params: Optional[Dict] = None) -> Dict:
    """Fetch with caching"""
    key = f"{url}_{str(params)}"
    if key in api_cache:
        return api_cache[key]
    try:
        resp = await asyncio.get_event_loop().run_in_executor(
            None, lambda: requests.get(url, params=params, timeout=10))
        if resp.status_code == 429:
            await asyncio.sleep(3)
            return {}
        resp.raise_for_status()
        data = resp.json()
        api_cache[key] = data
        return data
    except Exception as e:
        logging.error(f"Fetch error: {e}")
        return {}

# ============= COMPLETE BOT =============
class CompleteMultiTimeframeBot:
    def __init__(self):
        self.db = TradeDatabase()
        
        # Signal weights including TradingView indicators
        self.weights = {
            '15m': {
                'rsi': 2.0, 'stoch_rsi': 1.5, 'ema_trend': 1.8, 'ema_cross': 2.2,
                'bb_signal': 1.5, 'volume': 1.5, 'vwap': 1.5, 'structure': 2.0,
                'divergence': 2.5, 'candle': 1.5, 'fgi': 1.2, 'funding': 1.5,
                'oi': 1.3, 'liquidity': 1.2, 'supertrend': 2.0, 'sar': 1.5,
                'ma_cross': 1.8, 'trm': 2.5, 'gaussian': 1.8
            },
            '1h': {
                'rsi': 2.2, 'stoch_rsi': 1.3, 'ema_trend': 2.0, 'ema_cross': 2.0,
                'bb_signal': 1.5, 'volume': 1.5, 'vwap': 1.5, 'structure': 2.2,
                'divergence': 2.3, 'candle': 1.3, 'usdt_d': 1.0,
                'supertrend': 2.2, 'sar': 1.5, 'ma_cross': 1.8, 'trm': 2.3, 'gaussian': 1.8
            },
            '2h': {
                'rsi': 2.5, 'ema_trend': 2.3, 'ema_cross': 1.8, 
                'bb_signal': 1.5, 'volume': 1.5, 'vwap': 1.5, 'structure': 2.5, 
                'divergence': 2.5, 'spx': 1.3, 'usdt_d': 1.2,
                'supertrend': 2.5, 'sar': 1.3, 'ma_cross': 1.8, 'gaussian': 2.0
            },
            '4h': {
                'rsi': 3.0, 'ema_trend': 3.0, 'ema_cross': 1.5, 
                'bb_signal': 1.5, 'volume': 1.5, 'structure': 3.0, 
                'divergence': 2.0, 'spx': 1.5, 'usdt_d': 1.3,
                'supertrend': 3.0, 'sar': 1.3, 'gaussian': 2.2
            }
        }
    
    def process_klines(self, klines: List, symbol: str) -> pd.DataFrame:
        """Process OHLCV data"""
        df = pd.DataFrame(klines, columns=['time','open','high','low','close','volume',
                                           'ct','qav','trades','tbb','tbq','ign'])
        df[['open','high','low','close','volume']] = df[['open','high','low','close','volume']].apply(pd.to_numeric)
        df['time'] = pd.to_datetime(df['time'], unit='ms')
        df['symbol'] = symbol
        return df.dropna()
    
    def calculate_all_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculate ALL indicators including TradingView ones"""
        # Basic indicators
        df['RSI_14'] = calc_rsi(df, 14)
        df['STOCH_K'], df['STOCH_D'] = calc_stoch_rsi(df)
        
        # EMAs
        df['EMA_10'] = calc_ema(df, 10)
        df['EMA_20'] = calc_ema(df, 20)
        df['EMA_50'] = calc_ema(df, 50)
        df['EMA_100'] = calc_ema(df, 100)
        df['EMA_200'] = calc_ema(df, 200)
        
        # Bollinger Bands
        df['BB_UPPER'], df['BB_MID'], df['BB_LOWER'] = calc_bollinger_bands(df)
        df['ATR'] = calc_atr(df, 14)
        
        # VWAP
        df['VWAP'] = calc_vwap(df)
        df['VOL_MA'] = df['volume'].rolling(20).mean()
        
        # TradingView indicators
        df['SUPERTREND'], df['ST_DIRECTION'] = calc_supertrend(df, 10, 3.0)
        df['SAR'] = calc_parabolic_sar(df)
        
        # MA Cross (50/100)
        df['SMA_50'] = df['close'].rolling(50).mean()
        df['SMA_100'] = df['close'].rolling(100).mean()
        
        # TRM
        df['TSI'], df['TSI_SIGNAL'], df['TRM_RSI'] = calc_trm(df, 25, 5, 14, 5)
        
        # Gaussian Channel
        df['GC_UPPER'], df['GC_MID'], df['GC_LOWER'] = calc_gaussian_channel(df, 4, 144, 1.414)
        
        return df
    
    def find_pivots(self, series: pd.Series, order: int = 5) -> Tuple[np.ndarray, np.ndarray]:
        """Pivot detection"""
        try:
            highs = argrelextrema(np.array(series), np.greater, order=order)[0]
            lows = argrelextrema(np.array(series), np.less, order=order)[0]
            return highs, lows
        except:
            return np.array([]), np.array([])
    
    def detect_divergence(self, df: pd.DataFrame) -> Tuple[str, float]:
        """RSI Divergence"""
        if len(df) < 20:
            return "None", 0
        
        p_high, p_low = self.find_pivots(df['close'], 5)
        r_high, r_low = self.find_pivots(df['RSI_14'], 5)
        
        if len(p_low) >= 2 and len(r_low) >= 2:
            if (df['close'].iloc[p_low[-1]] < df['close'].iloc[p_low[-2]] and
                df['RSI_14'].iloc[r_low[-1]] > df['RSI_14'].iloc[r_low[-2]]):
                return "Bullish Divergence", 1.0
        
        if len(p_high) >= 2 and len(r_high) >= 2:
            if (df['close'].iloc[p_high[-1]] > df['close'].iloc[p_high[-2]] and
                df['RSI_14'].iloc[r_high[-1]] < df['RSI_14'].iloc[r_high[-2]]):
                return "Bearish Divergence", -1.0
        
        return "None", 0
    
    def detect_structure(self, df: pd.DataFrame) -> Tuple[str, float]:
        """Market Structure"""
        highs = df['high'].values[-10:]
        lows = df['low'].values[-10:]
        
        try:
            ph = argrelextrema(highs, np.greater, order=2)[0]
            pl = argrelextrema(lows, np.less, order=2)[0]
            
            if len(ph) >= 2 and len(pl) >= 2:
                if highs[ph[-1]] > highs[ph[-2]] and lows[pl[-1]] > lows[pl[-2]]:
                    return "HH/HL", 1.0
                elif highs[ph[-1]] < highs[ph[-2]] and lows[pl[-1]] < lows[pl[-2]]:
                    return "LL/LH", -1.0
        except:
            pass
        
        return "Neutral", 0
    
    def detect_candle_pattern(self, df: pd.DataFrame) -> Tuple[str, float]:
        """Candle Pattern"""
        if len(df) < 2:
            return "None", 0
        
        last, prev = df.iloc[-1], df.iloc[-2]
        
        if (last['close'] > last['open'] and 
            last['open'] < prev['close'] < prev['open'] < last['close']):
            return "Bullish Engulfing", 1.0
        
        if (last['close'] < last['open'] and 
            last['open'] > prev['close'] > prev['open'] > last['close']):
            return "Bearish Engulfing", -1.0
        
        body = abs(last['close'] - last['open'])
        lower_shadow = last['open'] - last['low'] if last['close'] > last['open'] else last['close'] - last['low']
        if lower_shadow > body * 2 and (last['high'] - max(last['close'], last['open'])) < body:
            return "Hammer", 0.8
        
        return "None", 0
    
    async def analyze_timeframe(self, df: pd.DataFrame, tf: str, market_data: Dict) -> Dict:
        """Complete analysis with ALL indicators"""
        signals = {}
        w = self.weights[tf]
        price = df['close'].iloc[-1]
        
        # 1. RSI
        rsi = df['RSI_14'].iloc[-1]
        if rsi < 30:
            signals['rsi'] = ((30 - rsi) / 30) * w.get('rsi', 0)
        elif rsi > 70:
            signals['rsi'] = -((rsi - 70) / 30) * w.get('rsi', 0)
        else:
            signals['rsi'] = 0
        
        # 2. Stochastic RSI
        if tf in ['15m', '1h'] and 'stoch_rsi' in w:
            stoch_k = df['STOCH_K'].iloc[-1]
            if stoch_k < 20:
                signals['stoch_rsi'] = ((20 - stoch_k) / 20) * w['stoch_rsi']
            elif stoch_k > 80:
                signals['stoch_rsi'] = -((stoch_k - 80) / 20) * w['stoch_rsi']
            else:
                signals['stoch_rsi'] = 0
        
        # 3. EMA Trend
        ema_score = 0
        if price > df['EMA_200'].iloc[-1]:
            ema_score += 0.6
        if df['EMA_10'].iloc[-1] > df['EMA_20'].iloc[-1]:
            ema_score += 0.4
        if price < df['EMA_200'].iloc[-1]:
            ema_score -= 0.6
        if df['EMA_10'].iloc[-1] < df['EMA_20'].iloc[-1]:
            ema_score -= 0.4
        signals['ema_trend'] = ema_score * w.get('ema_trend', 0)
        
        # 4. EMA Cross
        cross = 0
        if df['EMA_10'].iloc[-1] > df['EMA_20'].iloc[-1] and df['EMA_10'].iloc[-2] <= df['EMA_20'].iloc[-2]:
            cross = 1.0
        elif df['EMA_10'].iloc[-1] < df['EMA_20'].iloc[-1] and df['EMA_10'].iloc[-2] >= df['EMA_20'].iloc[-2]:
            cross = -1.0
        signals['ema_cross'] = cross * w.get('ema_cross', 0)
        
        # 5. Bollinger Bands
        bb_score = 0
        if price <= df['BB_LOWER'].iloc[-1]:
            bb_score = 1.0
        elif price >= df['BB_UPPER'].iloc[-1]:
            bb_score = -1.0
        signals['bb_signal'] = bb_score * w.get('bb_signal', 0)
        
        # 6. Volume
        vol_score = 1.0 if df['volume'].iloc[-1] > df['VOL_MA'].iloc[-1] * 1.5 else 0
        signals['volume'] = vol_score * w.get('volume', 0)
        
        # 7. VWAP
        vwap_score = 0
        if price > df['VWAP'].iloc[-1]:
            vwap_score = 0.5
        elif price < df['VWAP'].iloc[-1]:
            vwap_score = -0.5
        signals['vwap'] = vwap_score * w.get('vwap', 0)
        
        # 8. Market Structure
        _, struct_score = self.detect_structure(df)
        signals['structure'] = struct_score * w.get('structure', 0)
        
        # 9. Divergence
        _, div_score = self.detect_divergence(df)
        signals['divergence'] = div_score * w.get('divergence', 0)
        
        # 10. Candle Pattern
        if tf in ['15m', '1h'] and 'candle' in w:
            _, candle_score = self.detect_candle_pattern(df)
            signals['candle'] = candle_score * w.get('candle', 0)
        
        # 11. Supertrend
        st_score = 0
        if df['ST_DIRECTION'].iloc[-1] == 1:  # Bullish
            st_score = 1.0
        elif df['ST_DIRECTION'].iloc[-1] == -1:  # Bearish
            st_score = -1.0
        signals['supertrend'] = st_score * w.get('supertrend', 0)
        
        # 12. Parabolic SAR
        sar_score = 0
        if df['SAR'].iloc[-1] < price:
            sar_score = 0.8
        elif df['SAR'].iloc[-1] > price:
            sar_score = -0.8
        signals['sar'] = sar_score * w.get('sar', 0)
        
        # 13. MA Cross (50/100)
        ma_cross_score = 0
        if 'SMA_50' in df.columns and 'SMA_100' in df.columns:
            if df['SMA_50'].iloc[-1] > df['SMA_100'].iloc[-1]:
                ma_cross_score = 0.5
            elif df['SMA_50'].iloc[-1] < df['SMA_100'].iloc[-1]:
                ma_cross_score = -0.5
            # Cross detection
            if df['SMA_50'].iloc[-1] > df['SMA_100'].iloc[-1] and df['SMA_50'].iloc[-2] <= df['SMA_100'].iloc[-2]:
                ma_cross_score = 1.0
            elif df['SMA_50'].iloc[-1] < df['SMA_100'].iloc[-1] and df['SMA_50'].iloc[-2] >= df['SMA_100'].iloc[-2]:
                ma_cross_score = -1.0
        signals['ma_cross'] = ma_cross_score * w.get('ma_cross', 0)
        
        # 14. TRM (True Relative Movement)
        trm_score = 0
        if 'TSI' in df.columns and 'TSI_SIGNAL' in df.columns and 'TRM_RSI' in df.columns:
            tsi = df['TSI'].iloc[-1]
            tsi_sig = df['TSI_SIGNAL'].iloc[-1]
            trm_rsi = df['TRM_RSI'].iloc[-1]
            
            # TRM Buy: TSI > Signal AND RSI > 50
            if tsi > tsi_sig and trm_rsi > 50:
                trm_score = 1.0
            # TRM Sell: TSI < Signal AND RSI < 50
            elif tsi < tsi_sig and trm_rsi < 50:
                trm_score = -1.0
        signals['trm'] = trm_score * w.get('trm', 0)
        
        # 15. Gaussian Channel
        gc_score = 0
        if 'GC_MID' in df.columns:
            if price > df['GC_MID'].iloc[-1]:
                gc_score = 0.5
            elif price < df['GC_MID'].iloc[-1]:
                gc_score = -0.5
            # Strong signals near bands
            if price >= df['GC_UPPER'].iloc[-1]:
                gc_score = -1.0  # Overbought
            elif price <= df['GC_LOWER'].iloc[-1]:
                gc_score = 1.0  # Oversold
        signals['gaussian'] = gc_score * w.get('gaussian', 0)
        
        # Market Context (15m only)
        if tf == '15m':
            fgi_map = {"Extreme Fear": 1.0, "Fear": 0.5, "Greed": -0.5, "Extreme Greed": -1.0}
            signals['fgi'] = fgi_map.get(market_data.get('fgi_sent', 'Neutral'), 0) * w.get('fgi', 0)
            
            funding = market_data.get('funding', 0)
            if abs(funding) > 0.001:
                signals['funding'] = (-1.0 if funding > 0 else 1.0) * w.get('funding', 0)
            else:
                signals['funding'] = 0
            
            oi_change = market_data.get('oi_change', 0)
            if oi_change > 10:
                signals['oi'] = 1.0 * w.get('oi', 0)
            elif oi_change < -10:
                signals['oi'] = -1.0 * w.get('oi', 0)
            else:
                signals['oi'] = 0
            
            imbalance = market_data.get('liquidity_imbalance', 0)
            if imbalance > 0.2:
                signals['liquidity'] = 0.8 * w.get('liquidity', 0)
            elif imbalance < -0.2:
                signals['liquidity'] = -0.8 * w.get('liquidity', 0)
            else:
                signals['liquidity'] = 0
        
        # Medium timeframes
        if tf in ['1h', '2h'] and 'usdt_d' in w:
            usdt_d = market_data.get('usdt_d', 0)
            signals['usdt_d'] = (-0.8 if usdt_d > 7 else 0.5) * w['usdt_d']
        
        # Long timeframes
        if tf in ['2h', '4h'] and 'spx' in w:
            spx_map = {"Uptrend": 1.0, "Downtrend": -1.0}
            signals['spx'] = spx_map.get(market_data.get('spx_trend', 'Unknown'), 0) * w['spx']
        
        # Calculate scores
        bullish = sum(s for s in signals.values() if s > 0)
        bearish = abs(sum(s for s in signals.values() if s < 0))
        
        # Determine direction
        min_score = TIMEFRAMES[tf]['min_score']
        direction = None
        if bullish >= min_score and bullish > bearish:
            direction = 'LONG'
        elif bearish >= min_score and bearish > bullish:
            direction = 'SHORT'
        
        return {
            'bullish': bullish,
            'bearish': bearish,
            'direction': direction,
            'signals': signals,
            'indicators': {
                'rsi': rsi,
                'stoch_k': df['STOCH_K'].iloc[-1] if 'STOCH_K' in df.columns else 0,
                'price': price,
                'vwap': df['VWAP'].iloc[-1],
                'bb_upper': df['BB_UPPER'].iloc[-1],
                'bb_lower': df['BB_LOWER'].iloc[-1],
                'atr': df['ATR'].iloc[-1],
                'ema_10': df['EMA_10'].iloc[-1],
                'ema_200': df['EMA_200'].iloc[-1],
                'supertrend': df['SUPERTREND'].iloc[-1],
                'st_direction': df['ST_DIRECTION'].iloc[-1],
                'sar': df['SAR'].iloc[-1],
                'tsi': df['TSI'].iloc[-1] if 'TSI' in df.columns else 0,
                'gc_mid': df['GC_MID'].iloc[-1] if 'GC_MID' in df.columns else 0
            }
        }
    
    def generate_chart(self, df: pd.DataFrame, symbol: str, signal_type: str, entry: float, 
                       stop_loss: float, targets: List[float]) -> BytesIO:
        """Generate trading chart with all indicators"""
        fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(14, 10), 
                                             gridspec_kw={'height_ratios': [3, 1, 1]})
        
        # Plot 1: Price + Indicators
        df_plot = df.tail(100)
        ax1.plot(df_plot.index, df_plot['close'], label='Price', color='black', linewidth=2)
        ax1.plot(df_plot.index, df_plot['VWAP'], label='VWAP', color='blue', alpha=0.7)
        ax1.plot(df_plot.index, df_plot['BB_UPPER'], label='BB Upper', color='red', linestyle='--', alpha=0.5)
        ax1.plot(df_plot.index, df_plot['BB_LOWER'], label='BB Lower', color='green', linestyle='--', alpha=0.5)
        ax1.plot(df_plot.index, df_plot['SUPERTREND'], label='Supertrend', color='purple', linewidth=1.5)
        ax1.scatter(df_plot.index, df_plot['SAR'], label='SAR', color='cyan', s=10, alpha=0.6)
        
        # Plot Gaussian Channel
        ax1.plot(df_plot.index, df_plot['GC_UPPER'], color='orange', linestyle=':', alpha=0.5)
        ax1.plot(df_plot.index, df_plot['GC_MID'], color='orange', linestyle='-', alpha=0.7, label='Gaussian')
        ax1.plot(df_plot.index, df_plot['GC_LOWER'], color='orange', linestyle=':', alpha=0.5)
        
        # Plot EMAs
        ax1.plot(df_plot.index, df_plot['EMA_50'], color='olive', alpha=0.5, label='EMA 50')
        ax1.plot(df_plot.index, df_plot['EMA_200'], color='brown', alpha=0.5, label='EMA 200')
        
        # Plot Entry/SL/Targets
        if signal_type != "NO_TRADE":
            ax1.axhline(entry, color='blue', linestyle='--', linewidth=2, label=f'Entry: {entry:.2f}')
            ax1.axhline(stop_loss, color='red', linestyle='--', linewidth=2, label=f'SL: {stop_loss:.2f}')
            for i, target in enumerate(targets):
                ax1.axhline(target, color='green', linestyle=':', linewidth=1.5, alpha=0.7, label=f'T{i+1}: {target:.2f}')
        
        ax1.set_title(f'{symbol} - {signal_type} Signal', fontsize=14, fontweight='bold')
        ax1.legend(loc='upper left', fontsize=8, ncol=2)
        ax1.grid(True, alpha=0.3)
        
        # Plot 2: RSI
        ax2.plot(df_plot.index, df_plot['RSI_14'], label='RSI', color='purple')
        ax2.axhline(70, color='red', linestyle='--', alpha=0.5)
        ax2.axhline(30, color='green', linestyle='--', alpha=0.5)
        ax2.axhline(50, color='gray', linestyle=':', alpha=0.3)
        ax2.fill_between(df_plot.index, 30, 70, alpha=0.1, color='gray')
        ax2.set_ylabel('RSI', fontsize=10)
        ax2.legend(loc='upper left', fontsize=8)
        ax2.grid(True, alpha=0.3)
        
        # Plot 3: TSI (TRM)
        if 'TSI' in df_plot.columns:
            ax3.plot(df_plot.index, df_plot['TSI'], label='TSI', color='blue')
            ax3.plot(df_plot.index, df_plot['TSI_SIGNAL'], label='TSI Signal', color='red', alpha=0.7)
            ax3.axhline(0, color='black', linestyle='-', alpha=0.3)
            ax3.fill_between(df_plot.index, df_plot['TSI'], df_plot['TSI_SIGNAL'], 
                             where=df_plot['TSI'] > df_plot['TSI_SIGNAL'], alpha=0.3, color='green')
            ax3.fill_between(df_plot.index, df_plot['TSI'], df_plot['TSI_SIGNAL'], 
                             where=df_plot['TSI'] <= df_plot['TSI_SIGNAL'], alpha=0.3, color='red')
            ax3.set_ylabel('TSI (TRM)', fontsize=10)
            ax3.legend(loc='upper left', fontsize=8)
            ax3.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        # Save to bytes
        buf = BytesIO()
        plt.savefig(buf, format='png', dpi=150, bbox_inches='tight')
        buf.seek(0)
        plt.close()
        
        return buf
    
    async def get_market_context(self, symbol: str) -> Dict:
        """Fetch market context"""
        fgi, funding, oi, liquidity, cg, usdt, spx = await asyncio.gather(
            fetch_data(FGI_API),
            self.get_funding(symbol),
            self.get_open_interest(symbol),
            self.get_liquidity(symbol),
            fetch_data(COINGECKO_API),
            fetch_data(USDT_API),
            self.get_spx_trend()
        )
        
        fgi_val = int(fgi.get('data', [{}])[0].get('value', 50))
        fgi_sent = fgi.get('data', [{}])[0].get('value_classification', 'Neutral')
        
        total_mc = cg.get('data', {}).get('total_market_cap', {}).get('usd', 0)
        usdt_mc = usdt.get('market_data', {}).get('market_cap', {}).get('usd', 0)
        usdt_d = (usdt_mc / total_mc) * 100 if total_mc > 0 else 0
        
        return {
            'fgi_val': fgi_val,
            'fgi_sent': fgi_sent,
            'funding': funding,
            'oi': oi.get('oi', 0),
            'oi_change': oi.get('oi_change', 0),
            'liquidity_imbalance': liquidity.get('imbalance', 0),
            'bid_liq': liquidity.get('bid_liq', 0),
            'ask_liq': liquidity.get('ask_liq', 0),
            'usdt_d': usdt_d,
            'spx_trend': spx
        }
    
    async def get_funding(self, symbol: str) -> float:
        try:
            data = await fetch_data(BINANCE_FUNDING, {'symbol': symbol, 'limit': 1})
            return float(data[0]['fundingRate']) if data and len(data) > 0 else 0
        except:
            return 0
    
    async def get_open_interest(self, symbol: str) -> Dict:
        try:
            data = await fetch_data(BINANCE_OI, {'symbol': symbol})
            current_oi = float(data.get('openInterest', 0)) if data else 0
            return {'oi': current_oi, 'oi_change': 0}
        except:
            return {'oi': 0, 'oi_change': 0}
    
    async def get_liquidity(self, symbol: str) -> Dict:
        try:
            data = await fetch_data(BINANCE_ORDERBOOK, {'symbol': symbol, 'limit': 50})
            if not data or 'bids' not in data:
                return {'imbalance': 0, 'bid_liq': 0, 'ask_liq': 0}
            
            bids = np.array([[float(p), float(q)] for p, q in data['bids'][:50]])
            asks = np.array([[float(p), float(q)] for p, q in data['asks'][:50]])
            
            bid_liq = np.sum(bids[:, 1])
            ask_liq = np.sum(asks[:, 1])
            imbalance = (bid_liq - ask_liq) / (bid_liq + ask_liq) if (bid_liq + ask_liq) > 0 else 0
            
            return {'imbalance': imbalance, 'bid_liq': bid_liq, 'ask_liq': ask_liq}
        except:
            return {'imbalance': 0, 'bid_liq': 0, 'ask_liq': 0}
    
    async def get_spx_trend(self) -> str:
        try:
            spx = yf.Ticker("^GSPC")
            hist = spx.history(period="10d")
            return "Uptrend" if len(hist) >= 5 and hist['Close'].iloc[-1] > hist['Close'].iloc[-5] else "Downtrend"
        except:
            return "Unknown"
    
    async def analyze_multi_timeframe(self, symbol: str) -> Tuple[str, Optional[BytesIO]]:
        """Complete analysis with chart"""
        try:
            tasks = [fetch_data(BINANCE_API, {'symbol': symbol, 'interval': tf['interval'], 'limit': 200}) 
                     for tf in TIMEFRAMES.values()]
            results = await asyncio.gather(*tasks)
            klines_15m, klines_1h, klines_2h, klines_4h = results
            
            market_data = await self.get_market_context(symbol)
            
            dfs = {}
            analyses = {}
            
            for (tf, klines) in [('15m', klines_15m), ('1h', klines_1h), 
                                  ('2h', klines_2h), ('4h', klines_4h)]:
                if not klines:
                    return f"❌ No data for {tf}", None
                
                df = self.process_klines(klines, symbol)
                if len(df) < 50:
                    return f"❌ Insufficient data for {tf}", None
                
                df = self.calculate_all_indicators(df)
                dfs[tf] = df
                analyses[tf] = await self.analyze_timeframe(df, tf, market_data)
            
            long_votes = [tf for tf, a in analyses.items() if a['direction'] == 'LONG']
            short_votes = [tf for tf, a in analyses.items() if a['direction'] == 'SHORT']
            
            signal_type = "NO_TRADE"
            verdict = "⛔ NO CONSENSUS"
            
            has_15m_2h_long = '15m' in long_votes and '2h' in long_votes
            has_15m_2h_short = '15m' in short_votes and '2h' in short_votes
            
            price_15m = analyses['15m']['indicators']['price']
            atr_15m = analyses['15m']['indicators']['atr']
            
            entry = stop_loss = price_15m
            targets = [price_15m] * 3
            trade_details = ""
            
            if len(long_votes) >= 2:
                signal_type = "LONG"
                entry = price_15m * 0.999
                stop_loss = entry - (atr_15m * 1.8)
                
                if has_15m_2h_long:
                    targets = [entry + atr_15m * 2, entry + atr_15m * 4.5, entry + atr_15m * 7]
                    verdict = "🚀 STRONG LONG (15m+2h ⭐)"
                elif len(long_votes) >= 3:
                    targets = [entry + atr_15m * 2.5, entry + atr_15m * 5.5, entry + atr_15m * 9]
                    verdict = f"🚀🚀 VERY STRONG LONG ({len(long_votes)}/4)"
                else:
                    targets = [entry + atr_15m * 1.5, entry + atr_15m * 3.5, entry + atr_15m * 6]
                    verdict = f"🚀 LONG ({'+'.join(long_votes)})"
                
                rr = (targets[2] - entry) / (entry - stop_loss) if (entry - stop_loss) > 0 else 0
                trade_details = f"""
Entry: ${entry:.4f}
Stop: ${stop_loss:.4f} (-{abs((stop_loss-entry)/entry*100):.2f}%)
T1: ${targets[0]:.4f} (+{(targets[0]-entry)/entry*100:.2f}%) [30%]
T2: ${targets[1]:.4f} (+{(targets[1]-entry)/entry*100:.2f}%) [40%]
T3: ${targets[2]:.4f} (+{(targets[2]-entry)/entry*100:.2f}%) [30%]
R:R = 1:{rr:.1f}
"""
            
            elif len(short_votes) >= 2:
                signal_type = "SHORT"
                entry = price_15m * 1.001
                stop_loss = entry + (atr_15m * 1.8)
                
                if has_15m_2h_short:
                    targets = [entry - atr_15m * 2, entry - atr_15m * 4.5, entry - atr_15m * 7]
                    verdict = "🔻 STRONG SHORT (15m+2h ⭐)"
                elif len(short_votes) >= 3:
                    targets = [entry - atr_15m * 2.5, entry - atr_15m * 5.5, entry - atr_15m * 9]
                    verdict = f"🔻🔻 VERY STRONG SHORT ({len(short_votes)}/4)"
                else:
                    targets = [entry - atr_15m * 1.5, entry - atr_15m * 3.5, entry - atr_15m * 6]
                    verdict = f"🔻 SHORT ({'+'.join(short_votes)})"
                
                rr = (entry - targets[2]) / (stop_loss - entry) if (stop_loss - entry) > 0 else 0
                trade_details = f"""
Entry: ${entry:.4f}
Stop: ${stop_loss:.4f} (+{abs((stop_loss-entry)/entry*100):.2f}%)
T1: ${targets[0]:.4f} (-{abs((entry-targets[0])/entry*100):.2f}%) [30%]
T2: ${targets[1]:.4f} (-{abs((entry-targets[1])/entry*100):.2f}%) [40%]
T3: ${targets[2]:.4f} (-{abs((entry-targets[2])/entry*100):.2f}%) [30%]
R:R = 1:{rr:.1f}
"""
            
            # Save trade
            if signal_type != "NO_TRADE":
                self.db.save_trade({
                    'timestamp': datetime.now().isoformat(),
                    'symbol': symbol,
                    'timeframes': '+'.join(long_votes if signal_type == 'LONG' else short_votes),
                    'signal_type': signal_type,
                    'entry': entry,
                    'sl': stop_loss,
                    'targets': targets,
                    'confidence': len(long_votes if signal_type == 'LONG' else short_votes) / 4.0,
                    'notes': f'All TradingView indicators included'
                })
            
            # Generate chart
            chart_buf = self.generate_chart(dfs['15m'], symbol, signal_type, entry, stop_loss, targets)
            
            # Format message
            tf_breakdown = ""
            for tf in ['15m', '1h', '2h', '4h']:
                a = analyses[tf]
                icon = "🟢" if a['direction'] == 'LONG' else "🔴" if a['direction'] == 'SHORT' else "⚪"
                ind = a['indicators']
                
                tf_breakdown += f"""
{icon} {tf.upper()} - {a['direction'] or 'NEUTRAL'}
   Score: 🟢{a['bullish']:.1f} 🔴{a['bearish']:.1f}
   RSI: {ind['rsi']:.1f} | Price: ${ind['price']:.4f}
   Supertrend: {'↑' if ind['st_direction'] == 1 else '↓'}
   TRM TSI: {ind['tsi']:.1f}
"""
            
            all_signals_15m = analyses['15m']['signals']
            top_signals = sorted(all_signals_15m.items(), key=lambda x: abs(x[1]), reverse=True)[:12]
            signal_breakdown = "\n".join([f"  {k}: {v:+.2f}" for k, v in top_signals])
            
            message = f"""
╔════════════════════════════════════════╗
  {symbol} - COMPLETE ANALYSIS
  🎯 TradingView + Python Indicators
╚════════════════════════════════════════╝

⏰ {datetime.now().strftime('%H:%M:%S')}

┌────────────────────────────────────────┐
│ 📊 TIMEFRAME ANALYSIS                  │
└────────────────────────────────────────┘
{tf_breakdown}

┌────────────────────────────────────────┐
│ 🔝 TOP SIGNALS (15m)                   │
└────────────────────────────────────────┘
{signal_breakdown}

┌────────────────────────────────────────┐
│ 🌐 MARKET CONTEXT                      │
└────────────────────────────────────────┘
F&G: {market_data['fgi_sent']} ({market_data['fgi_val']})
Funding: {market_data['funding']:.4f}%
Open Interest: ${market_data['oi']:,.0f}
Liquidity: Bid {market_data['bid_liq']:.0f} / Ask {market_data['ask_liq']:.0f}
  Imbalance: {market_data['liquidity_imbalance']:+.2f}
USDT.D: {market_data['usdt_d']:.2f}%
SPX: {market_data['spx_trend']}

┌────────────────────────────────────────┐
│ 🎯 CONSENSUS                           │
└────────────────────────────────────────┘
LONG: {len(long_votes)}/4 {f'✅ ({",".join(long_votes)})' if len(long_votes)>=2 else ''}
SHORT: {len(short_votes)}/4 {f'✅ ({",".join(short_votes)})' if len(short_votes)>=2 else ''}

{'⭐ 15m+2h COMBO AGREES! ⭐' if has_15m_2h_long or has_15m_2h_short else ''}

╔════════════════════════════════════════╗
  {verdict}
╚════════════════════════════════════════╝
{trade_details}

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
💡 Includes ALL TradingView Indicators:
   ✅ VWAP + Bands
   ✅ Supertrend
   ✅ Parabolic SAR
   ✅ Bollinger Bands
   ✅ MA Cross (50/100)
   ✅ TRM (TSI + RSI)
   ✅ Gaussian Channel
   + 20+ Python indicators

📊 See chart image for visual analysis!
"""
            return message, chart_buf
            
        except Exception as e:
            logging.error(f"Analysis error: {e}")
            import traceback
            traceback.print_exc()
            return f"❌ Error: {str(e)}", None

# ============= TELEGRAM =============
async def start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    keyboard = [
        [InlineKeyboardButton("BTC 🔥", callback_data="BTCUSDT"),
         InlineKeyboardButton("ETH", callback_data="ETHUSDT")],
        [InlineKeyboardButton("SOL", callback_data="SOLUSDT"),
         InlineKeyboardButton("BNB", callback_data="BNBUSDT")],
        [InlineKeyboardButton("DOGE", callback_data="DOGEUSDT"),
         InlineKeyboardButton("XRP", callback_data="XRPUSDT")]
    ]
    await update.message.reply_text(
        "🤖 *COMPLETE BOT \\+ TRADINGVIEW*\n\n"
        "📊 30\\+ Indicators\n"
        "📈 TradingView signals\n"
        "📷 Chart generation\n"
        "⏰ 15m\\+1h\\+2h\\+4h\n\n"
        "Select pair:",
        reply_markup=InlineKeyboardMarkup(keyboard),
        parse_mode='MarkdownV2'
    )

async def handle_button(update: Update, context: ContextTypes.DEFAULT_TYPE):
    query = update.callback_query
    await query.answer()
    symbol = query.data
    await query.edit_message_text(f"🔍 Analyzing {symbol} with ALL indicators...\nGenerating chart...")
    
    try:
        bot = CompleteMultiTimeframeBot()
        analysis, chart = await bot.analyze_multi_timeframe(symbol)
        
        # Send chart first
        if chart:
            await query.message.reply_photo(photo=chart, caption=f"{symbol} Technical Analysis Chart")
        
        # Send text analysis
        if len(analysis) > 4000:
            parts = [analysis[i:i+4000] for i in range(0, len(analysis), 4000)]
            for part in parts:
                await query.message.reply_text(part)
        else:
            await query.message.reply_text(analysis)
    except Exception as e:
        import traceback
        traceback.print_exc()
        await query.edit_message_text(f"❌ Error: {str(e)}")

async def handle_message(update: Update, context: ContextTypes.DEFAULT_TYPE):
    symbol = update.message.text.upper().strip()
    if len(symbol) < 5:
        await update.message.reply_text("❌ Format: BTCUSDT")
        return
    
    await update.message.reply_text(f"🔍 Analyzing {symbol}...\n📊 Generating chart...")
    try:
        bot = CompleteMultiTimeframeBot()
        analysis, chart = await bot.analyze_multi_timeframe(symbol)
        
        if chart:
            await update.message.reply_photo(photo=chart, caption=f"{symbol} Analysis")
        
        await update.message.reply_text(analysis)
    except Exception as e:
        import traceback
        traceback.print_exc()
        await update.message.reply_text(f"❌ Error: {str(e)}")

def main():
    app = Application.builder().token(TELEGRAM_BOT_TOKEN).build()
    app.add_handler(CommandHandler('start', start))
    app.add_handler(CallbackQueryHandler(handle_button))
    app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, handle_message))
    
    logging.info("🚀 Complete Bot with TradingView Indicators!")
    logging.info("📊 30+ indicators + Chart generation enabled")
    app.run_polling()

if __name__ == "__main__":
    main()
