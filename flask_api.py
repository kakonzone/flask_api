"""
COMPLETE Flask API v6.0 - AUTO-START READY
All endpoints + Background process support
"""

from flask import Flask, jsonify, request
from flask_cors import CORS
import requests
import numpy as np
from datetime import datetime
import logging
import sqlite3
from typing import Dict
from scipy.ndimage import gaussian_filter1d
import yfinance as yf
import sys
import os

# ✅ Add this for auto-start support
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('flask_api.log'),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

app = Flask(__name__)
CORS(app)

# ============= ALL APIs =============
BINANCE_API = "https://api.binance.com/api/v3/klines"
BINANCE_FUNDING = "https://fapi.binance.com/fapi/v1/fundingRate"
BINANCE_OI = "https://fapi.binance.com/fapi/v1/openInterest"
BINANCE_ORDERBOOK = "https://api.binance.com/api/v3/depth"
FGI_API = "https://api.alternative.me/fng/?limit=1"
COINGECKO_API = "https://api.coingecko.com/api/v3/global"
USDT_API = "https://api.coingecko.com/api/v3/coins/tether"

# ============= DATABASE =============
class TradeDatabase:
    def __init__(self):
        self.conn = sqlite3.connect('trades.db', check_same_thread=False)
        self.init_db()
    
    def init_db(self):
        c = self.conn.cursor()
        c.execute('''CREATE TABLE IF NOT EXISTS trades
                     (id INTEGER PRIMARY KEY AUTOINCREMENT,
                      timestamp TEXT, symbol TEXT, timeframe TEXT,
                      signal_type TEXT, entry_price REAL, stop_loss REAL,
                      targets TEXT, confidence REAL, bullish_score REAL,
                      bearish_score REAL, status TEXT DEFAULT 'ACTIVE',
                      exit_price REAL, pnl REAL, notes TEXT)''')
        self.conn.commit()
    
    def save_trade(self, data: Dict):
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
        return c.lastrowid
    
    def get_all_trades(self, limit=100, status=None):
        c = self.conn.cursor()
        if status:
            c.execute('SELECT * FROM trades WHERE status=? ORDER BY id DESC LIMIT ?', (status, limit))
        else:
            c.execute('SELECT * FROM trades ORDER BY id DESC LIMIT ?', (limit,))
        
        columns = ['id', 'timestamp', 'symbol', 'timeframe', 'signal_type',
                   'entry_price', 'stop_loss', 'targets', 'confidence',
                   'bullish_score', 'bearish_score', 'status', 'exit_price', 'pnl', 'notes']
        return [dict(zip(columns, row)) for row in c.fetchall()]
    
    def update_trade(self, trade_id: int, exit_price: float, pnl: float, status: str):
        c = self.conn.cursor()
        c.execute('UPDATE trades SET exit_price=?, pnl=?, status=? WHERE id=?',
                  (exit_price, pnl, status, trade_id))
        self.conn.commit()

db = TradeDatabase()

# ============= INDICATORS =============

def calc_rsi(prices, period=14):
    deltas = np.diff(prices)
    gains = np.where(deltas > 0, deltas, 0)
    losses = np.where(deltas < 0, -deltas, 0)
    avg_gain = np.mean(gains[-period:])
    avg_loss = np.mean(losses[-period:])
    if avg_loss == 0:
        return 100
    rs = avg_gain / avg_loss
    return round(100 - (100 / (1 + rs)), 2)

def calc_ema(prices, period):
    multiplier = 2 / (period + 1)
    ema = [prices[0]]
    for price in prices[1:]:
        ema.append((price - ema[-1]) * multiplier + ema[-1])
    return ema

def calc_sma(prices, period):
    if len(prices) < period:
        return prices[-1]
    return sum(prices[-period:]) / period

def calc_macd(prices):
    ema_12 = calc_ema(prices, 12)
    ema_26 = calc_ema(prices, 26)
    macd_line = [ema_12[i] - ema_26[i] for i in range(len(prices))]
    signal_line = calc_ema(macd_line, 9)
    return macd_line[-1], signal_line[-1]

def calc_atr(highs, lows, closes, period=14):
    tr_list = []
    for i in range(1, len(closes)):
        hl = highs[i] - lows[i]
        hc = abs(highs[i] - closes[i-1])
        lc = abs(lows[i] - closes[i-1])
        tr = max(hl, hc, lc)
        tr_list.append(tr)
    return sum(tr_list[-period:]) / period if tr_list else 0

def calc_supertrend(highs, lows, closes, period=10, multiplier=3.0):
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

def calc_sar(highs, lows):
    if len(highs) < 2:
        return lows[0]
    trend = 1 if highs[-1] > highs[-2] else -1
    sar = lows[-2] if trend == 1 else highs[-2]
    return round(sar, 4)

def calc_gaussian_channel(prices, length=100, std_dev=2.0):
    if len(prices) < length:
        return prices[-1], prices[-1]
    smoothed = gaussian_filter1d(prices[-length:], sigma=length/10)
    mean = smoothed[-1]
    std = np.std(prices[-length:])
    upper = mean + (std_dev * std)
    lower = mean - (std_dev * std)
    return round(upper, 4), round(lower, 4)

def calc_trm(highs, lows, closes, length=20):
    if len(closes) < length:
        return 0
    momentum = closes[-1] - closes[-length]
    volatility = max(highs[-length:]) - min(lows[-length:])
    if volatility == 0:
        return 0
    trm = (momentum / volatility) * 100
    return round(trm, 2)

# ============= MARKET DATA FUNCTIONS =============

def get_orderbook_liquidity(symbol: str) -> Dict:
    try:
        response = requests.get(
            BINANCE_ORDERBOOK,
            params={'symbol': symbol.upper(), 'limit': 50},
            timeout=5
        )
        
        if response.status_code != 200:
            return {'imbalance': 0, 'bid_liq': 0, 'ask_liq': 0}
        
        data = response.json()
        
        bids = np.array([[float(p), float(q)] for p, q in data['bids'][:50]])
        asks = np.array([[float(p), float(q)] for p, q in data['asks'][:50]])
        
        bid_liq = np.sum(bids[:, 1])
        ask_liq = np.sum(asks[:, 1])
        
        imbalance = (bid_liq - ask_liq) / (bid_liq + ask_liq) if (bid_liq + ask_liq) > 0 else 0
        
        return {
            'imbalance': round(imbalance, 4),
            'bid_liq': round(bid_liq, 2),
            'ask_liq': round(ask_liq, 2)
        }
    except Exception as e:
        logger.error(f"Orderbook error: {e}")
        return {'imbalance': 0, 'bid_liq': 0, 'ask_liq': 0}

def get_btc_usdt_dominance() -> Dict:
    try:
        cg_response = requests.get(COINGECKO_API, timeout=5)
        cg_data = cg_response.json()
        total_mc = cg_data.get('data', {}).get('total_market_cap', {}).get('usd', 0)
        
        usdt_response = requests.get(USDT_API, timeout=5)
        usdt_data = usdt_response.json()
        usdt_mc = usdt_data.get('market_data', {}).get('market_cap', {}).get('usd', 0)
        
        usdt_dominance = (usdt_mc / total_mc * 100) if total_mc > 0 else 0
        btc_dominance = cg_data.get('data', {}).get('market_cap_percentage', {}).get('btc', 48.5)
        
        return {
            'btc_dominance': round(btc_dominance, 2),
            'usdt_dominance': round(usdt_dominance, 2),
            'total_market_cap': total_mc
        }
    except Exception as e:
        logger.error(f"Dominance error: {e}")
        return {'btc_dominance': 48.5, 'usdt_dominance': 5.2, 'total_market_cap': 0}

def get_spx_trend() -> str:
    try:
        spx = yf.Ticker('^GSPC')
        hist = spx.history(period='10d')
        if len(hist) >= 5:
            return 'Uptrend' if hist['Close'].iloc[-1] > hist['Close'].iloc[-5] else 'Downtrend'
        return 'Unknown'
    except Exception as e:
        logger.error(f"SPX error: {e}")
        return 'Unknown'

# ============= ANALYSIS =============

def analyze_symbol_full(symbol: str, timeframe: str = '15m'):
    try:
        response = requests.get(
            BINANCE_API,
            params={'symbol': symbol.upper(), 'interval': timeframe, 'limit': 200},
            timeout=10
        )
        
        if response.status_code != 200:
            return {'error': 'Failed to fetch data'}
        
        klines = response.json()
        if not klines:
            return {'error': 'No data'}
        
        opens = [float(k[1]) for k in klines]
        highs = [float(k[2]) for k in klines]
        lows = [float(k[3]) for k in klines]
        closes = [float(k[4]) for k in klines]
        volumes = [float(k[5]) for k in klines]
        
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
        bullish_score = 0
        bearish_score = 0
        
        if rsi < 30:
            bullish_score += 3
        elif rsi > 70:
            bearish_score += 3
        elif rsi < 40:
            bullish_score += 1.5
        elif rsi > 60:
            bearish_score += 1.5
        
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
        
        if macd > macd_signal:
            bullish_score += 2
        else:
            bearish_score += 2
        
        if current_price <= bb_lower:
            bullish_score += 2
        elif current_price >= bb_upper:
            bearish_score += 2
        
        if current_price > vwap:
            bullish_score += 1.5
        else:
            bearish_score += 1.5
        
        if st_direction == 1:
            bullish_score += 2.5
        else:
            bearish_score += 2.5
        
        if current_price > sar:
            bullish_score += 1.5
        else:
            bearish_score += 1.5
        
        if current_price < gaussian_lower:
            bullish_score += 1
        elif current_price > gaussian_upper:
            bearish_score += 1
        
        if trm > 20:
            bullish_score += 1
        elif trm < -20:
            bearish_score += 1
        
        min_score = 7
        
        if bullish_score >= min_score and bullish_score > bearish_score:
            signal_type = 'LONG'
            entry = current_price * 0.999
            stop_loss = entry - (atr * 1.8)
            targets = [entry + (atr * 2), entry + (atr * 4), entry + (atr * 6)]
        elif bearish_score >= min_score and bearish_score > bullish_score:
            signal_type = 'SHORT'
            entry = current_price * 1.001
            stop_loss = entry + (atr * 1.8)
            targets = [entry - (atr * 2), entry - (atr * 4), entry - (atr * 6)]
        else:
            signal_type = 'NO_TRADE'
            entry = current_price
            stop_loss = current_price
            targets = []
        
        confidence = round(max(bullish_score, bearish_score) / 20 * 100, 1)
        
        return {
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
        
    except Exception as e:
        logger.error(f"Analysis error: {e}")
        return {'error': str(e)}

# ============= FLASK ROUTES =============

@app.route('/')
def home():
    return jsonify({
        'status': 'online',
        'version': '6.0 AUTO-START',
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
            'trades_update': '/api/trades/<id> [PUT]'
        }
    })

@app.route('/api/analysis/<symbol>')
def get_analysis(symbol):
    timeframe = request.args.get('timeframe', '15m')
    return jsonify(analyze_symbol_full(symbol, timeframe))

@app.route('/api/timeframes/<symbol>')
def get_timeframes(symbol):
    try:
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
        return jsonify({'error': str(e)}), 500

@app.route('/api/symbols')
def search_symbols():
    query = request.args.get('search', '').upper()
    all_symbols = [
        'BTCUSDT', 'ETHUSDT', 'BNBUSDT', 'SOLUSDT', 'XRPUSDT',
        'ADAUSDT', 'DOGEUSDT', 'DOTUSDT', 'MATICUSDT', 'AVAXUSDT',
        'LINKUSDT', 'UNIUSDT', 'ATOMUSDT', 'LTCUSDT', 'SHIBUSDT'
    ]
    if query:
        all_symbols = [s for s in all_symbols if query in s]
    return jsonify({'symbols': all_symbols})

@app.route('/api/market-context/<symbol>')
def get_market_context(symbol):
    try:
        context = {}
        
        try:
            fgi = requests.get(FGI_API, timeout=5).json()
            context['fear_greed_index'] = int(fgi['data'][0]['value'])
            context['fgi_classification'] = fgi['data'][0]['value_classification']
        except:
            context['fear_greed_index'] = 50
            context['fgi_classification'] = 'Neutral'
        
        try:
            funding = requests.get(BINANCE_FUNDING, params={'symbol': symbol.upper(), 'limit': 1}, timeout=5).json()
            context['funding_rate'] = float(funding[0]['fundingRate'])
        except:
            context['funding_rate'] = 0.0001
        
        try:
            oi = requests.get(BINANCE_OI, params={'symbol': symbol.upper()}, timeout=5).json()
            context['open_interest'] = float(oi['openInterest'])
        except:
            context['open_interest'] = 15000000000
        
        liquidity = get_orderbook_liquidity(symbol)
        context['liquidity_imbalance'] = liquidity['imbalance']
        context['bid_liquidity'] = liquidity['bid_liq']
        context['ask_liquidity'] = liquidity['ask_liq']
        
        dominance = get_btc_usdt_dominance()
        context['btc_dominance'] = dominance['btc_dominance']
        context['usdt_dominance'] = dominance['usdt_dominance']
        context['total_market_cap'] = dominance['total_market_cap']
        
        context['spx_trend'] = get_spx_trend()
        context['oi_change'] = 0
        
        return jsonify(context)
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/orderbook/<symbol>')
def get_orderbook(symbol):
    return jsonify(get_orderbook_liquidity(symbol))

@app.route('/api/dominance')
def get_dominance():
    return jsonify(get_btc_usdt_dominance())

@app.route('/api/spx-trend')
def get_spx():
    return jsonify({'spx_trend': get_spx_trend()})

@app.route('/api/funding/<symbol>')
def get_funding(symbol):
    try:
        funding = requests.get(BINANCE_FUNDING, params={'symbol': symbol.upper(), 'limit': 1}, timeout=5).json()
        return jsonify({'funding_rate': float(funding[0]['fundingRate'])})
    except:
        return jsonify({'funding_rate': 0.0001})

@app.route('/api/open-interest/<symbol>')
def get_open_interest(symbol):
    try:
        oi = requests.get(BINANCE_OI, params={'symbol': symbol.upper()}, timeout=5).json()
        return jsonify({'open_interest': float(oi['openInterest'])})
    except:
        return jsonify({'open_interest': 15000000000})

@app.route('/api/fear-greed')
def get_fear_greed():
    try:
        fgi = requests.get(FGI_API, timeout=5).json()
        return jsonify({
            'value': int(fgi['data'][0]['value']),
            'classification': fgi['data'][0]['value_classification']
        })
    except:
        return jsonify({'value': 50, 'classification': 'Neutral'})

@app.route('/api/trades', methods=['GET', 'POST'])
def handle_trades():
    if request.method == 'POST':
        try:
            data = request.json
            trade_id = db.save_trade(data)
            return jsonify({'success': True, 'trade_id': trade_id})
        except Exception as e:
            return jsonify({'success': False, 'error': str(e)}), 500
    else:
        status = request.args.get('status')
        limit = int(request.args.get('limit', 100))
        trades = db.get_all_trades(limit, status)
        return jsonify({'trades': trades})

@app.route('/api/trades/<int:trade_id>', methods=['PUT'])
def update_trade(trade_id):
    try:
        data = request.json
        db.update_trade(trade_id, data['exit_price'], data['pnl'], data['status'])
        return jsonify({'success': True})
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500

# ✅ Health check endpoint for auto-start verification
@app.route('/api/health')
def health_check():
    return jsonify({'status': 'healthy', 'timestamp': datetime.now().isoformat()})

if __name__ == '__main__':
    print("\n" + "="*80)
    print("🚀 Flask API v6.0 - AUTO-START READY")
    print("="*80)
    print("✅ Server: http://0.0.0.0:5000")
    print("✅ Logs: flask_api.log")
    print("✅ Health Check: /api/health")
    print("="*80 + "\n")
    
    # ✅ Disable debug in production for auto-start
    app.run(host='0.0.0.0', port=5000, debug=False, threaded=True, use_reloader=False)