"""
Minimal Flask API for Flutter Trading App
No heavy dependencies
"""

from flask import Flask, jsonify, request
from flask_cors import CORS
import requests
import pandas as pd
import numpy as np
from datetime import datetime
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)
CORS(app)

# APIs
BINANCE_API = "https://api.binance.com/api/v3/klines"
BINANCE_FUNDING = "https://fapi.binance.com/fapi/v1/fundingRate"
FGI_API = "https://api.alternative.me/fng/?limit=1"

# ============= SIMPLE INDICATORS =============

def calc_rsi(prices, period=14):
    """Simple RSI calculation"""
    deltas = np.diff(prices)
    gains = np.where(deltas > 0, deltas, 0)
    losses = np.where(deltas < 0, -deltas, 0)
    
    avg_gain = np.mean(gains[-period:])
    avg_loss = np.mean(losses[-period:])
    
    if avg_loss == 0:
        return 100
    
    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))
    return round(rsi, 2)

def calc_ema(prices, period):
    """Simple EMA calculation"""
    multiplier = 2 / (period + 1)
    ema = [prices[0]]
    
    for price in prices[1:]:
        ema.append((price - ema[-1]) * multiplier + ema[-1])
    
    return ema

def calc_sma(prices, period):
    """Simple Moving Average"""
    if len(prices) < period:
        return prices[-1]
    return sum(prices[-period:]) / period

# ============= API ENDPOINTS =============

@app.route('/')
def home():
    """Health check"""
    return jsonify({
        'status': 'online',
        'service': 'Trading Bot API',
        'version': '1.0',
        'timestamp': datetime.now().isoformat()
    })

@app.route('/api/analyze/<symbol>')
def analyze_symbol(symbol):
    """Analyze a trading symbol"""
    try:
        timeframe = request.args.get('timeframe', '15m')
        
        # Fetch data
        response = requests.get(
            BINANCE_API,
            params={
                'symbol': symbol.upper(),
                'interval': timeframe,
                'limit': 100
            },
            timeout=10
        )
        
        if response.status_code != 200:
            return jsonify({
                'success': False,
                'error': 'Failed to fetch data from Binance'
            }), 500
        
        # Process data
        klines = response.json()
        
        closes = [float(k[4]) for k in klines]
        highs = [float(k[2]) for k in klines]
        lows = [float(k[3]) for k in klines]
        volumes = [float(k[5]) for k in klines]
        times = [k[0] for k in klines]
        
        # Calculate indicators
        rsi = calc_rsi(closes)
        ema_10 = calc_ema(closes, 10)[-1]
        ema_20 = calc_ema(closes, 20)[-1]
        ema_50 = calc_ema(closes, 50)[-1]
        sma_20 = calc_sma(closes, 20)
        
        # Bollinger Bands
        std = np.std(closes[-20:])
        bb_upper = sma_20 + (2 * std)
        bb_lower = sma_20 - (2 * std)
        
        # ATR
        tr_list = []
        for i in range(1, len(closes)):
            hl = highs[i] - lows[i]
            hc = abs(highs[i] - closes[i-1])
            lc = abs(lows[i] - closes[i-1])
            tr = max(hl, hc, lc)
            tr_list.append(tr)
        atr = sum(tr_list[-14:]) / 14
        
        current_price = closes[-1]
        
        # Generate signal
        bullish_score = 0
        bearish_score = 0
        
        # RSI
        if rsi < 30:
            bullish_score += 3
        elif rsi > 70:
            bearish_score += 3
        
        # EMA Trend
        if current_price > ema_50:
            bullish_score += 2
        else:
            bearish_score += 2
        
        # EMA Cross
        if ema_10 > ema_20:
            bullish_score += 2
        else:
            bearish_score += 2
        
        # BB Position
        if current_price <= bb_lower:
            bullish_score += 2
        elif current_price >= bb_upper:
            bearish_score += 2
        
        # Determine signal
        if bullish_score >= 5 and bullish_score > bearish_score:
            signal_type = 'LONG'
            entry = current_price * 0.999
            stop_loss = entry - (atr * 1.8)
            targets = [
                entry + (atr * 2),
                entry + (atr * 4),
                entry + (atr * 6)
            ]
        elif bearish_score >= 5 and bearish_score > bullish_score:
            signal_type = 'SHORT'
            entry = current_price * 1.001
            stop_loss = entry + (atr * 1.8)
            targets = [
                entry - (atr * 2),
                entry - (atr * 4),
                entry - (atr * 6)
            ]
        else:
            signal_type = 'NO_TRADE'
            entry = current_price
            stop_loss = current_price
            targets = [current_price, current_price, current_price]
        
        return jsonify({
            'success': True,
            'symbol': symbol.upper(),
            'timeframe': timeframe,
            'timestamp': datetime.now().isoformat(),
            'signal': {
                'type': signal_type,
                'entry': round(entry, 2),
                'stop_loss': round(stop_loss, 2),
                'targets': [round(t, 2) for t in targets],
                'bullish_score': bullish_score,
                'bearish_score': bearish_score,
                'confidence': round(max(bullish_score, bearish_score) / 10, 2)
            },
            'indicators': {
                'price': round(current_price, 2),
                'rsi': round(rsi, 2),
                'ema_10': round(ema_10, 2),
                'ema_20': round(ema_20, 2),
                'ema_50': round(ema_50, 2),
                'bb_upper': round(bb_upper, 2),
                'bb_mid': round(sma_20, 2),
                'bb_lower': round(bb_lower, 2),
                'atr': round(atr, 2)
            },
            'chart_data': {
                'timestamps': times[-50:],
                'prices': [round(p, 2) for p in closes[-50:]],
                'volumes': [round(v, 2) for v in volumes[-50:]],
                'ema_10': [round(p, 2) for p in ema_10[-50:]] if isinstance(ema_10, list) else [],
                'ema_20': [round(p, 2) for p in ema_20[-50:]] if isinstance(ema_20, list) else []
            }
        })
        
    except Exception as e:
        logger.error(f"Error: {e}")
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

@app.route('/api/signals')
def get_signals():
    """Get signals for multiple symbols"""
    try:
        symbols_param = request.args.get('symbols', 'BTCUSDT,ETHUSDT,SOLUSDT')
        symbols = symbols_param.split(',')
        
        signals = []
        for symbol in symbols:
            try:
                response = requests.get(
                    BINANCE_API,
                    params={'symbol': symbol.upper(), 'interval': '15m', 'limit': 20},
                    timeout=5
                )
                
                if response.status_code == 200:
                    klines = response.json()
                    current = float(klines[-1][4])
                    prev = float(klines[-2][4])
                    change = ((current - prev) / prev) * 100
                    
                    signals.append({
                        'symbol': symbol.upper(),
                        'price': round(current, 2),
                        'change_24h': round(change, 2),
                        'signal': 'LONG' if change > 0 else 'SHORT',
                        'strength': min(abs(change) * 10, 100)
                    })
            except:
                continue
        
        return jsonify({
            'success': True,
            'signals': signals
        })
        
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500

@app.route('/api/market-data/<symbol>')
def get_market_data(symbol):
    """Get market data"""
    try:
        # Fear & Greed
        fgi = requests.get(FGI_API, timeout=5).json()
        fgi_value = int(fgi.get('data', [{}])[0].get('value', 50))
        fgi_sentiment = fgi.get('data', [{}])[0].get('value_classification', 'Neutral')
        
        # Funding
        try:
            funding = requests.get(
                BINANCE_FUNDING,
                params={'symbol': symbol.upper(), 'limit': 1},
                timeout=5
            ).json()
            funding_rate = float(funding[0]['fundingRate']) * 100
        except:
            funding_rate = 0.0
        
        return jsonify({
            'success': True,
            'data': {
                'fgi_value': fgi_value,
                'fgi_sentiment': fgi_sentiment,
                'funding_rate': round(funding_rate, 4)
            }
        })
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500

@app.route('/api/timeframes/<symbol>')
def get_timeframes(symbol):
    """Multi-timeframe analysis"""
    try:
        timeframes = ['15m', '1h', '2h', '4h']
        results = []
        
        for tf in timeframes:
            response = requests.get(
                BINANCE_API,
                params={'symbol': symbol.upper(), 'interval': tf, 'limit': 50},
                timeout=5
            )
            
            if response.status_code == 200:
                klines = response.json()
                closes = [float(k[4]) for k in klines]
                rsi = calc_rsi(closes)
                
                # Simple scoring
                if rsi < 30:
                    direction = 'LONG'
                    bullish = 8.0
                    bearish = 2.0
                elif rsi > 70:
                    direction = 'SHORT'
                    bullish = 2.0
                    bearish = 8.0
                else:
                    direction = 'NEUTRAL'
                    bullish = 5.0
                    bearish = 5.0
                
                results.append({
                    'timeframe': tf,
                    'direction': direction,
                    'bullish_score': bullish,
                    'bearish_score': bearish,
                    'rsi': round(rsi, 1)
                })
        
        return jsonify({
            'success': True,
            'timeframes': results
        })
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500

if __name__ == '__main__':
    print("\n" + "="*50)
    print("🚀 Trading Bot API Server")
    print("="*50)
    print("📡 Server: http://0.0.0.0:5000")
    print("\n📊 Available Endpoints:")
    print("   GET  /")
    print("   GET  /api/analyze/<symbol>")
    print("   GET  /api/signals")
    print("   GET  /api/market-data/<symbol>")
    print("   GET  /api/timeframes/<symbol>")
    print("\n✅ Server starting...\n")
    
    app.run(host='0.0.0.0', port=5000, debug=True, threaded=True)
