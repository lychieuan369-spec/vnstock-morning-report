import os
import time
import numpy as np
import pandas as pd
import requests
import yfinance as yf
from datetime import datetime, timedelta

BOT_TOKEN = os.environ['TELEGRAM_BOT_TOKEN']
CHAT_ID = os.environ['TELEGRAM_CHAT_ID']

TODAY = datetime.today()
START_DATE = (TODAY - timedelta(days=365)).strftime('%Y-%m-%d')  # 1 year for EMA200
END_DATE = TODAY.strftime('%Y-%m-%d')

STOCK_SYMBOLS = ['VCB', 'HPG', 'FPT', 'MBB', 'VIC', 'SSI', 'VHM', 'TCB', 'ACB', 'BID', 'CTG', 'GAS', 'MSN', 'PLX', 'VRE', 'VPB', 'MSR']

def send_telegram(text):
    url = f"https://api.telegram.org/bot{BOT_TOKEN}/sendMessage"
    payload = {'chat_id': CHAT_ID, 'text': text, 'parse_mode': 'HTML'}
    try:
        resp = requests.post(url, json=payload, timeout=15)
        resp.raise_for_status()
        print(f"[OK] Sent (status {resp.status_code})")
    except Exception as e:
        print(f"[ERROR] Telegram send failed: {e}")

def fetch_vnstock(symbol, data_type='stock'):
    try:
        from vnstock import Vnstock
        source = 'TCBS' if data_type == 'index' else 'VCI'
        stock = Vnstock().stock(symbol=symbol, source=source)
        df = stock.quote.history(start=START_DATE, end=END_DATE, interval='1D')
        if df is None or df.empty:
            return None
        df.columns = [c.lower() for c in df.columns]
        col_map = {}
        for col in df.columns:
            if col in ('close', 'c', 'adjclose'): col_map[col] = 'close'
            elif col in ('low', 'l'): col_map[col] = 'low'
            elif col in ('high', 'h'): col_map[col] = 'high'
            elif col in ('open', 'o'): col_map[col] = 'open'
            elif col in ('volume', 'vol', 'v'): col_map[col] = 'volume'
        if col_map:
            df = df.rename(columns=col_map)
        if 'time' not in df.columns and 'date' in df.columns:
            df = df.rename(columns={'date': 'time'})
        df = df.sort_values('time').reset_index(drop=True)
        return df
    except Exception as e:
        print(f"[ERROR] fetch failed for {symbol}: {e}")
        return None

def calc_ema(series, period):
    return series.ewm(span=period, adjust=False).mean()

def calc_macd(close):
    ema12 = calc_ema(close, 12)
    ema26 = calc_ema(close, 26)
    macd_line = ema12 - ema26
    signal_line = calc_ema(macd_line, 9)
    histogram = macd_line - signal_line
    return macd_line, signal_line, histogram

def calc_adx(df, period=14):
    try:
        high = df['high']
        low = df['low']
        close = df['close']
        tr = pd.concat([
            high - low,
            (high - close.shift()).abs(),
            (low - close.shift()).abs()
        ], axis=1).max(axis=1)
        dm_plus = ((high - high.shift()) > (low.shift() - low)).astype(float) * (high - high.shift()).clip(lower=0)
        dm_minus = ((low.shift() - low) > (high - high.shift())).astype(float) * (low.shift() - low).clip(lower=0)
        atr = tr.ewm(alpha=1/period, adjust=False).mean()
        di_plus = 100 * dm_plus.ewm(alpha=1/period, adjust=False).mean() / atr
        di_minus = 100 * dm_minus.ewm(alpha=1/period, adjust=False).mean() / atr
        dx = (100 * (di_plus - di_minus).abs() / (di_plus + di_minus)).fillna(0)
        adx = dx.ewm(alpha=1/period, adjust=False).mean()
        return adx.iloc[-1], di_plus.iloc[-1], di_minus.iloc[-1]
    except:
        return None, None, None

def analyze_weekly(symbol, data_type='stock'):
    df = fetch_vnstock(symbol, data_type)
    if df is None or len(df) < 200:
        print(f"[WARN] Not enough data for {symbol} (got {len(df) if df is not None else 0} rows)")
        return None

    close = df['close']
    ema200 = calc_ema(close, 200)
    ema50 = calc_ema(close, 50)
    macd_line, signal_line, histogram = calc_macd(close)
    adx, di_plus, di_minus = calc_adx(df)

    last_close = close.iloc[-1]
    last_ema200 = ema200.iloc[-1]
    last_ema50 = ema50.iloc[-1]
    last_macd = macd_line.iloc[-1]
    last_signal = signal_line.iloc[-1]
    last_hist = histogram.iloc[-1]
    prev_hist = histogram.iloc[-2]

    # Weekly change
    if len(close) >= 6:
        chg_week = (close.iloc[-1] / close.iloc[-6] - 1) * 100
    else:
        chg_week = 0.0

    # Monthly change
    if len(close) >= 22:
        chg_month = (close.iloc[-1] / close.iloc[-22] - 1) * 100
    else:
        chg_month = 0.0

    # Market regime
    above_ema200 = last_close > last_ema200
    above_ema50 = last_close > last_ema50

    if above_ema200 and above_ema50:
        regime = '🐂 BULL'
    elif not above_ema200 and not above_ema50:
        regime = '🐻 BEAR'
    else:
        regime = '⚪ TRUNG TÍNH'

    # MACD signal
    if last_macd > last_signal and last_hist > 0 and last_hist > prev_hist:
        macd_signal = '📈 Tăng mạnh'
    elif last_macd > last_signal:
        macd_signal = '📈 Tăng'
    elif last_macd < last_signal and last_hist < 0 and last_hist < prev_hist:
        macd_signal = '📉 Giảm mạnh'
    else:
        macd_signal = '📉 Giảm'

    # Entry, target, stoploss, R/R
    high_52w = df['high'].iloc[-252:].max() if 'high' in df.columns else close.iloc[-252:].max()
    low_10d = df['low'].iloc[-10:].min() if 'low' in df.columns else close.iloc[-10:].min()

    entry = last_close
    pullback_entry = last_ema50  # buy on pullback to EMA50
    target = high_52w
    stop_loss = min(last_ema200 * 0.99, low_10d)  # just below EMA200 or recent low

    risk = entry - stop_loss
    reward = target - entry
    rr = reward / risk if risk > 0 else 0

    # ADX trend strength
    if adx is not None:
        if adx > 30:
            adx_str = f'ADX {adx:.0f} 💪 Trend mạnh'
        elif adx > 20:
            adx_str = f'ADX {adx:.0f} Trend vừa'
        else:
            adx_str = f'ADX {adx:.0f} ⚠️ Trend yếu/sideway'
    else:
        adx_str = 'ADX N/A'

    # Strategy recommendation
    if regime == '🐂 BULL' and 'Tăng' in macd_signal and adx is not None and adx > 20:
        strategy = '✅ Giữ/Mua thêm — trend còn mạnh'
    elif regime == '🐂 BULL' and adx is not None and adx < 20:
        strategy = '⏸ Giữ — chờ xác nhận momentum'
    elif regime == '🐻 BEAR' and 'Giảm' in macd_signal:
        strategy = '🚫 Không mua — bear market'
    elif regime == '🐻 BEAR' and 'Tăng' in macd_signal:
        strategy = '⚠️ Phục hồi tạm — cẩn thận bẫy tăng'
    else:
        strategy = '👁 Theo dõi thêm'

    return {
        'symbol': symbol,
        'price': last_close,
        'ema200': last_ema200,
        'ema50': last_ema50,
        'chg_week': chg_week,
        'chg_month': chg_month,
        'regime': regime,
        'macd_signal': macd_signal,
        'adx_str': adx_str,
        'strategy': strategy,
        'entry': entry,
        'pullback_entry': pullback_entry,
        'target': target,
        'stop_loss': stop_loss,
        'rr': rr,
    }

def fetch_macro_weekly():
    result = {}
    try:
        sp = yf.Ticker('^GSPC')
        h = sp.history(period='1mo')
        if not h.empty:
            result['sp_month'] = (h['Close'].iloc[-1] / h['Close'].iloc[0] - 1) * 100
        else:
            result['sp_month'] = None
    except:
        result['sp_month'] = None
    try:
        dxy = yf.Ticker('DX-Y.NYB')
        h = dxy.history(period='5d')
        result['dxy'] = h['Close'].iloc[-1] if not h.empty else None
    except:
        result['dxy'] = None
    return result

def main():
    date_str = TODAY.strftime('%d/%m/%Y')
    week_num = TODAY.isocalendar()[1]

    print("Fetching VNINDEX weekly...")
    vnindex = analyze_weekly('VNINDEX', data_type='index')
    time.sleep(4)

    stock_results = []
    for sym in STOCK_SYMBOLS:
        print(f"Fetching {sym}...")
        r = analyze_weekly(sym)
        stock_results.append((sym, r))
        time.sleep(4)

    macro = fetch_macro_weekly()

    lines = []
    lines.append(f'📊 <b>BÁO CÁO XU HƯỚNG TUẦN {week_num}</b>')
    lines.append(f'🗓 {date_str} | 16:00 ICT')
    lines.append('')

    # VNINDEX
    lines.append('🇻🇳 <b>VN-INDEX — XU HƯỚNG</b>')
    if vnindex:
        lines.append(f"Điểm: {vnindex['price']:,.0f} | Tuần: {vnindex['chg_week']:+.2f}% | Tháng: {vnindex['chg_month']:+.2f}%")
        lines.append(f"EMA50: {vnindex['ema50']:,.0f} | EMA200: {vnindex['ema200']:,.0f}")
        lines.append(f"Regime: {vnindex['regime']} | MACD: {vnindex['macd_signal']}")
        lines.append(f"{vnindex['adx_str']}")
        lines.append(f"→ {vnindex['strategy']}")
    else:
        lines.append('Không lấy được dữ liệu VNINDEX')
    lines.append('')

    # Stocks — group by regime
    bull = [(s, r) for s, r in stock_results if r and r['regime'] == '🐂 BULL']
    bear = [(s, r) for s, r in stock_results if r and r['regime'] == '🐻 BEAR']
    neutral = [(s, r) for s, r in stock_results if r and r['regime'] == '⚪ TRUNG TÍNH']
    no_data = [s for s, r in stock_results if r is None]

    if bull:
        lines.append(f'🐂 <b>BULL MARKET ({len(bull)} mã)</b>')
        for s, r in bull:
            lines.append(f"<b>{s}</b> | {r['macd_signal']} | Tuần: {r['chg_week']:+.2f}%")
            lines.append(f"  📥 Mua: {r['entry']:,.0f} | Pullback: {r['pullback_entry']:,.0f}")
            lines.append(f"  🎯 Target: {r['target']:,.0f} | ✂️ StopLoss: {r['stop_loss']:,.0f}")
            rr_str = f"{r['rr']:.1f}" if r['rr'] > 0 else 'N/A'
            lines.append(f"  ⚖️ R/R: 1:{rr_str} | {r['strategy']}")
        lines.append('')

    if neutral:
        lines.append(f'⚪ <b>TRUNG TÍNH ({len(neutral)} mã)</b>')
        for s, r in neutral:
            lines.append(f"{s} | {r['macd_signal']} | Tuần: {r['chg_week']:+.2f}%")
        lines.append('')

    if bear:
        lines.append(f'🐻 <b>BEAR MARKET ({len(bear)} mã)</b>')
        for s, r in bear:
            lines.append(f"{s} | {r['macd_signal']} | Tuần: {r['chg_week']:+.2f}%")
        lines.append('')

    if no_data:
        lines.append(f"⚠️ Không lấy được: {', '.join(no_data)}")
        lines.append('')

    # Macro
    lines.append('🌐 <b>MACRO TUẦN</b>')
    sp_str = f"{macro['sp_month']:+.2f}%" if macro.get('sp_month') is not None else 'N/A'
    dxy_str = f"{macro['dxy']:.1f}" if macro.get('dxy') is not None else 'N/A'
    lines.append(f"S&P500 1 tháng: {sp_str} | DXY: {dxy_str}")
    lines.append('')

    # Combined strategy
    lines.append('⚡ <b>CHIẾN LƯỢC TUẦN TỚI</b>')
    bull_count = len(bull)
    bear_count = len(bear)
    total = len([r for _, r in stock_results if r])

    if vnindex and vnindex['regime'] == '🐂 BULL' and bull_count > total * 0.6:
        lines.append(f"Thị trường bullish rộng ({bull_count}/{total} mã). Duy trì danh mục, mua thêm mã có MACD tăng mạnh và ADX > 25.")
    elif vnindex and vnindex['regime'] == '🐻 BEAR' and bear_count > total * 0.5:
        lines.append(f"Thị trường bearish ({bear_count}/{total} mã). Giảm exposure, ưu tiên bảo vệ vốn.")
    elif bull_count > bear_count:
        lines.append(f"Thị trường phân hóa — {bull_count} mã bull, {bear_count} mã bear. Chọn lọc mã bull có ADX > 20.")
    else:
        lines.append("Thị trường sideway/phân hóa. Giảm size, chờ tín hiệu rõ hơn tuần tới.")

    lines.append('')
    lines.append('⚠️ <i>Không phải tư vấn tài chính</i>')

    message = '\n'.join(lines)
    print("--- MESSAGE PREVIEW ---")
    print(message)
    print("--- END PREVIEW ---")
    send_telegram(message)

if __name__ == '__main__':
    main()
