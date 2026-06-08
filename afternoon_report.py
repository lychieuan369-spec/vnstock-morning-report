import os
import time
import requests
import pandas as pd
from datetime import datetime, timedelta

BOT_TOKEN = os.environ['TELEGRAM_BOT_TOKEN']
CHAT_ID = os.environ['TELEGRAM_CHAT_ID']

TODAY = datetime.today()
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

def fetch_intraday(symbol):
    """Fetch today's intraday price data."""
    try:
        from vnstock import Vnstock
        stock = Vnstock().stock(symbol=symbol, source='KBS')
        # Get last 2 days daily to compute today vs yesterday
        end = TODAY.strftime('%Y-%m-%d')
        start = (TODAY - timedelta(days=5)).strftime('%Y-%m-%d')
        df = stock.quote.history(start=start, end=end, interval='1D')
        if df is None or df.empty or len(df) < 1:
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
        df = df.sort_values('time').reset_index(drop=True)

        last = df.iloc[-1]
        prev = df.iloc[-2] if len(df) >= 2 else None

        last_close = last['close']
        last_open = last.get('open', last_close)
        last_high = last.get('high', last_close)
        last_low = last.get('low', last_close)
        last_vol = last.get('volume', 0)

        prev_close = prev['close'] if prev is not None else last_close

        chg_pct = (last_close - prev_close) / prev_close * 100 if prev_close > 0 else 0
        chg_from_open = (last_close - last_open) / last_open * 100 if last_open > 0 else 0

        # Volume vs yesterday
        prev_vol = prev.get('volume', last_vol) if prev is not None else last_vol
        vol_ratio = last_vol / prev_vol if prev_vol > 0 else 1.0

        return {
            'symbol': symbol,
            'close': last_close,
            'open': last_open,
            'high': last_high,
            'low': last_low,
            'chg_pct': chg_pct,
            'chg_from_open': chg_from_open,
            'vol_ratio': vol_ratio,
        }
    except Exception as e:
        print(f"[ERROR] {symbol}: {e}")
        return None

def price_arrow(chg):
    if chg >= 2: return '🚀'
    elif chg >= 0.5: return '📈'
    elif chg > -0.5: return '➡️'
    elif chg > -2: return '📉'
    else: return '💥'

def vol_str(ratio):
    if ratio >= 1.5: return f'✅ Vol {ratio:.1f}x'
    elif ratio < 0.7: return f'⚠️ Vol {ratio:.1f}x'
    else: return f'Vol {ratio:.1f}x'

def main():
    date_str = TODAY.strftime('%d/%m/%Y')

    results = []
    for sym in STOCK_SYMBOLS:
        print(f"Fetching {sym}...")
        r = fetch_intraday(sym)
        results.append(r)
        time.sleep(3)

    valid = [r for r in results if r is not None]
    no_data = [STOCK_SYMBOLS[i] for i, r in enumerate(results) if r is None]

    # Sort by % change
    gainers = sorted([r for r in valid if r['chg_pct'] >= 0.5], key=lambda x: x['chg_pct'], reverse=True)
    losers = sorted([r for r in valid if r['chg_pct'] <= -0.5], key=lambda x: x['chg_pct'])
    neutral = [r for r in valid if -0.5 < r['chg_pct'] < 0.5]

    lines = []
    lines.append('📊 <b>BÁO CÁO CHIỀU — INTRADAY</b>')
    lines.append(f'🗓 {date_str} | 14:00 ICT')
    lines.append('')

    if gainers:
        lines.append(f'🚀 <b>TĂNG ({len(gainers)} mã)</b>')
        for r in gainers:
            arrow = price_arrow(r['chg_pct'])
            lines.append(f"{arrow} <b>{r['symbol']}</b> {r['chg_pct']:+.2f}% | Giá: {r['close']:,.0f} | {vol_str(r['vol_ratio'])}")
            lines.append(f"   Open: {r['open']:,.0f} | High: {r['high']:,.0f} | Low: {r['low']:,.0f}")
        lines.append('')

    if losers:
        lines.append(f'💥 <b>GIẢM ({len(losers)} mã)</b>')
        for r in losers:
            arrow = price_arrow(r['chg_pct'])
            lines.append(f"{arrow} <b>{r['symbol']}</b> {r['chg_pct']:+.2f}% | Giá: {r['close']:,.0f} | {vol_str(r['vol_ratio'])}")
            lines.append(f"   Open: {r['open']:,.0f} | High: {r['high']:,.0f} | Low: {r['low']:,.0f}")
        lines.append('')

    if neutral:
        lines.append(f'➡️ <b>ĐI NGANG ({len(neutral)} mã)</b>')
        lines.append(', '.join([f"{r['symbol']} {r['chg_pct']:+.1f}%" for r in neutral]))
        lines.append('')

    # Market summary
    avg_chg = sum(r['chg_pct'] for r in valid) / len(valid) if valid else 0
    if avg_chg >= 1:
        market_mood = '🟢 Thị trường tích cực — xem xét giữ qua ATC'
    elif avg_chg <= -1:
        market_mood = '🔴 Thị trường tiêu cực — cân nhắc chốt lời/cắt lỗ trước ATC'
    else:
        market_mood = '⚪ Thị trường trung tính — giữ nguyên, chờ tín hiệu rõ hơn'

    lines.append('⚡ <b>NHẬN ĐỊNH 14H</b>')
    lines.append(f"TB danh mục: {avg_chg:+.2f}%")
    lines.append(market_mood)

    if no_data:
        lines.append('')
        lines.append(f"⚠️ Không lấy được: {', '.join(no_data)}")

    lines.append('')
    lines.append('⚠️ <i>Không phải tư vấn tài chính</i>')

    message = '\n'.join(lines)
    print("--- MESSAGE PREVIEW ---")
    print(message)
    print("--- END PREVIEW ---")
    send_telegram(message)

if __name__ == '__main__':
    main()
