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
    """Fetch today's intraday 1H data — phiên sáng 9:00-11:30."""
    try:
        from vnstock import Vnstock
        stock = Vnstock().stock(symbol=symbol, source='KBS')
        # Fetch last 2 days of 1H data
        end = TODAY.strftime('%Y-%m-%d')
        start = (TODAY - timedelta(days=2)).strftime('%Y-%m-%d')
        df = stock.quote.history(start=start, end=end, interval='1H')
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
        df['time'] = pd.to_datetime(df['time'])
        df = df.sort_values('time').reset_index(drop=True)

        # Filter today's session only (9:00-12:00 local)
        today_str = TODAY.strftime('%Y-%m-%d')
        today_bars = df[df['time'].dt.strftime('%Y-%m-%d') == today_str]

        if today_bars.empty:
            # Fallback: use last available bars
            today_bars = df.tail(4)

        if today_bars.empty:
            return None

        open_price = today_bars['open'].iloc[0]
        last_close = today_bars['close'].iloc[-1]
        session_high = today_bars['high'].max() if 'high' in today_bars.columns else last_close
        session_low = today_bars['low'].min() if 'low' in today_bars.columns else last_close
        total_vol = today_bars['volume'].sum() if 'volume' in today_bars.columns else 0

        chg_from_open = (last_close - open_price) / open_price * 100 if open_price > 0 else 0

        # RSI on H1 closes (need at least 15 bars — use all available)
        all_today = df.tail(20)
        rsi_val = None
        if len(all_today) >= 5:
            close_s = all_today['close']
            delta = close_s.diff()
            gain = delta.clip(lower=0).ewm(alpha=1/14, adjust=False).mean()
            loss = (-delta.clip(upper=0)).ewm(alpha=1/14, adjust=False).mean()
            rs = gain / loss
            rsi_series = 100 - (100 / (1 + rs))
            rsi_val = rsi_series.iloc[-1]

        # Momentum: last 2 bars direction
        if len(today_bars) >= 2:
            momentum = today_bars['close'].iloc[-1] - today_bars['close'].iloc[-2]
        else:
            momentum = 0

        return {
            'symbol': symbol,
            'open': open_price,
            'last': last_close,
            'high': session_high,
            'low': session_low,
            'chg_pct': chg_from_open,
            'volume': total_vol,
            'rsi_h1': rsi_val,
            'momentum': momentum,
            'bars': len(today_bars),
        }
    except Exception as e:
        print(f"[ERROR] {symbol}: {e}")
        return None


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

    lines = []
    lines.append('📊 <b>BÁO CÁO PHIÊN SÁNG — DỰ ĐOÁN CHIỀU</b>')
    lines.append(f'🗓 {date_str} | 14:00 ICT')
    lines.append('')

    gainers = sorted([r for r in valid if r['chg_pct'] >= 0.5], key=lambda x: x['chg_pct'], reverse=True)
    losers = sorted([r for r in valid if r['chg_pct'] <= -0.5], key=lambda x: x['chg_pct'])
    neutral = [r for r in valid if -0.5 < r['chg_pct'] < 0.5]

    if gainers:
        lines.append(f'🚀 <b>TĂNG PHIÊN SÁNG ({len(gainers)} mã)</b>')
        for r in gainers:
            rsi_str = f"RSI H1: {r['rsi_h1']:.0f}" if r['rsi_h1'] is not None else ''
            mom = '↑' if r['momentum'] > 0 else '↓'
            lines.append(f"<b>{r['symbol']}</b> {r['chg_pct']:+.2f}% {mom} | Giá: {r['last']:,.0f} | {rsi_str}")
            lines.append(f"   Open: {r['open']:,.0f} | High: {r['high']:,.0f} | Low: {r['low']:,.0f}")
            # Afternoon prediction
            if r['rsi_h1'] is not None and r['rsi_h1'] >= 70 and r['momentum'] < 0:
                pred = '⚠️ RSI cao + momentum yếu — có thể điều chỉnh chiều'
            elif r['chg_pct'] >= 2 and r['momentum'] > 0:
                pred = '💪 Đà tăng mạnh — có thể tiếp tục chiều'
            else:
                pred = '➡️ Theo dõi thêm'
            lines.append(f"   → {pred}")
        lines.append('')

    if losers:
        lines.append(f'💥 <b>GIẢM PHIÊN SÁNG ({len(losers)} mã)</b>')
        for r in losers:
            rsi_str = f"RSI H1: {r['rsi_h1']:.0f}" if r['rsi_h1'] is not None else ''
            mom = '↑' if r['momentum'] > 0 else '↓'
            lines.append(f"<b>{r['symbol']}</b> {r['chg_pct']:+.2f}% {mom} | Giá: {r['last']:,.0f} | {rsi_str}")
            lines.append(f"   Open: {r['open']:,.0f} | High: {r['high']:,.0f} | Low: {r['low']:,.0f}")
            if r['rsi_h1'] is not None and r['rsi_h1'] <= 30 and r['momentum'] > 0:
                pred = '🔄 RSI thấp + momentum đảo — có thể phục hồi chiều'
            elif r['chg_pct'] <= -2 and r['momentum'] < 0:
                pred = '🔴 Áp lực bán mạnh — tránh bắt đáy'
            else:
                pred = '➡️ Theo dõi thêm'
            lines.append(f"   → {pred}")
        lines.append('')

    if neutral:
        lines.append(f'➡️ <b>ĐI NGANG ({len(neutral)} mã)</b>')
        lines.append(', '.join([f"{r['symbol']} {r['chg_pct']:+.1f}%" for r in neutral]))
        lines.append('')

    avg_chg = sum(r['chg_pct'] for r in valid) / len(valid) if valid else 0
    bull_count = len(gainers)
    bear_count = len(losers)

    lines.append('⚡ <b>DỰ ĐOÁN PHIÊN CHIỀU</b>')
    lines.append(f"Phiên sáng: {bull_count} tăng / {bear_count} giảm | TB: {avg_chg:+.2f}%")
    if avg_chg >= 1 and bull_count > bear_count:
        lines.append('🟢 Xu hướng tích cực — chiều có thể tiếp đà tăng')
    elif avg_chg <= -1 and bear_count > bull_count:
        lines.append('🔴 Áp lực bán — chiều thận trọng, hạn chế mua đuổi')
    elif bull_count > bear_count:
        lines.append('⚪ Phân hóa nghiêng tăng — chọn lọc mã mạnh')
    else:
        lines.append('⚪ Thị trường phân hóa — chờ tín hiệu rõ hơn')

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
