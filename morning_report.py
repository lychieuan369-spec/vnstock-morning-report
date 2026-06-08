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
START_DATE = (TODAY - timedelta(days=180)).strftime('%Y-%m-%d')
END_DATE = TODAY.strftime('%Y-%m-%d')


def calc_rsi(prices, period=14):
    delta = prices.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.ewm(alpha=1/period, min_periods=period, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1/period, min_periods=period, adjust=False).mean()
    rs = avg_gain / avg_loss
    return 100 - (100 / (1 + rs))


def calc_ema(series, period=9):
    return series.ewm(span=period, adjust=False).mean()


def calc_wma(series, period=45):
    weights = np.arange(1, period + 1)
    return series.rolling(period).apply(lambda x: np.dot(x, weights) / weights.sum(), raw=True)


def get_signal(rsi, ema9, wma45):
    if rsi <= 25 and ema9 > wma45:
        return '🟢 MUA'
    elif rsi >= 75 and ema9 < wma45:
        return '🔴 BÁN'
    elif rsi <= 35:
        return '🟡 TÍCH LŨY'
    elif ema9 > wma45:
        return '📈 UPTREND'
    elif ema9 < wma45:
        return '📉 DOWNTREND'
    else:
        return '⚪ TRUNG TÍNH'


def vol_badge(vol_ratio):
    if vol_ratio is None:
        return ''
    if vol_ratio >= 1.5:
        return f'✅ Vol {vol_ratio:.1f}x'
    elif vol_ratio < 0.7:
        return f'⚠️ Vol yếu {vol_ratio:.1f}x'
    else:
        return f'Vol {vol_ratio:.1f}x'


def fetch_vnstock(symbol, data_type='stock'):
    for source in ['VCI', 'MSN']:
        try:
            from vnstock import Vnstock
            stock = Vnstock().stock(symbol=symbol, source=source)
            df = stock.quote.history(start=START_DATE, end=END_DATE, interval='1D')
            if df is None or df.empty:
                print(f"[WARN] No data for {symbol} from {source}")
                continue
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
            if len(df) >= 10:
                print(f"[OK] {symbol} from {source}: {len(df)} rows")
                return df
        except Exception as e:
            print(f"[ERROR] {symbol} from {source}: {e}")
            time.sleep(2)
    return None


def analyze_ticker(symbol, data_type='stock'):
    df = fetch_vnstock(symbol, data_type)
    if df is None or len(df) < 50:
        return None
    close = df['close']

    # Use 'low' and 'high' columns if available, else fallback to close
    low_col = df['low'] if 'low' in df.columns else close
    high_col = df['high'] if 'high' in df.columns else close

    rsi = calc_rsi(close)
    ema9 = calc_ema(rsi)
    wma45 = calc_wma(rsi)
    last_rsi = rsi.iloc[-1]
    last_ema9 = ema9.iloc[-1]
    last_wma45 = wma45.iloc[-1]
    last_close = close.iloc[-1]

    if len(close) >= 6:
        chg5d = (close.iloc[-1] / close.iloc[-6] - 1) * 100
    else:
        chg5d = 0.0

    stop_loss = low_col.iloc[-5:].min()
    target = high_col.iloc[-20:].max()

    # Volume confirmation
    if 'volume' in df.columns:
        vol_ma20 = df['volume'].rolling(20).mean()
        last_vol = df['volume'].iloc[-1]
        last_vol_ma20 = vol_ma20.iloc[-1]
        vol_ratio = last_vol / last_vol_ma20 if last_vol_ma20 > 0 else 1.0
    else:
        vol_ratio = None

    signal = get_signal(last_rsi, last_ema9, last_wma45)

    return {
        'symbol': symbol,
        'price': last_close,
        'rsi': last_rsi,
        'ema9': last_ema9,
        'wma45': last_wma45,
        'chg5d': chg5d,
        'signal': signal,
        'stop_loss': stop_loss,
        'target': target,
        'vol_ratio': vol_ratio,
    }


def fetch_macro():
    result = {}
    try:
        brent = yf.Ticker('BZ=F')
        brent_hist = brent.history(period='5d')
        result['brent'] = brent_hist['Close'].iloc[-1] if not brent_hist.empty else None
    except Exception as e:
        print(f"[ERROR] Brent fetch failed: {e}")
        result['brent'] = None

    try:
        dxy = yf.Ticker('DX-Y.NYB')
        dxy_hist = dxy.history(period='5d')
        result['dxy'] = dxy_hist['Close'].iloc[-1] if not dxy_hist.empty else None
    except Exception as e:
        print(f"[ERROR] DXY fetch failed: {e}")
        result['dxy'] = None

    try:
        sp = yf.Ticker('^GSPC')
        sp_hist = sp.history(period='10d')
        if not sp_hist.empty and len(sp_hist) >= 6:
            sp_chg = (sp_hist['Close'].iloc[-1] / sp_hist['Close'].iloc[-6] - 1) * 100
        elif not sp_hist.empty:
            sp_chg = (sp_hist['Close'].iloc[-1] / sp_hist['Close'].iloc[0] - 1) * 100
        else:
            sp_chg = None
        result['sp_chg'] = sp_chg
    except Exception as e:
        print(f"[ERROR] S&P500 fetch failed: {e}")
        result['sp_chg'] = None

    return result


def macro_score(brent, dxy, sp_chg):
    score = 0
    if brent is not None:
        if 70 <= brent <= 95:
            score += 1
        elif brent > 95:
            score -= 1
        # else 0 (below 70 is also concerning but not explicitly negative in spec)
    if dxy is not None:
        if dxy < 100:
            score += 1
        elif dxy > 103:
            score -= 1
    if sp_chg is not None:
        if sp_chg > 1:
            score += 1
        elif sp_chg < -1:
            score -= 1
    return score


def build_recommendation(vnindex_data, stock_results):
    all_signals = []
    if vnindex_data:
        all_signals.append(vnindex_data['signal'])
    for r in stock_results:
        if r:
            all_signals.append(r['signal'])

    mua_tickers = [r['symbol'] for r in stock_results if r and r['signal'] == '🟢 MUA']
    downtrend_count = sum(1 for s in all_signals if 'DOWNTREND' in s)
    uptrend_count = sum(1 for s in all_signals if 'UPTREND' in s)

    if mua_tickers:
        tickers_str = ', '.join(mua_tickers)
        return f"Có tín hiệu mua xuất hiện tại {tickers_str}. Kiểm tra thêm volume và xác nhận xu hướng trước khi vào lệnh."
    elif downtrend_count >= len(all_signals) * 0.6:
        return "Thị trường downtrend, chưa có tín hiệu mua. Quan sát vùng hỗ trợ và chờ RSI về vùng tích lũy."
    elif uptrend_count >= len(all_signals) * 0.6:
        return "Thị trường đang trong uptrend. Duy trì danh mục, tránh mua đuổi khi RSI cao."
    else:
        return "Thị trường phân hóa. Ưu tiên cổ phiếu có RSI <= 35 và EMA9 hướng lên. Quản lý rủi ro chặt."


def send_telegram(text):
    url = f"https://api.telegram.org/bot{BOT_TOKEN}/sendMessage"
    payload = {
        'chat_id': CHAT_ID,
        'text': text,
        'parse_mode': 'HTML',
    }
    try:
        resp = requests.post(url, json=payload, timeout=15)
        resp.raise_for_status()
        print(f"[OK] Telegram message sent (status {resp.status_code})")
    except Exception as e:
        print(f"[ERROR] Telegram send failed: {e}")


def main():
    date_str = TODAY.strftime('%d/%m/%Y')

    print("Fetching VNINDEX...")
    vnindex = analyze_ticker('VNINDEX', data_type='index')
    time.sleep(4)

    stock_symbols = ['VCB', 'HPG', 'FPT', 'MBB', 'VIC', 'SSI', 'VHM', 'TCB', 'ACB', 'BID', 'CTG', 'GAS', 'MSN', 'PLX', 'VRE', 'VPB', 'MSR']
    stock_results = []
    for sym in stock_symbols:
        print(f"Fetching {sym}...")
        result = analyze_ticker(sym, data_type='stock')
        stock_results.append(result)
        time.sleep(4)

    print("Fetching macro data...")
    macro = fetch_macro()
    brent = macro.get('brent')
    dxy = macro.get('dxy')
    sp_chg = macro.get('sp_chg')

    score = macro_score(brent, dxy, sp_chg)
    if score >= 2:
        macro_bias = 'BULLISH 🟢'
    elif score <= -1:
        macro_bias = 'BEARISH 🔴'
    else:
        macro_bias = 'NEUTRAL ⚪'

    # Build message
    lines = []
    lines.append('📊 <b>BÁO CÁO CHỨNG KHOÁN VN</b>')
    lines.append(f'🗓 {date_str} | 08:30 ICT')
    lines.append('')

    # VNINDEX section
    lines.append('🇻🇳 <b>VN-INDEX</b>')
    if vnindex:
        lines.append(
            f"Điểm: {vnindex['price']:.0f} | 5d: {vnindex['chg5d']:+.2f}%"
        )
        lines.append(
            f"RSI: {vnindex['rsi']:.1f} | EMA9: {vnindex['ema9']:.1f} | WMA45: {vnindex['wma45']:.1f}"
        )
        lines.append(f"→ {vnindex['signal']}")
        if vnindex['signal'] in ['🟢 MUA', '🔴 BÁN']:
            lines.append(f"  🎯 Mục tiêu: {vnindex['target']:,.0f} | ✂️ Cắt lỗ: {vnindex['stop_loss']:,.0f}")
    else:
        lines.append('Không lấy được dữ liệu VNINDEX')
    lines.append('')

    # Stock section
    lines.append('📈 <b>TOP CỔ PHIẾU</b>')

    action_stocks = [r for r in stock_results if r and r['signal'] in ['🟢 MUA', '🔴 BÁN', '🟡 TÍCH LŨY']]
    watch_stocks = [r for r in stock_results if r and r['signal'] not in ['🟢 MUA', '🔴 BÁN', '🟡 TÍCH LŨY']]
    no_data = [stock_symbols[i] for i, r in enumerate(stock_results) if r is None]

    if action_stocks:
        for r in action_stocks:
            lines.append(f"<b>{r['symbol']}</b> | {r['signal']}")
            vol_str = vol_badge(r.get('vol_ratio'))
            lines.append(f"  Giá: {r['price']:,.0f} | RSI: {r['rsi']:.1f} | 5d: {r['chg5d']:+.2f}% | {vol_str}")
            if r['signal'] == '🟢 MUA':
                lines.append(f"  🎯 Mục tiêu: {r['target']:,.0f} | ✂️ Cắt lỗ: {r['stop_loss']:,.0f}")
            elif r['signal'] == '🔴 BÁN':
                lines.append(f"  ✂️ Cắt lỗ nếu giữ: {r['stop_loss']:,.0f}")
            elif r['signal'] == '🟡 TÍCH LŨY':
                lines.append(f"  ✂️ Cắt lỗ: {r['stop_loss']:,.0f} | Chờ RSI tăng")
    else:
        lines.append('Không có tín hiệu MUA/BÁN/TÍCH LŨY hôm nay.')

    if watch_stocks:
        lines.append('')
        lines.append('👁 <b>Theo dõi:</b>')
        for r in watch_stocks:
            vol_str = vol_badge(r.get('vol_ratio'))
            lines.append(f"{r['symbol']} | {r['signal']} | RSI: {r['rsi']:.1f} | 5d: {r['chg5d']:+.2f}% | {vol_str}")

    if no_data:
        lines.append(f"⚠️ Không lấy được: {', '.join(no_data)}")

    lines.append('')

    # Macro section
    lines.append('🌐 <b>MACRO</b>')
    brent_str = f'${brent:.1f}' if brent is not None else 'N/A'
    dxy_str = f'{dxy:.1f}' if dxy is not None else 'N/A'
    sp_str = f'{sp_chg:+.2f}%' if sp_chg is not None else 'N/A'
    lines.append(f'Brent: {brent_str} | DXY: {dxy_str}')
    lines.append(f'S&P500 5d: {sp_str}')
    lines.append(f'Macro: {macro_bias}')
    lines.append('')

    # Recommendation
    lines.append('⚡ <b>KHUYẾN NGHỊ</b>')
    recommendation = build_recommendation(vnindex, stock_results)
    lines.append(recommendation)
    lines.append('')
    lines.append('⚠️ <i>Không phải tư vấn tài chính</i>')

    message = '\n'.join(lines)
    print("--- MESSAGE PREVIEW ---")
    print(message)
    print("--- END PREVIEW ---")

    send_telegram(message)


if __name__ == '__main__':
    main()
