"""
VN Stock Market Analyzer — Egan Methodology
RSI(14) + EMA(9) on RSI + WMA(45) on RSI + VN Macro Indicators
"""

import argparse
import sys
import warnings
from datetime import datetime, timedelta

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

try:
    from colorama import Fore, Style, init as colorama_init
    colorama_init(autoreset=True)
    HAS_COLOR = True
except ImportError:
    HAS_COLOR = False

    class _FakeColor:
        def __getattr__(self, name):
            return ""

    Fore = _FakeColor()
    Style = _FakeColor()


# ---------------------------------------------------------------------------
# Macro data (update with --update-macro flag or edit here)
# ---------------------------------------------------------------------------
MACRO_DATA = {
    "vnindex_rsi": None,   # calculated live
    "usdvnd": 25450,       # approximate
    "sbv_rate": 4.5,       # %
    "cpi": 3.2,            # % YoY
    "foreign_flow": 0,     # net M VND, fetch if possible
    "brent": 0,            # fetch from yfinance
    "china_pmi": 51.0,     # approximate
    "fed_rate": 4.25,      # %
    "vn_gdp": 6.8,         # % YoY
    "credit_growth": 13.5, # % YoY
}

# Sector tickers mapping
SECTOR_TICKERS = {
    "banking": ["VCB", "BID", "CTG", "MBB", "TCB", "ACB", "STB", "VPB", "HDB", "LPB"],
    "realestate": ["VIC", "VHM", "NVL", "PDR", "KDH", "DXG", "NLG", "BCM", "AGG", "DIG"],
    "steels": ["HPG", "HSG", "NKG", "TLH", "VGS", "POM", "TVN"],
    "retail": ["MWG", "FRT", "PNJ", "DGW", "HAX"],
    "energy": ["GAS", "PLX", "PVS", "BSR", "OIL", "PVC", "PVD", "PVT"],
    "tech": ["FPT", "CMG", "ELC", "VGI", "ITD"],
    "food": ["VNM", "MSN", "SAB", "QNS", "KDC", "MCM"],
    "industrial": ["VGC", "REE", "GEX", "SCI", "PHR"],
}

# ANSI color helpers (works even without colorama)
def green(s):
    return f"{Fore.GREEN}{s}{Style.RESET_ALL}"

def red(s):
    return f"{Fore.RED}{s}{Style.RESET_ALL}"

def yellow(s):
    return f"{Fore.YELLOW}{s}{Style.RESET_ALL}"

def cyan(s):
    return f"{Fore.CYAN}{s}{Style.RESET_ALL}"

def bold(s):
    return f"{Style.BRIGHT}{s}{Style.RESET_ALL}"

def color_value(val, good_above=None, bad_above=None, neutral_fmt=None):
    """Return colored string: green if good, red if bad, yellow otherwise."""
    if good_above is not None and val >= good_above:
        return green(str(val))
    if bad_above is not None and val >= bad_above:
        return red(str(val))
    return yellow(str(val))


# ---------------------------------------------------------------------------
# Indicators
# ---------------------------------------------------------------------------
def calc_rsi(prices: pd.Series, period: int = 14) -> pd.Series:
    delta = prices.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.ewm(alpha=1 / period, min_periods=period, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1 / period, min_periods=period, adjust=False).mean()
    rs = avg_gain / avg_loss
    return 100 - (100 / (1 + rs))


def calc_ema(series: pd.Series, period: int = 9) -> pd.Series:
    return series.ewm(span=period, adjust=False).mean()


def calc_wma(series: pd.Series, period: int = 45) -> pd.Series:
    weights = np.arange(1, period + 1)
    return series.rolling(period).apply(
        lambda x: np.dot(x, weights) / weights.sum(), raw=True
    )


def detect_signal(df: pd.DataFrame) -> str:
    """Detect Egan signal from the last few candles of df."""
    if len(df) < 3:
        return "NO SIGNAL"

    rsi = df["rsi"]
    ema9 = df["ema9"]
    wma45 = df["wma45"]

    last_rsi = rsi.iloc[-1]
    last_ema = ema9.iloc[-1]
    last_wma = wma45.iloc[-1]

    if pd.isna(last_ema) or pd.isna(last_wma):
        return "NO SIGNAL"

    # Check cross within last 3 candles
    def ema_crossed_above_wma(window=3):
        for i in range(-window, 0):
            prev_ema = ema9.iloc[i - 1]
            prev_wma = wma45.iloc[i - 1]
            cur_ema = ema9.iloc[i]
            cur_wma = wma45.iloc[i]
            if pd.isna(prev_ema) or pd.isna(prev_wma):
                continue
            if prev_ema <= prev_wma and cur_ema > cur_wma:
                return True
        return False

    def ema_crossed_below_wma(window=3):
        for i in range(-window, 0):
            prev_ema = ema9.iloc[i - 1]
            prev_wma = wma45.iloc[i - 1]
            cur_ema = ema9.iloc[i]
            cur_wma = wma45.iloc[i]
            if pd.isna(prev_ema) or pd.isna(prev_wma):
                continue
            if prev_ema >= prev_wma and cur_ema < cur_wma:
                return True
        return False

    if last_rsi <= 25 and ema_crossed_above_wma():
        return "BUY"
    if last_rsi >= 75 and ema_crossed_below_wma():
        return "SELL"
    if last_rsi <= 35 and last_ema < last_wma and (last_wma - last_ema) < 2:
        return "NEAR BUY"
    return "NO SIGNAL"


def detect_divergence(df: pd.DataFrame, lookback: int = 20) -> str:
    """Basic divergence detection."""
    if len(df) < lookback + 1:
        return "NONE"

    window = df.tail(lookback)
    prices = window["close"]
    rsi = window["rsi"]

    # Find swing lows for bullish divergence
    price_low_idx = prices.idxmin()
    rsi_low_idx = rsi.idxmin()

    # Find swing highs for bearish divergence
    price_high_idx = prices.idxmax()
    rsi_high_idx = rsi.idxmax()

    last_price = prices.iloc[-1]
    last_rsi = rsi.iloc[-1]

    price_at_low = prices[price_low_idx]
    rsi_at_low = rsi[rsi_low_idx]
    price_at_high = prices[price_high_idx]
    rsi_at_high = rsi[rsi_high_idx]

    divergences = []

    # Bullish: price lower low but RSI higher low
    if last_price < price_at_low and last_rsi > rsi_at_low:
        divergences.append("BULLISH DIV")

    # Bearish: price higher high but RSI lower high
    if last_price > price_at_high and last_rsi < rsi_at_high:
        divergences.append("BEARISH DIV")

    return ", ".join(divergences) if divergences else "NONE"


def add_indicators(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["rsi"] = calc_rsi(df["close"])
    df["ema9"] = calc_ema(df["rsi"])
    df["wma45"] = calc_wma(df["rsi"])
    return df


# ---------------------------------------------------------------------------
# Data fetching
# ---------------------------------------------------------------------------
def fetch_vnstock(ticker: str, start: str, end: str, resolution: str = "1D") -> pd.DataFrame:
    """Fetch from vnstock library."""
    try:
        from vnstock import stock_historical_data
        df = stock_historical_data(
            symbol=ticker,
            start_date=start,
            end_date=end,
            resolution=resolution,
            type="stock" if ticker not in ("VNINDEX",) else "index",
        )
        if df is None or df.empty:
            raise ValueError("Empty result from vnstock")
        # Normalize columns
        col_map = {}
        for c in df.columns:
            cl = c.lower()
            if cl in ("open", "o"):
                col_map[c] = "open"
            elif cl in ("high", "h"):
                col_map[c] = "high"
            elif cl in ("low", "l"):
                col_map[c] = "low"
            elif cl in ("close", "c"):
                col_map[c] = "close"
            elif cl in ("volume", "vol", "v"):
                col_map[c] = "volume"
            elif cl in ("time", "tradingdate", "date", "datetime"):
                col_map[c] = "date"
        df = df.rename(columns=col_map)
        if "date" in df.columns:
            df["date"] = pd.to_datetime(df["date"])
            df = df.set_index("date")
        df = df.sort_index()
        return df[["open", "high", "low", "close", "volume"]].dropna(subset=["close"])
    except Exception as e:
        raise RuntimeError(f"vnstock fetch failed: {e}")


def fetch_yfinance(ticker: str, start: str, end: str, interval: str = "1d") -> pd.DataFrame:
    """Fetch from yfinance."""
    import yfinance as yf
    yf_ticker = ticker
    if not any(c in ticker for c in [".", "^", "=", "-"]):
        yf_ticker = ticker + ".VN"
    t = yf.Ticker(yf_ticker)
    df = t.history(start=start, end=end, interval=interval)
    if df.empty:
        raise RuntimeError(f"yfinance: no data for {yf_ticker}")
    df.columns = [c.lower() for c in df.columns]
    df.index = pd.to_datetime(df.index)
    df.index.name = "date"
    return df[["open", "high", "low", "close", "volume"]].dropna(subset=["close"])


def fetch_data(ticker: str, start: str, end: str, timeframe: str = "1d") -> pd.DataFrame:
    """Try vnstock first, fallback to yfinance."""
    resolution_map = {"1d": "1D", "1h": "1H", "1w": "1W"}
    yf_interval_map = {"1d": "1d", "1h": "1h", "1w": "1wk"}
    resolution = resolution_map.get(timeframe, "1D")
    yf_interval = yf_interval_map.get(timeframe, "1d")

    vn_ticker = ticker.upper().replace("^VNINDEX", "VNINDEX")

    try:
        df = fetch_vnstock(vn_ticker, start, end, resolution)
        print(f"  {green('✓')} Data fetched via vnstock ({len(df)} candles)")
        return df
    except Exception as e1:
        print(f"  {yellow('!')} vnstock failed ({e1}), trying yfinance...")

    try:
        yf_ticker = "^VNINDEX" if vn_ticker == "VNINDEX" else vn_ticker
        df = fetch_yfinance(yf_ticker, start, end, yf_interval)
        print(f"  {green('✓')} Data fetched via yfinance ({len(df)} candles)")
        return df
    except Exception as e2:
        raise RuntimeError(
            f"All data sources failed.\n"
            f"  vnstock: {e1}\n"
            f"  yfinance: {e2}\n\n"
            f"Install vnstock: pip install vnstock>=3.0.0"
        )


def fetch_macro_live() -> dict:
    """Fetch live macro values from yfinance where possible."""
    import yfinance as yf
    macro = MACRO_DATA.copy()

    try:
        brent = yf.Ticker("BZ=F")
        h = brent.history(period="5d")
        if not h.empty:
            macro["brent"] = round(float(h["Close"].iloc[-1]), 2)
    except Exception:
        pass

    try:
        dxy = yf.Ticker("DX-Y.NYB")
        h = dxy.history(period="5d")
        if not h.empty:
            macro["dxy"] = round(float(h["Close"].iloc[-1]), 2)
    except Exception:
        macro["dxy"] = None

    try:
        fxi = yf.Ticker("FXI")
        h = fxi.history(period="5d")
        if not h.empty:
            macro["fxi"] = round(float(h["Close"].iloc[-1]), 2)
    except Exception:
        macro["fxi"] = None

    return macro


# ---------------------------------------------------------------------------
# Macro scoring
# ---------------------------------------------------------------------------
def score_macro(macro: dict) -> dict:
    scores = {}
    details = {}

    # usdvnd
    usd = macro.get("usdvnd", 25450)
    if usd > 25500:
        scores["usdvnd"] = -1
        details["usdvnd"] = f"{usd:,.0f} — VND weak {red('▼')}"
    elif usd < 24500:
        scores["usdvnd"] = +1
        details["usdvnd"] = f"{usd:,.0f} — VND strong {green('▲')}"
    else:
        scores["usdvnd"] = 0
        details["usdvnd"] = f"{usd:,.0f} — Neutral {yellow('→')}"

    # sbv_rate
    sbv = macro.get("sbv_rate", 4.5)
    scores["sbv_rate"] = 0
    details["sbv_rate"] = f"{sbv}% — Stable {yellow('→')}"

    # cpi
    cpi = macro.get("cpi", 3.2)
    if cpi < 3:
        scores["cpi"] = +1
        details["cpi"] = f"{cpi}% — Low inflation {green('▲')}"
    elif cpi > 4.5:
        scores["cpi"] = -1
        details["cpi"] = f"{cpi}% — High inflation {red('▼')}"
    else:
        scores["cpi"] = 0
        details["cpi"] = f"{cpi}% — Moderate {yellow('→')}"

    # foreign_flow
    ff = macro.get("foreign_flow", 0)
    if ff > 100:
        scores["foreign_flow"] = +1
        details["foreign_flow"] = f"{ff:+.0f}M VND — Net buy {green('▲')}"
    elif ff < -100:
        scores["foreign_flow"] = -1
        details["foreign_flow"] = f"{ff:+.0f}M VND — Net sell {red('▼')}"
    else:
        scores["foreign_flow"] = 0
        details["foreign_flow"] = f"{ff:+.0f}M VND — Neutral {yellow('→')}"

    # brent
    brent = macro.get("brent", 75)
    if 70 <= brent <= 90:
        scores["brent"] = +1
        details["brent"] = f"${brent:.1f} — Favorable range {green('▲')}"
    elif 90 < brent <= 100:
        scores["brent"] = 0
        details["brent"] = f"${brent:.1f} — Elevated {yellow('→')}"
    else:
        scores["brent"] = -1
        details["brent"] = f"${brent:.1f} — Unfavorable {red('▼')}"

    # china_pmi
    pmi = macro.get("china_pmi", 51.0)
    if pmi > 51:
        scores["china_pmi"] = +1
        details["china_pmi"] = f"{pmi} — Expanding {green('▲')}"
    elif pmi >= 50:
        scores["china_pmi"] = 0
        details["china_pmi"] = f"{pmi} — Borderline {yellow('→')}"
    else:
        scores["china_pmi"] = -1
        details["china_pmi"] = f"{pmi} — Contracting {red('▼')}"

    # fed_rate
    fed = macro.get("fed_rate", 4.25)
    scores["fed_rate"] = 0
    details["fed_rate"] = f"{fed}% — Stable/Watch {yellow('→')}"

    # vn_gdp
    gdp = macro.get("vn_gdp", 6.8)
    if gdp > 6.5:
        scores["vn_gdp"] = +1
        details["vn_gdp"] = f"{gdp}% — Strong growth {green('▲')}"
    elif gdp >= 5:
        scores["vn_gdp"] = 0
        details["vn_gdp"] = f"{gdp}% — Moderate {yellow('→')}"
    else:
        scores["vn_gdp"] = -1
        details["vn_gdp"] = f"{gdp}% — Weak {red('▼')}"

    # credit_growth
    cg = macro.get("credit_growth", 13.5)
    if 12 <= cg <= 18:
        scores["credit_growth"] = +1
        details["credit_growth"] = f"{cg}% — Healthy range {green('▲')}"
    elif 8 <= cg < 12:
        scores["credit_growth"] = 0
        details["credit_growth"] = f"{cg}% — Moderate {yellow('→')}"
    else:
        scores["credit_growth"] = -1
        details["credit_growth"] = f"{cg}% — Concern {red('▼')}"

    # vnindex_rsi
    vrsi = macro.get("vnindex_rsi")
    if vrsi is not None:
        if vrsi > 50:
            scores["vnindex_rsi"] = +1
            details["vnindex_rsi"] = f"{vrsi:.1f} — Bullish {green('▲')}"
        elif vrsi >= 40:
            scores["vnindex_rsi"] = 0
            details["vnindex_rsi"] = f"{vrsi:.1f} — Neutral {yellow('→')}"
        else:
            scores["vnindex_rsi"] = -1
            details["vnindex_rsi"] = f"{vrsi:.1f} — Bearish {red('▼')}"
    else:
        scores["vnindex_rsi"] = 0
        details["vnindex_rsi"] = "N/A — Not calculated"

    total = sum(scores.values())
    max_score = len(scores)

    return {
        "scores": scores,
        "details": details,
        "total": total,
        "max": max_score,
        "pct": round(total / max_score * 100, 1),
    }


def print_macro_dashboard(macro: dict):
    result = score_macro(macro)
    total = result["total"]
    pct = result["pct"]

    print()
    print(bold(cyan("=" * 60)))
    print(bold(cyan("  VN MACRO DASHBOARD — EGAN FRAMEWORK")))
    print(bold(cyan("=" * 60)))
    print(f"  Date: {datetime.now().strftime('%Y-%m-%d %H:%M')}")
    print()

    labels = {
        "vnindex_rsi":   "VNINDEX RSI",
        "usdvnd":        "USD/VND",
        "sbv_rate":      "SBV Rate",
        "cpi":           "CPI YoY",
        "foreign_flow":  "Foreign Flow",
        "brent":         "Brent Oil",
        "china_pmi":     "China PMI",
        "fed_rate":      "Fed Rate",
        "vn_gdp":        "VN GDP YoY",
        "credit_growth": "Credit Growth",
    }

    for key, label in labels.items():
        score = result["scores"].get(key, 0)
        detail = result["details"].get(key, "N/A")
        arrow = green("+1") if score > 0 else (red("-1") if score < 0 else yellow(" 0"))
        print(f"  {label:<18} {arrow}  {detail}")

    print()
    print(bold(cyan("-" * 60)))

    if total >= 5:
        rating = green("BULLISH")
    elif total >= 1:
        rating = yellow("MILD BULLISH")
    elif total >= -2:
        rating = yellow("NEUTRAL")
    elif total >= -5:
        rating = red("MILD BEARISH")
    else:
        rating = red("BEARISH")

    print(f"  Total Score: {bold(str(total))}/{result['max']}  ({pct}%)  —  {rating}")
    print(bold(cyan("=" * 60)))
    print()


# ---------------------------------------------------------------------------
# Signal mode
# ---------------------------------------------------------------------------
def print_signal(ticker: str, df: pd.DataFrame):
    if len(df) < 60:
        print(yellow(f"  Warning: only {len(df)} candles — need ≥60 for WMA(45). Results may be incomplete."))

    df = add_indicators(df)
    signal = detect_signal(df)
    div = detect_divergence(df)

    last = df.iloc[-1]
    rsi_val = last["rsi"]
    ema9_val = last["ema9"]
    wma45_val = last["wma45"]
    close_val = last["close"]

    if pd.isna(rsi_val):
        rsi_str = "N/A"
    else:
        rsi_str = f"{rsi_val:.2f}"

    if pd.isna(ema9_val):
        ema9_str = "N/A"
    else:
        ema9_str = f"{ema9_val:.2f}"

    if pd.isna(wma45_val):
        wma45_str = "N/A"
    else:
        wma45_str = f"{wma45_val:.2f}"

    print()
    print(bold(cyan("=" * 60)))
    print(bold(cyan(f"  SIGNAL ANALYSIS — {ticker.upper()}")))
    print(bold(cyan("=" * 60)))
    print(f"  Date     : {df.index[-1].strftime('%Y-%m-%d') if hasattr(df.index[-1], 'strftime') else df.index[-1]}")
    print(f"  Close    : {close_val:,.2f}")
    print(f"  RSI(14)  : {rsi_str}")
    print(f"  EMA(9)   : {ema9_str}")
    print(f"  WMA(45)  : {wma45_str}")
    print()

    if signal == "BUY":
        sig_str = green("*** BUY ***")
    elif signal == "SELL":
        sig_str = red("*** SELL ***")
    elif signal == "NEAR BUY":
        sig_str = yellow("~ NEAR BUY ~")
    else:
        sig_str = "NO SIGNAL"

    print(f"  Signal   : {bold(sig_str)}")
    print(f"  Divergence: {div}")
    print(bold(cyan("=" * 60)))
    print()


# ---------------------------------------------------------------------------
# Backtest mode
# ---------------------------------------------------------------------------
def run_backtest(ticker: str, df: pd.DataFrame, sl_pct: float = 0.03, tp_pct: float = 0.03):
    if len(df) < 60:
        print(red(f"  Not enough data for backtest ({len(df)} candles). Need ≥60."))
        return

    df = add_indicators(df)

    trades = []
    in_trade = False
    entry_price = None
    entry_date = None
    direction = None

    for i in range(46, len(df)):
        window = df.iloc[: i + 1]
        signal = detect_signal(window)

        if not in_trade:
            if signal in ("BUY", "NEAR BUY"):
                in_trade = True
                entry_price = df["close"].iloc[i]
                entry_date = df.index[i]
                direction = "LONG"
            elif signal == "SELL":
                in_trade = True
                entry_price = df["close"].iloc[i]
                entry_date = df.index[i]
                direction = "SHORT"
        else:
            current_price = df["close"].iloc[i]

            if direction == "LONG":
                pct_change = (current_price - entry_price) / entry_price
                if pct_change <= -sl_pct:
                    trades.append({
                        "entry": entry_date, "exit": df.index[i],
                        "direction": direction, "entry_price": entry_price,
                        "exit_price": current_price, "pnl_pct": pct_change,
                        "result": "LOSS"
                    })
                    in_trade = False
                elif pct_change >= tp_pct:
                    trades.append({
                        "entry": entry_date, "exit": df.index[i],
                        "direction": direction, "entry_price": entry_price,
                        "exit_price": current_price, "pnl_pct": pct_change,
                        "result": "WIN"
                    })
                    in_trade = False
            elif direction == "SHORT":
                pct_change = (entry_price - current_price) / entry_price
                if pct_change <= -sl_pct:
                    trades.append({
                        "entry": entry_date, "exit": df.index[i],
                        "direction": direction, "entry_price": entry_price,
                        "exit_price": current_price, "pnl_pct": -sl_pct,
                        "result": "LOSS"
                    })
                    in_trade = False
                elif pct_change >= tp_pct:
                    trades.append({
                        "entry": entry_date, "exit": df.index[i],
                        "direction": direction, "entry_price": entry_price,
                        "exit_price": current_price, "pnl_pct": pct_change,
                        "result": "WIN"
                    })
                    in_trade = False

    # Close any open trade at end
    if in_trade:
        current_price = df["close"].iloc[-1]
        pct_change = (current_price - entry_price) / entry_price if direction == "LONG" else (entry_price - current_price) / entry_price
        trades.append({
            "entry": entry_date, "exit": df.index[-1],
            "direction": direction, "entry_price": entry_price,
            "exit_price": current_price, "pnl_pct": pct_change,
            "result": "WIN" if pct_change > 0 else "LOSS"
        })

    print()
    print(bold(cyan("=" * 60)))
    print(bold(cyan(f"  BACKTEST RESULTS — {ticker.upper()}")))
    print(bold(cyan("=" * 60)))

    if not trades:
        print(yellow("  No trades generated in this period."))
        print(bold(cyan("=" * 60)))
        return

    total = len(trades)
    wins = sum(1 for t in trades if t["result"] == "WIN")
    losses = total - wins
    win_rate = wins / total * 100
    total_return = sum(t["pnl_pct"] for t in trades) * 100

    # Max drawdown
    equity = [0]
    for t in trades:
        equity.append(equity[-1] + t["pnl_pct"] * 100)
    peak = equity[0]
    max_dd = 0
    for e in equity:
        if e > peak:
            peak = e
        dd = peak - e
        if dd > max_dd:
            max_dd = dd

    avg_win = np.mean([t["pnl_pct"] * 100 for t in trades if t["result"] == "WIN"]) if wins > 0 else 0
    avg_loss = np.mean([t["pnl_pct"] * 100 for t in trades if t["result"] == "LOSS"]) if losses > 0 else 0
    rr = abs(avg_win / avg_loss) if avg_loss != 0 else float("inf")

    print(f"  Total Trades : {total}")
    print(f"  Wins / Losses: {green(str(wins))} / {red(str(losses))}")
    print(f"  Win Rate     : {color_value(round(win_rate, 1), good_above=55, bad_above=40)}%")
    print(f"  Avg Win      : {green(f'{avg_win:.2f}%')}")
    print(f"  Avg Loss     : {red(f'{avg_loss:.2f}%')}")
    print(f"  Avg R:R      : {bold(f'{rr:.2f}')}")
    print(f"  Max Drawdown : {red(f'{max_dd:.2f}%')}")
    print(f"  Total Return : {green(f'{total_return:.2f}%') if total_return > 0 else red(f'{total_return:.2f}%')}")
    print()
    print(f"  Last 5 trades:")
    for t in trades[-5:]:
        entry_str = t["entry"].strftime("%Y-%m-%d") if hasattr(t["entry"], "strftime") else str(t["entry"])
        exit_str = t["exit"].strftime("%Y-%m-%d") if hasattr(t["exit"], "strftime") else str(t["exit"])
        pnl = t["pnl_pct"] * 100
        res = green("WIN ") if t["result"] == "WIN" else red("LOSS")
        print(f"    {entry_str} → {exit_str}  {t['direction']:5}  {pnl:+.2f}%  {res}")

    print(bold(cyan("=" * 60)))
    print()


# ---------------------------------------------------------------------------
# Scan mode
# ---------------------------------------------------------------------------
def run_scan(sector: str, timeframe: str):
    tickers = SECTOR_TICKERS.get(sector.lower())
    if not tickers:
        available = ", ".join(SECTOR_TICKERS.keys())
        print(red(f"  Unknown sector '{sector}'. Available: {available}"))
        return

    end = datetime.now().strftime("%Y-%m-%d")
    start = (datetime.now() - timedelta(days=120)).strftime("%Y-%m-%d")

    print()
    print(bold(cyan("=" * 60)))
    print(bold(cyan(f"  SECTOR SCAN — {sector.upper()}")))
    print(bold(cyan("=" * 60)))
    print(f"  {'Ticker':<10} {'RSI':>7} {'Signal':<14} {'Close':>10}")
    print(f"  {'-'*10} {'-'*7} {'-'*14} {'-'*10}")

    results = []
    for ticker in tickers:
        try:
            df = fetch_data(ticker, start, end, timeframe)
            df = add_indicators(df)
            if len(df) < 20:
                raise ValueError("too few candles")
            signal = detect_signal(df)
            rsi_val = df["rsi"].iloc[-1]
            close_val = df["close"].iloc[-1]
            results.append((ticker, rsi_val, signal, close_val))
        except Exception as e:
            results.append((ticker, None, f"ERR: {str(e)[:20]}", None))

    # Sort: BUY first, then NEAR BUY, then rest
    order = {"BUY": 0, "NEAR BUY": 1, "NO SIGNAL": 2, "SELL": 3}
    results.sort(key=lambda x: order.get(x[2], 4))

    for ticker, rsi_val, signal, close_val in results:
        rsi_str = f"{rsi_val:.1f}" if rsi_val is not None else "N/A"
        close_str = f"{close_val:,.2f}" if close_val is not None else "N/A"

        if signal == "BUY":
            sig_colored = green(f"{signal:<14}")
        elif signal == "NEAR BUY":
            sig_colored = yellow(f"{signal:<14}")
        elif signal == "SELL":
            sig_colored = red(f"{signal:<14}")
        else:
            sig_colored = f"{signal:<14}"

        print(f"  {ticker:<10} {rsi_str:>7} {sig_colored} {close_str:>10}")

    print(bold(cyan("=" * 60)))
    print()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(
        description="VN Stock Market Analyzer — Egan Methodology (RSI + EMA9 + WMA45)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python analyze_vn.py --ticker VCB --timeframe 1d --mode signal
  python analyze_vn.py --ticker VNINDEX --mode signal
  python analyze_vn.py --mode macro
  python analyze_vn.py --ticker HPG --mode backtest
  python analyze_vn.py --mode scan --sector banking
  python analyze_vn.py --mode macro --update-macro
        """,
    )
    parser.add_argument("--ticker", default="VNINDEX", help="Stock ticker symbol (default: VNINDEX)")
    parser.add_argument("--timeframe", choices=["1d", "1h", "1w"], default="1d", help="Timeframe (default: 1d)")
    parser.add_argument("--mode", choices=["signal", "macro", "backtest", "scan"], default="signal", help="Analysis mode (default: signal)")
    parser.add_argument("--sector", default="banking", help="Sector for scan mode (default: banking)")
    parser.add_argument("--update-macro", action="store_true", help="Fetch live macro data from yfinance")
    parser.add_argument("--days", type=int, default=365, help="Lookback days for data (default: 365)")

    args = parser.parse_args()

    print()
    print(bold(cyan("  VN STOCK ANALYZER — EGAN RSI METHODOLOGY")))
    print(bold(cyan(f"  Mode: {args.mode.upper()}  |  {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")))
    print()

    # Fetch macro if needed or requested
    macro = MACRO_DATA.copy()
    if args.update_macro or args.mode == "macro":
        print(f"  {cyan('Fetching live macro data...')}")
        try:
            live = fetch_macro_live()
            macro.update(live)
            print(f"  {green('✓')} Brent: ${macro.get('brent', 'N/A')}  DXY: {macro.get('dxy', 'N/A')}  FXI: {macro.get('fxi', 'N/A')}")
        except Exception as e:
            print(f"  {yellow('!')} Live macro fetch failed: {e}")

    # Fetch VNINDEX RSI for macro scoring
    if args.mode == "macro" or (args.mode == "signal" and macro.get("vnindex_rsi") is None):
        try:
            end = datetime.now().strftime("%Y-%m-%d")
            start = (datetime.now() - timedelta(days=120)).strftime("%Y-%m-%d")
            print(f"  {cyan('Fetching VNINDEX for RSI calculation...')}")
            vni_df = fetch_data("VNINDEX", start, end, "1d")
            vni_df = add_indicators(vni_df)
            macro["vnindex_rsi"] = round(float(vni_df["rsi"].iloc[-1]), 2)
            print(f"  {green('✓')} VNINDEX RSI: {macro['vnindex_rsi']}")
        except Exception as e:
            print(f"  {yellow('!')} Could not fetch VNINDEX RSI: {e}")

    if args.mode == "macro":
        print_macro_dashboard(macro)
        return

    if args.mode == "scan":
        run_scan(args.sector, args.timeframe)
        return

    # Signal or backtest — need ticker data
    ticker = args.ticker.upper()
    end = datetime.now().strftime("%Y-%m-%d")
    start = (datetime.now() - timedelta(days=args.days)).strftime("%Y-%m-%d")

    print(f"  {cyan(f'Fetching data for {ticker}...')}  ({start} → {end}, {args.timeframe})")

    try:
        df = fetch_data(ticker, start, end, args.timeframe)
    except RuntimeError as e:
        print(f"\n  {red('ERROR:')} {e}")
        sys.exit(1)

    if len(df) < 60:
        print(yellow(f"  Warning: Only {len(df)} candles available — WMA(45) needs ≥60 for reliable results."))

    if args.mode == "signal":
        print_signal(ticker, df)

        # Also show macro summary
        macro_result = score_macro(macro)
        total = macro_result["total"]
        pct = macro_result["pct"]
        print(f"  Macro Score: {bold(str(total))}/10  ({pct}%)  |  ", end="")
        if total >= 3:
            print(green("Macro BULLISH"))
        elif total >= -2:
            print(yellow("Macro NEUTRAL"))
        else:
            print(red("Macro BEARISH"))
        print()

    elif args.mode == "backtest":
        run_backtest(ticker, df)


if __name__ == "__main__":
    main()
