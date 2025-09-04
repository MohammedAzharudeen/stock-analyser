import os
import json
from datetime import datetime, timedelta
from typing import Dict, Any, Optional
import numpy as np
import pandas as pd
import yfinance as yf
import streamlit as st
from dotenv import load_dotenv
from ta.momentum import RSIIndicator
from ta.trend import MACD

# =========================
# ENV / PAGE
# =========================
load_dotenv()

st.set_page_config(page_title="📈 Stock Analyst (Rule-Based)", page_icon="📈", layout="wide")
st.title("📈 Stock Analyst - Rule-Based Analysis")
st.caption("Enter a ticker symbol (e.g., AAPL, MSFT, TSLA). I'll analyze it using technical indicators and provide BUY/HOLD/SELL recommendations.")

# =========================
# Data / Indicator Helpers
# =========================
def fetch_stock_data(symbol: str) -> Dict[str, Any]:
    """
    Pull price history (1y daily), fast info, and basic financial ratios if available.
    """
    tk = yf.Ticker(symbol)

    # Price history
    end = datetime.utcnow()
    start = end - timedelta(days=365 + 30)  # little buffer
    hist = tk.history(start=start.date(), end=end.date(), interval="1d", auto_adjust=True)

    # Fast info (robust vs legacy .info)
    fast = {}
    try:
        fast = tk.fast_info or {}
    except Exception:
        fast = {}

    # Fundamentals (best-effort; yfinance varies by ticker)
    fin = {}
    try:
        # yfinance v0.2+ exposes .get_financials, .get_income_stmt, etc. as DataFrames
        income = tk.income_stmt
        bal = tk.balance_sheet
        cash = tk.cashflow
        fin = {
            "income_stmt_cols": list(income.columns) if isinstance(income, pd.DataFrame) else [],
            "balance_sheet_cols": list(bal.columns) if isinstance(bal, pd.DataFrame) else [],
            "cashflow_cols": list(cash.columns) if isinstance(cash, pd.DataFrame) else [],
        }
    except Exception:
        pass

    return {"hist": hist, "fast": fast, "fin": fin}


def compute_indicators(hist: pd.DataFrame) -> Dict[str, Any]:
    if hist is None or hist.empty:
        return {"error": "No price history"}

    close = hist["Close"].dropna()
    if len(close) < 60:
        return {"error": "Insufficient history (need ~60+ days)"}

    # Simple MAs
    sma20 = close.rolling(20).mean()
    sma50 = close.rolling(50).mean()
    sma200 = close.rolling(200).mean()

    # RSI
    rsi = RSIIndicator(close, window=14).rsi()

    # MACD
    macd_calc = MACD(close, window_slow=26, window_fast=12, window_sign=9)
    macd = macd_calc.macd()
    macd_signal = macd_calc.macd_signal()
    macd_diff = macd_calc.macd_diff()

    # 52w high/low
    last_252 = close.tail(252)
    high_52w = float(last_252.max())
    low_52w = float(last_252.min())

    last = float(close.iloc[-1])
    above_50 = last > float(sma50.iloc[-1]) if not np.isnan(sma50.iloc[-1]) else False
    above_200 = last > float(sma200.iloc[-1]) if not np.isnan(sma200.iloc[-1]) else False

    # Volume trend
    vol = hist["Volume"].dropna()
    vol_avg_20 = float(vol.tail(20).mean()) if len(vol) >= 20 else float(vol.mean())
    vol_last = float(vol.iloc[-1]) if len(vol) else np.nan
    vol_ratio = (vol_last / vol_avg_20) if vol_avg_20 else np.nan

    return {
        "last_price": last,
        "sma20": float(sma20.iloc[-1]) if not np.isnan(sma20.iloc[-1]) else None,
        "sma50": float(sma50.iloc[-1]) if not np.isnan(sma50.iloc[-1]) else None,
        "sma200": float(sma200.iloc[-1]) if not np.isnan(sma200.iloc[-1]) else None,
        "rsi14": float(rsi.iloc[-1]) if not np.isnan(rsi.iloc[-1]) else None,
        "macd": float(macd.iloc[-1]) if not np.isnan(macd.iloc[-1]) else None,
        "macd_signal": float(macd_signal.iloc[-1]) if not np.isnan(macd_signal.iloc[-1]) else None,
        "macd_hist": float(macd_diff.iloc[-1]) if not np.isnan(macd_diff.iloc[-1]) else None,
        "52w_high": high_52w,
        "52w_low": low_52w,
        "above_sma50": above_50,
        "above_sma200": above_200,
        "volume_last": vol_last,
        "volume_avg20": vol_avg_20,
        "volume_ratio": float(vol_ratio) if not np.isnan(vol_ratio) else None,
    }


def extract_fundamentals(fast: Dict[str, Any]) -> Dict[str, Any]:
    """
    Pull whatever is reliably present in .fast_info (yfinance's newer API).
    Fields may be missing per ticker.
    """
    def safe(k, default=None):
        v = fast.get(k, default)
        try:
            return float(v)
        except Exception:
            return v

    out = {
        "market_cap": safe("market_cap"),
        "pe_ratio": safe("trailing_pe"),
        "forward_pe": safe("forward_pe"),
        "pb_ratio": safe("price_to_book"),
        "dividend_yield": safe("dividend_yield"),
    }
    return out


# =========================
# Rule-Based Analysis Engine
# =========================
def analyze_stock(technical: Dict[str, Any], fundamental: Dict[str, Any], symbol: str) -> Dict[str, Any]:
    """
    Rule-based stock analysis without AI.
    Returns recommendation based on technical and fundamental indicators.
    """
    
    # Initialize scoring system
    bullish_signals = 0
    bearish_signals = 0
    neutral_signals = 0
    
    technical_summary = []
    fundamental_summary = []
    risks = []
    notes = []
    
    # Technical Analysis Rules
    price = technical.get("last_price")
    sma20 = technical.get("sma20")
    sma50 = technical.get("sma50")
    sma200 = technical.get("sma200")
    rsi = technical.get("rsi14")
    macd = technical.get("macd")
    macd_signal = technical.get("macd_signal")
    macd_hist = technical.get("macd_hist")
    above_sma50 = technical.get("above_sma50", False)
    above_sma200 = technical.get("above_sma200", False)
    vol_ratio = technical.get("volume_ratio")
    high_52w = technical.get("52w_high")
    low_52w = technical.get("52w_low")
    
    # Moving Average Analysis
    if above_sma50 and above_sma200:
        bullish_signals += 2
        technical_summary.append("Strong uptrend: Price above both 50-day and 200-day moving averages")
    elif above_sma50:
        bullish_signals += 1
        technical_summary.append("Moderate uptrend: Price above 50-day moving average")
    elif above_sma200:
        neutral_signals += 1
        technical_summary.append("Mixed signals: Price above 200-day but below 50-day MA")
    else:
        bearish_signals += 2
        technical_summary.append("Downtrend: Price below major moving averages")
        risks.append("Price trading below key support levels")
    
    # RSI Analysis
    if rsi:
        if rsi > 70:
            bearish_signals += 1
            technical_summary.append(f"RSI overbought at {rsi:.1f}")
            risks.append("Potential short-term pullback due to overbought conditions")
        elif rsi < 30:
            bullish_signals += 1
            technical_summary.append(f"RSI oversold at {rsi:.1f} - potential bounce")
        elif 45 <= rsi <= 65:
            bullish_signals += 1
            technical_summary.append(f"RSI healthy at {rsi:.1f}")
        else:
            neutral_signals += 1
            technical_summary.append(f"RSI neutral at {rsi:.1f}")
    
    # MACD Analysis
    if macd and macd_signal and macd_hist:
        if macd > macd_signal and macd_hist > 0:
            bullish_signals += 1
            technical_summary.append("MACD bullish crossover - momentum positive")
        elif macd < macd_signal and macd_hist < 0:
            bearish_signals += 1
            technical_summary.append("MACD bearish crossover - momentum negative")
        else:
            neutral_signals += 1
            technical_summary.append("MACD mixed signals")
    
    # Volume Analysis
    if vol_ratio:
        if vol_ratio > 1.5:
            notes.append(f"High volume activity ({vol_ratio:.1f}x average)")
        elif vol_ratio < 0.5:
            risks.append("Low volume - lack of conviction in price moves")
    
    # 52-week Range Analysis
    if price and high_52w and low_52w:
        range_position = (price - low_52w) / (high_52w - low_52w)
        if range_position > 0.8:
            risks.append("Trading near 52-week highs - limited upside potential")
        elif range_position < 0.2:
            bullish_signals += 1
            technical_summary.append("Trading near 52-week lows - potential value opportunity")
        
        notes.append(f"Trading at {range_position*100:.1f}% of 52-week range")
    
    # Fundamental Analysis
    pe_ratio = fundamental.get("pe_ratio")
    pb_ratio = fundamental.get("pb_ratio")
    market_cap = fundamental.get("market_cap")
    dividend_yield = fundamental.get("dividend_yield")
    
    if pe_ratio:
        if pe_ratio < 15:
            bullish_signals += 1
            fundamental_summary.append(f"Attractive valuation: P/E ratio of {pe_ratio:.1f}")
        elif pe_ratio > 30:
            bearish_signals += 1
            fundamental_summary.append(f"High valuation: P/E ratio of {pe_ratio:.1f}")
            risks.append("High valuation multiple may limit returns")
        else:
            neutral_signals += 1
            fundamental_summary.append(f"Moderate valuation: P/E ratio of {pe_ratio:.1f}")
    
    if pb_ratio:
        if pb_ratio < 1.5:
            bullish_signals += 1
            fundamental_summary.append(f"Trading below book value: P/B ratio of {pb_ratio:.1f}")
        elif pb_ratio > 3:
            bearish_signals += 1
            fundamental_summary.append(f"High price-to-book: P/B ratio of {pb_ratio:.1f}")
    
    if dividend_yield and dividend_yield > 0.02:  # 2%+
        bullish_signals += 1
        fundamental_summary.append(f"Dividend yield of {dividend_yield*100:.1f}% provides income")
    
    # Decision Logic
    total_signals = bullish_signals + bearish_signals + neutral_signals
    if total_signals == 0:
        action = "HOLD"
        confidence = 50
        notes.append("Insufficient data for strong recommendation")
    else:
        bullish_ratio = bullish_signals / total_signals
        bearish_ratio = bearish_signals / total_signals
        
        if bullish_ratio >= 0.6:
            action = "BUY"
            confidence = min(90, 50 + int(bullish_ratio * 40))
        elif bearish_ratio >= 0.6:
            action = "SELL"
            confidence = min(90, 50 + int(bearish_ratio * 40))
        else:
            action = "HOLD"
            confidence = 60
    
    # Add general market context
    notes.append(f"Analysis based on {bullish_signals} bullish, {bearish_signals} bearish, {neutral_signals} neutral signals")
    
    if not technical_summary:
        technical_summary.append("Limited technical data available")
    if not fundamental_summary:
        fundamental_summary.append("Limited fundamental data available")
    if not risks:
        risks.append("General market risk applies")
    
    return {
        "action": action,
        "confidence": confidence,
        "technical_summary": " | ".join(technical_summary),
        "fundamental_summary": " | ".join(fundamental_summary),
        "risks": risks,
        "notes": " | ".join(notes)
    }


# =========================
# UI
# =========================
col = st.columns([2, 1, 1])
with col[0]:
    symbol = st.text_input("Ticker symbol", value="", placeholder="e.g., AAPL")

with col[1]:
    lookback = st.selectbox("Price lookback", ["1y", "6mo", "3mo"], index=0)

with col[2]:
    run_btn = st.button("Analyze", type="primary", use_container_width=True, disabled=(not symbol.strip()))

st.divider()

if run_btn:
    ticker = symbol.strip().upper()
    with st.spinner(f"Fetching and analyzing {ticker}…"):
        try:
            data = fetch_stock_data(ticker)
            hist = data["hist"]
            fast = data["fast"]

            if hist is None or hist.empty:
                st.error("No price data found for this symbol.")
            else:
                # Trim lookback for display
                if lookback == "6mo":
                    hist_disp = hist.tail(126)  # ~126 trading days
                elif lookback == "3mo":
                    hist_disp = hist.tail(63)
                else:
                    hist_disp = hist

                inds = compute_indicators(hist)
                fins = extract_fundamentals(fast)
                
                # Show raw metrics
                st.subheader(f"{ticker} – Key Metrics")
                c1, c2 = st.columns(2)
                with c1:
                    st.markdown("**Technical**")
                    st.json(inds, expanded=False)
                with c2:
                    st.markdown("**Fundamental**")
                    st.json(fins, expanded=False)

                # Chart
                st.markdown("**Price (Adj Close)**")
                st.line_chart(hist_disp["Close"])

                # Rule-based analysis
                decision = analyze_stock(inds, fins, ticker)
                
                a = decision.get("action", "HOLD")
                conf = decision.get("confidence", 50)
                
                # Color-code the recommendation
                if a == "BUY":
                    st.success(f"**Recommendation: {a}** (confidence: {conf}/100)")
                elif a == "SELL":
                    st.error(f"**Recommendation: {a}** (confidence: {conf}/100)")
                else:
                    st.info(f"**Recommendation: {a}** (confidence: {conf}/100)")
                
                c1, c2 = st.columns(2)
                with c1:
                    st.markdown("**Technical Summary**")
                    st.write(decision.get("technical_summary", ""))
                with c2:
                    st.markdown("**Fundamental Summary**")
                    st.write(decision.get("fundamental_summary", ""))

                if decision.get("risks"):
                    st.markdown("**Risks**")
                    st.write("- " + "\n- ".join(decision["risks"]))
                if decision.get("notes"):
                    st.markdown("**Notes**")
                    st.write(decision["notes"])

        except Exception as e:
            st.error(f"Error: {e}")

# Footer
st.divider()
st.caption("Data via yfinance. Rule-based analysis - not financial advice.")
st.info("💡 **No API Key Required!** This version uses algorithmic rules based on technical and fundamental analysis.")
