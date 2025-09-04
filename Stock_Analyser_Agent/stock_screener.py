import os
import json
import time
import asyncio
from datetime import datetime, timezone, timedelta
from typing import Dict, Any, List, Tuple
import numpy as np
import pandas as pd
import yfinance as yf
import streamlit as st
from dotenv import load_dotenv
from ta.momentum import RSIIndicator
from ta.trend import MACD
from groq import Groq
import concurrent.futures
from threading import Thread

# =========================
# ENV / PAGE
# =========================
load_dotenv()
GROQ_API_KEY = os.getenv("GROQ_API_KEY", "")

st.set_page_config(page_title="📊 Stock Screener AI", page_icon="📊", layout="wide")
st.title("📊 AI Stock Screener - Top 5 Buy Recommendations")
st.caption("Automated analysis of multiple stocks with AI-powered buy recommendations updated every 10 minutes")

if not GROQ_API_KEY:
    st.error("⚠️ **GROQ_API_KEY not found!** Please add it to your .env file.")
    st.stop()

# =========================
# Stock Universe
# =========================
STOCK_UNIVERSE = {
    "US_LARGE_CAP": ["AAPL", "MSFT", "GOOGL", "AMZN", "TSLA", "NVDA", "META", "NFLX", "JPM", "JNJ"],
    "US_GROWTH": ["TSLA", "NVDA", "AMD", "CRM", "SHOP", "ROKU", "ZM", "PLTR", "SNOW", "COIN"],
    "US_VALUE": ["BRK-B", "JPM", "BAC", "WFC", "XOM", "CVX", "KO", "PG", "JNJ", "PFE"],
    "INDIAN_LARGE_CAP": ["RELIANCE.NS", "TCS.NS", "INFY.NS", "HDFCBANK.NS", "ITC.NS", "BHARTIARTL.NS", "SBIN.NS", "LT.NS", "WIPRO.NS", "MARUTI.NS"],
    "CRYPTO_RELATED": ["COIN", "MSTR", "RIOT", "MARA", "SQ", "PYPL"],
    "ETF_POPULAR": ["SPY", "QQQ", "VOO", "VTI", "ARKK", "XLK"]
}

# Flatten all stocks into one list
ALL_STOCKS = []
for category, stocks in STOCK_UNIVERSE.items():
    ALL_STOCKS.extend(stocks)
ALL_STOCKS = list(set(ALL_STOCKS))  # Remove duplicates

# =========================
# Session State
# =========================
def _init_state():
    if "groq_client" not in st.session_state:
        st.session_state.groq_client = None
    if "last_analysis_time" not in st.session_state:
        st.session_state.last_analysis_time = None
    if "top_stocks" not in st.session_state:
        st.session_state.top_stocks = []
    if "analysis_running" not in st.session_state:
        st.session_state.analysis_running = False

_init_state()

# =========================
# Analysis Functions (from stock_groq.py)
# =========================
def fetch_stock_data_quick(symbol: str) -> Dict[str, Any]:
    """Quick stock data fetch for screening"""
    try:
        tk = yf.Ticker(symbol)
        end = datetime.now(timezone.utc)
        start = end - timedelta(days=365 + 30)
        hist = tk.history(start=start.date(), end=end.date(), interval="1d", auto_adjust=True)
        
        if hist.empty:
            return {"error": f"No data for {symbol}"}
            
        fast = {}
        try:
            fast = tk.fast_info or {}
        except:
            pass
            
        return {"hist": hist, "fast": fast, "symbol": symbol}
    except Exception as e:
        return {"error": str(e), "symbol": symbol}

def compute_indicators_quick(hist: pd.DataFrame) -> Dict[str, Any]:
    """Quick technical indicators computation"""
    if hist is None or hist.empty or len(hist) < 60:
        return {"error": "Insufficient data"}

    close = hist["Close"].dropna()
    
    # Moving averages
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
    
    # Price metrics
    last = float(close.iloc[-1])
    above_50 = last > float(sma50.iloc[-1]) if not np.isnan(sma50.iloc[-1]) else False
    above_200 = last > float(sma200.iloc[-1]) if not np.isnan(sma200.iloc[-1]) else False
    
    # 52-week range
    high_52w = float(close.tail(252).max())
    low_52w = float(close.tail(252).min())
    
    # Volume
    vol = hist["Volume"].dropna()
    vol_avg_20 = float(vol.tail(20).mean()) if len(vol) >= 20 else 0
    vol_last = float(vol.iloc[-1]) if len(vol) else 0
    vol_ratio = (vol_last / vol_avg_20) if vol_avg_20 > 0 else 1
    
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
        "volume_ratio": float(vol_ratio) if not np.isnan(vol_ratio) else None,
    }

def extract_fundamentals_quick(ticker_obj, fast: Dict[str, Any]) -> Dict[str, Any]:
    """Quick fundamental data extraction"""
    def safe(v, default=None):
        try:
            return float(v) if v is not None else default
        except (ValueError, TypeError):
            return default

    out = {
        "market_cap": safe(fast.get("market_cap")),
        "pe_ratio": safe(fast.get("trailing_pe")),
        "pb_ratio": safe(fast.get("price_to_book")),
        "dividend_yield": safe(fast.get("dividend_yield")),
    }
    
    # Quick fallback to legacy info if needed
    if all(v is None for v in out.values()):
        try:
            info = ticker_obj.info
            if info:
                out = {
                    "market_cap": safe(info.get("marketCap")),
                    "pe_ratio": safe(info.get("trailingPE")),
                    "pb_ratio": safe(info.get("priceToBook")),
                    "dividend_yield": safe(info.get("dividendYield")),
                }
        except:
            pass
    
    return out

# =========================
# Scoring System
# =========================
def calculate_stock_score(technical: Dict[str, Any], fundamental: Dict[str, Any]) -> Tuple[float, List[str]]:
    """
    Calculate a buy score (0-100) for a stock based on technical and fundamental analysis
    Returns: (score, reasons)
    """
    score = 0
    reasons = []
    
    # Technical Analysis (60% weight)
    
    # Trend Analysis (25 points)
    if technical.get("above_sma200"):
        score += 15
        reasons.append("Above 200-day MA (strong trend)")
    if technical.get("above_sma50"):
        score += 10
        reasons.append("Above 50-day MA (short-term bullish)")
    
    # RSI Analysis (15 points)
    rsi = technical.get("rsi14")
    if rsi:
        if 30 <= rsi <= 70:  # Healthy range
            score += 15
            reasons.append(f"Healthy RSI ({rsi:.1f})")
        elif rsi < 30:  # Oversold - potential bounce
            score += 10
            reasons.append(f"Oversold RSI ({rsi:.1f}) - potential bounce")
    
    # MACD Analysis (10 points)
    macd_hist = technical.get("macd_hist")
    if macd_hist and macd_hist > 0:
        score += 10
        reasons.append("MACD bullish momentum")
    
    # Volume Analysis (10 points)
    vol_ratio = technical.get("volume_ratio")
    if vol_ratio and vol_ratio > 1.2:
        score += 10
        reasons.append(f"High volume ({vol_ratio:.1f}x avg)")
    
    # Fundamental Analysis (40% weight)
    
    # Valuation (20 points)
    pe_ratio = fundamental.get("pe_ratio")
    if pe_ratio:
        if pe_ratio < 15:  # Undervalued
            score += 20
            reasons.append(f"Low P/E ({pe_ratio:.1f}) - undervalued")
        elif pe_ratio < 25:  # Fair value
            score += 15
            reasons.append(f"Reasonable P/E ({pe_ratio:.1f})")
        elif pe_ratio < 35:  # Slightly expensive
            score += 10
            reasons.append(f"Moderate P/E ({pe_ratio:.1f})")
    
    # Dividend Yield (10 points)
    div_yield = fundamental.get("dividend_yield")
    if div_yield and div_yield > 0.02:  # 2%+
        score += 10
        reasons.append(f"Good dividend yield ({div_yield*100:.1f}%)")
    
    # Market Cap Stability (10 points)
    market_cap = fundamental.get("market_cap")
    if market_cap and market_cap > 10_000_000_000:  # $10B+
        score += 10
        reasons.append("Large cap stability")
    elif market_cap and market_cap > 2_000_000_000:  # $2B+
        score += 5
        reasons.append("Mid cap growth potential")
    
    return min(score, 100), reasons

# =========================
# Multi-Stock Analysis
# =========================
def analyze_single_stock(symbol: str) -> Dict[str, Any]:
    """Analyze a single stock and return score + data"""
    try:
        # Fetch data
        data = fetch_stock_data_quick(symbol)
        if "error" in data:
            return {"symbol": symbol, "error": data["error"], "score": 0}
        
        # Calculate indicators
        technical = compute_indicators_quick(data["hist"])
        if "error" in technical:
            return {"symbol": symbol, "error": technical["error"], "score": 0}
        
        # Get fundamentals
        ticker_obj = yf.Ticker(symbol)
        fundamental = extract_fundamentals_quick(ticker_obj, data["fast"])
        
        # Calculate score
        score, reasons = calculate_stock_score(technical, fundamental)
        
        return {
            "symbol": symbol,
            "score": score,
            "reasons": reasons,
            "technical": technical,
            "fundamental": fundamental,
            "last_price": technical.get("last_price", 0),
            "error": None
        }
        
    except Exception as e:
        return {"symbol": symbol, "error": str(e), "score": 0}

def analyze_stock_universe(stock_list: List[str], max_workers: int = 10) -> List[Dict[str, Any]]:
    """Analyze multiple stocks in parallel"""
    results = []
    
    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        future_to_symbol = {executor.submit(analyze_single_stock, symbol): symbol for symbol in stock_list}
        
        for future in concurrent.futures.as_completed(future_to_symbol):
            result = future.result()
            if result["score"] > 0:  # Only include stocks with valid scores
                results.append(result)
    
    # Sort by score (highest first)
    results.sort(key=lambda x: x["score"], reverse=True)
    return results

# =========================
# UI Components
# =========================
def display_stock_card(stock_data: Dict[str, Any], rank: int):
    """Display a stock recommendation card"""
    symbol = stock_data["symbol"]
    score = stock_data["score"]
    reasons = stock_data["reasons"]
    price = stock_data["last_price"]
    
    # Color coding based on score
    if score >= 80:
        color = "🟢"
        badge_color = "success"
    elif score >= 60:
        color = "🟡"
        badge_color = "warning"
    else:
        color = "🔴"
        badge_color = "error"
    
    with st.container():
        col1, col2, col3 = st.columns([1, 3, 1])
        
        with col1:
            st.markdown(f"### #{rank}")
            st.markdown(f"{color} **{symbol}**")
        
        with col2:
            st.markdown(f"**Score: {score}/100**")
            st.markdown(f"**Price: ${price:.2f}**")
            
            # Display reasons
            if reasons:
                st.markdown("**Why Buy:**")
                for reason in reasons[:3]:  # Show top 3 reasons
                    st.markdown(f"• {reason}")
        
        with col3:
            if st.button(f"Analyze {symbol}", key=f"analyze_{symbol}_{rank}"):
                st.session_state[f"analyze_{symbol}"] = True
        
        st.divider()

# =========================
# Main UI
# =========================

# Sidebar controls
with st.sidebar:
    st.markdown("## 🎛️ Screener Controls")
    
    # Stock universe selection
    selected_categories = st.multiselect(
        "Select Stock Categories",
        options=list(STOCK_UNIVERSE.keys()),
        default=["US_LARGE_CAP", "US_GROWTH"],
        help="Choose which stock categories to analyze"
    )
    
    # Build stock list
    stocks_to_analyze = []
    for category in selected_categories:
        stocks_to_analyze.extend(STOCK_UNIVERSE[category])
    stocks_to_analyze = list(set(stocks_to_analyze))  # Remove duplicates
    
    st.markdown(f"**Stocks to analyze:** {len(stocks_to_analyze)}")
    
    # Auto-refresh toggle
    auto_refresh = st.checkbox("Auto-refresh every 10 minutes", value=True)
    
    # Manual refresh button
    if st.button("🔄 Refresh Now", type="primary"):
        st.session_state.last_analysis_time = None  # Force refresh

# Main content
if not stocks_to_analyze:
    st.warning("Please select at least one stock category from the sidebar.")
    st.stop()

# Check if we need to run analysis
current_time = datetime.now()
should_analyze = (
    st.session_state.last_analysis_time is None or
    (current_time - st.session_state.last_analysis_time).total_seconds() > 600 or  # 10 minutes
    not st.session_state.top_stocks
)

# Analysis section
if should_analyze and not st.session_state.analysis_running:
    st.session_state.analysis_running = True
    
    with st.spinner(f"🔍 Analyzing {len(stocks_to_analyze)} stocks... This may take 1-2 minutes."):
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        # Run analysis
        results = analyze_stock_universe(stocks_to_analyze)
        
        # Update session state
        st.session_state.top_stocks = results[:5]  # Top 5
        st.session_state.last_analysis_time = current_time
        st.session_state.analysis_running = False
        
        progress_bar.progress(100)
        status_text.success(f"✅ Analysis complete! Found {len(results)} valid stocks.")
        time.sleep(1)
        progress_bar.empty()
        status_text.empty()

# Display results
if st.session_state.top_stocks:
    # Header with last update time
    col1, col2 = st.columns([3, 1])
    with col1:
        st.markdown("## 🏆 Top 5 Buy Recommendations")
    with col2:
        if st.session_state.last_analysis_time:
            time_ago = (datetime.now() - st.session_state.last_analysis_time).total_seconds() / 60
            st.markdown(f"*Updated {time_ago:.0f} min ago*")
    
    # Display top 5 stocks
    for i, stock_data in enumerate(st.session_state.top_stocks, 1):
        display_stock_card(stock_data, i)
    
    # Summary statistics
    st.markdown("## 📊 Analysis Summary")
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        avg_score = np.mean([s["score"] for s in st.session_state.top_stocks])
        st.metric("Average Score", f"{avg_score:.1f}/100")
    
    with col2:
        high_score_count = len([s for s in st.session_state.top_stocks if s["score"] >= 80])
        st.metric("Strong Buys", high_score_count)
    
    with col3:
        total_analyzed = len(stocks_to_analyze)
        st.metric("Stocks Analyzed", total_analyzed)
    
    with col4:
        next_refresh = 10 - ((datetime.now() - st.session_state.last_analysis_time).total_seconds() / 60) if st.session_state.last_analysis_time else 0
        if auto_refresh and next_refresh > 0:
            st.metric("Next Refresh", f"{next_refresh:.0f} min")
        else:
            st.metric("Next Refresh", "Manual")

else:
    st.info("Click 'Refresh Now' to start the analysis!")

# Auto-refresh logic
if auto_refresh and st.session_state.last_analysis_time:
    time_since_last = (datetime.now() - st.session_state.last_analysis_time).total_seconds()
    if time_since_last > 600:  # 10 minutes
        st.rerun()

# Footer
st.divider()
st.caption("🤖 Powered by Groq AI • Data via yfinance • This is educational, not financial advice")
st.caption("⚡ Automated screening updates every 10 minutes when auto-refresh is enabled")
