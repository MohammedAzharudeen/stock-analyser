import os
import json
from datetime import datetime, timedelta, timezone
from typing import Dict, Any, Optional
import numpy as np
import pandas as pd
import yfinance as yf
import streamlit as st
from dotenv import load_dotenv
from ta.momentum import RSIIndicator
from ta.trend import MACD
from groq import Groq

# =========================
# ENV / PAGE
# =========================
load_dotenv()
GROQ_API_KEY = os.getenv("GROQ_API_KEY", "")

st.set_page_config(page_title="📈 Stock Analyst Agent (Groq)", page_icon="📈", layout="wide")
st.title("📈 Stock Analyst Agent (Powered by Groq AI)")
st.caption("Enter a ticker symbol (e.g., AAPL, MSFT, TSLA). I'll fetch data, analyze it, and provide AI-powered BUY/HOLD/SELL recommendations using Groq's fast AI models.")

if not GROQ_API_KEY:
    st.warning("⚠️ **Set GROQ_API_KEY in your environment or .env file.**", icon="⚠️")
    st.info("""
    **To get your FREE Groq API key:**
    1. Visit: https://console.groq.com/keys
    2. Sign up (free account)
    3. Create an API key
    4. Add to your .env file: `GROQ_API_KEY=your_key_here`
    
    **Groq Benefits:**
    - ✅ FREE (14,400 requests/day)
    - ⚡ Ultra-fast inference
    - 🤖 Multiple AI models available
    """)

# =========================
# Session State
# =========================
def _init_state():
    if "groq_client" not in st.session_state:
        st.session_state.groq_client: Optional[Groq] = None

_init_state()

# =========================
# Data / Indicator Helpers
# =========================
def fetch_stock_data(symbol: str) -> Dict[str, Any]:
    """
    Pull price history (1y daily), fast info, and basic financial ratios if available.
    """
    tk = yf.Ticker(symbol)

    # Price history
    end = datetime.now(timezone.utc)
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


def extract_fundamentals(ticker_obj, fast: Dict[str, Any]) -> Dict[str, Any]:
    """
    Pull fundamental data from both fast_info and legacy info APIs.
    Falls back to legacy info if fast_info doesn't have the data.
    """
    def safe(v, default=None):
        try:
            return float(v) if v is not None else default
        except (ValueError, TypeError):
            return default

    # Try fast_info first
    out = {
        "market_cap": safe(fast.get("market_cap")),
        "pe_ratio": safe(fast.get("trailing_pe")),
        "forward_pe": safe(fast.get("forward_pe")),
        "pb_ratio": safe(fast.get("price_to_book")),
        "dividend_yield": safe(fast.get("dividend_yield")),
    }
    
    # If fast_info doesn't have the data, try legacy info
    if all(v is None for v in out.values()):
        try:
            info = ticker_obj.info
            if info:
                out = {
                    "market_cap": safe(info.get("marketCap")),
                    "pe_ratio": safe(info.get("trailingPE")),
                    "forward_pe": safe(info.get("forwardPE")),
                    "pb_ratio": safe(info.get("priceToBook")),
                    "dividend_yield": safe(info.get("dividendYield")),
                }
        except Exception as e:
            # If legacy info fails, keep the fast_info results (even if None)
            pass
    
    return out


# =========================
# Groq AI Analysis
# =========================
def build_groq_client() -> Groq:
    if st.session_state.groq_client is None:
        st.session_state.groq_client = Groq(api_key=GROQ_API_KEY)
    return st.session_state.groq_client


def ask_groq_agent(payload: Dict[str, Any], model: str = "llama-3.1-8b-instant") -> str:
    """
    Send stock data to Groq AI for analysis and get investment recommendation.
    """
    client = build_groq_client()
    
    system_message = """You are a disciplined equity research assistant. You will be given:
1) Technical indicators (RSI, MACD, SMAs, 52w range, volume ratio)
2) Fundamental signals (PE, PB, market cap, dividend yield)

Output strict JSON with fields:
{
  "action": "BUY" | "HOLD" | "SELL",
  "confidence": 0-100,
  "technical_summary": "...",
  "fundamental_summary": "...",
  "risks": ["...", "..."],
  "notes": "..."
}

Rules:
- Combine both technical + fundamental signals.
- Favor BUY if trend is positive (price > SMA50 & SMA200, MACD>=0, RSI 45–65) and valuation not excessive (PE or PB reasonable vs sector).
- Favor SELL if trend is negative (price < SMA200, MACD<0, RSI<40) or valuation/risks are severe.
- Otherwise HOLD.
- Be conservative if data is missing.
- JSON only. No markdown, no extra text."""

    user_message = json.dumps(payload, indent=2)
    
    try:
        response = client.chat.completions.create(
            model=model,  # Use selected model
            messages=[
                {"role": "system", "content": system_message},
                {"role": "user", "content": user_message}
            ],
            temperature=0.3,  # More deterministic for financial decisions
            max_tokens=1000,
        )
        
        return response.choices[0].message.content
    
    except Exception as e:
        return json.dumps({
            "action": "HOLD",
            "confidence": 50,
            "technical_summary": "Error occurred during analysis",
            "fundamental_summary": "Unable to process fundamental data",
            "risks": ["Technical analysis error", "Data processing issue"],
            "notes": f"Analysis failed: {str(e)}"
        })


# =========================
# UI
# =========================
col = st.columns([2, 1, 1])
with col[0]:
    symbol = st.text_input("Ticker symbol", value="", placeholder="e.g., AAPL")

with col[1]:
    lookback = st.selectbox("Price lookback", ["1y", "6mo", "3mo"], index=0)

with col[2]:
    # Model selection
    model_choice = st.selectbox("AI Model", [
        "llama-3.1-8b-instant",    # Fast and reliable
        "llama-3.2-90b-text-preview",  # Most capable
        "llama-3.2-11b-text-preview",  # Good balance
        "mixtral-8x7b-32768",      # Alternative
        "gemma2-9b-it"             # Lightweight
    ], index=0)

st.divider()

run_btn = st.button("🚀 Analyze with Groq AI", type="primary", use_container_width=True, disabled=(not symbol.strip() or not GROQ_API_KEY))

if run_btn:
    ticker = symbol.strip().upper()
    with st.spinner(f"Fetching data and analyzing {ticker} with Groq AI..."):
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
                fins = extract_fundamentals(yf.Ticker(ticker), fast)
                payload = {
                    "symbol": ticker,
                    "as_of": datetime.now(timezone.utc).isoformat() + "Z",
                    "technical": inds,
                    "fundamental": fins,
                }

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

                # Call Groq AI for decision
                raw = ask_groq_agent(payload, model_choice)

                # Try parse
                decision = None
                try:
                    decision = json.loads(raw)
                except Exception:
                    # Sometimes models add stray characters—try a crude fix:
                    try:
                        start = raw.find("{")
                        end = raw.rfind("}")
                        if start != -1 and end != -1:
                            decision = json.loads(raw[start:end+1])
                    except Exception:
                        decision = None

                if not decision or not isinstance(decision, dict) or "action" not in decision:
                    st.error("AI returned an invalid response. Showing raw output.")
                    st.code(raw)
                else:
                    a = decision.get("action", "HOLD")
                    conf = decision.get("confidence", 50)
                    
                    # Color-code the recommendation
                    if a == "BUY":
                        st.success(f"**🟢 Recommendation: {a}** (confidence: {conf}/100)")
                    elif a == "SELL":
                        st.error(f"**🔴 Recommendation: {a}** (confidence: {conf}/100)")
                    else:
                        st.info(f"**🟡 Recommendation: {a}** (confidence: {conf}/100)")
                    
                    c1, c2 = st.columns(2)
                    with c1:
                        st.markdown("**Technical Summary**")
                        st.write(decision.get("technical_summary", ""))
                    with c2:
                        st.markdown("**Fundamental Summary**")
                        st.write(decision.get("fundamental_summary", ""))

                    if decision.get("risks"):
                        st.markdown("**⚠️ Risks**")
                        st.write("- " + "\n- ".join(decision["risks"]))
                    if decision.get("notes"):
                        st.markdown("**📝 Notes**")
                        st.write(decision["notes"])

        except Exception as e:
            st.error(f"Error: {e}")

# Footer / debug
st.divider()
st.caption("Data via yfinance • AI analysis via Groq • This is educational, not financial advice.")

# Groq info sidebar
with st.sidebar:
    st.markdown("## 🚀 Groq AI Info")
    st.success("**Ultra-fast AI inference**")
    st.info(f"**Model:** {model_choice}")
    
    if GROQ_API_KEY:
        st.success("✅ Groq API Key configured")
    else:
        st.error("❌ Groq API Key missing")
        st.markdown("""
        **Get FREE Groq API Key:**
        1. Visit [console.groq.com](https://console.groq.com/keys)
        2. Sign up (free)
        3. Create API key
        4. Add to .env: `GROQ_API_KEY=your_key`
        """)
    
    st.markdown("""
    **Groq Benefits:**
    - 🆓 Free (14,400 requests/day)
    - ⚡ Ultra-fast inference
    - 🤖 Latest LLaMA 3.2 models
    - 🔒 Secure & reliable
    
    **Available Models:**
    - **llama-3.1-8b-instant** - Fastest
    - **llama-3.2-90b-text-preview** - Most capable
    - **llama-3.2-11b-text-preview** - Balanced
    - **mixtral-8x7b-32768** - Alternative
    """)
