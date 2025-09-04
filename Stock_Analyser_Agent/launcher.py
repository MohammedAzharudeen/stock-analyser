import streamlit as st
import subprocess
import sys
import os

st.set_page_config(page_title="🚀 Stock Analysis Suite", page_icon="🚀", layout="centered")

st.title("🚀 Stock Analysis Suite")
st.markdown("Choose your analysis tool:")

col1, col2 = st.columns(2)

with col1:
    st.markdown("### 📈 Individual Stock Analyzer")
    st.markdown("""
    **Features:**
    - Analyze specific stocks
    - Detailed technical & fundamental analysis
    - AI-powered BUY/HOLD/SELL recommendations
    - Interactive charts
    - Multiple AI models
    """)
    
    if st.button("🔍 Launch Stock Analyzer", type="primary", use_container_width=True):
        st.markdown("**Starting Individual Stock Analyzer...**")
        st.markdown("The app will open in a new browser tab.")
        st.code("streamlit run stock_groq.py --server.port 8501")

with col2:
    st.markdown("### 📊 AI Stock Screener")
    st.markdown("""
    **Features:**
    - Analyze 50+ stocks automatically
    - Top 5 buy recommendations
    - Auto-refresh every 10 minutes
    - Multi-category screening
    - Scoring & ranking system
    """)
    
    if st.button("🏆 Launch Stock Screener", type="primary", use_container_width=True):
        st.markdown("**Starting AI Stock Screener...**")
        st.markdown("The app will open in a new browser tab.")
        st.code("streamlit run stock_screener.py --server.port 8502")

st.divider()

st.markdown("### 🛠️ Manual Launch Commands")
st.markdown("You can also run these commands directly in your terminal:")

col1, col2 = st.columns(2)
with col1:
    st.code("""
# Individual Stock Analyzer
source .venv/bin/activate
streamlit run stock_groq.py
    """)

with col2:
    st.code("""
# AI Stock Screener  
source .venv/bin/activate
streamlit run stock_screener.py
    """)

st.markdown("### 📋 Requirements")
st.info("""
**Both apps require:**
- ✅ Groq API key in `.env` file
- ✅ All dependencies installed (`pip install -r requirements.txt`)
- ✅ Virtual environment activated
""")

# Check if Groq API key exists
if os.path.exists('.env'):
    with open('.env', 'r') as f:
        env_content = f.read()
        if 'GROQ_API_KEY=' in env_content and 'your_groq_api_key_here' not in env_content:
            st.success("✅ Groq API key configured")
        else:
            st.warning("⚠️ Groq API key not properly configured in .env file")
else:
    st.error("❌ .env file not found")

st.divider()
st.caption("🤖 Powered by Groq AI • Built with Streamlit • Educational purposes only")
