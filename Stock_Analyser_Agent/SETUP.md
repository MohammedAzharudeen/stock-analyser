# 🚀 Stock Analyst Agent - Setup Guide

## 📋 Prerequisites
- Python 3.8+
- OpenAI API account and API key

## 🔧 Environment Setup

### 1. Create Environment File
Create a `.env` file in the project root directory:

```bash
# Method 1: Create manually
touch .env

# Method 2: Use the template
cp env_template.txt .env
```

### 2. Add Your OpenAI API Key
Edit the `.env` file and add your OpenAI API key:

```bash
# Required: Your OpenAI API Key
OPENAI_API_KEY=sk-proj-your-actual-api-key-here

# Optional configurations (uncomment to use)
# OPENAI_MODEL=gpt-4o-mini
# OPENAI_TEMPERATURE=0.3
```

**⚠️ Important**: 
- Get your API key from: https://platform.openai.com/api-keys
- Never commit your `.env` file to version control
- Your API key should start with `sk-proj-` or `sk-`

### 3. Install Dependencies
```bash
pip install streamlit yfinance pandas numpy python-dotenv ta autogen-agentchat autogen-ext
```

### 4. Run the Application
```bash
streamlit run stock.py
```

## 🔐 Security Notes
- The `.env` file is automatically ignored by git (see `.gitignore`)
- Never share your API key publicly
- Monitor your OpenAI API usage to avoid unexpected charges

## 🎯 Usage
1. Enter a stock ticker symbol (e.g., AAPL, MSFT, TSLA)
2. Select analysis timeframe (1y, 6mo, 3mo)
3. Click "Analyze" to get AI-powered investment recommendations

## 🆘 Troubleshooting
- **"Set OPENAI_API_KEY" warning**: Make sure your `.env` file exists and contains a valid API key
- **No data found**: Try a different ticker symbol or check if markets are open
- **API errors**: Check your OpenAI account credits and API key validity
