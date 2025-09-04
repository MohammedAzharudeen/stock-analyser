# 📈 Stock Analyst Agent

> AI-powered stock analysis suite with automated screening and detailed individual stock analysis

![Python](https://img.shields.io/badge/python-v3.8+-blue.svg)
![Streamlit](https://img.shields.io/badge/streamlit-v1.28+-red.svg)
![Groq](https://img.shields.io/badge/AI-Groq-green.svg)

---

## 🚨 **EDUCATIONAL USE ONLY - READ DISCLAIMER FIRST**

**⚠️ This tool is for educational purposes only and requires proper stock market knowledge to use responsibly. DO NOT make investment decisions without understanding technical analysis, fundamental analysis, and market risks. See [full disclaimer](#️-important-disclaimer) below.**

---

## 🚀 Features

### 📊 AI Stock Screener
- **Automated Analysis** of 50+ stocks across multiple categories
- **Top 5 Buy Recommendations** with AI-powered scoring (0-100)
- **Auto-refresh** every 10 minutes for fresh insights
- **Multi-category Support**: US Large Cap, Growth, Value, Indian stocks, ETFs
- **Parallel Processing** for fast analysis
- **Smart Scoring Algorithm** combining technical and fundamental analysis

### 📈 Individual Stock Analyzer
- **Detailed Analysis** of specific stocks with AI recommendations
- **Technical Indicators**: RSI, MACD, Moving Averages, Volume Analysis
- **Fundamental Metrics**: P/E, P/B ratios, Market Cap, Dividend Yield
- **AI-Powered Decisions**: BUY/HOLD/SELL with confidence scores
- **Interactive Charts** and real-time data
- **Multiple AI Models** via Groq (LLaMA 3.1/3.2, Mixtral, Gemma)

### 🎛️ Additional Tools
- **Rule-Based Analyzer** (no API key required)
- **App Launcher** for easy navigation between tools
- **Comprehensive Setup Guide**

## 🛠️ Installation

### Prerequisites
- Python 3.8+
- Groq API key (free at [console.groq.com](https://console.groq.com/keys))

### Quick Setup

1. **Clone the repository**
   ```bash
   git clone https://github.com/yourusername/Stock_Analyser_Agent.git
   cd Stock_Analyser_Agent
   ```

2. **Create virtual environment**
   ```bash
   python -m venv .venv
   source .venv/bin/activate  # On Windows: .venv\Scripts\activate
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

4. **Set up environment variables**
   ```bash
   # Create .env file
   echo 'GROQ_API_KEY=your_groq_api_key_here' > .env
   ```
   
   Get your free Groq API key from: https://console.groq.com/keys

## 🚀 Usage

### Launch Applications

#### Option 1: Individual Commands
```bash
# AI Stock Screener (Top 5 recommendations)
source .venv/bin/activate && streamlit run stock_screener.py --server.port 8504

# Individual Stock Analyzer (Detailed analysis)
source .venv/bin/activate && streamlit run stock_groq.py --server.port 8503

# App Launcher (Choose between apps)
source .venv/bin/activate && streamlit run launcher.py --server.port 8501
```

#### Option 2: Run Both Simultaneously
```bash
# Terminal 1 - Stock Screener
source .venv/bin/activate && streamlit run stock_screener.py --server.port 8504

# Terminal 2 - Individual Analyzer  
source .venv/bin/activate && streamlit run stock_groq.py --server.port 8503
```

### Default Ports
- **Stock Screener**: Port 8504
- **Individual Analyzer**: Port 8503  
- **App Launcher**: Port 8501

## 📊 How It Works

### Stock Screener Algorithm

**Technical Analysis (60% weight):**
- **Trend Analysis**: Price vs 50/200-day moving averages
- **Momentum**: RSI indicators for overbought/oversold conditions
- **MACD Signals**: Bullish/bearish crossovers
- **Volume**: Confirmation through volume analysis

**Fundamental Analysis (40% weight):**
- **Valuation**: P/E ratio assessment
- **Income**: Dividend yield evaluation  
- **Stability**: Market cap considerations

**Scoring System:**
- 80-100: 🟢 Strong Buy
- 60-79: 🟡 Moderate Buy
- 0-59: 🔴 Weak/Hold

### Supported Stock Categories

| Category | Examples | Count |
|----------|----------|-------|
| **US Large Cap** | AAPL, MSFT, GOOGL, AMZN | 10 |
| **US Growth** | TSLA, NVDA, AMD, CRM | 10 |
| **US Value** | BRK-B, JPM, BAC, XOM | 10 |
| **Indian Large Cap** | RELIANCE.NS, TCS.NS, INFY.NS | 10 |
| **Crypto-Related** | COIN, MSTR, RIOT | 6 |
| **Popular ETFs** | SPY, QQQ, VOO, VTI | 6 |

## 🎯 Example Usage

### Stock Screener Workflow
1. **Select Categories**: Choose US Large Cap + Growth stocks
2. **Start Analysis**: Click "🔄 Refresh Now"
3. **View Results**: See top 5 ranked recommendations
4. **Enable Auto-refresh**: Get updates every 10 minutes

### Individual Analyzer Workflow
1. **Enter Ticker**: Type stock symbol (e.g., AAPL, TSLA, ITC.NS)
2. **Select Timeframe**: Choose 1y, 6mo, or 3mo analysis
3. **Choose AI Model**: Pick from available Groq models
4. **Get Analysis**: Receive detailed BUY/HOLD/SELL recommendation

## 🔧 Configuration

### Environment Variables
```bash
# Required
GROQ_API_KEY=your_groq_api_key_here

# Optional
GROQ_MODEL=llama-3.1-8b-instant
TEMPERATURE=0.3
```

### Available AI Models
- `llama-3.1-8b-instant` - Fastest (default)
- `llama-3.2-90b-text-preview` - Most capable
- `llama-3.2-11b-text-preview` - Balanced
- `mixtral-8x7b-32768` - Alternative
- `gemma2-9b-it` - Lightweight

## 📁 Project Structure

```
Stock_Analyser_Agent/
├── stock_screener.py      # AI Stock Screener app
├── stock_groq.py          # Individual Stock Analyzer (Groq AI)
├── stock_no_ai.py         # Rule-based analyzer (no API needed)
├── stock.py               # Original analyzer (AutoGen)
├── launcher.py            # App launcher/selector
├── requirements.txt       # Python dependencies
├── .env                   # Environment variables (create this)
├── .gitignore            # Git ignore rules
├── SETUP.md              # Detailed setup instructions
└── README.md             # This file
```

## 🔍 Technical Details

### Dependencies
- **Streamlit**: Web interface framework
- **yfinance**: Real-time stock data
- **pandas/numpy**: Data manipulation
- **ta**: Technical analysis indicators
- **groq**: AI model integration
- **python-dotenv**: Environment management

### Data Sources
- **Stock Data**: Yahoo Finance via yfinance
- **AI Analysis**: Groq API (LLaMA, Mixtral models)
- **Technical Indicators**: TA-Lib compatible library

### Performance
- **Parallel Processing**: Analyzes 10 stocks simultaneously
- **Caching**: Session state management for efficiency
- **Auto-refresh**: Configurable update intervals
- **Error Handling**: Graceful fallbacks for missing data

## 🚨 Troubleshooting

### Common Issues

**"No module named 'groq'"**
```bash
source .venv/bin/activate
pip install groq
```

**"Port already in use"**
```bash
# Use different port
streamlit run stock_screener.py --server.port 8505
```

**"GROQ_API_KEY not found"**
```bash
# Check .env file exists and contains your API key
cat .env
echo 'GROQ_API_KEY=your_actual_key' > .env
```

**"No price data found"**
- Verify ticker symbol format (e.g., use `ITC.NS` for Indian stocks)
- Check if markets are open
- Try different ticker symbols

### Performance Tips
- Enable auto-refresh during market hours only
- Select fewer stock categories for faster analysis
- Use `llama-3.1-8b-instant` model for speed
- Close unused browser tabs to free memory

## 📈 Example Output

### Stock Screener Results
```
🏆 Top 5 Buy Recommendations

#1 🟢 AAPL - Score: 87/100
Price: $174.50
Why Buy:
• Above 200-day MA (strong trend)
• Healthy RSI (58.2)
• Reasonable P/E (28.5)

#2 🟢 MSFT - Score: 84/100  
Price: $412.30
Why Buy:
• Above 50-day MA (short-term bullish)
• Good dividend yield (2.1%)
• Large cap stability
```

### Individual Analysis
```json
{
  "action": "BUY",
  "confidence": 78,
  "technical_summary": "Strong uptrend with price above key MAs",
  "fundamental_summary": "Reasonable valuation with solid fundamentals",
  "risks": ["Market volatility", "Sector rotation risk"],
  "notes": "Good entry point for long-term investors"
}
```

## 🔄 Git Setup

### Initialize Repository
```bash
# Initialize git repository
git init
git add .
git commit -m "Initial commit: Stock Analyst Agent"

# Add remote repository (replace with your GitHub repo)
git remote add origin https://github.com/yourusername/Stock_Analyser_Agent.git
git branch -M main
git push -u origin main
```

### Important Files
- `.env` - Contains API keys (already in .gitignore)
- `.venv/` - Virtual environment (already in .gitignore)  
- `__pycache__/` - Python cache files (already in .gitignore)

## 🤝 Contributing

1. Fork the repository
2. Create feature branch (`git checkout -b feature/amazing-feature`)
3. Commit changes (`git commit -m 'Add amazing feature'`)
4. Push to branch (`git push origin feature/amazing-feature`)
5. Open Pull Request

## ⚠️ Important Disclaimer

### 🎓 Educational Purpose Only

This Stock Analyst Agent is designed **strictly for educational and learning purposes**. It is NOT intended for actual investment decisions.

### 📚 Prerequisites Required

**DO NOT USE this tool without proper stock market knowledge:**

- ✅ **Understand Technical Analysis** - RSI, MACD, Moving Averages
- ✅ **Know Fundamental Analysis** - P/E ratios, Market Cap, Financial Statements  
- ✅ **Grasp Market Risks** - Volatility, Economic factors, Company-specific risks
- ✅ **Have Investment Experience** - Portfolio management, Risk assessment
- ✅ **Understand AI Limitations** - Models can be wrong, Past performance ≠ Future results

### 🚨 Critical Warnings

**NEVER make investment decisions based solely on this tool:**

- 🚫 **Not Financial Advice** - This is educational software, not professional guidance
- 🚫 **No Guarantees** - AI predictions can be completely wrong
- 🚫 **Market Volatility** - Stocks can lose value rapidly and unpredictably
- 🚫 **Data Limitations** - Analysis based on historical data only
- 🚫 **No Liability** - Authors not responsible for any financial losses

### ✅ Responsible Usage

**If you choose to invest (at your own risk):**

- 📖 **Educate Yourself First** - Learn stock analysis fundamentals
- 💼 **Consult Professionals** - Seek advice from licensed financial advisors
- 🔍 **Do Independent Research** - Verify all information from multiple sources
- 💰 **Start Small** - Only invest money you can afford to lose completely
- 📊 **Diversify** - Never put all money in one stock or sector
- ⏰ **Long-term Perspective** - Avoid day trading without expertise
- 📋 **Keep Records** - Track all investments and tax implications

### 🎯 Intended Learning Outcomes

Use this tool to:
- Learn how technical indicators work
- Understand fundamental analysis concepts
- Practice reading financial data
- Explore AI applications in finance
- Build programming and data analysis skills

### 🔒 No Personal Data Collection

This tool processes only public stock market data and does not collect personal financial information.


## 🙏 Acknowledgments

- **Groq** for providing fast, free AI inference
- **Yahoo Finance** for reliable stock data
- **Streamlit** for the amazing web framework
- **TA-Lib** community for technical analysis tools

## 📞 Support

- **Issues**: Open a GitHub issue
- **Groq API**: [Groq Documentation](https://console.groq.com/docs)
- **Streamlit**: [Streamlit Documentation](https://docs.streamlit.io)

---

**⭐ If you find this project helpful, please give it a star!**

Built with ❤️ using Python, Streamlit, and Groq AI
