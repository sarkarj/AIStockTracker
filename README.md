# 💫 AI Stock Tracker

A sophisticated real-time stock analysis dashboard powered by dual AI models, providing comprehensive technical analysis and market insights through an elegant web interface.

![Stock Tracker Demo](https://img.shields.io/badge/Status-Active-brightgreen) ![Python](https://img.shields.io/badge/Python-3.8+-blue) ![Streamlit](https://img.shields.io/badge/Streamlit-Latest-red) ![Docker](https://img.shields.io/badge/Docker-Required-blue)

## 🚀 Features

### 📊 Real-Time Stock Analysis
- **Live Market Data**: Fetches real-time stock prices using Yahoo Finance API
- **Technical Indicators**: Advanced calculations including RSI, MACD, EMA, and Bollinger Bands
- **Interactive Charts**: Beautiful Plotly visualizations with technical overlays
- **Price Tracking**: Real-time price changes with visual trend indicators

### 🤖 Dual AI-Powered Insights
- **Local LLM**: Llama 3.2 1B model running on [Docker Model Runner](https://hub.docker.com/r/ai/llama3.2)
- **Google Gemini 2.0**: Cloud-based AI for comparative analysis
- **Smart Caching**: Intelligent caching system to optimize API calls
- **Recommendation Engine**: Buy/Sell/Hold recommendations from both models

### 🎨 Modern UI/UX
- **Responsive Design**: Optimized for desktop and mobile devices
- **Real-time Updates**: Live refresh capabilities with loading states

### ⚡ Performance Optimized
- **Efficient Caching**: SHA256-based caching for AI predictions
- **Batch Processing**: Simultaneous analysis of multiple stocks
- **Error Handling**: Robust retry mechanisms and fallback systems
- **Response Time Tracking**: Performance metrics for both AI models

## 🛠️ Technology Stack

### Backend
- **Python 3.9+**: Core application language
- **Streamlit**: Web framework for rapid dashboard development
- **yfinance**: Yahoo Finance API for market data
- **pandas**: Data manipulation and analysis
- **plotly**: Interactive charting library

### AI Models
- **Local LLM**: Llama 3.2 1B (Q4_0 Quantized) via Docker Model Runner
- **Google Gemini 2.0**: Cloud-based generative AI model
- **Caching**: Shelve-based persistent storage for predictions

### Infrastructure
- **Docker**: Containerized local LLM deployment
- **REST APIs**: HTTP-based model communication
- **Environment Variables**: Secure configuration management

## 📦 Installation & Setup

### Prerequisites
- Python 3.9 or higher
- Docker Desktop
- Git

### 1. Clone the Repository
```bash
git clone https://github.com/sarkarj/AIStockTracker.git
cd AIStockTracker
```

### 2. Install Python Dependencies
```bash
pip install -r requirements.txt
```

### 3. Set Up Docker Model Runner
```bash
# Pull and run the Docker Model Runner
docker model pull ai/llama3.2:1B-Q4_0
docker desktop enable model-runner --tcp 12434

# Verify the model is running
docker model list
```

Expected output:
```
MODEL NAME            PARAMETERS  QUANTIZATION  ARCHITECTURE  MODEL ID      CREATED      SIZE       
ai/llama3.2:1B-Q4_0   1.24 B      Q4_0          llama         c682bd9c5f3d  4 weeks ago  727.75 MiB
```

### 4. Configure Environment Variables
Create a `.env` file in the project root:

```env
# Cache directory for AI predictions
CACHE_PATH=./insight_cache

# Local LLM Configuration
LLM_MODEL=ai/llama3.2:1B-Q4_0
LLM_BASE_URL=http://host.docker.internal:12434/engines/llama.cpp/v1/chat/completions

# Google Gemini API Key (get from Google AI Studio)
GEMINI_API_KEY=your_gemini_api_key_here
```

### 5. Create Cache Directory
```bash
mkdir insight_cache
```

## 🚀 Running the Application

### Start the Streamlit Dashboard
```bash
streamlit run stock_tracker.py
```

The application will be available at `http://localhost:8501`

### Default Stock Symbols
The app comes pre-configured with popular stocks:
- **AAPL** (Apple Inc.)
- **NVDA** (NVIDIA Corporation)
- **TSLA** (Tesla, Inc.)
- **GOOG** (Alphabet Inc.)
- **MSFT** (Microsoft Corporation)
- **COST** (Costco Wholesale Corporation)
- **META** (Meta Platforms, Inc.)
- **VOO** (Vanguard S&P 500 ETF)
- **SPY** (SPDR S&P 500 ETF Trust)

## 🔧 Configuration

### Local LLM Setup (Docker Model Runner)

The application uses Docker Model Runner to host Llama 3.2 locally:

```bash
# List available models
docker model list

# Access model API
curl http://localhost:12434/engines/llama.cpp/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "ai/llama3.2:1B-Q4_0",
    "messages": [{"role": "user", "content": "Hello"}]
  }'
```

### Google Gemini API Setup

1. Visit [Google AI Studio](https://aistudio.google.com/)
2. Create a new API key
3. Add the key to your `.env` file as `GEMINI_API_KEY`

### Cache Configuration

The application uses persistent caching to optimize performance:
- Cache location: `./insight_cache/`
- Cache key: SHA256 hash of symbol + indicators
- Storage: Python's `shelve` module for persistence

## 📊 Technical Indicators

### Implemented Indicators
- **RSI (Relative Strength Index)**: Momentum oscillator (0-100)
- **MACD (Moving Average Convergence Divergence)**: Trend-following momentum indicator
- **EMA (Exponential Moving Average)**: 20-period exponential moving average
- **Bollinger Bands**: Statistical analysis of price volatility

### Calculation Methods
```python
# EMA Calculation
df['EMA'] = df['Close'].ewm(span=20).mean()

# RSI Calculation
rsi = 100 - (100 / (1 + (gains.sum() / losses.sum())))

# MACD Calculation
short_ema = df['Close'].ewm(span=12).mean()
long_ema = df['Close'].ewm(span=26).mean()
macd = short_ema - long_ema
signal = macd.ewm(span=9).mean()

# Bollinger Bands
upper_band = 20_MA + (2 * 20_STD)
lower_band = 20_MA - (2 * 20_STD)
```

## 🤖 AI Model Integration

### Local LLM (Llama 3.2)
- **Model**: ai/llama3.2:1B-Q4_0
- **Size**: 727.75 MiB
- **Response Time**: Typically 1-3 seconds
- **Advantages**: Privacy, no API costs, offline capability

### Google Gemini 2.0
- **Model**: gemini-2.0-flash
- **API**: REST-based Google AI API
- **Response Time**: Typically 2-5 seconds
- **Advantages**: Latest AI technology, high accuracy

### Prediction Format
Both models provide:
- Market sentiment (Bullish/Bearish/Neutral)
- Action recommendation (Buy/Sell/Hold)
- Technical analysis reasoning
- Price movement predictions

## 🎨 UI Components

### Stock Cards
Each stock is displayed in a sophisticated card featuring:
- **Header**: Symbol, status badge, current price with trend arrow
- **Chart Section**: Interactive price chart with technical indicators
- **Indicators Panel**: RSI with progress bar, EMA, MACD values
- **AI Insights**: Dual predictions with recommendations and timing

### Visual Elements
- **Color Coding**: Green (bullish), Red (bearish), Orange (neutral)
- **Responsive Grid**: 2-column layout on desktop, single column on mobile

## 📈 Performance Metrics

### Caching Strategy
- **Cache Hit Rate**: ~80-90% for repeated requests
- **Storage Efficiency**: Persistent shelve database
- **Key Generation**: SHA256 hash for unique identification

### Response Times
- **Data Fetching**: 1-2 seconds per stock (Yahoo Finance)
- **Technical Analysis**: <1 second per stock
- **Local LLM**: 1-3 seconds per prediction
- **Gemini API**: 2-5 seconds per prediction

## 🔒 Security & Privacy

### Data Protection
- **Local Processing**: Technical analysis computed locally
- **API Security**: Environment variables for sensitive keys
- **No Data Storage**: Stock prices not permanently stored
- **Cache Isolation**: Predictions cached by unique hash

### Privacy Benefits of Local LLM
- **No Data Transmission**: Analysis stays on your machine
- **Offline Capability**: Works without internet for predictions
- **Cost Efficiency**: No per-request API charges

## 🐛 Troubleshooting

### Common Issues

#### Docker Model Runner Not Responding
```bash
# List models available locally
docker model list

# Check if the model runner is running
docker model status

Docker Model Runner is running
Status:
```

#### Gemini API Errors
- Verify API key is correct in `.env`
- Check API quotas in Google AI Studio
- Ensure internet connectivity

#### Yahoo Finance Data Issues
- Some symbols might be delisted or unavailable
- Try alternative ticker symbols
- Check market hours (data might be delayed)

#### Cache Issues
```bash
# Clear cache directory
rm -rf ./insight_cache/*

# Restart application
```

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🙏 Acknowledgments

- **Yahoo Finance** for providing free market data
- **Google AI** for Gemini API access
- **Docker Model Runner** for local LLM hosting
- **Streamlit** for the amazing web framework
- **Plotly** for interactive visualizations
- **Claude AI** for application coding 

---

**⚠️ Disclaimer**: This tool is for educational and informational purposes only. It does not constitute financial advice. Always consult with qualified financial advisors before making investment decisions. Past performance does not guarantee future results.
