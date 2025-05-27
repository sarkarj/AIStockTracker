import streamlit as st
import pandas as pd
import plotly.graph_objs as go
import yfinance as yf
import requests
import time
import json
import os
import hashlib
import shelve
from dotenv import load_dotenv

# --- LOAD CONFIG ---
load_dotenv()
LLM_API = os.getenv("LLM_BASE_URL")
LLM_MODEL = os.getenv("LLM_MODEL")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
CACHE_PATH = os.getenv("CACHE_PATH")
os.makedirs(CACHE_PATH, exist_ok=True)

# --- STREAMLIT SETUP ---
st.set_page_config(page_title="Stock Insights AI", layout="wide")

# Enhanced CSS for better UI/UX
st.markdown("""
<style>
    /* Reset and base styles */
    .main > div {
        padding-top: 1rem !important;
    }
    
    /* Compact header */
    .compact-header {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        border-radius: 12px;
        padding: 1rem 1.5rem;
        margin-bottom: 1.5rem;
        display: flex;
        justify-content: space-between;
        align-items: center;
        box-shadow: 0 4px 15px rgba(102, 126, 234, 0.3);
    }
    
    .header-content {
        display: flex;
        align-items: center;
        gap: 1rem;
    }
    
    .header-title {
        font-size: 1.5rem;
        font-weight: 700;
        margin: 0;
    }
    
    .header-subtitle {
        font-size: 0.9rem;
        opacity: 0.9;
        margin: 0;
        font-weight: 400;
    }
    
    .header-actions {
        display: flex;
        align-items: center;
        gap: 1rem;
    }
    
    /* Compact controls */
    .controls-section {
        background: linear-gradient(135deg, #f8f9fa 0%, #e9ecef 100%);
        padding: 1rem 1.5rem;
        border-radius: 12px;
        margin-bottom: 1.5rem;
        border: 1px solid #dee2e6;
        box-shadow: 0 2px 8px rgba(0,0,0,0.05);
    }
    
    .controls-grid {
        display: grid;
        grid-template-columns: 1fr auto;
        gap: 1rem;
        align-items: center;
    }
    
    /* Stock cards */
    .stock-card {
        background: linear-gradient(135deg, #ffffff 0%, #f8f9fa 100%);
        padding: 1.5rem;
        border-radius: 15px;
        box-shadow: 0 4px 20px rgba(0,0,0,0.08);
        border: 1px solid #e9ecef;
        margin-bottom: 1.5rem;
        height: 100%;
        transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
        position: relative;
        overflow: hidden;
    }
    
    .stock-card::before {
        content: '';
        position: absolute;
        top: 0;
        left: 0;
        right: 0;
        height: 3px;
        background: linear-gradient(90deg, #667eea, #764ba2);
        transform: scaleX(0);
        transition: transform 0.3s ease;
    }
    
    .stock-card:hover {
        transform: translateY(-4px);
        box-shadow: 0 12px 30px rgba(0,0,0,0.15);
    }
    
    .stock-card:hover::before {
        transform: scaleX(1);
    }
    
    .stock-header {
        display: flex;
        justify-content: space-between;
        align-items: center;
        margin-bottom: 1.2rem;
        padding-bottom: 1rem;
        border-bottom: 2px solid #e9ecef;
    }
    
    .stock-symbol {
        font-size: 1.8rem;
        font-weight: 700;
        color: #2c3e50;
        margin-right: 1rem;
    }
    
    .stock-price {
        font-size: 1.4rem;
        font-weight: 600;
        display: flex;
        align-items: center;
        gap: 0.5rem;
    }
    
    .price-positive {
        color: #27ae60;
    }
    
    .price-negative {
        color: #e74c3c;
    }
    
    .price-neutral {
        color: #95a5a6;
    }
    
    .status-badge {
        padding: 0.3rem 0.8rem;
        border-radius: 20px;
        font-size: 0.8rem;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 0.5px;
        animation: pulse 2s infinite;
    }
    
    @keyframes pulse {
        0%, 100% { opacity: 1; }
        50% { opacity: 0.8; }
    }
    
    .status-bullish {
        background: linear-gradient(45deg, #27ae60, #2ecc71);
        color: white;
    }
    
    .status-bearish {
        background: linear-gradient(45deg, #e74c3c, #c0392b);
        color: white;
    }
    
    .status-neutral {
        background: linear-gradient(45deg, #f39c12, #e67e22);
        color: white;
    }
    
    .card-section {
        margin-bottom: 1.5rem;
    }
    
    .section-title {
        font-size: 1rem;
        font-weight: 700;
        color: #34495e;
        margin-bottom: 1rem;
        display: flex;
        align-items: center;
        gap: 0.5rem;
        text-transform: uppercase;
        letter-spacing: 0.5px;
    }
    
    .indicator-grid {
        display: grid;
        grid-template-columns: 1fr 1fr;
        gap: 0.8rem;
        margin-bottom: 1rem;
    }
    
    .indicator-item {
        padding: 1rem;
        background: linear-gradient(135deg, #ffffff 0%, #f8f9fa 100%);
        border-radius: 10px;
        border: 1px solid #e9ecef;
        text-align: center;
        transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
        position: relative;
        overflow: hidden;
    }
    
    .indicator-item::before {
        content: '';
        position: absolute;
        top: 0;
        left: -100%;
        width: 100%;
        height: 100%;
        background: linear-gradient(90deg, transparent, rgba(255,255,255,0.3), transparent);
        transition: left 0.5s;
    }
    
    .indicator-item:hover {
        border-color: #3498db;
        transform: scale(1.05);
        box-shadow: 0 4px 15px rgba(52, 152, 219, 0.2);
    }
    
    .indicator-item:hover::before {
        left: 100%;
    }
    
    .indicator-label {
        font-weight: 600;
        color: #7f8c8d;
        font-size: 0.8rem;
        margin-bottom: 0.5rem;
        text-transform: uppercase;
        letter-spacing: 0.5px;
    }
    
    .indicator-value {
        font-weight: 700;
        font-size: 1.1rem;
        color: #2c3e50;
    }
    
    .rsi-value {
        color: #e74c3c;
    }
    
    .rsi-value.rsi-oversold {
        color: #27ae60;
    }
    
    .rsi-value.rsi-overbought {
        color: #e74c3c;
    }
    
    .rsi-value.rsi-neutral {
        color: #f39c12;
    }
    
    .macd-positive {
        color: #27ae60;
    }
    
    .macd-negative {
        color: #e74c3c;
    }
    
    .rsi-progress {
        width: 100%;
        height: 6px;
        background: #ecf0f1;
        border-radius: 3px;
        margin-top: 0.5rem;
        overflow: hidden;
    }
    
    .rsi-bar {
        height: 100%;
        border-radius: 3px;
        transition: width 0.8s cubic-bezier(0.4, 0, 0.2, 1);
        animation: slideIn 1s ease-out;
    }
    
    @keyframes slideIn {
        0% { width: 0; }
    }
    
    .prediction-container {
        margin-top: 1.5rem;
        padding-top: 1rem;
        border-top: 2px solid #e9ecef;
    }
    
    .prediction-grid {
        display: grid;
        grid-template-columns: 1fr 1fr;
        gap: 1rem;
    }
    
    .prediction-box {
        background: linear-gradient(135deg, #ffffff 0%, #f8f9fa 100%);
        padding: 1rem;
        border-radius: 12px;
        border: 2px solid transparent;
        position: relative;
        transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
        overflow: hidden;
    }
    
    .prediction-box::after {
        content: '';
        position: absolute;
        top: 0;
        left: 0;
        right: 0;
        bottom: 0;
        border-radius: 12px;
        padding: 2px;
        background: linear-gradient(45deg, #3498db, #2980b9);
        mask: linear-gradient(#fff 0 0) content-box, linear-gradient(#fff 0 0);
        mask-composite: xor;
        -webkit-mask-composite: xor;
        opacity: 0;
        transition: opacity 0.3s ease;
    }
    
    .prediction-box.llm-box::after {
        background: linear-gradient(45deg, #3498db, #2980b9);
    }
    
    .prediction-box.gemini-box::after {
        background: linear-gradient(45deg, #9b59b6, #8e44ad);
    }
    
    .prediction-box:hover {
        transform: translateY(-3px);
        box-shadow: 0 8px 25px rgba(0,0,0,0.15);
    }
    
    .prediction-box:hover::after {
        opacity: 1;
    }
    
    .prediction-header {
        font-weight: 700;
        font-size: 0.9rem;
        margin-bottom: 0.8rem;
        color: #2c3e50;
        display: flex;
        align-items: center;
        gap: 0.5rem;
    }
    
    .prediction-content {
        font-size: 0.85rem;
        line-height: 1.5;
        color: #5a6c7d;
        margin-bottom: 0.8rem;
    }
    
    .prediction-recommendation {
        display: inline-block;
        padding: 0.3rem 0.8rem;
        border-radius: 15px;
        font-size: 0.75rem;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 0.5px;
        margin-bottom: 0.5rem;
        animation: glow 2s infinite alternate;
    }
    
    @keyframes glow {
        0% { box-shadow: 0 0 5px rgba(0,0,0,0.1); }
        100% { box-shadow: 0 0 15px rgba(0,0,0,0.2); }
    }
    
    .rec-buy {
        background: linear-gradient(45deg, #27ae60, #2ecc71);
        color: white;
    }
    
    .rec-sell {
        background: linear-gradient(45deg, #e74c3c, #c0392b);
        color: white;
    }
    
    .rec-hold {
        background: linear-gradient(45deg, #f39c12, #e67e22);
        color: white;
    }
    
    .timing-info {
        font-size: 0.7rem;
        color: #95a5a6;
        text-align: right;
        display: flex;
        align-items: center;
        justify-content: flex-end;
        gap: 0.3rem;
    }
    
    .loading-skeleton {
        background: linear-gradient(90deg, #f0f0f0 25%, #e0e0e0 50%, #f0f0f0 75%);
        background-size: 200% 100%;
        animation: loading 1.5s infinite;
        border-radius: 10px;
        height: 20px;
        margin: 0.5rem 0;
    }
    
    @keyframes loading {
        0% { background-position: 200% 0; }
        100% { background-position: -200% 0; }
    }
    
    .trend-arrow {
        font-size: 1.2rem;
        margin-left: 0.5rem;
        animation: bounce 2s infinite;
    }
    
    @keyframes bounce {
        0%, 20%, 50%, 80%, 100% { transform: translateY(0); }
        40% { transform: translateY(-3px); }
        60% { transform: translateY(-2px); }
    }
    
    .trend-up {
        color: #27ae60;
    }
    
    .trend-down {
        color: #e74c3c;
    }
    
    .chart-container {
        border-radius: 10px;
        overflow: hidden;
        background: white;
        box-shadow: 0 2px 10px rgba(0,0,0,0.05);
        transition: box-shadow 0.3s ease;
    }
    
    .chart-container:hover {
        box-shadow: 0 4px 20px rgba(0,0,0,0.1);
    }
    
    /* Responsive design */
    @media (max-width: 768px) {
        .compact-header {
            flex-direction: column;
            gap: 1rem;
            text-align: center;
        }
        
        .controls-grid {
            grid-template-columns: 1fr;
            gap: 1rem;
        }
        
        .prediction-grid {
            grid-template-columns: 1fr;
        }
        
        .indicator-grid {
            grid-template-columns: 1fr;
        }
        
        .stock-header {
            flex-direction: column;
            align-items: flex-start;
            gap: 1rem;
        }
    }
    
    /* Custom button styling */
    .stButton > button {
        background: linear-gradient(45deg, #667eea, #764ba2) !important;
        color: white !important;
        border: none !important;
        border-radius: 8px !important;
        padding: 0.5rem 1.5rem !important;
        font-weight: 600 !important;
        transition: all 0.3s ease !important;
        box-shadow: 0 2px 10px rgba(102, 126, 234, 0.3) !important;
    }
    
    .stButton > button:hover {
        transform: translateY(-2px) !important;
        box-shadow: 0 4px 15px rgba(102, 126, 234, 0.4) !important;
    }
    
    /* Input styling */
    .stTextInput > div > div > input {
        border-radius: 8px !important;
        border: 2px solid #e9ecef !important;
        transition: border-color 0.3s ease !important;
    }
    
    .stTextInput > div > div > input:focus {
        border-color: #667eea !important;
        box-shadow: 0 0 0 0.2rem rgba(102, 126, 234, 0.25) !important;
    }
    
    /* Loading spinner enhancement */
    .stSpinner {
        text-align: center;
        padding: 2rem;
    }
    
    /* Status indicators */
    .status-indicator {
        display: inline-flex;
        align-items: center;
        gap: 0.5rem;
        padding: 0.3rem 0.8rem;
        border-radius: 20px;
        font-size: 0.8rem;
        font-weight: 600;
    }
    
    /* Smooth scrolling */
    html {
        scroll-behavior: smooth;
    }
</style>
""", unsafe_allow_html=True)

# --- COMPACT HEADER SECTION ---
st.markdown("""
<div class="compact-header">
    <div class="header-content">
        <div>
            <div class="header-title">💫 AI Stock Tracker</div>
            <div class="header-subtitle">Real-time insights powered by AI models</div>
        </div>
    </div>
    <div class="header-actions">
        <div style="color: rgba(255,255,255,0.8); font-size: 0.8rem;">
            Live Market Analysis
        </div>
    </div>
</div>
""", unsafe_allow_html=True)

# --- COMPACT CONTROLS SECTION ---
col1, col2 = st.columns([5, 1])
with col1:
    stock_symbols = st.text_input(
        "🔍 Enter stock symbols (comma separated):", 
        "AAPL,NVDA,TSLA,GOOG,MSFT,COST,META,VOO,SPY", 
        key="stocks_input",
        help="Enter stock symbols separated by commas (e.g., AAPL,GOOGL,MSFT)"
    ).split(",")

with col2:
    # Add some spacing to align button with input
    st.markdown("<br>", unsafe_allow_html=True)
    if st.button("⚡ Refresh", type="primary", key="refresh_btn", use_container_width=True):
        st.rerun()

# --- HELPER FUNCTIONS (unchanged) ---
def fetch_historical_data(symbol, period="90d", interval="1d", retries=3, delay=2):
    for i in range(retries):
        try:
            stock = yf.Ticker(symbol)
            data = stock.history(period=period, interval=interval)
            if data.empty:
                raise ValueError("No data returned")
            return data
        except Exception as e:
            print(f"[{symbol}] Retry {i+1}: {e}")
            time.sleep(delay)
    return None

def calculate_indicators(df):
    df['EMA'] = df['Close'].ewm(span=20).mean()
    df['20_MA'] = df['Close'].rolling(20).mean()
    df['20_STD'] = df['Close'].rolling(20).std()
    df['Upper Band'] = df['20_MA'] + 2 * df['20_STD']
    df['Lower Band'] = df['20_MA'] - 2 * df['20_STD']
    return df

def calculate_all(stock):
    data = fetch_historical_data(stock)
    if data is None or data.empty or len(data) < 30:
        return None, None, None
    df = calculate_indicators(data)

    ema = df['EMA'].iloc[-1]
    rsi = 100 - (100 / (1 + (df['Close'].diff().gt(0).sum() / max(df['Close'].diff().lt(0).sum(), 1))))
    lower_band = df['Lower Band'].iloc[-1]
    upper_band = df['Upper Band'].iloc[-1]
    close_price = df['Close'].iloc[-1]
    prev_close = df['Close'].iloc[-2]

    short_ema = df['Close'].ewm(span=12).mean()
    long_ema = df['Close'].ewm(span=26).mean()
    macd = short_ema - long_ema
    signal = macd.ewm(span=9).mean()

    # Calculate price change
    price_change = ((close_price - prev_close) / prev_close) * 100

    indicators = {
        "EMA": round(float(ema), 2),
        "RSI": round(float(rsi), 2),
        "Bollinger Bands": (round(float(lower_band), 2), round(float(upper_band), 2)),
        "MACD": round(float(macd.iloc[-1]), 2),
        "MACD Signal": round(float(signal.iloc[-1]), 2),
        "Current Price": round(float(close_price), 2),
        "Price Change": round(float(price_change), 2)
    }

    return indicators, df, price_change

def get_cache_key(symbol, indicators):
    key = f"{symbol}-{json.dumps(indicators, sort_keys=True)}"
    return hashlib.sha256(key.encode()).hexdigest()

def format_prompt(symbol, indicators):
    return (
        f"Given the technical indicators for {symbol}, provide a concise market prediction in 2-3 sentences.\n\n"
        f"- EMA: {indicators['EMA']}\n"
        f"- RSI: {indicators['RSI']}\n"
        f"- Bollinger Bands: Lower={indicators['Bollinger Bands'][0]}, Upper={indicators['Bollinger Bands'][1]}\n"
        f"- MACD: {indicators['MACD']}, Signal: {indicators['MACD Signal']}\n"
        f"- Current Price: {indicators['Current Price']}\n"
        f"Return if the stock is bullish, bearish, or neutral, and recommend action (buy/sell/hold)."
    )

def get_llm_prediction(symbol, indicators):
    cache_key = get_cache_key(symbol + "_llama", indicators)
    with shelve.open(os.path.join(CACHE_PATH, "insights")) as db:
        if cache_key in db:
            return db[cache_key]
        prompt = format_prompt(symbol, indicators)
        try:
            response = requests.post(LLM_API, json={
                "model": LLM_MODEL,
                "messages": [
                    {"role": "system", "content": "You are a financial assistant."},
                    {"role": "user", "content": prompt}
                ]
            }, headers={"Content-Type": "application/json"}, timeout=30)
            response.raise_for_status()
            content = response.json()["choices"][0]["message"]["content"]
            db[cache_key] = content.strip()
            return content.strip()
        except Exception as e:
            return f"LLM Error: {e}"

def get_gemini_prediction(symbol, indicators):
    if not GEMINI_API_KEY:
        return "Gemini API key not provided."
    cache_key = get_cache_key(symbol + "_gemini", indicators)
    with shelve.open(os.path.join(CACHE_PATH, "insights")) as db:
        if cache_key in db:
            return db[cache_key]
        prompt = format_prompt(symbol, indicators)
        try:
            response = requests.post(
                f"https://generativelanguage.googleapis.com/v1beta/models/gemini-2.0-flash:generateContent?key={GEMINI_API_KEY}",
                headers={"Content-Type": "application/json"},
                json={"contents": [{"parts": [{"text": prompt}]}]},
                timeout=30
            )
            response.raise_for_status()
            content = response.json()["candidates"][0]["content"]["parts"][0]["text"]
            db[cache_key] = content.strip()
            return content.strip()
        except Exception as e:
            return f"Gemini Error: {e}"

def get_status_from_indicators(indicators):
    """Determine bullish/bearish/neutral status from indicators"""
    rsi = indicators['RSI']
    macd = indicators['MACD']
    price = indicators['Current Price']
    bb_lower, bb_upper = indicators['Bollinger Bands']
    
    bullish_signals = 0
    bearish_signals = 0
    
    # RSI analysis
    if rsi < 30:
        bullish_signals += 1  # Oversold
    elif rsi > 70:
        bearish_signals += 1  # Overbought
    
    # MACD analysis
    if macd > indicators['MACD Signal']:
        bullish_signals += 1
    else:
        bearish_signals += 1
    
    # Bollinger Bands analysis
    if price < bb_lower:
        bullish_signals += 1  # Near lower band
    elif price > bb_upper:
        bearish_signals += 1  # Near upper band
    
    if bullish_signals > bearish_signals:
        return "bullish"
    elif bearish_signals > bullish_signals:
        return "bearish"
    else:
        return "neutral"

def extract_recommendation(prediction_text):
    """Extract buy/sell/hold recommendation from AI prediction"""
    text_lower = prediction_text.lower()
    if "buy" in text_lower and "sell" not in text_lower:
        return "buy"
    elif "sell" in text_lower and "buy" not in text_lower:
        return "sell"
    else:
        return "hold"

# --- ENHANCED CARD COMPONENT ---
def create_stock_card(stock, indicators, data, llm_prediction, gemini_prediction, llm_time, gemini_time, price_change):
    """Create a professional stock card with enhanced visual elements"""
    
    # Determine status and styling
    status = get_status_from_indicators(indicators)
    price_class = "price-positive" if price_change > 0 else "price-negative" if price_change < 0 else "price-neutral"
    trend_arrow = "↗️" if price_change > 0 else "↘️" if price_change < 0 else "➡️"
    
    card_html = f"""
    <div class="stock-card">
        <div class="stock-header">
            <div style="display: flex; align-items: center;">
                <div class="stock-symbol">{stock}</div>
                <div class="status-badge status-{status}">{status}</div>
            </div>
            <div class="stock-price {price_class}">
                ${indicators['Current Price']}
                <span class="trend-arrow">{trend_arrow}</span>
                <small>({price_change:+.2f}%)</small>
            </div>
        </div>
    """
    
    st.markdown(card_html, unsafe_allow_html=True)
    
    # Create two columns: Chart and Indicators
    chart_col, indicators_col = st.columns([3, 2])
    
    # Chart section
    with chart_col:
        st.markdown('<div class="section-title">📊 Price Chart</div>', unsafe_allow_html=True)
        st.markdown('<div class="chart-container">', unsafe_allow_html=True)
        
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=data.index, y=data['Close'], name="Close", 
                                line=dict(color="#3498db", width=2)))
        fig.add_trace(go.Scatter(x=data.index, y=data['EMA'], name="EMA", 
                                line=dict(color="#e74c3c", width=2)))
        fig.add_trace(go.Scatter(x=data.index, y=data['Upper Band'], name="Upper Band", 
                                line=dict(dash="dot", color="#95a5a6", width=1)))
        fig.add_trace(go.Scatter(x=data.index, y=data['Lower Band'], name="Lower Band", 
                                line=dict(dash="dot", color="#95a5a6", width=1)))

        fig.update_layout(
            height=280,
            margin=dict(l=10, r=10, t=10, b=10),
            showlegend=True,
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
            xaxis=dict(showgrid=False),
            yaxis=dict(showgrid=True, gridcolor="rgba(128,128,128,0.2)"),
            plot_bgcolor="rgba(0,0,0,0)",
            paper_bgcolor="rgba(0,0,0,0)"
        )
        st.plotly_chart(fig, use_container_width=True, config={'displayModeBar': False})
        st.markdown('</div>', unsafe_allow_html=True)
    
    # Technical indicators section
    with indicators_col:
        st.markdown('<div class="section-title">📈 Technical Indicators</div>', unsafe_allow_html=True)
        
        # RSI with color coding and progress bar
        rsi = indicators['RSI']
        rsi_class = "rsi-oversold" if rsi < 30 else "rsi-overbought" if rsi > 70 else "rsi-neutral"
        rsi_color = "#27ae60" if rsi < 30 else "#e74c3c" if rsi > 70 else "#f39c12"
        
        # MACD with color coding
        macd = indicators['MACD']
        macd_class = "macd-positive" if macd > 0 else "macd-negative"
        
        indicators_html = f"""
        <div class="indicator-grid">
            <div class="indicator-item">
                <div class="indicator-label">RSI</div>
                <div class="indicator-value rsi-value {rsi_class}">{rsi}</div>
                <div class="rsi-progress">
                    <div class="rsi-bar" style="width: {rsi}%; background-color: {rsi_color};"></div>
                </div>
            </div>
            <div class="indicator-item">
                <div class="indicator-label">EMA (20)</div>
                <div class="indicator-value">${indicators['EMA']}</div>
            </div>
            <div class="indicator-item">
                <div class="indicator-label">MACD</div>
                <div class="indicator-value {macd_class}">{indicators['MACD']}</div>
            </div>
            <div class="indicator-item">
                <div class="indicator-label">Signal</div>
                <div class="indicator-value">{indicators['MACD Signal']}</div>
            </div>
        </div>
        <div class="indicator-item">
            <div class="indicator-label">Bollinger Bands</div>
            <div class="indicator-value">${indicators['Bollinger Bands'][0]} - ${indicators['Bollinger Bands'][1]}</div>
        </div>
        """
        st.markdown(indicators_html, unsafe_allow_html=True)
    
    # AI Predictions section (full width under chart and indicators)
    st.markdown('<div class="prediction-container">', unsafe_allow_html=True)
    st.markdown('<div class="section-title">✨ AI Market Insights</div>', unsafe_allow_html=True)
    
    # Extract recommendations from predictions
    llm_rec = extract_recommendation(llm_prediction)
    gemini_rec = extract_recommendation(gemini_prediction)
    
    predictions_html = f"""
    <div class="prediction-grid">
        <div class="prediction-box llm-box">
            <div class="prediction-header">🐳 Local LLM (Llama 3.2)</div>
            <div class="prediction-content">{llm_prediction}</div>
            <div class="prediction-recommendation rec-{llm_rec}">Recommend: {llm_rec.upper()}</div>
            <div class="timing-info">
                <span>⚡</span>
                <span>{llm_time:.2f}s</span>
            </div>
        </div>
        <div class="prediction-box gemini-box">
            <div class="prediction-header">🧠 Google Gemini 2.0</div>
            <div class="prediction-content">{gemini_prediction}</div>
            <div class="prediction-recommendation rec-{gemini_rec}">Recommend: {gemini_rec.upper()}</div>
            <div class="timing-info">
                <span>⚡</span>
                <span>{gemini_time:.2f}s</span>
            </div>
        </div>
    </div>
    """
    st.markdown(predictions_html, unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)
    
    st.markdown('</div>', unsafe_allow_html=True)

# --- MAIN DASHBOARD ---
def display_dashboard():
    # Create responsive grid - 2 cards per row on desktop, 1 on mobile
    stocks_data = []
    
    # Show loading state
    if len([s.strip() for s in stock_symbols if s.strip()]) > 0:
        with st.spinner('🔄 Loading stock data and generating AI insights...'):
            # Collect all stock data first
            for stock in stock_symbols:
                stock = stock.strip().upper()
                if not stock:
                    continue
                    
                indicators, data, price_change = calculate_all(stock)
                if not indicators:
                    st.warning(f"⚠️ No data available for {stock}")
                    continue
                    
                # Get AI predictions
                llm_start = time.time()
                llm_prediction = get_llm_prediction(stock, indicators)
                llm_time = time.time() - llm_start
                
                gemini_start = time.time()
                gemini_prediction = get_gemini_prediction(stock, indicators)
                gemini_time = time.time() - gemini_start
                
                stocks_data.append({
                    'stock': stock,
                    'indicators': indicators,
                    'data': data,
                    'llm_prediction': llm_prediction,
                    'gemini_prediction': gemini_prediction,
                    'llm_time': llm_time,
                    'gemini_time': gemini_time,
                    'price_change': price_change
                })
    
    # Display cards in grid layout
    if stocks_data:
        st.markdown(f"### 📊 Tracking {len(stocks_data)} stocks")
        
        # Create rows of 2 cards each
        for i in range(0, len(stocks_data), 2):
            cols = st.columns(2)
            
            # First card
            with cols[0]:
                stock_info = stocks_data[i]
                create_stock_card(
                    stock_info['stock'],
                    stock_info['indicators'],
                    stock_info['data'],
                    stock_info['llm_prediction'],
                    stock_info['gemini_prediction'],
                    stock_info['llm_time'],
                    stock_info['gemini_time'],
                    stock_info['price_change']
                )
            
            # Second card (if exists)
            if i + 1 < len(stocks_data):
                with cols[1]:
                    stock_info = stocks_data[i + 1]
                    create_stock_card(
                        stock_info['stock'],
                        stock_info['indicators'],
                        stock_info['data'],
                        stock_info['llm_prediction'],
                        stock_info['gemini_prediction'],
                        stock_info['llm_time'],
                        stock_info['gemini_time'],
                        stock_info['price_change']
                    )
    else:
        st.info("👆 Enter stock symbols above to start tracking")

# --- RUN DASHBOARD ---
display_dashboard()

# Performance metrics display
if len([s.strip() for s in stock_symbols if s.strip()]) > 0:
    st.markdown("---")
    st.markdown("### 📊 Quick Stats")
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Stocks Tracked", len([s.strip() for s in stock_symbols if s.strip()]))
    with col2:
        st.metric("Data Source", "Yahoo Finance")
    with col3:
        st.metric("AI Models", "2 Active")
    with col4:
        current_time = time.strftime("%H:%M:%S")
        st.metric("Last Update", current_time)