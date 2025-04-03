import streamlit as st
import yfinance as yf
import pandas as pd
import plotly.graph_objects as go
import ollama
import tempfile
import base64
import os
import time
from datetime import datetime

# Configuration
st.set_page_config(layout="wide", page_title="AI Stock Analyst")
st.title("📈 AI-Powered Stock Technical Analysis Dashboard")

# --- Ollama Setup (Local) ---
def setup_ollama():
    """Initialize Ollama connection with error handling"""
    OLLAMA_HOST = os.getenv("OLLAMA_HOST", "http://localhost:11434")  # Default port
    
    try:
        client = ollama.Client(host=OLLAMA_HOST)
        # Verify connection by listing models
        models = client.list()
        if not models.get('models'):
            st.warning("No models found in Ollama. Please pull a model first (e.g., `ollama pull llama3`)")
        return client
    except Exception as e:
        st.error(f"🚨 Failed to connect to Ollama at {OLLAMA_HOST}")
        st.error(f"Error details: {str(e)}")
        st.warning("""
            Troubleshooting:
            1. Ensure Ollama is installed (https://ollama.ai/)
            2. Run `ollama serve` in terminal
            3. Pull a model (e.g., `ollama pull llama3`)
            4. For vision models: `ollama pull llava`
            """)
        st.stop()

ollama_client = setup_ollama()

# --- Sidebar Controls ---
st.sidebar.header("Settings")
ticker = st.sidebar.text_input("Stock Symbol", "AAPL").upper()
col1, col2 = st.sidebar.columns(2)
start_date = col1.date_input("Start Date", datetime(2023, 1, 1))
end_date = col2.date_input("End Date", datetime.today())

# Fetch data with caching
@st.cache_data(ttl=3600)
def get_stock_data(ticker, start, end):
    try:
        data = yf.download(ticker, start=start, end=end, progress=False)
        if data.empty:
            st.error("No data found for this ticker/date range")
            return None
        return data
    except Exception as e:
        st.error(f"YFinance error: {str(e)}")
        return None

if st.sidebar.button("📥 Load Data", type="primary"):
    with st.spinner(f"Fetching {ticker} data..."):
        st.session_state.stock_data = get_stock_data(ticker, start_date, end_date)

# --- Main Chart ---
if "stock_data" in st.session_state and st.session_state.stock_data is not None:
    data = st.session_state.stock_data
    
    # Create candlestick chart
    fig = go.Figure(data=[
        go.Candlestick(
            x=data.index,
            open=data['Open'],
            high=data['High'],
            low=data['Low'],
            close=data['Close'],
            name="Price"
        )
    ])

    # Technical indicators
    st.sidebar.subheader("Technical Indicators")
    indicators = st.sidebar.multiselect(
        "Add Indicators:",
        ["50-Day SMA", "200-Day SMA", "20-Day EMA", "Bollinger Bands", "RSI", "MACD"],
        default=["50-Day SMA"]
    )

    # Add selected indicators
    if "50-Day SMA" in indicators:
        fig.add_trace(go.Scatter(
            x=data.index,
            y=data['Close'].rolling(50).mean(),
            name="50-Day SMA",
            line=dict(color='royalblue')
        ))
    
    if "200-Day SMA" in indicators:
        fig.add_trace(go.Scatter(
            x=data.index,
            y=data['Close'].rolling(200).mean(),
            name="200-Day SMA",
            line=dict(color='orange')
        ))
    
    if "20-Day EMA" in indicators:
        fig.add_trace(go.Scatter(
            x=data.index,
            y=data['Close'].ewm(span=20).mean(),
            name="20-Day EMA",
            line=dict(color='green')
        ))
    
    if "Bollinger Bands" in indicators:
        sma = data['Close'].rolling(20).mean()
        std = data['Close'].rolling(20).std()
        fig.add_trace(go.Scatter(x=data.index, y=sma + 2*std, name='Upper Band', line=dict(color='gray')))
        fig.add_trace(go.Scatter(x=data.index, y=sma - 2*std, name='Lower Band', line=dict(color='gray'), fill='tonexty'))

    # Chart formatting
    fig.update_layout(
        height=700,
        xaxis_rangeslider_visible=False,
        title=f"{ticker} Price Chart",
        hovermode="x unified",
        template="plotly_dark"
    )
    st.plotly_chart(fig, use_container_width=True)

    # --- AI Analysis Section ---
    st.divider()
    st.subheader("🤖 AI Technical Analysis")
    
    analysis_type = st.radio("Analysis Mode:", 
                           ["Basic Technical Analysis", "Advanced Pattern Recognition"],
                           horizontal=True)
    
    if st.button("🔍 Generate AI Report", type="primary"):
        with st.spinner("🧠 Analyzing market trends..."):
            try:
                # Save chart as temp image
                with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp:
                    fig.write_image(tmp.name, scale=2)
                    img_b64 = base64.b64encode(tmp.read()).decode()
                
                # Dynamic prompt based on selection
                prompt = """
                As a senior technical analyst with 20 years experience, analyze this stock chart.
                Provide:
                1. Clear buy/hold/sell recommendation with reasoning
                2. Key support/resistance levels
                3. Notable chart patterns
                4. Confidence level (low/medium/high)
                """ if analysis_type == "Basic Technical Analysis" else """
                Perform advanced technical analysis:
                1. Identify harmonic patterns (Gartley, Bat, Butterfly)
                2. Detect Elliott Wave patterns if visible
                3. Analyze volume-price relationship
                4. Predict next likely price movement
                """
                
                messages = [{
                    'role': 'user',
                    'content': prompt,
                    'images': [img_b64]
                }]
                
                # Get response with retry logic
                response = None
                for attempt in range(3):
                    try:
                        response = ollama_client.chat(
                            model='llava' if 'vision' in analysis_type else 'llama3',
                            messages=messages
                        )
                        break
                    except Exception as e:
                        if attempt == 2:
                            raise
                        time.sleep(2)
                
                # Display results
                st.success("✅ Analysis Complete")
                st.markdown("### AI Findings")
                st.write(response['message']['content'])
                
                # Add disclaimer
                st.warning("⚠️ This is AI-generated analysis, not financial advice. Always do your own research.")
                
            except Exception as e:
                st.error(f"AI analysis failed: {str(e)}")
            finally:
                if 'tmp' in locals():
                    os.unlink(tmp.name)

# --- How to Use Section ---
with st.expander("ℹ️ How to use this dashboard"):
    st.markdown("""
    1. Enter a stock symbol (e.g. AAPL, TSLA)
    2. Select date range
    3. Click "Load Data"
    4
