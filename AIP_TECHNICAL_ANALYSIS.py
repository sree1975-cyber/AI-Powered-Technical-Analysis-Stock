import streamlit as st
import yfinance as yf
import pandas as pd
import plotly.graph_objects as go
import ollama
import tempfile
import base64
import os
import time

# Configuration
st.set_page_config(layout="wide")
st.title("AI-Powered Technical Stock Analysis Dashboard")

# --- Ollama Setup (Works Locally + Cloud) ---
OLLAMA_HOST = os.getenv("OLLAMA_HOST", "http://localhost:127.0.0.1:11434")  # Default local, override with env var

try:
    ollama_client = ollama.Client(host=OLLAMA_HOST)
    # Test connection
    ollama_client.list()
except Exception as e:
    st.error(f"⚠️ Failed to connect to Ollama at {OLLAMA_HOST}. Error: {str(e)}")
    st.warning("""
        To fix this:
        1. For local runs: Ensure Ollama is installed and running (`ollama serve`).
        2. For Streamlit Cloud: Host Ollama on a cloud server and set `OLLAMA_HOST` in secrets.
    """)
    st.stop()

# --- Sidebar Controls ---
st.sidebar.header("Configuration")
ticker = st.sidebar.text_input("Enter Stock Ticker (e.g., AAPL):", "AAPL")
start_date = st.sidebar.date_input("Start Date", pd.to_datetime("2023-01-01"))
end_date = st.sidebar.date_input("End Date", pd.to_datetime("2024-12-14"))

# Fetch stock data
if st.sidebar.button("Fetch Data"):
    try:
        st.session_state["stock_data"] = yf.download(ticker, start=start_date, end=end_date)
        st.success(f"Successfully loaded {ticker} data!")
    except Exception as e:
        st.error(f"Failed to fetch data: {str(e)}")

# --- Main Chart Logic ---
if "stock_data" in st.session_state:
    data = st.session_state["stock_data"]

    # Candlestick chart
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
        "Select Indicators:",
        ["20-Day SMA", "20-Day EMA", "20-Day Bollinger Bands", "VWAP"],
        default=["20-Day SMA"]
    )

    def add_indicator(indicator):
        if indicator == "20-Day SMA":
            sma = data['Close'].rolling(20).mean()
            fig.add_trace(go.Scatter(x=data.index, y=sma, name='SMA (20)'))
        elif indicator == "20-Day EMA":
            ema = data['Close'].ewm(span=20).mean()
            fig.add_trace(go.Scatter(x=data.index, y=ema, name='EMA (20)'))
        elif indicator == "20-Day Bollinger Bands":
            sma = data['Close'].rolling(20).mean()
            std = data['Close'].rolling(20).std()
            fig.add_trace(go.Scatter(x=data.index, y=sma + 2*std, name='BB Upper'))
            fig.add_trace(go.Scatter(x=data.index, y=sma - 2*std, name='BB Lower'))
        elif indicator == "VWAP":
            data['VWAP'] = (data['Close'] * data['Volume']).cumsum() / data['Volume'].cumsum()
            fig.add_trace(go.Scatter(x=data.index, y=data['VWAP'], name='VWAP'))

    for indicator in indicators:
        add_indicator(indicator)

    fig.update_layout(
        height=600,
        xaxis_rangeslider_visible=False,
        title=f"{ticker} Stock Analysis"
    )
    st.plotly_chart(fig, use_container_width=True)

    # --- AI Analysis Section ---
    st.subheader("AI Technical Analysis")
    if st.button("Generate AI Analysis"):
        with st.spinner("Analyzing chart with AI..."):
            try:
                # Save chart as image
                with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmpfile:
                    fig.write_image(tmpfile.name, format='png', width=1200)
                    tmp_path = tmpfile.name

                # Encode image
                with open(tmp_path, "rb") as f:
                    img_b64 = base64.b64encode(f.read()).decode("utf-8")

                # AI Prompt
                messages = [{
                    'role': 'user',
                    'content': """As a senior technical analyst, analyze this stock chart.
                        Provide: 1) Clear buy/hold/sell recommendation, 2) Key support/resistance levels,
                        3) Notable patterns, and 4) Confidence level (low/medium/high).""",
                    'images': [img_b64]
                }]

                # Retry logic for Ollama
                max_retries = 3
                for attempt in range(max_retries):
                    try:
                        response = ollama_client.chat(model='llama3.2-vision', messages=messages)
                        break
                    except Exception as e:
                        if attempt == max_retries - 1:
                            raise
                        time.sleep(2)

                st.success("AI Analysis Complete!")
                st.markdown(f"**Recommendation:** {response['message']['content']}")

            except Exception as e:
                st.error(f"AI analysis failed: {str(e)}")
            finally:
                if os.path.exists(tmp_path):
                    os.remove(tmp_path)
