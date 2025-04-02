## NOTE: Set yfinance to the following version to get chart working: "pip install yfinance==0.2.40"

import streamlit as st
import yfinance as yf
import pandas as pd
import plotly.graph_objects as go
import ollama
import tempfile
import base64
import os
import time

# Set up Streamlit app
st.set_page_config(layout="wide")
st.title("AI-Powered Technical Stock Analysis Dashboard")
st.sidebar.header("Configuration")

# Initialize Ollama client with error handling
try:
    ollama_client = ollama.Client(host='http://localhost:11434')  # Default Ollama port
    available_models = [model['name'] for model in ollama_client.list()['models']]
    if 'llama3.2-vision' not in available_models:
        st.warning("llama3.2-vision model not found in Ollama. Please pull it first.")
except Exception as e:
    st.error(f"Failed to connect to Ollama: {str(e)}")
    st.stop()

# Input for stock ticker and date range
ticker = st.sidebar.text_input("Enter Stock Ticker (e.g., AAPL):", "AAPL")
start_date = st.sidebar.date_input("Start Date", value=pd.to_datetime("2023-01-01"))
end_date = st.sidebar.date_input("End Date", value=pd.to_datetime("2024-12-14"))

# Fetch stock data
if st.sidebar.button("Fetch Data"):
    try:
        st.session_state["stock_data"] = yf.download(ticker, start=start_date, end=end_date)
        st.success("Stock data loaded successfully!")
    except Exception as e:
        st.error(f"Error fetching stock data: {str(e)}")

# Check if data is available
if "stock_data" in st.session_state:
    data = st.session_state["stock_data"]

    # Plot candlestick chart
    fig = go.Figure(data=[
        go.Candlestick(
            x=data.index,
            open=data['Open'],
            high=data['High'],
            low=data['Low'],
            close=data['Close'],
            name="Candlestick"
        )
    ])

    # Sidebar: Select technical indicators
    st.sidebar.subheader("Technical Indicators")
    indicators = st.sidebar.multiselect(
        "Select Indicators:",
        ["20-Day SMA", "20-Day EMA", "20-Day Bollinger Bands", "VWAP"],
        default=["20-Day SMA"]
    )

    # Helper function to add indicators to the chart
    def add_indicator(indicator):
        if indicator == "20-Day SMA":
            sma = data['Close'].rolling(window=20).mean()
            fig.add_trace(go.Scatter(x=data.index, y=sma, mode='lines', name='SMA (20)'))
        elif indicator == "20-Day EMA":
            ema = data['Close'].ewm(span=20).mean()
            fig.add_trace(go.Scatter(x=data.index, y=ema, mode='lines', name='EMA (20)'))
        elif indicator == "20-Day Bollinger Bands":
            sma = data['Close'].rolling(window=20).mean()
            std = data['Close'].rolling(window=20).std()
            bb_upper = sma + 2 * std
            bb_lower = sma - 2 * std
            fig.add_trace(go.Scatter(x=data.index, y=bb_upper, mode='lines', name='BB Upper'))
            fig.add_trace(go.Scatter(x=data.index, y=bb_lower, mode='lines', name='BB Lower'))
        elif indicator == "VWAP":
            data['VWAP'] = (data['Close'] * data['Volume']).cumsum() / data['Volume'].cumsum()
            fig.add_trace(go.Scatter(x=data.index, y=data['VWAP'], mode='lines', name='VWAP'))

    # Add selected indicators to the chart
    for indicator in indicators:
        add_indicator(indicator)

    fig.update_layout(
        xaxis_rangeslider_visible=False,
        height=600,
        title=f"{ticker} Stock Price with Technical Indicators"
    )
    st.plotly_chart(fig, use_container_width=True)

    # Analyze chart with LLaMA 3.2 Vision
    st.subheader("AI-Powered Analysis")
    if st.button("Run AI Analysis"):
        with st.spinner("Analyzing the chart, please wait..."):
            try:
                # Save chart as a temporary image
                with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmpfile:
                    fig.write_image(tmpfile.name, format='png', width=1200, height=800)
                    tmpfile_path = tmpfile.name

                # Read image and encode to Base64
                with open(tmpfile_path, "rb") as image_file:
                    image_data = base64.b64encode(image_file.read()).decode('utf-8')

                # Prepare AI analysis request
                messages = [{
                    'role': 'user',
                    'content': """You are a Stock Trader specializing in Technical Analysis at a top financial institution.
                                Analyze the stock chart's technical indicators and provide a buy/hold/sell recommendation.
                                Base your recommendation only on the candlestick chart and the displayed technical indicators.
                                First, provide the recommendation, then, provide your detailed reasoning.
                                Be concise but thorough in your analysis.
                    """,
                    'images': [image_data]
                }]
                
                # Add retry logic for the Ollama call
                max_retries = 3
                for attempt in range(max_retries):
                    try:
                        response = ollama.chat(model='llama3.2-vision', messages=messages)
                        break
                    except Exception as e:
                        if attempt == max_retries - 1:
                            raise
                        time.sleep(2)  # Wait before retrying

                # Display AI analysis result
                st.write("**AI Analysis Results:**")
                st.write(response["message"]["content"])

            except Exception as e:
                st.error(f"Error during AI analysis: {str(e)}")
            finally:
                # Clean up temporary file
                if 'tmpfile_path' in locals() and os.path.exists(tmpfile_path):
                    os.remove(tmpfile_path)
