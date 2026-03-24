import streamlit as st
options = [
    "My Portfolio",
    "Market Signals",
    "Symbol Technical Analysis",
    "All Symbols Technical Analysis",
    "What to Buy Tomorrow",
    "What to Sell Tomorrow",
    "Buy Sector Wise",
]
st.radio("Options", options, horizontal=True)
