# pages/3_回測.py
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from StockViewer_2.modlus.fetch_stock_list import get_stock_lists
from StockViewer_2.modlus.stock_data_processing import download_data
from StockViewer_2.modlus.investment_indicators import calculate_irr_from_prices

# from investment_indicators import calculate_max_drawdown, ...

def page_backtest():
    st.header("簡易回測")
    st.write("在此可進行如定期定額 IRR、Shapre、MDD 等指標回測。")

    # 下載或讀取股價
    get_stock_lists()  # 先更新上市、上櫃清單
    default_stock_list = ["2412", "00713", "^IRX"]
    stock_list = st.multiselect("選擇要回測的標的", default_stock_list, default=default_stock_list)

    st.write("下載股價中...")
    stock_data = download_data(stock_list)
    st.success("股價下載完成。")

    invest_amt = st.number_input("每月定期定額(負值表示支出)", value=-1000, step=100)

    irr_dict = {}
    for s in stock_list:
        if s not in stock_data.columns:
            irr_dict[s] = np.nan
            continue
        px = stock_data[s].dropna()
        if len(px) < 2:
            irr_dict[s] = np.nan
        else:
            irr_dict[s] = calculate_irr_from_prices(px, investment_per_month=invest_amt)

    st.write("定期定額IRR(年化)")
    df_irr = pd.DataFrame({"IRR(年化)": irr_dict})
    st.dataframe(df_irr.style.format("{:.2%}"))


page_backtest()
