# pages/4_比較.py
import streamlit as st
import pandas as pd
import plotly.express as px
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

from fetch_stock_list import get_stock_lists
from stock_data_processing import download_data
from investment_indicators import calculate_sharpe
from investment_indicators import (
    calculate_irr_from_prices,
    calculate_sharpe,
    calculate_max_drawdown,
    calculate_annualized_return
)

def page_compare():
    st.header("多標的比較 (相關係數、Sharpe、MDD)")
    st.write("在此可一次比較多檔股票或ETF的表現。")

    # 下載或讀取股價
    get_stock_lists()
    default_stock_list = ["2412", "2303", "2330", "^IRX"]
    stock_list = st.multiselect("比較標的", default_stock_list, default=default_stock_list)

    stock_data = download_data(stock_list)
    st.write("資料範圍:", stock_data.index.min(), "~", stock_data.index.max())
    st.write(f"資料筆數: {len(stock_data)}")

    if not stock_data.empty:
        period = st.selectbox("計算期間(交易日)", [60, 120, 240, 480], index=1)
        subset = stock_data[-period:].dropna(how='all', axis=1)
        if subset.empty:
            st.warning("近期無可用股價資料。")
            return
        corr_matrix = subset.corr()
        fig_corr = plt.figure(figsize=(6,5))
        sns.heatmap(corr_matrix, annot=True, cmap="coolwarm", square=True)
        plt.title(f"Correlation Heatmap (最後{period}交易日)")
        st.pyplot(fig_corr)

        # 計算 Sharpe, MDD, 年化報酬
        sharpe_dict = {}
        mdd_dict = {}
        ann_dict = {}
        for s in subset.columns:
            sh = calculate_sharpe(subset[s], period=period)
            mdd = calculate_max_drawdown(subset[s])
            ann = calculate_annualized_return(subset[s], period=period)
            sharpe_dict[s] = sh
            mdd_dict[s] = mdd
            ann_dict[s] = ann

        df_compare = pd.DataFrame({
            "Sharpe": sharpe_dict,
            "MDD": mdd_dict,
            "AnnualReturn": ann_dict
        })
        st.dataframe(df_compare.style.format("{:.4f}"))

page_compare()
