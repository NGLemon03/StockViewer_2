# pages/5_空頭.py
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px

from fetch_stock_list import get_stock_lists
from stock_data_processing import download_data
from investment_indicators import calculate_sharpe
from investment_indicators import (
    calculate_irr_from_prices,
    calculate_sharpe,
    calculate_max_drawdown,
    calculate_annualized_return
)

from bear_market_analysis import analyze_bear_market


def page_bear():
    st.header("空頭期間分析")
    get_stock_lists()
    default_stock_list = ["2412", "2330"]
    stock_list = st.multiselect("要分析的標的", default_stock_list, default=default_stock_list)

    stock_data = download_data(stock_list)
    if stock_data.empty:
        st.warning("無可用股價資料。")
        return

    start_date_1 = st.date_input("空頭1開始", pd.to_datetime("2020-01-01"))
    end_date_1 = st.date_input("空頭1結束", pd.to_datetime("2020-05-01"))
    start_date_2 = st.date_input("空頭2開始", pd.to_datetime("2022-01-01"))
    end_date_2 = st.date_input("空頭2結束", pd.to_datetime("2022-12-31"))

    bear_periods = {
        "疫情": (start_date_1, end_date_1),
        "FED升息": (start_date_2, end_date_2)
    }
    for name, (sd, ed) in bear_periods.items():
        st.write(f"#### {name}: {sd} ~ {ed}")
        df_res = analyze_bear_market(stock_data, str(sd), str(ed))
        if df_res.empty:
            st.write("該期間無效或無資料。")
            continue
        df_res_show = df_res.loc[df_res.index.intersection(stock_list)]
        st.dataframe(df_res_show.style.format("{:.4f}"))

        for col in ["最大回撤", "波動率", "總收益率"]:
            if col in df_res_show.columns:
                fig_bar = px.bar(
                    df_res_show,
                    x=df_res_show.index,
                    y=col,
                    title=f"{name} - {col}"
                )
                st.plotly_chart(fig_bar, use_container_width=True)
