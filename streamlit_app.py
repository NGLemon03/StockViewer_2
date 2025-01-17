# streamlit_app.py
import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns
import plotly.express as px
import matplotlib.pyplot as plt

from config import work_dir
from fetch_stock_list import get_stock_lists
from stock_data_processing import download_data
from investment_indicators import (
    calculate_irr_from_prices,
    calculate_sharpe,
    calculate_max_drawdown,
    calculate_annualized_return
)
from bear_market_analysis import analyze_bear_market

def streamlit_app():
    st.title("整合版：單一股票研究 & 多標的比較")

    # 抓取上市/上櫃/興櫃清單
    get_stock_lists()

    # ──────────────────────────
    # A. 基本股票/ETF下載 & IRR計算
    # ──────────────────────────
    default_stock_list = [
        '2412', '00713', '00701', '006208', '00710B',
        '^IRX','^TYX'
    ]
    stock_list = st.sidebar.multiselect(
        "選擇股票或ETF代號",
        options=default_stock_list,
        default=default_stock_list
    )
    invest_amt = st.sidebar.number_input("定期定額金額(負數表示支出)", value=-1000, step=100)
    st.write("### 下載與整理股價資料中 ...")
    stock_data = download_data(stock_list)
    st.success("下載完成！")

    irr_dict = {}
    listing_times = {}
    for stock in stock_list:
        if stock not in stock_data.columns:
            irr_dict[stock] = np.nan
            listing_times[stock] = 0
            continue
        prices = stock_data[stock].dropna()
        listing_times[stock] = len(prices)
        if len(prices) < 2:
            irr_dict[stock] = np.nan
        else:
            irr_dict[stock] = calculate_irr_from_prices(prices, investment_per_month=invest_amt)

    irr_df = pd.DataFrame({
        '上市天數': listing_times,
        'IRR(年化)': [irr_dict[s] for s in stock_list]
    }, index=stock_list)
    st.dataframe(irr_df.style.format({"IRR(年化)": "{:.2%}"}))

    # ──────────────────────────
    # B. 投資指標計算 (Sharpe, MDD, AnnualReturn)
    # ──────────────────────────
    period_options = [60, 120, 240]
    selected_periods = st.sidebar.multiselect(
        "計算指標的期間(交易日)", period_options, default=[60, 120]
    )
    all_metrics_df = pd.DataFrame(index=stock_data.columns)
    all_metrics_df['IRR'] = [irr_dict.get(s, np.nan) for s in all_metrics_df.index]

    for period in selected_periods:
        subset = stock_data[-period:].dropna(how='all', axis=1)
        if subset.empty:
            continue
        corr_matrix = subset.corr()
        fig_corr = plt.figure(figsize=(6, 5))
        sns.heatmap(corr_matrix, annot=True, cmap="coolwarm", square=True)
        plt.title(f"Correlation Heatmap ({period} days)")
        st.pyplot(fig_corr)

        for stock in subset.columns:
            sh = calculate_sharpe(subset[stock], period)
            mdd = calculate_max_drawdown(subset[stock])
            ann_ret = calculate_annualized_return(subset[stock], period)
            all_metrics_df.loc[stock, f"{period}_Sharpe"] = sh
            all_metrics_df.loc[stock, f"{period}_MDD"] = mdd
            all_metrics_df.loc[stock, f"{period}_AnnualReturn"] = ann_ret

    st.write("### 投資指標匯總")
    st.dataframe(all_metrics_df.loc[stock_list].style.format("{:.4f}"))

    # ──────────────────────────
    # C. 股價走勢與利率基準比較
    # ──────────────────────────
    st.subheader("股價與基準利率之走勢/累積報酬比較")
    selected_for_plot = st.multiselect(
        "選擇要比較的標的",
        options=stock_data.columns,
        default=stock_data.columns[:3]
    )
    plot_type = st.radio("圖形類型", ["累積報酬 (%)", "價格"], index=0)
    if selected_for_plot:
        plot_df = stock_data[selected_for_plot].dropna()
        if not plot_df.empty:
            start_date = st.date_input("走勢 - 開始日期", value=plot_df.index[0])
            end_date = st.date_input("走勢 - 結束日期", value=plot_df.index[-1])
            mask = (plot_df.index >= pd.to_datetime(start_date)) & (plot_df.index <= pd.to_datetime(end_date))
            plot_df = plot_df.loc[mask]

            if plot_type == "累積報酬 (%)":
                ret_df = plot_df.pct_change().fillna(0).cumsum() * 100
                fig_line = px.line(ret_df, x=ret_df.index, y=ret_df.columns,
                                   title="累積報酬(%)", labels={"value": "累積報酬(%)"})
                st.plotly_chart(fig_line, use_container_width=True)
            else:
                fig_line = px.line(plot_df, x=plot_df.index, y=plot_df.columns,
                                   title="價格走勢", labels={"value": "價格"})
                st.plotly_chart(fig_line, use_container_width=True)
        else:
            st.warning("該時間範圍內沒有資料。")

    # ──────────────────────────
    # D. 空頭市場分析
    # ──────────────────────────
    st.subheader("空頭市場分析")
    bear1_start = st.sidebar.date_input("範例-空頭期間1開始", value=pd.to_datetime("2020-01-01"))
    bear1_end   = st.sidebar.date_input("範例-空頭期間1結束", value=pd.to_datetime("2020-05-01"))
    bear2_start = st.sidebar.date_input("範例-空頭期間2開始", value=pd.to_datetime("2022-01-01"))
    bear2_end   = st.sidebar.date_input("範例-空頭期間2結束", value=pd.to_datetime("2022-12-31"))
    bear_markets = {
        "2020 新冠疫情": (bear1_start, bear1_end),
        "2022 FED 升息": (bear2_start, bear2_end),
    }
    for market, (start_date, end_date) in bear_markets.items():
        st.write(f"#### {market}: {start_date} ~ {end_date}")
        bear_results = analyze_bear_market(stock_data, str(start_date), str(end_date))
        if bear_results.empty:
            st.write("該期間沒有有效數據。")
            continue
        st.dataframe(bear_results.loc[stock_list].style.format("{:.4f}"))
        for col in ["最大回撤", "波動率", "總收益率"]:
            if col in bear_results.columns:
                subset_bear = bear_results.loc[stock_list].dropna(subset=[col])
                if not subset_bear.empty:
                    fig_bar = px.bar(
                        subset_bear,
                        x=subset_bear.index,
                        y=col,
                        title=f"{market} - {col}",
                        labels={"x": "標的", "y": col}
                    )
                    st.plotly_chart(fig_bar, use_container_width=True)

    # ──────────────────────────
    # E. 單一股票的財報分析 (示意位置)
    # ──────────────────────────
    st.subheader("單一股票：財報分析 (示意入口)")
    st.write("此處可擴充顯示：財報的估價、預估股利、預估EPS、壓力線等等功能...")

    st.info("分析結束。可於左側選單重新選擇標的或調整參數。")
