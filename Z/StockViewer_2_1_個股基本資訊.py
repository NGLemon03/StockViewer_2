# pages/1_個股基本資訊.py

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import os
from datetime import datetime
import yfinance as yf

# 專案內部模組
from config import work_dir, DL_dir
from investment_indicators import (
    calc_metrics_for_range,
    calc_multiple_period_metrics,
    calculate_alpha_beta
)
from financial_statements_fetcher import (
    fetch_all_data,
    close_driver
)
from DL_Y import download_stock_price  # 假設您有此檔案專門只下載股價

st.set_page_config(
    page_title="個股基本資訊與多股票比較",
    layout="wide",
)

def page_basic_info():
    st.title("個股基本資訊與多股票比較")

    # 區塊1: 下載參數
    with st.expander("1) 資料下載參數", expanded=True):
        colA, colB, colC = st.columns(3)
        with colA:
            stock_id = st.text_input("主要股票代號 (會完整抓取 Goodinfo + 三大財報)", value="2412")
        with colB:
            market_id = st.text_input("市場基準代號 (計算Alpha/Beta用)", value="^TWII")
        with colC:
            other_ids_input = st.text_input("其他比較股票代號 (逗號分隔)", value="2330,00713,006208")

        colD, colE = st.columns(2)
        with colD:
            start_date = st.date_input("股價開始日期", pd.to_datetime("2000-01-01"))
            end_date   = st.date_input("股價結束日期", pd.to_datetime("2025-01-01"))
        with colE:
            start_yq = st.text_input("財報起始季度(YYYY-Q)", value="2000-1")
            end_yq   = st.text_input("財報結束季度(YYYY-Q)", value="2024-4")

        # 解析
        try:
            sy, sq = map(int, start_yq.split("-"))
            ey, eq = map(int, end_yq.split("-"))
        except:
            st.error("財報季度格式錯誤(YYYY-Q)")
            return

        # 按鈕: 下載
        if st.button("下載或更新資料 (Goodinfo + 股價)"):
            st.write("開始下載...請稍候")
            # 主要股票:
            main_dir = os.path.join(DL_dir, stock_id)
            os.makedirs(main_dir, exist_ok=True)
            fetch_all_data(
                stockID=stock_id,
                base_dir=main_dir,
                start_date=str(start_date),
                end_date=str(end_date),
                start_year=sy,
                start_quarter=sq,
                end_year=ey,
                end_quarter=eq
            )

            # 市場指數(用於Alpha/Beta)
            if market_id.strip():
                market_dir = os.path.join(DL_dir, market_id)
                os.makedirs(market_dir, exist_ok=True)
                download_stock_price(
                    stockID=market_id,
                    base_dir=market_dir,
                    start_date=str(start_date),
                    end_date=str(end_date)
                )

            # 其他股票
            other_ids = [x.strip() for x in other_ids_input.split(",") if x.strip()]
            for sid in other_ids:
                sid_dir = os.path.join(DL_dir, sid)
                os.makedirs(sid_dir, exist_ok=True)
                download_stock_price(
                    stockID=sid,
                    base_dir=sid_dir,
                    start_date=str(start_date),
                    end_date=str(end_date)
                )

            st.success("下載完畢！")


    # 區塊2: 指標與比較
    with st.expander("2) 多股票指標比較", expanded=True):
        col1, col2 = st.columns(2)
        with col1:
            analysis_start_date = st.date_input("分析起始日期", pd.to_datetime("2010-01-01"))
        with col2:
            analysis_end_date   = st.date_input("分析結束日期", pd.to_datetime("2025-01-01"))

        rebase_data = {}
        metrics_data = []
        cumulative_df = pd.DataFrame()

        # 多檔股票 = [主要股票] + [市場基準(可選)] + [其他股票]
        all_stocks = [stock_id]  # main
        if market_id.strip():
            all_stocks.append(market_id.strip())
        other_ids = [x.strip() for x in other_ids_input.split(",") if x.strip()]
        all_stocks.extend(other_ids)

        for sid in all_stocks:
            price_file = os.path.join(DL_dir, sid, f"{sid}_price.csv")
            if not os.path.exists(price_file):
                st.warning(f"{sid}_price.csv 不存在, 請先下載")
                continue
            try:
                df_price = pd.read_csv(price_file, parse_dates=["Date"], index_col="Date")
                df_price.index = df_price.index.tz_localize(None)
                # 篩選分析區間
                df_sub = df_price.loc[analysis_start_date:analysis_end_date].copy()
                if df_sub.empty:
                    st.warning(f"{sid} 在分析期間無交易資料，跳過")
                    continue

                # Rebase：若該股票的第一筆日期 > analysis_start_date，就從那天開始 = 1
                first_idx = df_sub.index[0]
                df_sub["pct"] = df_sub["Close"].pct_change().fillna(0)
                # 累積報酬：從「該檔實際開始日」當天Close當作 1
                # 先計算 factor = 1 + daily_ret 累乘
                cum = (1 + df_sub["pct"]).cumprod()
                # 讓第一天 = 1
                # (若您想讓 "analysis_start_date" 當天=1，即使股票沒上市，也可以再行調整)
                base_price = cum.iloc[0]
                df_sub["cumulative_return"] = cum / base_price


                # 累積報酬存至 cumulative_df
                # 以 df_sub.index 做 join
                cumulative_df[sid] = df_sub["cumulative_return"]

            except Exception as e:
                st.error(f"讀取/處理 {sid} 時出錯: {e}")

        # 畫累積報酬
        if not cumulative_df.empty:
            st.subheader("累積報酬 (Rebase) 比較圖")
            fig = px.line(
                cumulative_df,
                x=cumulative_df.index,
                y=cumulative_df.columns,
                labels={"value": "累積報酬", "variable": "股票"},
                title="多股票累積報酬比較 (Rebase)"
            )
            fig.update_xaxes(autorange=True)
            fig.update_yaxes(rangemode="tozero")  # 讓y從0開始
            st.plotly_chart(fig, use_container_width=True)


    # 區塊3: 多期(3/5/10/15/20年) 指標
    with st.expander("3) 多期指標 (3/5/10/15/20年)", expanded=False):
        st.write("此處以所選日期區間的 `end_date` 作為當前時點 (若該檔股票資料無法涵蓋則顯示 NaN)。")
        # 使用 analysis_end_date 作為基準點
        as_of_dt = pd.Timestamp(analysis_end_date)
        # 在第3區塊內新增無風險收益率計算
        tnx_path = os.path.join(DL_dir, "^TNX", "^TNX_price.csv")
        daily_rf_return = 0.0
        if os.path.exists(tnx_path):
            df_tnx = pd.read_csv(tnx_path, parse_dates=["Date"], index_col="Date")

            yearly_yield = df_tnx["Close"].mean() / 100.0  # 假設Close是年化殖利率 (%)
            daily_rf_return = yearly_yield / 252  # 簡單估算日收益率
        else:
            st.warning("無法找到 ^TNX_price.csv，無風險利率暫設為0.0")


        multi_period_data = []
        for sid in all_stocks:
            price_file = os.path.join(DL_dir, sid, f"{sid}_price.csv")
            if not os.path.exists(price_file):
                continue
            try:
                df_price = pd.read_csv(price_file, parse_dates=["Date"], index_col="Date")
                df_price.index = df_price.index.tz_localize(None)
                # 篩選只包含 end_date 之前的數據
                sub_df = df_price.loc[:as_of_dt]
                if sub_df.empty:
                    continue

                # 市場數據
                market_path = os.path.join(DL_dir, market_id, f"{market_id}_price.csv")
                if not os.path.exists(market_path):
                    st.warning(f"找不到市場指數檔 {market_path}")
                    market_df = pd.DataFrame()
                else:
                    market_df = pd.read_csv(market_path, parse_dates=["Date"], index_col="Date")
                    market_df.index = market_df.index.tz_localize(None)
                                # 計算多期指標
                df_multi = calc_multiple_period_metrics(
                    stock_df=sub_df,
                    as_of_date=as_of_dt,
                    years_list=[3, 5, 10, 15, 20],
                    market_df=market_df,
                    daily_rf_return=daily_rf_return
                )

                alpha_val, beta_val = (np.nan, np.nan)
                if market_id.strip() and sid != market_id.strip():
                    market_path = os.path.join(DL_dir, market_id.strip(), f"{market_id.strip()}_price.csv")
                    if os.path.exists(market_path):
                        df_m = pd.read_csv(market_path, parse_dates=["Date"], index_col="Date")
                        df_m.index = df_m.index.tz_localize(None)
                        df_m = df_m.loc[sub_df.index.min():sub_df.index.max()]
                        def get_risk_free_rate(start_date: str, end_date: str, base_dir: str = DL_dir) -> float:
                            """
                            獲取 ^TNX 無風險利率的平均日收益率。
                            """
                            free_rate = "^TNX"
                            free_rate_dir = os.path.join(base_dir, free_rate)
                            os.makedirs(free_rate_dir, exist_ok=True)
                            free_rate_file = os.path.join(free_rate_dir, f"{free_rate}_price.csv")
                            print(f"嘗試獲取無風險利率數據，路徑：{free_rate_file}")

                            # 檢查是否已有下載數據
                            if not os.path.exists(free_rate_file):
                                download_stock_price(
                                    stockID=free_rate,
                                    base_dir=free_rate_dir,
                                    start_date=start_date,
                                    end_date=end_date
                                )

                            try:
                                df_free = pd.read_csv(free_rate_file, parse_dates=["Date"], index_col="Date")
                                df_free = df_free.loc[start_date:end_date]
                                print(df_free)
                                df_free["daily_return"] = df_free["Close"].pct_change().fillna(0)
                                return df_free["daily_return"].mean()
                            except Exception as e:
                                print(f"無法獲取 ^TNX 數據，使用默認值。錯誤：{e}")
                                return 0.0  # 默認無風險利率
                        risk_free_rate = get_risk_free_rate(
                            start_date=str(start_date),
                            end_date=str(end_date)
                        )

                        alpha_val, beta_val = calculate_alpha_beta(
                            stock_prices=sub_df['Close'],
                            market_prices=df_m['Close'],
                            daily_rf_return=risk_free_rate
                        )

                df_multi["股票"] = sid
                df_multi["Alpha"] = alpha_val
                df_multi["Beta"] = beta_val
                multi_period_data.append(df_multi)
            except Exception as e:
                st.error(f"處理 {sid} 時發生錯誤: {e}")

        if multi_period_data:
            # 合併所有數據
            df_combined = pd.concat(multi_period_data, ignore_index=True)
            df_pivot = df_combined.pivot(index="股票", columns="Years", values=["MDD", "Sharpe", "AnnualReturn", "DCA_IRR", "Alpha", "Beta"])
            # 重設多層列名，第一層為年份，第二層為指標
            df_pivot.columns.names = ["指標", "年份"]
            metric_name_map = {
                "MDD": "最大回撤",
                "Sharpe": "夏普值",
                "AnnualReturn": "年平均報酬",
                "DCA_IRR": "定期定額IRR",
                "Alpha": "Alpha",
                "Beta": "Beta",
            }
            df_pivot.rename(columns=metric_name_map, inplace=True)
            # 顯示表格
            def format_rule(metric, year):
                
                if metric == "最大回撤":
                    return "{:.0%}"
                elif metric in ["年平均報酬", "定期定額IRR"]:
                    return "{:.1%}"
                elif metric in ["夏普值", "Alpha", "Beta"]:
                    return "{:.2f}"
                else:
                    return "{:.1%}"  # 預設百分比格式

            st.dataframe(
                df_pivot.style.format({
                    (year, metric): format_rule(year, metric)
                    for year, metric in df_pivot.columns
                })
            )
            
        else:
            st.info("無可顯示之多期指標。")




def main():
    page_basic_info()

if __name__ == "__main__":
    main()
