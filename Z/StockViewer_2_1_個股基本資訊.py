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
    calc_multiple_period_metrics,
    get_risk_free_rate
)
from financial_statements_fetcher import (
    fetch_all_data,
    close_driver
)
from DL_Y import download_stock_price

today_date = pd.to_datetime(datetime.today().date())

st.set_page_config(page_title="個股基本資訊與多股票比較", layout="wide")

def page_basic_info():
    st.title("個股基本資訊與多股票比較")

    # 區塊1: 下載參數
    with st.expander("1) 資料下載參數", expanded=True):
        colA, colB, colC, colD = st.columns([1,1,2,2])
        with colA:
            stock_id = st.text_input("主要股票(抓取Goodinfo數據)", value="2412")
        with colB:
            market_id = st.text_input("市場基準(計算用)", value="^TWII")
        with colC:
            other_ids_input = st.text_input("比較股票(逗號分隔)", value="2330,00713,006208")
        with colD:
            # 使用 columns 排列處理進度文字和動態更新
            progress_col1, progress_col2 = st.columns([1, 3])
            with progress_col1:
                st.text("處理進度")  # 固定標題
            with progress_col2:
                progress_placeholder = st.empty()  # 動態更新的處理進度文字

            # 進度條放在下方
            progress_bar = st.progress(0)

        colE, colF, colG, colH = st.columns(4)
        with colE:
            start_date = st.date_input("股價開始日期", pd.to_datetime("2000-01-01"))
        with colF:
            end_date   = st.date_input("股價結束日期", today_date)
        with colG:
            start_yq = st.text_input("財報起始季度(YYYY-Q)", value="2000-1")
        with colH:
            end_yq   = st.text_input("財報結束季度(YYYY-Q)", value="2024-4")

        # 解析
        try:
            sy, sq = map(int, start_yq.split("-"))
            ey, eq = map(int, end_yq.split("-"))
        except:
            st.error("財報季度格式錯誤(YYYY-Q)")
            return

        # 按鈕: 下載
        if st.button("下載或更新資料"):
            all_stocks = [stock_id] + [market_id.strip()] + [x.strip() for x in other_ids_input.split(",") if x.strip()]
            total_tasks = len(all_stocks)

            for i, sid in enumerate(all_stocks):
                status = f"正在處理 {sid} ({i + 1}/{total_tasks})"
                progress_placeholder.text(status)  # 動態更新處理狀態
                progress_bar.progress((i + 1) / total_tasks)
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
            progress_placeholder.text("所有資料已處理完成！")
            progress_bar.progress(1.0)



    with st.expander("2) 多股票指標比較(Rebase圖)", expanded=True):
        col1, col2 = st.columns(2)
        with col1:
            analysis_start_date = st.date_input("分析起始日", pd.to_datetime("2010-01-01"))
        with col2:
            analysis_end_date = st.date_input("分析結束日", today_date)

        all_stocks = [stock_id]
        if market_id.strip():
            all_stocks.append(market_id.strip())
        others = [x.strip() for x in other_ids_input.split(",") if x.strip()]
        all_stocks.extend(others)

        cum_df = pd.DataFrame()
        for sid in all_stocks:
            csvf = os.path.join(DL_dir, sid, f"{sid}_price.csv")
            if not os.path.exists(csvf):
                st.warning(f"{sid}_price.csv 不存在，請先下載")
                continue
            try:
                dfp = pd.read_csv(csvf, parse_dates=["Date"], index_col="Date")
                dfp.index = dfp.index.tz_localize(None)
                subp = dfp.loc[analysis_start_date:analysis_end_date].copy()
                if subp.empty:
                    st.warning(f"{sid} 在此區間無資料")
                    continue
                subp["pct"] = subp["Close"].pct_change().fillna(0)
                cprod = (1 + subp["pct"]).cumprod()
                basev = cprod.iloc[0]
                subp["rebase"] = cprod / basev
                cum_df[sid] = subp["rebase"]
            except Exception as e:
                st.error(f"處理 {sid} 發生錯誤: {e}")

        if not cum_df.empty:
            fig = px.line(cum_df, x=cum_df.index, y=cum_df.columns,
                          title="多股票累積報酬(Rebase)比較",
                          labels={"value": "累積報酬","variable": "股票"})
            fig.update_layout(legend=dict(orientation="h"))
            st.plotly_chart(fig, use_container_width=True)

    with st.expander("3) 多期(3/5/10/15/20年) 指標", expanded=False):
        as_of_dt = pd.Timestamp(analysis_end_date)
        tnxf = os.path.join(DL_dir, "^TNX", "^TNX_price.csv")
        daily_rf = 0.0
        yearly_rf  = 0.01
        if os.path.exists(tnxf):
            try:
                dftnx = pd.read_csv(tnxf, parse_dates=["Date"], index_col="Date")
                dftnx.index = dftnx.index.tz_localize(None)
                daily_rf, yearly_rf = get_risk_free_rate(dftnx.loc[:as_of_dt])
            except:
                st.warning("解析 ^TNX_price.csv 失敗, rf=0")

        use_price_type = st.radio("計算使用", ["Close","Adj Close"], index=0)

        all_syms = [stock_id]
        if market_id.strip():
            all_syms.append(market_id.strip())
        all_syms.extend(others)

        market_df = pd.DataFrame()
        if market_id.strip():
            mkt_csv = os.path.join(DL_dir, market_id.strip(), f"{market_id.strip()}_price.csv")
            if os.path.exists(mkt_csv):
                mdf = pd.read_csv(mkt_csv, parse_dates=["Date"], index_col="Date")
                mdf.index = mdf.index.tz_localize(None)
                market_df = mdf

        from investment_indicators import calc_multiple_period_metrics
        multi_data = []
        for sid in all_syms:
            cf = os.path.join(DL_dir, sid, f"{sid}_price.csv")
            if not os.path.exists(cf):
                continue
            dfp = pd.read_csv(cf, parse_dates=["Date"], index_col="Date")
            dfp.index = dfp.index.tz_localize(None)
            subp = dfp.loc[:as_of_dt]
            if subp.empty:
                continue

            df_metrics = calc_multiple_period_metrics(
                stock_df=subp,
                as_of_date=as_of_dt,
                years_list=[3,5,10,15,20],
                market_df=market_df,
                daily_rf_return=daily_rf,
                use_adj_close=(use_price_type=="Adj Close"),
                freq_for_reg="W",       # 月度回歸
                rf_annual_rate=yearly_rf 
            )
            df_metrics["股票"] = sid
            multi_data.append(df_metrics)

        if not multi_data:
            st.info("無可顯示之多期指標")
            return

        # 合併
        merged_df = pd.concat(multi_data, ignore_index=True)
        # pivot:  index="股票", columns="Years"
        pivoted = merged_df.pivot(index="股票", columns="Years")
        # pivoted 將有多層欄位: ( 指標 , 年 ) 
        # st.write(pivoted)  # Debug

        # === 表1
        st.subheader("回報及風險表現")
        st.markdown("**Sharpe**：衡量每單位風險帶來的回報，數值越高代表回報效率越高", help="適合挑選夏普比率高的資產，因為這些資產能穩定獲利，值得投資")
        st.markdown("**Sortino**：專注於下行風險，僅考量虧損風險而忽略收益，數值越高代表虧損風險越小", help="適合降低虧損風險，挑選Sortino比率高的資產更能控制風險")
        st.markdown("**Alpha**：評估投資是否跑贏市場基準，正值表示超額回報", help="強調超額回報，但不考慮風險或波動")
        st.markdown("**Beta**：衡量資產的波動性與市場的聯動性數值=1時與市場同步，>1時波動性更高，<1時則更穩定", help="用來判斷資產相對於市場的波動程度")



        try:
            df_sh = pivoted[["Sharpe","Sortino","Alpha","Beta"]]
            # df_sh 會是一個 2-level columns: top=[Sharpe, Sortino], sub=3,5,10,15,20
            st.dataframe(df_sh.style.format("{:.2f}"))
        except KeyError:
            st.write("無資料")


        # === 表2
        st.subheader("資產表現")
        st.markdown("**最大回撤**：歷史上資產的最大虧損幅度，數值越大風險越高", help="選擇最大回撤小的資產，可以減輕虧損時的心理壓力")
        st.markdown("**年化波動率**：衡量資產價格的波動幅度，數值越高代表波動越大，風險越高", help="波動率低的資產更適合追求穩定回報的投資者")
        st.markdown("**定期定額年化報酬**：模擬每月固定金額投資的年化收益率，數值越高越有吸引力", help="可用於判斷資產是否適合採用定期定額的投資方式")
        st.markdown("**年平均報酬**：每年的平均收益率，最直觀的衡量回報表現的標準", help="年平均報酬越高，資產的投資吸引力越強")
        try:
            df_mvol = pivoted[["MDD","AnnualVol","DCA_IRR","AnnualReturn"]]
            # MDD,AnnualVol 都是 0.xx => 轉百分比
            st.dataframe(
                df_mvol.style.format("{:.1%}")
            )
        except KeyError:
            st.write("無資料")


def main():
    page_basic_info()

if __name__ == "__main__":
    main()
