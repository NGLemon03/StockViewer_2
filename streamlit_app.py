import streamlit as st
import pandas as pd
import plotly.express as px
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

from config import work_dir, output_dir
from fetch_stock_list import get_stock_lists
from stock_data_processing import download_data
from investment_indicators import (
    calculate_irr_from_prices,
    calculate_sharpe,
    calculate_max_drawdown,
    calculate_annualized_return
)
from bear_market_analysis import analyze_bear_market

# 這裡引入 financial_statements_fetcher 的主要功能，用於呼叫爬蟲下載
from financial_statements_fetcher import (
    driver,
    download_stock_price,
    parse_html_with_multi_layer_headers,
    parse_equity_distribution,
    save_html,
    save_html_nolink,
    scrape_financial_data,
    close_driver
)

# ====== 1. 個股基本資訊頁面 ======
def page_basic_info():
    st.header("個股基本資訊")
    st.write("在此輸入欲查詢的股票代碼，並可選擇是否執行爬蟲下載財報或股價。")

    stock_id = st.text_input("輸入股票代號", value="2412")
    base_dir = st.text_input("存放資料的目錄", value=f"{work_dir}/{stock_id}")
    start_date = st.date_input("股價下載開始日期", value=pd.to_datetime("2000-01-01"))
    end_date = st.date_input("股價下載結束日期", value=pd.to_datetime("2025-12-31"))

    if st.button("下載/更新 財報 & 股價"):
        st.write("開始下載或更新資料...")
        st.write("若本地已有檔案且不強制更新，會自動略過下載。")

        # 建立目錄
        import os
        os.makedirs(base_dir, exist_ok=True)

        # 下載股價
        download_stock_price(
            driver=driver,
            stockID=stock_id,
            base_dir=base_dir,
            start_date=str(start_date),
            end_date=str(end_date)
        )

        # 以下示意：依需求抓取主要財報(若想抓更多，請自行新增)
        scrape_financial_data(
            driver=driver,
            report_type='BS',  # 資產負債表
            start_year=2024, start_quarter=2,
            end_year=2020, end_quarter=1,
            stockID=stock_id,
            base_dir=base_dir
        )
        scrape_financial_data(
            driver=driver,
            report_type='IS',  # 損益表
            start_year=2024, start_quarter=2,
            end_year=2020, end_quarter=1,
            stockID=stock_id,
            base_dir=base_dir
        )
        scrape_financial_data(
            driver=driver,
            report_type='CF',  # 現金流量表
            start_year=2024, start_quarter=2,
            end_year=2020, end_quarter=1,
            stockID=stock_id,
            base_dir=base_dir
        )

        st.success("下載流程完成(或使用既有檔案)。")
        st.info("請至資料夾中檢視 CSV 或 HTML 檔案。")

    st.write("若需關閉瀏覽器，請於執行完畢後按下按鈕。")
    if st.button("關閉瀏覽器"):
        close_driver()
        st.write("瀏覽器已關閉。")

# ====== 2. 財報下載/預覽 ======
def page_financials():
    st.header("財報下載/預覽")
    st.write("此處可擴充讀取個股已下載的財報 CSV，並做進一步解析。")

    # 亦可加上資料路徑輸入
    stock_id = st.text_input("輸入股票代號以讀取其財報資料", value="2412")
    base_dir = f"{work_dir}/{stock_id}"
    st.write(f"目前預設讀取目錄：{base_dir}")

    # 範例：顯示可能存在的財報檔案列表
    import os
    files_in_dir = os.listdir(base_dir) if os.path.exists(base_dir) else []
    csv_files = [f for f in files_in_dir if f.endswith(".csv")]
    if csv_files:
        selected_csv = st.selectbox("選擇要預覽的 CSV 檔", options=csv_files)
        if selected_csv:
            csv_path = os.path.join(base_dir, selected_csv)
            df = pd.read_csv(csv_path)
            st.write(f"檔案：{selected_csv}")
            st.dataframe(df.head(50))
    else:
        st.warning("該資料夾尚無任何 CSV 檔。請先至『個股基本資訊』頁面下載。")

# ====== 3. 回測頁面 ======
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

# ====== 4. 多標的比較 ======
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

# ====== 5. 空頭分析 ======
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

# ====== 主程式 ======
def streamlit_app():
    st.title("整合式多分頁介面")
    st.sidebar.title("功能導覽")
    page = st.sidebar.radio("選擇分頁", [
        "個股基本資訊", 
        "財報下載/預覽", 
        "回測", 
        "多標的比較", 
        "空頭分析"
    ])

    if page == "個股基本資訊":
        page_basic_info()
    elif page == "財報下載/預覽":
        page_financials()
    elif page == "回測":
        page_backtest()
    elif page == "多標的比較":
        page_compare()
    elif page == "空頭分析":
        page_bear()
