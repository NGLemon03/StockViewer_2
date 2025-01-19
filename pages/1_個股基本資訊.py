# pages/1_個股基本資訊.py
import streamlit as st
import pandas as pd
import numpy as np
from config import work_dir
from fetch_stock_list import get_stock_lists
from stock_data_processing import download_data



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

def page_basic_info():
    st.header("個股基本資訊")
    st.write("在此輸入欲查詢的股票代碼，並可選擇是否執行爬蟲下載財報或股價。")

    stock_id = st.text_input("輸入股票代號", value="2412")
    base_dir = st.text_input("存放資料的目錄", value=f"{work_dir}/{stock_id}")
    start_date = st.date_input("股價下載開始日期", value=pd.to_datetime("2000-01-01"))
    end_date = st.date_input("股價下載結束日期", value=pd.to_datetime("2025-12-31"))
    start_year = st.number_input("財報下載起始年份", value=2024, step=1)
    start_quarter = st.selectbox("財報下載起始季度", options=[1, 2, 3, 4], index=1)
    end_year = st.number_input("財報下載結束年份", value=2020, step=1)
    end_quarter = st.selectbox("財報下載結束季度", options=[1, 2, 3, 4], index=0)    

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

        # 下載財報
        report_types = {"BS": "資產負債表", "IS": "損益表", "CF": "現金流量表"}
        for report_type, report_name in report_types.items():
            st.write(f"下載 {report_name}...")
            scrape_financial_data(
                driver=driver,
                report_type=report_type,
                start_year=start_year,
                start_quarter=start_quarter,
                end_year=end_year,
                end_quarter=end_quarter,
                stockID=stock_id,
                base_dir=base_dir
            )

        st.success("下載流程完成(或使用既有檔案)。")
        st.info("請至資料夾中檢視 CSV 或 HTML 檔案。")

    st.write("若需關閉瀏覽器，請於執行完畢後按下按鈕。")
    if st.button("關閉瀏覽器"):
        close_driver()
        st.write("瀏覽器已關閉。")
    st.header("個股基本資訊")
    st.write("這裡可顯示：")
    st.write("- 股票代碼與名稱")
    st.write("- 所屬產業、上市櫃別等")
    st.write("- 近期股價或其他基本面資訊...")



# Streamlit 會自動執行此檔案的所有頂層程式碼
# 但為了保持程式乾淨，最好將執行邏輯放在函式中，再在最底部呼叫。
page_basic_info()
