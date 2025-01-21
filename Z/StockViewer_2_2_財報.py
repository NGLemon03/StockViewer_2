# pages/2_財報.py
import streamlit as st
import pandas as pd
import numpy as np
from config import work_dir


# 可引用您的抓財報模組
# from financial_statements_fetcher import (
#     download_stock_price,
#     save_html,
#     parse_html_with_multi_layer_headers,
#     ...
# )

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

page_financials()