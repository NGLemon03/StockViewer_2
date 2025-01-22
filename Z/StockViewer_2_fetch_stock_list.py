# fetch_stock_list.py
import os
import requests
import pandas as pd
from bs4 import BeautifulSoup

from StockViewer_2.modlus.config import list_dir  # 引用 config.py 中的路徑

urls_and_filenames = {
    'https://isin.twse.com.tw/isin/C_public.jsp?strMode=2': os.path.join(list_dir, 'TWSE.csv'),
    'https://isin.twse.com.tw/isin/C_public.jsp?strMode=4': os.path.join(list_dir, 'OTC.csv'),
    'https://isin.twse.com.tw/isin/C_public.jsp?strMode=5': os.path.join(list_dir, 'emerging.csv')
}

def fetch_stock_list(url):
    """從指定網址抓取股票列表（HTML 解析）"""
    response = requests.get(url)
    soup = BeautifulSoup(response.content, 'html.parser')
    table = soup.select_one('body > table.h4')
    return pd.read_html(str(table))[0]

def save_stock_list(url, filename):
    """將抓取到的股票列表儲存成 CSV（若已存在則略過）"""
    if not os.path.exists(filename):
        df = fetch_stock_list(url)
        df.to_csv(filename, index=False, encoding='utf_8_sig')
        print(f"已儲存: {filename}")
    else:
        print(f"檔案已存在: {filename}")

def get_stock_lists():
    """統一抓取上市/上櫃/興櫃資料"""
    for url, filename in urls_and_filenames.items():
        save_stock_list(url, filename)
