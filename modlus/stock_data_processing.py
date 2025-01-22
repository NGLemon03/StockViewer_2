# stock_data_processing.py
import os
import re
import pandas as pd
import numpy as np
import yfinance as yf

from modlus.config import DL_DIR
from modlus.fetch_stock_list import urls_and_filenames

file_data_cache = {}
suffix_cache = {}

def load_and_process_file(filepath):
    """讀取上市/上櫃/興櫃清單 CSV，拆解出代號和名稱"""
    df = pd.read_csv(filepath, encoding='utf_8_sig', header=None, skiprows=1)
    df.columns = ['有價證券代號及名稱', 'ISIN', '上市日', '市場別', '產業別', 'CFICode', '備註']
    df[['Code', 'Name']] = df['有價證券代號及名稱'].str.split('　', expand=True)
    return df

def ensure_file_data_cache(filepath):
    """確保 file_data_cache 裏有上市/上櫃/興櫃的資料表"""
    if filepath not in file_data_cache:
        file_data_cache[filepath] = load_and_process_file(filepath)
    return file_data_cache[filepath]

def determine_suffix(code):
    """根據 stock code 判斷 .TW 或 .TWO"""
    if code in suffix_cache:
        return suffix_cache[code]

    for filepath in urls_and_filenames.values():
        df = ensure_file_data_cache(filepath)
        if code in df['Code'].values:
            suffix = '.TW' if 'TWSE' in filepath else '.TWO'
            suffix_cache[code] = suffix
            return suffix

    # 如果完全沒找到(可能是興櫃或國外標的)，給空字串
    suffix_cache[code] = ''
    return ''

def download_data(stock_list):
    """
    檢查本地是否已有 CSV，若無則從 yfinance 下載。
    回傳 DataFrame，columns = 每個標的；index = 日期
    """
    stock_data = {}
    for stock in stock_list:
        # 若是特殊符號開頭(指數)或僅字母(ETF代號?)，直接使用該符號
        if stock.startswith('^') or re.match(r'^[A-Za-z]{3,4}$', stock):
            stock_code = stock
            csv_path = os.path.join(DL_DIR, f"{stock_code}.csv")
            try:
                if os.path.exists(csv_path):
                    df = pd.read_csv(csv_path, index_col=0, parse_dates=True)
                    stock_data[stock] = df['Adj Close']
                else:
                    df = yf.download(stock_code, period="15y")
                    if not df.empty:
                        df['Adj Close'].to_csv(csv_path)
                        stock_data[stock] = df['Adj Close']
                    else:
                        print(f"無數據: {stock_code}, 跳過...")
            except Exception as e:
                print(f"下載 {stock_code} 時發生錯誤: {e}")
            continue

        # 判斷是否需要補後綴
        suffix = determine_suffix(stock)
        if not suffix:
            print(f"{stock} 無法判斷後綴，跳過...")
            continue

        stock_code = f"{stock}{suffix}"
        csv_path = os.path.join(DL_DIR, f"{stock_code}.csv")

        try:
            if os.path.exists(csv_path):
                df = pd.read_csv(csv_path, index_col=0, parse_dates=True)
                stock_data[stock] = df['Adj Close']
            else:
                df = yf.download(stock_code, period="15y")
                if not df.empty:
                    df['Adj Close'].to_csv(csv_path)
                    stock_data[stock] = df['Adj Close']
                else:
                    print(f"無數據: {stock_code}, 跳過...")
        except Exception as e:
            print(f"下載 {stock_code} 時發生錯誤: {e}")

    return pd.DataFrame(stock_data)
