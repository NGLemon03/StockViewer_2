import os
import re
import time
import pandas as pd
import yfinance as yf

from modlus.fetch_stock_list import urls_and_filenames

# -------------------------------------------------------
# 全域參數 (可自行調整或改成參數)
# -------------------------------------------------------
force_update = False

# -------------------------------------------------------
# 內部快取(避免重覆讀取檔案)
# -------------------------------------------------------
file_data_cache = {}

def load_and_process_file(filepath):
    """讀取 TWSE.csv / OTC.csv / emerging.csv，拆解出代號和名稱"""
    df = pd.read_csv(filepath, encoding='utf_8_sig', header=None, skiprows=1)
    df.columns = ['有價證券代號及名稱', 'ISIN', '上市日', '市場別', '產業別', 'CFICode', '備註']
    df[['Code', 'Name']] = df['有價證券代號及名稱'].str.split('　', expand=True)
    return df

def ensure_file_data_cache(filepath):
    """確保 file_data_cache 裏有上市/上櫃/興櫃的資料表"""
    if filepath not in file_data_cache:
        if os.path.exists(filepath):
            file_data_cache[filepath] = load_and_process_file(filepath)
        else:
            file_data_cache[filepath] = pd.DataFrame(columns=["Code", "Name"])
    return file_data_cache[filepath]

def is_index_or_foreign(symbol: str) -> bool:
    """
    判斷是否為指數或外國標的:
      - 以 ^ 開頭 (如 ^TWII, ^GSPC)
      - 或者為 3~4 碼英文字 (如 QQQ, SPY, TSLA)
    """
    if symbol.startswith('^'):
        return True
    if re.match(r'^[A-Za-z]{3,4}$', symbol):
        return True
    return False

def determine_suffix(code: str) -> str:
    """
    從上市/上櫃/興櫃清單中尋找該 code 是否存在:
      - 若在 TWSE -> .TW
      - 若在 OTC  -> .TWO
      - 若都找不到，回傳空字串
    """
    for filepath in urls_and_filenames.values():
        df = ensure_file_data_cache(filepath)
        if code in df['Code'].values:
            if "TWSE" in filepath:
                return ".TW"
            elif "OTC" in filepath:
                return ".TWO"
            # 若是 emerging.csv 就另行處理或直接跳過
    return ""

def calculate_adjusted_close(df: pd.DataFrame) -> pd.DataFrame:
    """
    根據昨收價、現金股利和股票股利計算調整後的昨收價。
    這裡只是示範，可以依實際需求調整公式。
    """
    if 'Close' not in df.columns or 'Dividends' not in df.columns or 'Stock Splits' not in df.columns:
        print("缺少必要的欄位 (Close, Dividends, Stock Splits)，無法計算調整後昨收價。")
        df['Adjusted Close'] = df.get('Close', pd.Series(index=df.index, dtype='float'))
        return df

    df['Adj Close'] = (df['Close'] - df['Dividends']) / (1 + df['Stock Splits'] / 10)
    return df

def download_stock_price(stockID, base_dir, start_date, end_date, interval='1d'):
    """
    下載股價到 {base_dir}/{stockID}_price.csv
      - 若是指數/外國 (以 ^ 開頭, 或3~4字母) -> 不加後綴
      - 否則 (台股) -> 檢查上市 / 上櫃，加 .TW / .TWO
    """
    price_file_path = os.path.join(base_dir, f"{stockID}_price.csv")
    if os.path.exists(price_file_path) and not force_update:
        print(f"{price_file_path} 已存在，使用已有文件，跳過下載股價數據。")
        return

    os.makedirs(base_dir, exist_ok=True)

    # 判斷是否為指數/外國
    if is_index_or_foreign(stockID):
        ticker_symbol = stockID
    else:
        suffix = determine_suffix(stockID)
        if suffix == "":
            print(f"無法判斷 {stockID} 是否上市/上櫃，或 yfinance 無法提供資料，跳過下載。")
            return
        ticker_symbol = f"{stockID}{suffix}"

    print(f"嘗試下載股價資料，代號：{ticker_symbol}")
    ticker = yf.Ticker(ticker_symbol)
    stock_data = ticker.history(start=start_date, end=end_date, interval=interval)
    time.sleep(1)

    if stock_data.empty:
        print(f"無法下載 {stockID} 的股價資料({ticker_symbol})。")
        return
    stock_data.index = stock_data.index.tz_convert('Asia/Taipei').tz_localize(None)

    actual_start = stock_data.index[0]
    actual_end = stock_data.index[-1]
    print(f"成功下載資料，日期範圍：{actual_start} 至 {actual_end}")

    # 計算 Adjusted Close
    stock_data = calculate_adjusted_close(stock_data)
    stock_data.to_csv(price_file_path, encoding='utf-8-sig')
    print(f"{stockID} 股價數據已保存到 {price_file_path}")
