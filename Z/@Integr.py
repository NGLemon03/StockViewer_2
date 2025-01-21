# ---------- Start of bear_market_analysis.py ----------
# bear_market_analysis.py
import pandas as pd
import numpy as np
from investment_indicators import calculate_max_drawdown

def analyze_bear_market(stock_data, start_date, end_date):
    """分析指定空頭期間的最大回撤、波動率、總收益率"""
    bear_data = stock_data[(stock_data.index >= start_date) & (stock_data.index <= end_date)].dropna(how='all')
    if bear_data.empty:
        return pd.DataFrame()

    def _volatility(series):
        returns = series.pct_change().dropna()
        return returns.std() * np.sqrt(252)

    results = {}
    for stock in stock_data.columns:
        prices = bear_data[stock].dropna()
        if len(prices) < 2:
            continue
        mdd = calculate_max_drawdown(prices)
        vol = _volatility(prices)
        total_return = (prices.iloc[-1] / prices.iloc[0]) - 1
        results[stock] = {
            "最大回撤": mdd,
            "波動率": vol,
            "總收益率": total_return
        }

    return pd.DataFrame(results).T

# ---------- End of bear_market_analysis.py ----------

# ---------- Start of config.py ----------
# config.py
import os
import matplotlib

# 設定工作資料夾（以執行該檔案所在路徑為基準）
work_dir = os.path.dirname(os.path.abspath(__file__))
os.chdir(work_dir)

# 建議使用者安裝「思源黑體」或其他中文字體，這裡示範使用微軟正黑體
matplotlib.rcParams['font.family'] = 'Microsoft YaHei'
matplotlib.rcParams['font.size'] = 10

# 資料夾結構
list_dir = os.path.join(work_dir, "list")
stock_dir = os.path.join(work_dir, "stock")
output_dir = os.path.join(work_dir, "output")
heatmap_dir = os.path.join(output_dir, "heatmaps")
DL_dir =  os.path.join(work_dir, "DL")

os.makedirs(list_dir, exist_ok=True)
os.makedirs(stock_dir, exist_ok=True)
os.makedirs(output_dir, exist_ok=True)
os.makedirs(heatmap_dir, exist_ok=True)
os.makedirs(DL_dir, exist_ok=True)

# ---------- End of config.py ----------

# ---------- Start of DL_Y.py ----------
import os
import re
import time
import pandas as pd
import yfinance as yf

from fetch_stock_list import urls_and_filenames

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

    df['Adjusted Close'] = (df['Close'] - df['Dividends']) / (1 + df['Stock Splits'] / 10)
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

# ---------- End of DL_Y.py ----------

# ---------- Start of fetch_stock_list.py ----------
# fetch_stock_list.py
import os
import requests
import pandas as pd
from bs4 import BeautifulSoup

from config import list_dir  # 引用 config.py 中的路徑

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

# ---------- End of fetch_stock_list.py ----------

# ---------- Start of financial_statements_fetcher.py ----------
import os
import time
import random
import pandas as pd
import yfinance as yf
from io import StringIO
from bs4 import BeautifulSoup
import re
# Selenium 相關
from selenium import webdriver
from selenium_stealth import stealth
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.common.by import By
from webdriver_manager.chrome import ChromeDriverManager
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from fetch_stock_list import urls_and_filenames  # {'上市網址': 'TWSE.csv', '上櫃網址': 'OTC.csv', ...}

# --------------------------------------------------------------------
# 以下為原先 good_info.py 的設定，改名為 financial_statements_fetcher.py
# 主要差異：改寫成函式 + 可以由外部指定 stockID、base_dir、start_date等參數
# 保留 force_update 與 fetch_delay 為全域變數（預設 False / 10）
# --------------------------------------------------------------------

# 全域參數(如要在前台改成可動態切換，可以再做擴充)
force_update = False   # 設置為 True 時，即使有檔案也會重新抓取
fetch_delay = 10       # 每次抓取等待秒數

# -------------------------------------------------------
# 內部快取(避免重覆讀取檔案)
# -------------------------------------------------------
file_data_cache = {}

def load_and_process_file(filepath):
    """讀取 TWSE.csv / OTC.csv / emerging.csv，拆解出代號和名稱。"""
    df = pd.read_csv(filepath, encoding='utf_8_sig', header=None, skiprows=1)
    df.columns = ['有價證券代號及名稱', 'ISIN', '上市日', '市場別', '產業別', 'CFICode', '備註']
    # '有價證券代號及名稱' 例如 "2330　台積電"
    # 用全形空白分割
    df[['Code', 'Name']] = df['有價證券代號及名稱'].str.split('　', n=1, expand=True)
    return df

def ensure_file_data_cache(filepath):
    """確保 file_data_cache 裏有上市/上櫃/興櫃的資料表"""
    if filepath not in file_data_cache:
        if os.path.exists(filepath):
            file_data_cache[filepath] = load_and_process_file(filepath)
        else:
            # 可能使用者尚未fetch，或是該檔案不在本機
            file_data_cache[filepath] = pd.DataFrame(columns=["Code", "Name"])
    return file_data_cache[filepath]

def is_index_or_foreign(symbol: str) -> bool:
    """
    判斷是否為指數或外國標的:
      - 以 ^ 開頭，如 ^TWII, ^GSPC, ^IRX
      - 或者為 3~4 碼英文字 (如 QQQ, SPY, TSLA(4碼?), etc.)
    """
    if symbol.startswith('^'):
        return True
    if re.match(r'^[A-Za-z]{3,4}$', symbol):
        return True
    return False

def determine_suffix(code: str) -> str:
    """
    從上市/上櫃/興櫃清單中尋找該 code 是否存在:
      - TWSE -> .TW
      - OTC  -> .TWO
      - emerging -> 也可能是興櫃，但實務上多數興櫃在 yfinance 不一定有資料
    若都找不到，回傳空字串。
    """
    for filepath in urls_and_filenames.values():
        df = ensure_file_data_cache(filepath)
        # 檢查 code 是否在 df['Code'] 裏
        if code in df['Code'].values:
            if "TWSE" in filepath:  # filepath 例如 ".../TWSE.csv"
                return ".TW"
            elif "OTC" in filepath: # ".../OTC.csv"
                return ".TWO"
            # 若是 emerging.csv 就按需求自行決定（這裡示範回傳空字串）
    return ""


# -------------------------------------------------------
# 1. 啟動 Selenium，連接已開啟的 Chrome (port=9222)
# -------------------------------------------------------
options = webdriver.ChromeOptions()
options.add_experimental_option("debuggerAddress", "127.0.0.1:9222")

driver = webdriver.Chrome(options=options)

stealth(
    driver,
    languages=["zh-TW", "zh"],
    vendor="Google Inc.",
    platform="Win32",
    webgl_vendor="Intel Inc.",
    renderer="Intel Iris OpenGL Engine",
    fix_hairline=True,
)

# -------------------------------------------------------
# 2. 通用輔助函式
# -------------------------------------------------------

def close_driver():
    """關閉瀏覽器 (供前台呼叫)."""
    try:
        driver.quit()
        print("瀏覽器已關閉")
    except:
        pass

def random_sleep(min_seconds=5, max_seconds=10):
    sleep_time = random.uniform(min_seconds, max_seconds)
    time.sleep(sleep_time)

def save_html_nolink(driver, base_dir, selector, filename, delay=5):
    """
    不重新載入網址情況下，抓取當前 driver 頁面指定 selector 的 HTML 並存檔.
    """
    file_path = os.path.join(base_dir, filename)
    if os.path.exists(file_path) and not force_update:
        print(f"{file_path} 已存在，使用已有文件，跳過等待時間。")
        return

    print(f"等待頁面加載 {delay} 秒...")
    time.sleep(delay)

    try:
        print(f"嘗試獲取元素: {selector}")
        element = driver.find_element(By.CSS_SELECTOR, selector)
        with open(file_path, 'w', encoding='utf-8') as file:
            file.write(element.get_attribute('outerHTML'))
        print(f"成功保存 {file_path}！")
    except Exception as e:
        print(f"錯誤: 無法保存 {file_path} - {e}")

    print(f"額外等待 {delay} 秒...")
    time.sleep(delay)


def save_html(driver, base_dir, url, selector, filename, delay=5):
    """
    前往指定網址，等待後抓取 selector 的內容保存.
    """
    file_path = os.path.join(base_dir, filename)
    if os.path.exists(file_path) and not force_update:
        print(f"{file_path} 已存在，使用已有文件，跳過等待時間。")
        return

    print(f"前往網址: {url}")
    driver.get(url)
    print(f"等待頁面加載 {delay} 秒...")
    time.sleep(delay)

    try:
        print(f"嘗試獲取元素: {selector}")
        element = driver.find_element(By.CSS_SELECTOR, selector)
        with open(file_path, 'w', encoding='utf-8') as file:
            file.write(element.get_attribute('outerHTML'))
        print(f"成功保存 {file_path}！")
    except Exception as e:
        print(f"錯誤: 無法保存 {file_path} - {e}")

    print(f"額外等待 {delay} 秒...")
    time.sleep(delay)

def download_stock_price(driver, stockID, base_dir, start_date, end_date, interval='1d'):
    """
    使用 yfinance 下載股價 (stockID TW/TWO)，輸出到 base_dir 下 {stockID}_price.csv
    """
    price_file_path = os.path.join(base_dir, f"{stockID}_price.csv")
    if os.path.exists(price_file_path) and not force_update:
        print(f"{price_file_path} 已存在，使用已有文件，跳過下載股價數據。")
        return

    # 1) 判斷是否為指數/外國
    if is_index_or_foreign(stockID):
        ticker_symbol = stockID
    else:
        # 2) 台股，檢查上市/上櫃
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
        print(f"無法下載 {stockID} 的股價資料（{ticker_symbol}）。")
        return
    stock_data.index = stock_data.index.tz_convert('Asia/Taipei').tz_localize(None)

    actual_start = stock_data.index[0]
    actual_end = stock_data.index[-1]
    print(f"成功下載資料，日期範圍：{actual_start} 至 {actual_end}")

    if stock_data is None or stock_data.empty:
        print(f"無法下載 {stockID} 的股價資料。")
        return

    def calculate_adjusted_close(df):
        """
        根據昨收價、現金股利和股票股利計算調整後的昨收價。
        :param df: 包含昨收價、現金股利、股票股利的 DataFrame
        """
        if 'Close' not in df.columns or 'Dividends' not in df.columns or 'Stock Splits' not in df.columns:
            print("缺少必要的欄位 (Close, Dividends, Stock Splits)，無法計算調整後昨收價。")
            return df

        # 根據公式計算昨收價（新）
        df['Adjusted Close'] = (df['Close'] - df['Dividends']) / (1 + df['Stock Splits'] / 10)
        return df

    stock_data = calculate_adjusted_close(stock_data)
    stock_data.to_csv(price_file_path, encoding='utf-8-sig')
    print(f"{stockID} 股價數據已保存到 {price_file_path}")

def parse_html_with_multi_layer_headers(base_dir, file_path, output_csv, table_id=None):
    """
    解析帶多層表頭的 HTML 表格, 輸出 csv
    """
    if not os.path.exists(file_path):
        print(f"在 parse_html_with_multi_layer_headers: 找不到 {file_path}, 無法解析.")
        return

    with open(file_path, 'r', encoding='utf-8') as file:
        soup = BeautifulSoup(file, 'html.parser')

    if table_id:
        table = soup.find('table', {'id': table_id})
    else:
        table = soup.find('table')
    if table is None:
        print(f"在 {file_path} 中未找到 id = {table_id} 的 <table>.")
        return

    rows = table.find_all('tr')
    # 移除 DummyTHead
    rows = [row for row in rows if 'DummyTHead' not in row.get('class', [])]

    # 收集表頭
    header_rows = []
    data_start_row = 0
    for i, row in enumerate(rows):
        ths = row.find_all('th')
        if ths:
            header_rows.append(row)
        else:
            data_start_row = i
            break

    if not header_rows:
        print(f"在 {file_path} 中未找到表頭(th).")
        return

    total_columns = 0
    for row in header_rows:
        cols = sum(int(th.get('colspan', 1)) for th in row.find_all(['th', 'td']))
        total_columns = max(total_columns, cols)

    num_header_rows = len(header_rows)
    header_matrix = [['' for _ in range(total_columns)] for _ in range(num_header_rows)]
    occupied = {}

    # 建立多層表頭
    for row_idx, row in enumerate(header_rows):
        col_idx = 0
        cells = row.find_all(['th', 'td'])
        for cell in cells:
            while occupied.get((row_idx, col_idx), False):
                col_idx += 1
            colspan = int(cell.get('colspan', 1))
            rowspan = int(cell.get('rowspan', 1))
            text = cell.get_text(strip=True)
            for i2 in range(rowspan):
                for j2 in range(colspan):
                    r = row_idx + i2
                    c = col_idx + j2
                    header_matrix[r][c] = text
                    occupied[(r, c)] = True
            col_idx += colspan

    # 合併表頭
    column_names = []
    for col in range(total_columns):
        headers = [header_matrix[row][col] for row in range(num_header_rows)]
        headers = [h for h in headers if h]
        column_name = '_'.join(headers)
        column_names.append(column_name)

    data_rows = rows[data_start_row:]
    data = []
    for row in data_rows:
        if row.find('th'):
            continue
        tds = row.find_all(['td', 'th'])
        data_row = [td.get_text(strip=True) for td in tds]
        if len(data_row) < total_columns:
            data_row.extend([''] * (total_columns - len(data_row)))
        elif len(data_row) > total_columns:
            data_row = data_row[:total_columns]
        data.append(data_row)

    df = pd.DataFrame(data, columns=column_names)
    output_csv_path = os.path.join(base_dir, output_csv)
    df.to_csv(output_csv_path, index=False, encoding='utf-8-sig')
    print(f"成功將 {file_path} 轉換為 {output_csv_path}")

def parse_equity_distribution(base_dir, file_path, output_csv, table_id='tblDetail'):
    """
    專門解析股數分級(含多層表頭)
    """
    if not os.path.exists(file_path):
        print(f"在 parse_equity_distribution: 找不到 {file_path}")
        return

    with open(file_path, 'r', encoding='utf-8') as f:
        html_content = f.read()

    df_list = pd.read_html(html_content, attrs={'id': table_id}, header=[0,1])
    if not df_list:
        print(f"在 {file_path} 中未找到 table id={table_id}.")
        return

    df = df_list[0]

    def flatten_column(col):
        labels = [str(s).strip() for s in col if str(s) != 'nan' and 'Unnamed' not in str(s)]
        return '_'.join(labels)

    df.columns = [flatten_column(col) for col in df.columns.values]
    # 刪除可能重複表頭行
    first_col = df.columns[0]
    df = df[~df[first_col].str.contains('週別', na=False)]

    output_csv_path = os.path.join(base_dir, output_csv)
    df.to_csv(output_csv_path, index=False, encoding='utf-8-sig')
    print(f"成功將 {file_path} 解析為 {output_csv_path}")

def generate_quarter_list(start_year, start_quarter, end_year, end_quarter):
    """
    生成季度 (由大到小)
    """
    quarters = []
    year, quarter = start_year, start_quarter
    while (year > end_year) or (year == end_year and quarter >= end_quarter):
        quarters.append(f"{year}Q{quarter}")
        quarter -= 1
        if quarter == 0:
            quarter = 4
            year -= 1
    return quarters

def calculate_qry_times(needed_quarters, quarters_per_page):
    """
    Goodinfo一次可帶入7或10季，所以計算需要幾次請求
    """
    qry_times = []
    index = 0
    total_quarters = len(needed_quarters)
    while index < total_quarters:
        quarter_str = needed_quarters[index]  # ex: "2024Q2"
        y = quarter_str[:4]
        q = quarter_str[-1]
        qry_time = f"{y}{q}"
        qry_times.append(qry_time)
        index += quarters_per_page
    return qry_times

def scrape_financial_data(driver, report_type, start_year, start_quarter, end_year, end_quarter, stockID, base_dir):
    """
    抓取財務報表(資產負債表/損益表/現金流量表) 並輸出 csv
    """
    if report_type == 'BS':
        RPT_CAT = 'BS_M_QUAR'
        quarters_per_page = 7
        file_name = "Financial_BalanceSheet.csv"
    elif report_type == 'IS':
        RPT_CAT = 'IS_M_QUAR_ACC'
        quarters_per_page = 7
        file_name = "Financial_IncomeStatement.csv"
    elif report_type == 'CF':
        RPT_CAT = 'CF_M_QUAR_ACC'
        quarters_per_page = 10
        file_name = "Financial_CashFlow.csv"
    else:
        print("未知的報表類型。")
        return

    file_path = os.path.join(base_dir, file_name)
    if os.path.exists(file_path) and not force_update:
        print(f"{file_path} 已存在，使用已有文件。")
        return

    needed_quarters = generate_quarter_list(start_year, start_quarter, end_year, end_quarter)
    qry_times = calculate_qry_times(needed_quarters, quarters_per_page)

    data_list = []
    collected_quarters = set()

    for qry_time in qry_times:
        url = f"https://goodinfo.tw/tw/StockFinDetail.asp?RPT_CAT={RPT_CAT}&STOCK_ID={stockID}&QRY_TIME={qry_time}"
        print(f"正在訪問網址：{url}")
        driver.get(url)
        random_sleep(1, 3)

        # 等待表格
        try:
            WebDriverWait(driver, 10).until(
                EC.presence_of_element_located((By.ID, 'tblFinDetail'))
            )
            time.sleep(1)
        except Exception as e:
            print(f"錯誤：未能加載表格 - {e}")
            continue

        table_element = driver.find_element(By.ID, 'tblFinDetail')
        table_html = table_element.get_attribute('outerHTML')

        df_list = pd.read_html(StringIO(table_html), header=[0,1])
        df = df_list[0]

        # 扁平化列名
        df.columns = ['_'.join(col).strip() if col[1] else col[0].strip()
                      for col in df.columns.values]

        # 取得當前頁面季度
        page_quarters = []
        for col in df.columns[1:]:
            quarter = col.split('_')[0]
            if quarter not in page_quarters:
                page_quarters.append(quarter)

        # 轉成長格式
        for quarter in page_quarters:
            if quarter in needed_quarters and quarter not in collected_quarters:
                amount_cols = [c for c in df.columns if c.startswith(quarter) and '百分比' not in c]
                percent_cols = [c for c in df.columns if c.startswith(quarter) and '百分比' in c]
                for idx, row in df.iterrows():
                    item = row[df.columns[0]]
                    for col_idx, amount_col in enumerate(amount_cols):
                        amount_val = row[amount_col]
                        data_entry = {
                            '項目': item,
                            '季度': quarter,
                            '金額': amount_val
                        }
                        if percent_cols:
                            # 找對應百分比欄
                            try:
                                percent_val = row[percent_cols[col_idx]]
                            except:
                                percent_val = None
                            data_entry['百分比'] = percent_val
                        data_list.append(data_entry)
                collected_quarters.add(quarter)

        print(f"已收集季度: {collected_quarters}")

    if not data_list:
        print("未獲取到任何數據。")
        return

    result_df = pd.DataFrame(data_list)
    if '百分比' in result_df.columns:
        pivot_df = result_df.pivot_table(index='項目', columns='季度',
                                         values=['金額','百分比'],
                                         aggfunc='first')
        pivot_df.columns = [f"{col[1]}_{col[0]}" for col in pivot_df.columns]
    else:
        pivot_df = result_df.pivot_table(index='項目', columns='季度',
                                         values='金額', aggfunc='first')
        pivot_df.columns = pivot_df.columns.tolist()

    pivot_df.reset_index(inplace=True)

    # 順序排列
    needed_cols = []
    for q in needed_quarters:
        needed_cols.append(q)

    # 先建基本Ordered
    ordered_cols = ['項目']
    if '百分比' in result_df.columns:
        for q in needed_cols:
            col_amount = f"{q}_金額"
            col_percent = f"{q}_百分比"
            if col_amount in pivot_df.columns:
                ordered_cols.append(col_amount)
            if col_percent in pivot_df.columns:
                ordered_cols.append(col_percent)
    else:
        for q in needed_cols:
            if q in pivot_df.columns:
                ordered_cols.append(q)

    existing_cols = [c for c in ordered_cols if c in pivot_df.columns]
    pivot_df = pivot_df[existing_cols]

    pivot_df.to_csv(file_path, index=False, encoding='utf-8-sig')
    print(f"數據已保存到 '{file_path}'")

# -------------------------------------------------------
# 3. 在檔案結尾，如用命令行執行，可直接做測試 (非必需)
# -------------------------------------------------------
def fetch_all_data(stockID, base_dir, start_date, end_date,
                   start_year, start_quarter, end_year, end_quarter):
    """
    一次抓取:
    1. 股價 (price.csv)
    2. Goodinfo其他頁面: CashFlow, EPS(quarter/year), PER_PBR, MonthlyRevenue, Dividend, EquityDistribution
        - HTML + 解析出對應CSV
    3. 三大財報 (BS, IS, CF)

    請在前台呼叫這個函式，就能下載出所有與 good_info.py 同樣的 CSV.
    """

    # 1) 下載股價
    download_stock_price(
        driver=driver,
        stockID=stockID,
        base_dir=base_dir,
        start_date=start_date,
        end_date=end_date
    )

    # 2) 下載 Goodinfo 頁面 (CashFlow、EPS、PER、月營收、股利、股數分級)
    print("開始抓取現金流量數據...")
    CashFlow_url = f"https://goodinfo.tw/tw/StockCashFlow.asp?STOCK_ID={stockID}&PRICE_ADJ=F&SCROLL2Y=517&RPT_CAT=M_QUAR"
    save_html(driver, base_dir, CashFlow_url, "#divDetail", "CashFlow.html", fetch_delay)

    print("開始抓取 EPS(季度)...")
    eps_quar_url = f"https://goodinfo.tw/tw/StockBzPerformance.asp?STOCK_ID={stockID}&YEAR_PERIOD=9999&PRICE_ADJ=F&SCROLL2Y=519&RPT_CAT=M_QUAR"
    save_html(driver, base_dir, eps_quar_url, "#txtFinDetailData", "EPS_Quar.html", fetch_delay)

    print("開始抓取 EPS(年度)...")
    eps_year_url = f"https://goodinfo.tw/tw/StockBzPerformance.asp?STOCK_ID={stockID}&YEAR_PERIOD=9999&PRICE_ADJ=F&SCROLL2Y=519&RPT_CAT=M_YEAR"
    save_html(driver, base_dir, eps_year_url, "#txtFinDetailData", "EPS_Year.html", fetch_delay)

    print("開始抓取 PER/PBR ...")
    per_pbr_html_path = os.path.join(base_dir, "PER_PBR.html")
    if os.path.exists(per_pbr_html_path) and not force_update:
        print(f"{per_pbr_html_path} 已存在，使用已有文件，略過...")
    else:
        # 先載入EPS_quar那頁
        driver.get(eps_quar_url)
        time.sleep(fetch_delay)
        try:
            select_element_2 = WebDriverWait(driver, fetch_delay).until(
                EC.presence_of_element_located((By.CSS_SELECTOR, "#selSheet"))
            )
            select_element_2.send_keys("PER/PBR")
            print("PER/PBR 選項已選擇。")
            time.sleep(fetch_delay)
            save_html_nolink(driver, base_dir, "#txtFinDetailData", "PER_PBR.html", fetch_delay)
        except Exception as e:
            print(f"錯誤: 無法選擇 PER/PBR - {e}")

    print("開始抓取月營收...")
    revenue_url = f"https://goodinfo.tw/tw/ShowSaleMonChart.asp?STOCK_ID={stockID}"
    save_html(driver, base_dir, revenue_url, "#divDetail", "MonthlyRevenue.html", fetch_delay)

    print("開始抓取股利資料...")
    dividend_url = f"https://goodinfo.tw/tw/StockDividendPolicy.asp?STOCK_ID={stockID}"
    save_html(driver, base_dir, dividend_url, "#divDividendDetail", "Dividend.html", fetch_delay)

    print("開始抓取股數分級(查5年)...")
    equity_dist_html_path = os.path.join(base_dir, "EquityDistribution.html")
    if os.path.exists(equity_dist_html_path) and not force_update:
        print(f"{equity_dist_html_path} 已存在，使用已有文件，略過...")
    else:
        eq_url = f"https://goodinfo.tw/tw/EquityDistributionClassHis.asp?STOCK_ID={stockID}"
        driver.get(eq_url)
        time.sleep(fetch_delay)
        try:
            five_year_button = WebDriverWait(driver, 10).until(
                EC.element_to_be_clickable((By.CSS_SELECTOR, "input[value='查5年']"))
            )
            print("點擊 '查5年'...")
            five_year_button.click()
            WebDriverWait(driver, 20).until(
                EC.presence_of_element_located((By.CSS_SELECTOR, "#divDetail"))
            )
            time.sleep(fetch_delay)
            element = driver.find_element(By.CSS_SELECTOR, "#divDetail")
            with open(equity_dist_html_path, 'w', encoding='utf-8') as f:
                f.write(element.get_attribute('outerHTML'))
            print(f"成功保存股數分級 -> {equity_dist_html_path}")
        except Exception as e:
            print(f"錯誤: 無法點擊 '查5年' - {e}")

    # 2.1) 解析上述 HTML -> CSV
    mapping = {
        "EPS_Quar":          {"html": "EPS_Quar.html",          "csv": "EPS_Quar.csv",          "table_id": None},
        "EPS_Year":          {"html": "EPS_Year.html",          "csv": "EPS_Year.csv",          "table_id": None},
        "CashFlow":          {"html": "CashFlow.html",          "csv": "CashFlow.csv",          "table_id": None},
        "PER_PBR":           {"html": "PER_PBR.html",           "csv": "PER_PBR.csv",           "table_id": None},
        "MonthlyRevenue":    {"html": "MonthlyRevenue.html",    "csv": "MonthlyRevenue.csv",    "table_id": None},
        "Dividend":          {"html": "Dividend.html",          "csv": "Dividend.csv",          "table_id": None},
        "EquityDistribution":{"html": "EquityDistribution.html","csv": "EquityDistribution.csv","table_id": 'tblDetail'},
    }
    for key, info in mapping.items():
        html_f = os.path.join(base_dir, info["html"])
        csv_f  = os.path.join(base_dir, info["csv"])
        table_id = info["table_id"]
        if os.path.exists(csv_f) and not force_update:
            print(f"{csv_f} 已存在 -> 略過解析")
            continue

        if key == "EquityDistribution":
            parse_equity_distribution(base_dir, html_f, info["csv"], table_id)
        else:
            parse_html_with_multi_layer_headers(base_dir, html_f, info["csv"], table_id)

    # 3) 再抓 三大財報(資產負債表/損益表/現金流量表)
    #    這裡直接呼叫 scrape_financial_data
    for rpt_type in ["BS", "IS", "CF"]:
        if rpt_type == "BS":
            rpt_name = "資產負債表"
        elif rpt_type == "IS":
            rpt_name = "損益表"
        else:
            rpt_name = "現金流量表"
        print(f"抓取 {rpt_name}...")
        scrape_financial_data(
            driver=driver,
            report_type=rpt_type,
            start_year=start_year,
            start_quarter=start_quarter,
            end_year=end_year,
            end_quarter=end_quarter,
            stockID=stockID,
            base_dir=base_dir
        )

    print("所有資料(股價+一般Goodinfo+三大財報)已下載/解析完成.")


# ---------------------------------------------------
# 讓使用者用 python financial_statements_fetcher.py 可單檔執行做測試
# ---------------------------------------------------
if __name__ == "__main__":
    # 測試範例
    stockID = "2412"
    base_dir = os.path.join(os.path.dirname(__file__), stockID)
    os.makedirs(base_dir, exist_ok=True)

    fetch_all_data(
        stockID=stockID,
        base_dir=base_dir,
        start_date="2000-01-01",
        end_date="2024-12-31",
        start_year=2024,
        start_quarter=2,
        end_year=2020,
        end_quarter=1
    )

    print("抓取完畢，關閉瀏覽器...")
    close_driver()
    print("完成！")


# ---------- End of financial_statements_fetcher.py ----------

# ---------- Start of stock_data_processing.py ----------
# stock_data_processing.py
import os
import re
import pandas as pd
import numpy as np
import yfinance as yf

from config import stock_dir
from fetch_stock_list import urls_and_filenames

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
            csv_path = os.path.join(stock_dir, f"{stock_code}.csv")
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
        csv_path = os.path.join(stock_dir, f"{stock_code}.csv")

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

# ---------- End of stock_data_processing.py ----------

# ---------- Start of investment_indicators.py ----------
import numpy as np
import pandas as pd
import numpy_financial as npf
from datetime import timedelta


import statsmodels.api as sm  # 用於回歸計算Alpha/Beta

def calculate_max_drawdown(prices_series: pd.Series) -> float:
    """
    計算最大回撤 (Max Drawdown)。
    prices_series 為時序股價 (index 為日期，value 為股價)。
    """
    prices = prices_series.dropna()
    if len(prices) < 2:
        return np.nan
    
    cumulative = (1 + prices.pct_change().fillna(0)).cumprod()
    rolling_max = cumulative.cummax()
    drawdown = (cumulative - rolling_max) / rolling_max
    return drawdown.min()  # 最大回撤(最深跌幅)，負值

def calculate_sharpe(prices_series: pd.Series, risk_free_rate=0.01) -> float:
    """
    計算年化 Sharpe Ratio
    - 一年預設 252 交易日
    - risk_free_rate 預設 1%
    """
    prices = prices_series.dropna()
    if len(prices) < 2:
        return np.nan

    returns = prices.pct_change().dropna()
    mean_return = returns.mean() * 252
    std_dev = returns.std() * np.sqrt(252)
    if std_dev == 0:
        return np.nan
    sharpe_ratio = (mean_return - risk_free_rate) / std_dev
    return sharpe_ratio

def calculate_annualized_return(prices_series: pd.Series) -> float:
    """
    計算年化報酬率 (以實際天數計算)。
    """
    prices = prices_series.dropna()
    if len(prices) < 2:
        return np.nan

    total_return = (prices.iloc[-1] / prices.iloc[0]) - 1
    days_count = (prices.index[-1] - prices.index[0]).days
    if days_count <= 0:
        return np.nan
    annual_factor = 365 / days_count
    annualized_return = (1 + total_return) ** annual_factor - 1
    return annualized_return


import yfinance as yf




def calculate_irr_from_prices(prices_series: pd.Series, investment_per_month=-1000) -> float:
    """
    定期定額 IRR (每月投資 investment_per_month，負值表示流出)
    以每月最後一個交易日的收盤價為基準。
    """
    monthly_prices = prices_series.resample('ME').last().dropna()
    if len(monthly_prices) < 2:
        return np.nan

    num_months = len(monthly_prices)
    monthly_investments = [investment_per_month] * num_months

    shares_purchased = []
    for i in range(num_months):
        if monthly_prices.iloc[i] <= 0:
            shares_purchased.append(0.0)
        else:
            shares_purchased.append(-investment_per_month / monthly_prices.iloc[i])

    total_shares = sum(shares_purchased)
    final_value = total_shares * monthly_prices.iloc[-1]  # 最後全部賣出
    # 前面 num_months 次 = 投資(負數)，最後一次 = 賣出(正數)
    cash_flows = monthly_investments + [final_value]

    irr = npf.irr(cash_flows)
    if irr is None or np.isnan(irr):
        return np.nan

    # 轉為年化
    annualized_irr = (1 + irr) ** 12 - 1
    return annualized_irr

def calculate_alpha_beta(stock_prices: pd.Series, market_prices: pd.Series, daily_rf_return: float = 0.0) -> tuple[float, float]:
    """
    stock_prices, market_prices: 時序股價 (index=日期, value=收盤價)
    daily_rf_return: 每日無風險利率 (例如把 ^TNX 換算成日收益率的平均)
    回傳 (alpha, beta)，若資料不足則 (NaN, NaN)
    """
    stock_returns = stock_prices.pct_change().dropna() - daily_rf_return
    market_returns = market_prices.pct_change().dropna() - daily_rf_return

    df = pd.concat([stock_returns, market_returns], axis=1).dropna()
    df.columns = ["stock", "market"]
    if len(df) < 2:
        return (np.nan, np.nan)

    X = sm.add_constant(df["market"])
    y = df["stock"]
    model = sm.OLS(y, X).fit()
    alpha = model.params["const"]
    beta  = model.params["market"]
    return alpha, beta



def calc_metrics_for_range(prices_df: pd.DataFrame, start_date=None, end_date=None) -> dict:
    """
    綜合計算該股在 [start_date, end_date] 的四大指標: 
    - MDD, Sharpe, AnnualReturn, DCA_IRR
    """
    sub_df = prices_df.loc[start_date:end_date].dropna(subset=["Close"])
    if sub_df.empty:
        return {
            "MDD": np.nan,
            "Sharpe": np.nan,
            "AnnualReturn": np.nan,
            "DCA_IRR": np.nan
        }

    close_series = sub_df["Close"].copy()
    mdd = calculate_max_drawdown(close_series)
    sharpe = calculate_sharpe(close_series)
    annual_rtn = calculate_annualized_return(close_series)
    dca_irr = calculate_irr_from_prices(close_series)

    return {
        "MDD": mdd,
        "Sharpe": sharpe,
        "AnnualReturn": annual_rtn,
        "DCA_IRR": dca_irr
    }

def calc_multiple_period_metrics(
    stock_df: pd.DataFrame,
    as_of_date: pd.Timestamp,
    years_list=[3,5,10,15,20],
    market_df: pd.DataFrame = None,
    daily_rf_return: float = 0.0
) -> pd.DataFrame:
    """
    依據 as_of_date，往回 3/5/10/15/20 年計算 MDD, Sharpe, AnnualReturn, DCA_IRR
    也在同一區間計算 alpha/beta (若有提供 market_df 與 daily_rf_return)
    market_df: 市場收盤價 (DataFrame, columns=["Close"])，index=日期
    daily_rf_return: 每日無風險利率(例如從 ^TNX 算出的平均日收益率)

    回傳 DataFrame，每行是一個 years: [Years, MDD, Sharpe, AnnualReturn, DCA_IRR, Alpha, Beta]
    """
    results = []
    for y in years_list:
        start_dt = as_of_date - pd.DateOffset(years=y)
        sub_stock = stock_df.loc[start_dt:as_of_date]
        if len(sub_stock) < 2:
            row = {
                "Years": y,
                "MDD": np.nan,
                "Sharpe": np.nan,
                "AnnualReturn": np.nan,
                "DCA_IRR": np.nan,
                "Alpha": np.nan,
                "Beta": np.nan
            }
        else:
            # 1) 先計算 MDD, Sharpe...
            metrics = calc_metrics_for_range(stock_df, start_dt, as_of_date)
            row = {
                "Years": y,
                **metrics,
                "Alpha": np.nan,
                "Beta": np.nan
            }

            # 2) 若有提供 market_df, 就計算 alpha/beta
            if market_df is not None and not market_df.empty:
                sub_market = market_df.loc[start_dt:as_of_date]
                if len(sub_market) > 1:
                    
                    alpha, beta = calculate_alpha_beta(
                        stock_prices=sub_stock["Close"],
                        market_prices=sub_market["Close"],
                        daily_rf_return=daily_rf_return
                    )
                    row["Alpha"] = alpha
                    row["Beta"] = beta

        results.append(row)
    return pd.DataFrame(results)

# ---------- End of investment_indicators.py ----------

# ---------- Start of main.py ----------
# main_app.py
import streamlit as st

def main():
    st.set_page_config(
        page_title="我的多頁版投資分析平台",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    st.title("歡迎使用投資分析平台 (多頁版)")
    st.write("""
        左側是多頁面的導覽列，可在「個股基本資訊」、「財報」、「回測」、「比較」、「空頭」之間切換。
        \n在每個分頁中，可各自使用您先前整合的邏輯及介面。
    """)

if __name__ == "__main__":
    main()

# ---------- End of main.py ----------

# ---------- Start of streamlit_app.py ----------
import streamlit as st



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

# ---------- End of streamlit_app.py ----------

# ---------- Start of pages/1_個股基本資訊.py ----------
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

# ---------- End of pages/1_個股基本資訊.py ----------

# ---------- Start of pages/2_財報.py ----------
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
# ---------- End of pages/2_財報.py ----------

# ---------- Start of pages/3_回測.py ----------
# pages/3_回測.py
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from fetch_stock_list import get_stock_lists
from stock_data_processing import download_data
from investment_indicators import calculate_irr_from_prices

# from investment_indicators import calculate_max_drawdown, ...

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


page_backtest()

# ---------- End of pages/3_回測.py ----------

# ---------- Start of pages/4_比較.py ----------
# pages/4_比較.py
import streamlit as st
import pandas as pd
import plotly.express as px
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

from fetch_stock_list import get_stock_lists
from stock_data_processing import download_data
from investment_indicators import calculate_sharpe
from investment_indicators import (
    calculate_irr_from_prices,
    calculate_sharpe,
    calculate_max_drawdown,
    calculate_annualized_return
)

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

page_compare()

# ---------- End of pages/4_比較.py ----------

# ---------- Start of pages/5_空頭.py ----------
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

# ---------- End of pages/5_空頭.py ----------

