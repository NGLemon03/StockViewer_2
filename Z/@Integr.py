# ---------- Start of modlus/bear_market_analysis.py ----------
# bear_market_analysis.py
import pandas as pd
import numpy as np
from StockViewer_2.modlus.investment_indicators import calculate_max_drawdown

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

# ---------- End of modlus/bear_market_analysis.py ----------

# ---------- Start of modlus/config.py ----------
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

# ---------- End of modlus/config.py ----------

# ---------- Start of modlus/DL_Y.py ----------
import os
import re
import time
import pandas as pd
import yfinance as yf

from StockViewer_2.modlus.fetch_stock_list import urls_and_filenames

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

# ---------- End of modlus/DL_Y.py ----------

# ---------- Start of modlus/fetch_stock_list.py ----------
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

# ---------- End of modlus/fetch_stock_list.py ----------

# ---------- Start of modlus/financial_statements_fetcher.py ----------
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
from StockViewer_2.modlus.fetch_stock_list import urls_and_filenames  # {'上市網址': 'TWSE.csv', '上櫃網址': 'OTC.csv', ...}

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
        df['Adj Close'] = (df['Close'] - df['Dividends']) / (1 + df['Stock Splits'] / 10)
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


# ---------- End of modlus/financial_statements_fetcher.py ----------

# ---------- Start of modlus/investment_indicators.py ----------
import numpy as np
import pandas as pd
import numpy_financial as npf
from datetime import timedelta
import statsmodels.api as sm
import os
from pandas import Timedelta

##########################################################
# 計算各種投資指標 (含無風險利率取得、Sharpe、Sortino、Alpha、Beta…)
##########################################################
TW_tradeday = 240
def get_risk_free_rate(tnx_df: pd.DataFrame) -> float:
    """
    從已讀取好的 ^TNX 檔案 (含 Close 欄位) 計算『平均日無風險收益率』。
    假設 ^TNX 的 Close 表示年化殖利率(%)，用最簡單的除法: daily_rf = (年化殖利率) / 252。
    """
    if tnx_df.empty or "Close" not in tnx_df.columns:
        return 0.0
    yearly_yield = tnx_df["Close"].mean() / 100.0
    daily_rf = yearly_yield / 252
    return daily_rf, yearly_yield

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

def calculate_sharpe(prices_series: pd.Series, daily_rf=0.0) -> float:
    prices = prices_series.dropna()
    if len(prices) < 2:
        return np.nan
    returns = prices.pct_change().dropna() - daily_rf
    mean_ret = returns.mean() * 252
    std_dev = returns.std() * np.sqrt(252)
    if std_dev == 0:
        return np.nan
    return (mean_ret) / std_dev

def calculate_sortino(prices_series: pd.Series, daily_rf=0.0) -> float:
    """
    計算年化報酬率 (以實際天數計算)。
    """
    prices = prices_series.dropna()
    if len(prices) < 2:
        return np.nan
    daily_ret = prices.pct_change().dropna() - daily_rf
    mean_ret = daily_ret.mean() * 252
    negative_ret = daily_ret[daily_ret < 0]
    if len(negative_ret) < 1:
        return np.nan
    downside_std = negative_ret.std() * np.sqrt(252)
    if downside_std == 0:
        return np.nan
    return mean_ret / downside_std

def calculate_annualized_return(prices_series: pd.Series) -> float:
    prices = prices_series.dropna()
    if len(prices) < 2:
        return np.nan
    total_ret = (prices.iloc[-1] / prices.iloc[0]) - 1
    days_count = (prices.index[-1] - prices.index[0]).days
    if days_count <= 0:
        return np.nan
    factor = 365 / days_count
    return (1 + total_ret) ** factor - 1

def calculate_annual_volatility(prices_series: pd.Series, daily_rf=0.0) -> float:
    """
    年化波動度 = (日收益率的std) * sqrt(252)
    """
    prices = prices_series.dropna()
    if len(prices) < 2:
        return np.nan
    daily_ret = prices.pct_change().dropna() - daily_rf
    return daily_ret.std() * np.sqrt(252)

def calculate_irr_from_prices(prices_series: pd.Series, investment_per_month=-1000) -> float:
    """
    定期定額 IRR (每月投資 investment_per_month，負值表示流出)
    以每月最後一個交易日的收盤價為基準。
    """
    monthly_prices = prices_series.resample('ME').last().dropna()
    if len(monthly_prices) < 2:
        return np.nan
    months = len(monthly_prices)
    invests = [investment_per_month]*months
    shares_purchased = []
    for i in range(months):
        px = monthly_prices.iloc[i]
        if px <= 0:
            shares_purchased.append(0.0)
        else:
            shares_purchased.append(-investment_per_month / px)
    total_shares = sum(shares_purchased)
    final_value = total_shares * monthly_prices.iloc[-1]
    cash_flows = invests + [final_value]
    irr = npf.irr(cash_flows)
    if irr is None or np.isnan(irr):
        return np.nan
    return (1 + irr)**12 - 1
def calc_alpha_beta_monthly(
    stock_px: pd.Series,
    market_px: pd.Series,
    freq: str = "M",
    rf_annual_rate: float = 0.0
) -> tuple[float, float]:
    """
    使用『月度』或『週度』的累積報酬率進行回歸，計算年化 Alpha / Beta。
    - freq: 'M' (月) 或 'W' (週) 等
    - rf_annual_rate: 年化無風險利率 (如 0.01 表示1%)
      會轉換為對應 freq 的無風險收益。
    回傳 (alpha_annual, beta)
    """

    # 1) 將股價、市場價格分別 resample 到 freq (例如每月最後一天)
    stock_m = stock_px.resample(freq).last().dropna()
    market_m = market_px.resample(freq).last().dropna()

    # 2) 將兩者對齊
    df = pd.concat([stock_m, market_m], axis=1).dropna()
    df.columns = ["stock","market"]
    if len(df)<2:
        return (np.nan, np.nan)

    # 3) 計算該 freq 對應的報酬
    #   freq='M' -> 12 periods/year, freq='W' -> ~52 periods/year
    #   periods_per_year: freq='M' => 12, 'W' => 52
    if freq.upper().startswith("M"):
        periods_per_year = 12
    elif freq.upper().startswith("W"):
        periods_per_year = 52
    else:
        # 預設 fallback
        periods_per_year = 12

    df["stock_ret"] = df["stock"].pct_change()
    df["mkt_ret"]   = df["market"].pct_change()
    df = df.dropna()

    # 4) 扣除無風險報酬 (若您有 daily_rf => 需先轉年化 => monthly => 於此可直接帶入 0.01 / 12 之類)
    #   假設 rf_annual_rate 是年化 => freq -> per_period rf
    rf_per_period = rf_annual_rate / periods_per_year
    df["stock_excess"] = df["stock_ret"] - rf_per_period
    df["mkt_excess"]   = df["mkt_ret"]   - rf_per_period

    # 5) 回歸
    X = sm.add_constant(df["mkt_excess"])
    y = df["stock_excess"]
    model = sm.OLS(y, X).fit()

    alpha_per_period = model.params.get("const", np.nan)  # freq 週期內的超額
    beta = model.params.get("mkt_excess", np.nan)

    # 6) 將 alpha_per_period => 年化
    #   如果 freq='M'，1 + alpha_per_period ^ 12 -1
    #   freq='W' => ^52 -1
    if np.isnan(alpha_per_period):
        alpha_annual = np.nan
    else:
        alpha_annual = (1 + alpha_per_period)**periods_per_year - 1

    return alpha_annual, beta

def calculate_alpha_beta(stock_prices: pd.Series, market_prices: pd.Series, daily_rf: float=0.0) -> tuple[float, float]:
    stk = stock_prices.dropna()
    mkt = market_prices.dropna()
    if len(stk) < 2 or len(mkt) < 2:
        return np.nan, np.nan
    stock_ret = stk.pct_change().dropna() - daily_rf
    market_ret = mkt.pct_change().dropna() - daily_rf
    df = pd.concat([stock_ret, market_ret], axis=1).dropna()
    df.columns = ["stock","market"]
    if len(df) < 2:
        return np.nan, np.nan
    X = sm.add_constant(df["market"])
    y = df["stock"]
    model = sm.OLS(y, X).fit()
    daily_alpha = model.params.get("const", np.nan)
    beta = model.params.get("market", np.nan)

    # Annualize alpha
    if not np.isnan(daily_alpha):
        annualized_alpha = (1 + daily_alpha) ** 252 - 1
    else:
        annualized_alpha = np.nan

    return annualized_alpha, beta





def calc_multiple_period_metrics(
    stock_df: pd.DataFrame,
    as_of_date: pd.Timestamp,
    years_list=[3,5,10,15,20],
    market_df: pd.DataFrame = None,
    daily_rf_return: float = 0.0,
    use_adj_close: bool = False,
    freq_for_reg: str = "M",       # 新增參數，用月or週做回歸
    rf_annual_rate: float = 1.0  # 新增參數，無風險利率
) -> pd.DataFrame:
    """
    以 as_of_date 為終點，往前 3/5/10/15/20 年：
      MDD, AnnualVol, Sharpe, Sortino, AnnualReturn, DCA_IRR, Alpha, Beta
    use_adj_close: True 表示使用 'Adj Close' 欄位計算，否則用 'Close'
    """
    col_name = "Adj Close" if use_adj_close else "Close"
    results = []
    for y in years_list:
        start_dt = as_of_date - pd.DateOffset(years=y)
        sub = stock_df.loc[start_dt:as_of_date]
        print(sub.index[0])
        print(start_dt - Timedelta(days=60))
        
        if len(sub) < 2 or col_name not in sub.columns:

            row = {
                "Years": y, "MDD": np.nan, "AnnualVol": np.nan,
                "Sharpe": np.nan, "Sortino": np.nan,
                "AnnualReturn": np.nan, "DCA_IRR": np.nan,
                "Alpha": np.nan, "Beta": np.nan
            }
        else:
            px = sub[col_name].dropna()
            if len(px)<2:
                row = {
                    "Years": y, "MDD": np.nan, "AnnualVol": np.nan,
                    "Sharpe": np.nan, "Sortino": np.nan,
                    "AnnualReturn": np.nan, "DCA_IRR": np.nan,
                    "Alpha": np.nan, "Beta": np.nan
                }
            else:
                alpha_val, beta_val = np.nan, np.nan
                if market_df is not None and not market_df.empty:
                    if col_name in market_df.columns:
                        sub_mkt = market_df.loc[start_dt:as_of_date, col_name].dropna()
                        if len(sub_mkt)>2:
                            # 這裡改用 "calc_alpha_beta_monthly"
                            alpha_val, beta_val = calc_alpha_beta_monthly(
                                stock_px=px,
                                market_px=sub_mkt,
                                freq=freq_for_reg,
                                rf_annual_rate=rf_annual_rate
                            )
                                
                mdd = calculate_max_drawdown(px)
                vol = calculate_annual_volatility(px, daily_rf_return)
                shp = calculate_sharpe(px, daily_rf_return)
                srt = calculate_sortino(px, daily_rf_return)
                ann_ret = calculate_annualized_return(px)
                irr = calculate_irr_from_prices(px)

                row = {
                  "Years": y,
                  "MDD": mdd*100, "AnnualVol": vol*100,
                  "Sharpe": shp, "Sortino": srt,
                  "AnnualReturn": ann_ret*100,
                  "DCA_IRR": irr*100,
                  "Alpha": alpha_val*100, 
                  "Beta": beta_val
                }
        results.append(row)
    return pd.DataFrame(results)

# ---------- End of modlus/investment_indicators.py ----------

# ---------- Start of modlus/stock_data_processing.py ----------
# stock_data_processing.py
import os
import re
import pandas as pd
import numpy as np
import yfinance as yf

from StockViewer_2.modlus.config import stock_dir
from StockViewer_2.modlus.fetch_stock_list import urls_and_filenames

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

# ---------- End of modlus/stock_data_processing.py ----------

# ---------- Start of pages/page1_basic_info.py ----------
import os
import dash_core_components as dcc
import plotly.graph_objs as go
import pandas as pd
from modlus.DL_Y import download_stock_price, fetch_all_data
from modlus.config import DL_dir
# 1) 個股基本資訊 Tab
def render_page1_basic_info(stock_id, market_id, other_ids, start_date, end_date,
                           start_yq, end_yq):
    """
    原先 pages/1_個股基本資訊.py 邏輯:
      - fetch_all_data() 下載資料
      - 比較多股票 Rebase 圖
      - 多期(3/5/10/15/20年) 指標
    這裡簡化示範: 只做「多股票 Rebase 圖」(and partial 3/5/10/15/20).
    您可自行擴充。
    """
    container = []

    # 下載 or 讀取 data
    # 在 Dash 中，可能不會像 Streamlit 用 button 事件立即下載
    # 若您仍要做 "下載" 按鈕，可以在callback中執行 fetch_all_data()

    # 這裡示範：檢查 local CSV 有無，若無再下載
    # 主要股票
    main_dir = os.path.join(DL_dir, stock_id)
    os.makedirs(main_dir, exist_ok=True)
    # 假設您要自動下載 (若已存在就跳過)
    fetch_all_data(
        stockID=stock_id,
        base_dir=main_dir,
        start_date=str(start_date),
        end_date=str(end_date),
        start_year=int(start_yq.split("-")[0]),
        start_quarter=int(start_yq.split("-")[1]),
        end_year=int(end_yq.split("-")[0]),
        end_quarter=int(end_yq.split("-")[1])
    )

    # 下載市場基準
    all_stocks = [stock_id]
    if market_id.strip():
        all_stocks.append(market_id.strip())
        market_dir = os.path.join(DL_dir, market_id.strip())
        os.makedirs(market_dir, exist_ok=True)
        download_stock_price(
            stockID=market_id.strip(),
            base_dir=market_dir,
            start_date=str(start_date),
            end_date=str(end_date)
        )

    # 下載其他股票
    others = []
    if other_ids:
        others = [x.strip() for x in other_ids.split(",") if x.strip()]
        for sid in others:
            sid_dir = os.path.join(DL_dir, sid)
            os.makedirs(sid_dir, exist_ok=True)
            download_stock_price(
                stockID=sid,
                base_dir=sid_dir,
                start_date=str(start_date),
                end_date=str(end_date)
            )
    all_stocks.extend(others)

    # -------------------------------
    # 2) Rebase 圖
    # -------------------------------
    rebase_fig = go.Figure()
    rebase_fig.update_layout(
        title="多股票累積報酬(Rebase)比較",
        hovermode='x unified'
    )
    for sid in all_stocks:
        csvf = os.path.join(DL_dir, sid, f"{sid}_price.csv")
        if not os.path.exists(csvf):
            continue
        dfp = pd.read_csv(csvf, parse_dates=["Date"], index_col="Date")
        if dfp.empty:
            continue
        subp = dfp.loc[start_date:end_date].copy()
        if subp.empty:
            continue
        if 'Close' not in subp.columns:
            continue
        subp['pct'] = subp['Close'].pct_change().fillna(0)
        subp['cum'] = (1+subp['pct']).cumprod()
        base = subp['cum'].iloc[0] if len(subp)>0 else 1
        subp['rebase'] = subp['cum']/base
        rebase_fig.add_trace(
            go.Scatter(
                x=subp.index,
                y=subp['rebase'],
                mode='lines',
                name=sid
            )
        )

    container.append(dcc.Graph(figure=rebase_fig))

    return container
# ---------- End of pages/page1_basic_info.py ----------

# ---------- Start of pages/page2_financial.py ----------
import os
import dash_core_components as dcc
import plotly.graph_objs as go
import pandas as pd
from modlus.config import DL_dir
from plotly.subplots import make_subplots

# 2) 財報下載/預覽 Tab (含 4 合 1 圖: 單季EPS&YOY、毛利率/營業利益率/淨利率、4Q EPS合計、月營收)
#   原先 pages/2_財報.py
def render_page2_financial(stock_id):
    """
    讀取EPS_Quar.csv, MonthlyRevenue.csv, 產生 2x2 subplot.
    """
    base_dir = os.path.join(DL_dir, stock_id)
    # --- 讀取EPS_Quar ---
    eps_f = os.path.join(base_dir, "EPS_Quar.csv")
    df_quarter = pd.DataFrame()
    if os.path.exists(eps_f):
        df_quarter = pd.read_csv(eps_f)
        # 簡化: 只抓 單季 EPS(元)_稅後EPS, 營業毛利, 營業收入, ...
        # 這裡可直接套您pages/2_財報.py的函式 => 先略寫
        # ... (與您pages/2_財報.py相同處理)...

    # (這裡為了示範，我們用簡化假資料)
    df_q = pd.DataFrame({
       'Date': pd.date_range("2023-01-01", periods=4, freq='Q'),
       'EPS': [1.2, 1.3, 1.1, 1.4],
       'YOY': [0.05,0.08,-0.1,0.15],
       'GrossMargin': [35,36,34,37],
       'OperatingMargin': [25,26,24,26],
       'NetMargin': [20,21,19,22],
    }).set_index('Date')

    # ---- 月營收 (示範用假資料) ----
    df_month = pd.DataFrame({
       'Date': pd.date_range("2023-01-01", periods=6, freq='MS'),
       'Revenue': [180,190,170,200,210,205]
    }).set_index('Date')

    # 建立 2x2 subplots
    fig = make_subplots(
        rows=2, cols=2,
        shared_xaxes=True,
        subplot_titles=["(A) 單季EPS & YOY","(B) 毛利/營業/淨利率",
                        "(C) 近四季EPS合計","(D) 月營收" ],
        vertical_spacing=0.12
    )

    # (A) 單季EPS (Bar) + YOY (Line,右軸)
    fig.add_trace(
        go.Bar(
            x=df_q.index, y=df_q['EPS'], name="EPS"
        ),
        row=1, col=1
    )
    fig.add_trace(
        go.Scatter(
            x=df_q.index,
            y=df_q['YOY']*100,
            name="EPSYOY(%)",
            yaxis='y2'
        ),
        row=1, col=1
    )
    # 設定右軸
    fig.update_layout(
        yaxis2=dict(
            overlaying='y',
            side='right'
        )
    )

    # (B) 三條線
    fig.add_trace(
        go.Scatter(
            x=df_q.index, y=df_q['GrossMargin'], name="毛利率(%)"
        ),
        row=1, col=2
    )
    fig.add_trace(
        go.Scatter(
            x=df_q.index, y=df_q['OperatingMargin'], name="營業利率(%)"
        ),
        row=1, col=2
    )
    fig.add_trace(
        go.Scatter(
            x=df_q.index, y=df_q['NetMargin'], name="淨利率(%)"
        ),
        row=1, col=2
    )

    # (C) 近四季 EPS 合計(假: 先把EPS累加)
    cumsum_val = df_q['EPS'].cumsum()
    fig.add_trace(
        go.Bar(
            x=cumsum_val.index, y=cumsum_val, name="近四季EPS合計"
        ),
        row=2, col=1
    )

    # (D) 月營收
    fig.add_trace(
        go.Scatter(
            x=df_month.index, y=df_month['Revenue'],
            name="月營收(億)"
        ),
        row=2, col=2
    )

    fig.update_layout(
        title=f"{stock_id} 財報分析 (示範)",
        hovermode='x unified',
        height=800
    )

    return dcc.Graph(figure=fig)
# ---------- End of pages/page2_financial.py ----------

# ---------- Start of pages/page3_backtest.py ----------

# ---------- End of pages/page3_backtest.py ----------

# ---------- Start of pages/page4_compare.py ----------

# ---------- End of pages/page4_compare.py ----------

# ---------- Start of pages/page5_bear.py ----------
from dash import html, dcc
from modlus.bear_market_analysis import analyze_bear_market
from modlus.stock_data_processing import download_data
def render_page5_bear(stock_list):
    """
    analyze_bear_market(stock_data, start_date, end_date)
    """
    df = download_data(stock_list)
    if df.empty:
        return html.Div("無可用股價資料")

    # 假設預設兩段區間
    # 這裡不做互動輸入, 直接寫死
    period_map = {
        "疫情": ("2020-01-01","2020-05-01"),
        "FED升息":("2022-01-01","2022-12-31")
    }

    children_list = []
    for label,(sd,ed) in period_map.items():
        subres = analyze_bear_market(df, sd, ed)
        if subres.empty:
            children_list.append(html.Div(f"{label} 無資料"))
            continue
        # Convert to table
        subres_html = subres.style.format("{:.4f}").to_html()
        children_list.append(html.H4(f"{label} : {sd} ~ {ed}"))
        children_list.append(html.Div([
            dcc.Markdown(subres_html, dangerously_allow_html=True)
        ]))
    return html.Div(children_list)

# ---------- End of pages/page5_bear.py ----------

# ---------- Start of dash_app.py ----------
# dash_app.py
import dash
from dash import dcc, html, Input, Output, State
import plotly.express as px
import pandas as pd
import numpy as np
import os
from datetime import datetime
from pages.page1_basic_info import render_page1_basic_info
from pages.page2_financial import render_page2_financial
from pages.page3_backtest import render_page3_backtest
from pages.page4_compare import render_page4_compare
from pages.page5_bear import render_page5_bear
# ====== 引用您原有的模組 =======
# (請確保 bear_market_analysis.py, config.py, DL_Y.py, fetch_stock_list.py,
#  financial_statements_fetcher.py, investment_indicators.py, stock_data_processing.py
#  都在同一個資料夾下，或是一個 modules/ 子資料夾中，也可以根據您實際需求來 import)

from StockViewer_2.modlus.config import DL_dir  # 例如 config.py 裡定義了 DL_dir
from StockViewer_2.modlus.DL_Y import download_stock_price
from StockViewer_2.modlus.fetch_stock_list import get_stock_lists
from StockViewer_2.modlus.financial_statements_fetcher import fetch_all_data, close_driver
from StockViewer_2.modlus.investment_indicators import (
    get_risk_free_rate,
    calc_multiple_period_metrics,
    calculate_sharpe,
    calculate_max_drawdown,
    calculate_annualized_return,
    calculate_irr_from_prices
)
from StockViewer_2.modlus.bear_market_analysis import analyze_bear_market
from StockViewer_2.modlus.stock_data_processing import download_data
# ==============================================

# ====== 建立 Dash App ======
app = dash.Dash(__name__)
app.title = "Dash投資分析平台"

# ----------------------------------------------------------------
#  下方函式為「過去 Streamlit pages/xxx.py」的核心邏輯 -> 現改造成Dash callbacks
# ----------------------------------------------------------------




# =================================================
# Dash App Layout (多 Tab)
# =================================================
app.layout = html.Div([
    html.H1("Dash 投資分析平台 (多Tab示範)"),

    html.Div([
        # 全域輸入
        html.Label("主要股票(抓取Goodinfo數據)"),
        dcc.Input(id='main_stock_id', type='text', value='2412', style={'width':'100px'}),
        html.Label("市場基準(計算用)"),
        dcc.Input(id='market_id', type='text', value='^TWII', style={'width':'100px'}),
        html.Label("比較股票(逗號分隔)"),
        dcc.Input(id='other_ids', type='text', value='2330,00713,006208', style={'width':'200px'}),
    ], style={'marginBottom':'10px'}),

    html.Div([
        # 日期區
        html.Label("股價開始日期"),
        dcc.Input(id='start_date', type='text', value='2000-01-01', style={'width':'120px'}),
        html.Label("股價結束日期"),
        dcc.Input(id='end_date', type='text', value=datetime.today().strftime("%Y-%m-%d"), style={'width':'120px'}),
        # 財報季度
        html.Label("財報起始季度(YYYY-Q)"),
        dcc.Input(id='start_yq', type='text', value='2000-1', style={'width':'80px'}),
        html.Label("財報結束季度(YYYY-Q)"),
        dcc.Input(id='end_yq', type='text', value='2024-4', style={'width':'80px'}),
    ], style={'marginBottom':'10px'}),

    dcc.Tabs(id='tabs', value='tab1', children=[
        dcc.Tab(label="個股基本資訊", value='tab1'),
        dcc.Tab(label="財報下載/預覽", value='tab2'),
        dcc.Tab(label="回測", value='tab3'),
        dcc.Tab(label="多標的比較", value='tab4'),
        dcc.Tab(label="空頭分析", value='tab5'),
    ]),
    html.Div(id='tabs_content', style={'marginTop':'20px'})
])


# =================================================
# Callbacks
# =================================================
@app.callback(
    Output('tabs_content','children'),
    Input('tabs','value'),
    State('main_stock_id','value'),
    State('market_id','value'),
    State('other_ids','value'),
    State('start_date','value'),
    State('end_date','value'),
    State('start_yq','value'),
    State('end_yq','value')
)
def render_tabs(tab, stock_id, market_id, other_ids, sdate, edate, syq, eyq):
    """
    切換 Tab 時，根據 tab=tabX 來呼叫對應的 render_xxx 函式
    """
    if tab == 'tab1':
        return html.Div(render_page1_basic_info(
            stock_id, market_id, other_ids,
            sdate, edate, syq, eyq
        ))
    elif tab == 'tab2':
        return render_page2_financial(stock_id)
    elif tab == 'tab3':
        # 簡化: 只回測 main_stock_id + other_ids
        all_stocks = [stock_id]
        if market_id.strip():
            all_stocks.append(market_id.strip())
        if other_ids:
            all_stocks.extend([x.strip() for x in other_ids.split(",") if x.strip()])
        return render_page3_backtest(all_stocks)
    elif tab == 'tab4':
        # 多標的比較
        all_stocks = [stock_id]
        if market_id.strip():
            all_stocks.append(market_id.strip())
        if other_ids:
            all_stocks.extend([x.strip() for x in other_ids.split(",") if x.strip()])
        return render_page4_compare(all_stocks)
    else:
        # tab5: 空頭
        all_stocks = [stock_id]
        if market_id.strip():
            all_stocks.append(market_id.strip())
        if other_ids:
            all_stocks.extend([x.strip() for x in other_ids.split(",") if x.strip()])
        return render_page5_bear(all_stocks)


if __name__ == "__main__":
    app.run_server(debug=True, port=27451)

# ---------- End of dash_app.py ----------

