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
from modlus.fetch_stock_list import urls_and_filenames  # {'上市網址': 'TWSE.csv', '上櫃網址': 'OTC.csv', ...}

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

