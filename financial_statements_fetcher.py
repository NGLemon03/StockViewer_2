# financial_statements_fetcher.py
import os
import time
import random
import pandas as pd
import yfinance as yf
from io import StringIO
from bs4 import BeautifulSoup
from selenium import webdriver
from selenium_stealth import stealth
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.common.by import By
from webdriver_manager.chrome import ChromeDriverManager
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC

from config import work_dir

# 初始化 WebDriver,強制使用chrome_deugger
options = webdriver.ChromeOptions()
options.add_experimental_option("debuggerAddress", "127.0.0.1:9222")
driver = webdriver.Chrome(options=options)

# 使用 stealth 函數來避免被目標網站的檢測到使用自動化工具
stealth(driver,
        languages=["zh-TW", "zh"],    # 設定瀏覽器語言
        vendor="Google Inc.",         # 設定裝置的供應商
        platform="Win32",             # 設定作業系統平台
        webgl_vendor="Intel Inc.",    # WebGL 渲染器供應商
        renderer="Intel Iris OpenGL Engine",  # 具體的渲染器
        fix_hairline=True,            # 修正線條問題
        )
# 強制更新設定,如果設為 True 則無視本地文件直接重新抓取數據
force_update = False
fetch_delay = 10

def random_sleep(min_seconds=5, max_seconds=10):
    """隨機等待函數"""
    sleep_time = random.uniform(min_seconds, max_seconds)
    time.sleep(sleep_time)

def save_html_nolink(driver, selector, file_path, delay=5):
    """
    僅抓取指定 selector 的 HTML (outerHTML),不跳轉
    若檔案已存在且不強制更新,則直接跳過。
    """
    if os.path.exists(file_path) and not force_update:
        print(f"{file_path} 已存在,使用已有文件,跳過等待時間。")
        return
    else:
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
    下載指定股票ID在特定日期範圍內的股價資料,並保存為 CSV 文件。
    可以指定數據抓取的時間間隔。

    :param driver: Selenium WebDriver 的實例。
    :param stockID: 股票代碼。
    :param base_dir: 文件保存的基本目錄。
    :param start_date: 開始日期。
    :param end_date: 結束日期。
    :param interval: 資料的時間間隔(默認為 '1d',即每天)。
    """
    ticker_symbol_1 = f"{stockID}.TW"
    ticker_symbol_2 = f"{stockID}.TWO"
    ticker_symbols = [ticker_symbol_1, ticker_symbol_2]
    stock_data = None

    for ticker_symbol in ticker_symbols:
        print(f"嘗試下載股價資料,代號：{ticker_symbol}")
        ticker = yf.Ticker(ticker_symbol)
        stock_data = ticker.history(start=start_date, end=end_date, interval=interval)
        if not stock_data.empty:
            actual_start = stock_data.index[0]
            actual_end = stock_data.index[-1]
            print(f"成功下載資料,日期範圍：{actual_start} 至 {actual_end}")
            break
        else:
            print(f"無法下載資料,代號：{ticker_symbol}")
            stock_data = None

    if stock_data is None or stock_data.empty:
        print(f"無法下載 {stockID} 的股價資料。")
        return

    price_file_path = os.path.join(base_dir, f"{stockID}_price.csv")
    stock_data.to_csv(price_file_path, encoding='utf-8-sig')
    print(f"{stockID} 股價數據已保存到 {price_file_path}")

def save_html(driver, url, selector, file_path, delay=5):
    """
    跳轉網址後,再抓取指定 selector 的 HTML。
    若檔案已存在且不強制更新,則直接跳過。
    """
    if os.path.exists(file_path) and not force_update:
        print(f"{file_path} 已存在,使用已有文件,跳過等待時間。")
        return
    else:
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

def parse_html_with_multi_layer_headers(file_path, output_csv, table_id=None):
    """
    解析具有多層表頭的 HTML 表格,並將其轉換成 CSV 文件。
    可以指定表格的 id 進行精確的選擇。

    :param file_path: 要解析的 HTML 文件的路徑。
    :param output_csv: 輸出 CSV 文件的路徑。
    :param table_id: HTML 表格的 id 屬性(可選)。
    """
    with open(file_path, 'r', encoding='utf-8') as file:
        soup = BeautifulSoup(file, 'html.parser')

    if table_id:
        table = soup.find('table', {'id': table_id})
    else:
        table = soup.find('table')
    if table is None:
        print(f"在 {file_path} 中未找到 id 為 '{table_id}' 的表格。")
        return
    rows = table.find_all('tr')

    rows = [row for row in rows if 'DummyTHead' not in row.get('class', [])]

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
        print(f"在 {file_path} 中未找到表頭。")
        return

    num_header_rows = len(header_rows)
    total_columns = 0
    for row in header_rows:
        cols = sum(int(th.get('colspan', 1)) for th in row.find_all(['th', 'td']))
        total_columns = max(total_columns, cols)

    header_matrix = [['' for _ in range(total_columns)] for _ in range(num_header_rows)]
    occupied = {}

    for row_idx, row in enumerate(header_rows):
        col_idx = 0
        cells = row.find_all(['th', 'td'])
        for cell in cells:
            while occupied.get((row_idx, col_idx), False):
                col_idx += 1
            colspan = int(cell.get('colspan', 1))
            rowspan = int(cell.get('rowspan', 1))
            text = cell.get_text(strip=True)
            for i in range(rowspan):
                for j in range(colspan):
                    r = row_idx + i
                    c = col_idx + j
                    header_matrix[r][c] = text
                    occupied[(r, c)] = True
            col_idx += colspan

    column_names = []
    for col in range(total_columns):
        headers = [header_matrix[row][col] for row in range(num_header_rows)]
        headers = [h for h in headers if h and 'Unnamed' not in h]
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
    df.to_csv(output_csv, index=False, encoding='utf-8-sig')
    print(f"成功將 {file_path} 轉換為 {output_csv}")

def parse_equity_distribution(file_path, output_csv, table_id='tblDetail'):
    """
    解析股數分級 HTML(多層表頭),轉為 CSV
    """
    with open(file_path, 'r', encoding='utf-8') as file:
        html_content = file.read()

    df_list = pd.read_html(html_content, attrs={'id': table_id}, header=[0,1])
    if not df_list:
        print(f"在 {file_path} 中未找到 id 為 '{table_id}' 的表格。")
        return

    df = df_list[0]
    def flatten_column(col):
        labels = [str(s).strip() for s in col if str(s) != 'nan' and 'Unnamed' not in str(s)]
        return '_'.join(labels)
    df.columns = [flatten_column(col) for col in df.columns.values]

    df = df[~df[df.columns[0]].str.contains('週別', na=False)]
    df.to_csv(output_csv, index=False, encoding='utf-8-sig')
    print(f"成功將 {file_path} 轉換為 {output_csv}")

def generate_quarter_list(start_year, start_quarter, end_year, end_quarter):
    """
    生成從指定的起始季度到結束季度的列表。
    :param start_year: 起始年份。
    :param start_quarter: 起始季度。
    :param end_year: 結束年份。
    :param end_quarter: 結束季度。
    :return: 包含季度標識的列表,如 ['2020Q1', '2019Q4', ...]。
    """
    quarters = []
    year, quarter = start_year, start_quarter
    while (year > end_year) or (year == end_year and quarter >= end_quarter):
        quarters.append(f"{year}Q{quarter}")  # 將每個季度以 '年Q季' 的格式加入列表
        quarter -= 1  # 減少季度數
        if quarter == 0:  # 如果季度數減到0,則年份減一,季度變為4
            quarter = 4
            year -= 1
    return quarters

def calculate_qry_times(needed_quarters, quarters_per_page):
    """
    根據季度列表計算所需的 QRY_TIME 列表,用於分批次查詢數據。

    :param needed_quarters: 需要查詢的季度列表,如 ['2020Q1', '2019Q4', ...]。
    :param quarters_per_page: 每個查詢能包含的最大季度數。
    :return: 每個批次查詢所需的 QRY_TIME 列表。
    """
    qry_times = []
    index = 0
    total_quarters = len(needed_quarters)
    while index < total_quarters:
        quarter_str = needed_quarters[index]  # 從列表中獲取當前季度
        year = quarter_str[:4]  # 季度字符串的前4位是年份
        quarter = quarter_str[-1]  # 季度字符串的最後一位是季度數
        qry_time = f"{year}{quarter}"  # 形成 QRY_TIME 格式,例如 '20201' for 2020年第一季度
        qry_times.append(qry_time)
        index += quarters_per_page  # 增加索引以跳過已經處理的季度
    return qry_times

def scrape_financial_data(driver, report_type, start_year, start_quarter, end_year, end_quarter, stockID, base_dir):
    """
    抓取指定股票ID和時間範圍內的財務報表數據。

    :param driver: Selenium WebDriver 的實例。
    :param report_type: 要抓取的報表類型 ('BS' for Balance Sheet, 'IS' for Income Statement, 'CF' for Cash Flow)。
    :param start_year: 抓取起始年份。
    :param start_quarter: 抓取起始季度。
    :param end_year: 抓取結束年份。
    :param end_quarter: 抓取結束季度。
    :param stockID: 股票代碼。
    :param base_dir: 文件存儲的基本路徑。
    """
    if report_type == 'BS':
        RPT_CAT = 'BS_M_QUAR'
        quarters_per_page = 7
        file_name = f"Financial_BalanceSheet.csv"
    elif report_type == 'IS':
        RPT_CAT = 'IS_M_QUAR_ACC'
        quarters_per_page = 7
        file_name = f"Financial_IncomeStatement.csv"
    elif report_type == 'CF':
        RPT_CAT = 'CF_M_QUAR_ACC'
        quarters_per_page = 10
        file_name = f"Financial_CashFlow.csv"
    else:
        print("未知的報表類型。")
        return

    file_path = os.path.join(base_dir, file_name)
    if os.path.exists(file_path) and not force_update:
        print(f"{file_path} 已存在,使用已有文件。")
        return

    needed_quarters = generate_quarter_list(start_year, start_quarter, end_year, end_quarter)
    qry_times = calculate_qry_times(needed_quarters, quarters_per_page)

    data_list = []
    collected_quarters = set()

    # 遍歷每個 QRY_TIME 並抓取相關數據
    for qry_time in qry_times:
        url = f"https://goodinfo.tw/tw/StockFinDetail.asp?RPT_CAT={RPT_CAT}&STOCK_ID={stockID}&QRY_TIME={qry_time}"
        print(f"正在訪問網址：{url}")
        driver.get(url)
        random_sleep(1, 3)
        try:
            WebDriverWait(driver, 10).until(EC.presence_of_element_located((By.ID, 'tblFinDetail')))
            time.sleep(1)
        except Exception as e:
            print(f"錯誤：未能加載表格 - {e}")
            continue

        # 解析頁面上的表格
        table_element = driver.find_element(By.ID, 'tblFinDetail')
        table_html = table_element.get_attribute('outerHTML')

        df_list = pd.read_html(StringIO(table_html), header=[0,1])
        df = df_list[0]

        df.columns = ['_'.join(col).strip() if col[1] else col[0].strip() for col in df.columns.values]

        # 確定該頁面包含哪些季度的數據並進行收集
        page_quarters = []
        for col in df.columns[1:]:  # 避開第一列(項目名稱)
            quarter = col.split('_')[0]
            if quarter not in page_quarters:
                page_quarters.append(quarter)

        for quarter in page_quarters:
            if quarter in needed_quarters and quarter not in collected_quarters:
                amount_cols = [col for col in df.columns if col.startswith(quarter) and '百分比' not in col]
                percent_cols = [col for col in df.columns if col.startswith(quarter) and '百分比' in col]

                for idx, row in df.iterrows():
                    item = row[df.columns[0]]
                    for amount_col in amount_cols:
                        amount = row[amount_col]
                        data_entry = {
                            '項目': item,
                            '季度': quarter,
                            '金額': amount,
                        }
                        if percent_cols:
                            percent_col = percent_cols[amount_cols.index(amount_col)]
                            percent = row[percent_col]
                            data_entry['百分比'] = percent
                        data_list.append(data_entry)
                collected_quarters.add(quarter)

        print(f"已收集季度: {', '.join(sorted(collected_quarters, reverse=True))}")

    if not data_list:
        print("未獲取到任何數據。")
        return

    # 將收集的數據整理為 DataFrame 並保存為 CSV 文件
    result_df = pd.DataFrame(data_list)
    if '百分比' in result_df.columns:
        pivot_df = result_df.pivot_table(index='項目', columns='季度', values=['金額', '百分比'], aggfunc='first')
        pivot_df.columns = [f"{col[1]}_{col[0]}" for col in pivot_df.columns]
    else:
        pivot_df = result_df.pivot_table(index='項目', columns='季度', values='金額', aggfunc='first')
        pivot_df.columns = [f"{col}" for col in pivot_df.columns]
    pivot_df.reset_index(inplace=True)

    ordered_columns = ['項目']
    for quarter in needed_quarters:
        if '百分比' in result_df.columns:
            ordered_columns.extend([f"{quarter}_金額", f"{quarter}_百分比"])
        else:
            ordered_columns.append(f"{quarter}")

    existing_columns = [col for col in ordered_columns if col in pivot_df.columns]
    pivot_df = pivot_df[existing_columns]
    pivot_df.to_csv(file_path, index=False, encoding="utf-8-sig")
    print(f"數據已保存到 '{file_path}'")

def close_driver():
    """關閉瀏覽器"""
    global driver
    print("關閉瀏覽器。")
    driver.quit()
    print("完成！")
