import os
import re
import pandas as pd
import numpy as np
import yfinance as yf
import matplotlib.pyplot as plt
import numpy_financial as npf  # 替代 np.irr
import matplotlib
import requests
from bs4 import BeautifulSoup
import seaborn as sns

#-------------------------------------------------------
# 1. 設定工作目錄和基本參數
#-------------------------------------------------------
work_dir = os.path.dirname(os.path.abspath(__file__))
os.chdir(work_dir)

matplotlib.rcParams['font.family'] = 'Microsoft YaHei'  # 支持中文字體
matplotlib.rcParams['font.size'] = 10

# 設定資料保存目錄
list_dir = os.path.join(work_dir, "list")
stock_dir = os.path.join(work_dir, "stock")
output_dir = os.path.join(work_dir, "output")  # 儲存結果的目錄
heatmap_dir = os.path.join(output_dir, "heatmaps")
os.makedirs(list_dir, exist_ok=True)
os.makedirs(stock_dir, exist_ok=True)
os.makedirs(output_dir, exist_ok=True)
os.makedirs(heatmap_dir, exist_ok=True)

#-------------------------------------------------------
# 2. 抓取台股上市、上櫃、興櫃列表
#-------------------------------------------------------
urls_and_filenames = {
    'https://isin.twse.com.tw/isin/C_public.jsp?strMode=2': os.path.join(list_dir, 'TWSE.csv'),
    'https://isin.twse.com.tw/isin/C_public.jsp?strMode=4': os.path.join(list_dir, 'OTC.csv'),
    'https://isin.twse.com.tw/isin/C_public.jsp?strMode=5': os.path.join(list_dir, 'emerging.csv')
}

def fetch_stock_list(url):
    response = requests.get(url)
    soup = BeautifulSoup(response.content, 'html.parser')
    table = soup.select_one('body > table.h4')
    return pd.read_html(str(table))[0]

def save_stock_list(url, filename):
    if not os.path.exists(filename):
        df = fetch_stock_list(url)
        df.to_csv(filename, index=False, encoding='utf_8_sig')
        print(f"已儲存: {filename}")
    else:
        print(f"檔案已存在: {filename}")

# 抓取並保存資料 (若已存在就不重複下載)
for url, filename in urls_and_filenames.items():
    save_stock_list(url, filename)

#-------------------------------------------------------
# 3. 初始化緩存
#-------------------------------------------------------
suffix_cache = {}
file_data_cache = {}  # 用於緩存已加載的上市/上櫃/興櫃清單

#-------------------------------------------------------
# 4. 畫相關性熱力圖
#-------------------------------------------------------
def plot_correlation_heatmap(correlation_matrix, period):
    plt.figure(figsize=(10, 8))
    sns.heatmap(
        correlation_matrix, 
        annot=True, 
        fmt=".2f", 
        cmap="coolwarm", 
        cbar=True,
        square=True
    )
    plt.title(f"Correlation Heatmap ({period} days)")
    plt.tight_layout()
    
    # 儲存圖表
    output_file = os.path.join(heatmap_dir, f"correlation_heatmap_{period}_days.png")
    plt.savefig(output_file, dpi=300)
    plt.show()
    print(f"相關性熱力圖已儲存到 {output_file}")

#-------------------------------------------------------
# 5. 載入與解析 CSV
#-------------------------------------------------------
def load_and_process_file(filepath):
    """
    載入上市/上櫃/興櫃清單 CSV，標準化欄位與分割代號、名稱。
    """
    df = pd.read_csv(filepath, encoding='utf_8_sig', header=None, skiprows=1)
    df.columns = ['有價證券代號及名稱', '國際證券辨識號碼(ISIN Code)', '上市日', '市場別', '產業別', 'CFICode', '備註']

    if '有價證券代號及名稱' in df.columns:
        df[['Code', 'Name']] = df['有價證券代號及名稱'].str.split('　', expand=True)
    return df

#-------------------------------------------------------
# 6. 判別股票代碼後綴
#-------------------------------------------------------
def determine_suffix(code):
    """
    根據 stock code 判斷應該加上 .TW 或 .TWO
    """
    if code in suffix_cache:
        return suffix_cache[code]
    
    for filepath in urls_and_filenames.values():
        if filepath not in file_data_cache:
            file_data_cache[filepath] = load_and_process_file(filepath)
        
        df = file_data_cache[filepath]
        # 如果該代碼有出現在該 CSV，就決定後綴
        if code in df['Code'].values:
            suffix = '.TW' if 'TWSE' in filepath else '.TWO'
            suffix_cache[code] = suffix
            return suffix

    # 如果找不到，就給空字串（可能為興櫃或其他）
    suffix_cache[code] = ''
    return ''

#-------------------------------------------------------
# 7. 下載股價數據
#-------------------------------------------------------
def download_data(stock_list):
    """
    依序檢查/下載各個股票的股價資料，存在本地 CSV 並讀取進資料表。
    支援以 ^ 開頭的指數或含 = 的匯率符號(如 USDTWD=X)。
    """
    stock_data = {}
    for stock in stock_list:
        # 若是特殊符號 (指數、匯率等)，用原代號直接下載
        if stock.startswith('^') or re.match(r'^[A-Za-z]{3,}=*[A-Za-zX]*$', stock):
            stock_code = f"{stock}"
            csv_path = os.path.join(stock_dir, f"{stock_code}.csv")
            try:
                if os.path.exists(csv_path):
                    print(f"檔案已存在: {csv_path}")
                    df = pd.read_csv(csv_path, index_col=0, parse_dates=True)
                    stock_data[stock] = df['Adj Close']
                else:
                    print(f"下載中: {stock_code}")
                    df = yf.download(stock_code, period="15y")
                    if not df.empty:
                        df['Adj Close'].to_csv(csv_path)
                        stock_data[stock] = df['Adj Close']
                    else:
                        print(f"無數據: {stock_code}, 跳過...")
            except Exception as e:
                print(f"處理 {stock_code} 時發生錯誤: {e}")
            continue

        # 否則判斷上市櫃後綴
        suffix = determine_suffix(stock)
        if not suffix:
            print(f"{stock} 無法判斷後綴，跳過...")
            continue

        stock_code = f"{stock}{suffix}"
        csv_path = os.path.join(stock_dir, f"{stock_code}.csv")

        try:
            if os.path.exists(csv_path):
                print(f"檔案已存在: {csv_path}")
                df = pd.read_csv(csv_path, index_col=0, parse_dates=True)
                stock_data[stock] = df['Adj Close']
            else:
                print(f"下載中: {stock_code}")
                df = yf.download(stock_code, period="15y")
                if not df.empty:
                    df['Adj Close'].to_csv(csv_path)
                    stock_data[stock] = df['Adj Close']
                else:
                    print(f"無數據: {stock_code}, 跳過...")
        except Exception as e:
            print(f"處理 {stock_code} 時發生錯誤: {e}")

    return pd.DataFrame(stock_data)

#-------------------------------------------------------
# 8. 各種計算函數
#-------------------------------------------------------
def calculate_max_drawdown(prices_series):
    """
    計算最大回撤
    """
    prices = prices_series.dropna()
    if len(prices) < 2:
        return np.nan
    cumulative = (1 + prices.pct_change()).cumprod()
    rolling_max = cumulative.cummax()
    drawdown = (cumulative - rolling_max) / rolling_max
    return drawdown.min()

def calculate_sharpe(prices_series, period, risk_free_rate=0.01):
    """
    計算年化夏普比
    """
    returns = prices_series.pct_change().dropna()[-period:]
    if len(returns) < 2:
        return np.nan
    mean_return = returns.mean() * 252
    std_dev = returns.std() * np.sqrt(252)
    sharpe_ratio = (mean_return - risk_free_rate) / std_dev if std_dev != 0 else np.nan
    return sharpe_ratio

def calculate_annualized_return(prices_series, period):
    """
    計算年化報酬率
    """
    prices = prices_series.dropna()[-period:]
    if len(prices) < 2:
        return np.nan
    total_return = (prices.iloc[-1] / prices.iloc[0]) - 1
    annualized_return = (1 + total_return) ** (252 / period) - 1
    return annualized_return

def calculate_irr_from_prices(prices, investment_per_month=-1000):
    """
    根據歷史價格模擬定期定額投資的 IRR 計算。
    """
    # 按月重採樣，確保至少有兩個月的數據
    prices = prices.resample('ME').last().dropna()
    if len(prices) < 2:
        return np.nan

    # 模擬每月投資金額
    num_months = len(prices)
    monthly_investments = [investment_per_month] * num_months

    # 累積持股數
    shares_purchased = [-investment_per_month / price for price in prices]
    total_shares = sum(shares_purchased)

    # 最終現金流
    total_value = total_shares * prices.iloc[-1]
    cash_flows = monthly_investments + [total_value]

    # 使用 numpy-financial 的 IRR 計算
    irr = npf.irr(cash_flows)
    if irr is None or np.isnan(irr):
        return np.nan

    # 轉換為年化 IRR
    annualized_irr = (1 + irr) ** 12 - 1
    return annualized_irr

#-------------------------------------------------------
# 9. 分析空頭市場
#-------------------------------------------------------
def analyze_bear_market(stock_data, start_date, end_date):
    """
    分析指定日期範圍(空頭期間)的最大回撤、波動率、總收益率。
    """
    bear_data = stock_data[(stock_data.index >= start_date) & (stock_data.index <= end_date)].dropna(how='all')
    
    if bear_data.empty:
        print(f"{start_date}到{end_date}期間沒有有效數據")
        return pd.DataFrame()

    def _volatility(series):
        returns = series.pct_change().dropna()
        return returns.std() * np.sqrt(252)

    results = {}
    for stock in stock_data.columns:
        prices = bear_data[stock].dropna()
        if len(prices) < 2:
            print(f"{stock}在{start_date} 到 {end_date} 期間沒有有效數據")
            continue

        max_dd = calculate_max_drawdown(prices)
        vol = _volatility(prices)
        total_return = (prices.iloc[-1] / prices.iloc[0]) - 1

        results[stock] = {
            "最大回撤": max_dd,
            "波動率": vol,
            "總收益率": total_return
        }

    return pd.DataFrame(results).T

#-------------------------------------------------------
# 9-1. USDTWD=X「上升 / 下降」區段分割 (簡易示範)
#-------------------------------------------------------
def identify_usdtwd_trend_periods(usd_series):
    """
    將 USDTWD=X 的日線數據，依據「日變化>0 =>上升, <0 =>下降」
    分割出多個連續區段 (start_date, end_date, trend_type)。
    trend_type 可為 "Up" 或 "Down"
    """
    usd_series = usd_series.dropna().sort_index()
    if len(usd_series) < 2:
        return []

    diff = usd_series.diff()
    trend_periods = []
    current_trend = None
    current_start = None

    for i in range(1, len(usd_series)):
        date_prev = usd_series.index[i - 1]
        date_curr = usd_series.index[i]
        change = diff.iloc[i]
        
        # 判斷漲跌
        if change > 0:
            trend_label = "Up"
        else:
            trend_label = "Down"

        if current_trend is None or trend_label != current_trend:
            if current_trend is not None and current_start is not None:
                trend_periods.append((current_start, date_prev, current_trend))
            current_trend = trend_label
            current_start = date_curr

    # 最後一段
    if current_trend is not None and current_start is not None:
        last_date = usd_series.index[-1]
        trend_periods.append((current_start, last_date, current_trend))

    return trend_periods


#-------------------------------------------------------
# 10. 主程式
#-------------------------------------------------------
if __name__ == "__main__":
    # 輸入股票代碼列表 (含 USDTWD=X)
    stock_list = [
        '00713', '00701', '006208', '2412','00710B','00727B','00741B','00860B','00859B','00864B','00719B',
        'SHY', 'BIL', 'MINT', 'FLRN', 'JPST', 'ICSH',
        'USDTWD=X'  # 關鍵：台幣對美元匯率
    ]
    rate_symbols = ['^IRX','^TYX'] # 90天 T-Bill = ^IRX，30年公債 = ^TYX

    # 10-1. 下載數據
    stock_data = download_data(stock_list)
    rate_data = download_data(rate_symbols)

    # 對齊索引
    stock_data.index = pd.to_datetime(stock_data.index)
    rate_data.index = pd.to_datetime(rate_data.index)
    common_idx = stock_data.index.intersection(rate_data.index)
    stock_data = stock_data.loc[common_idx]
    rate_data = rate_data.loc[common_idx]

    #---------------------------------------------------
    # 10-2. 計算定期定額 IRR 和上市時間
    #---------------------------------------------------
    irr_dict = {}
    listing_times = {}
    for stk in stock_list:
        if stk not in stock_data.columns:
            print(f"{stk}沒有數據,跳過...")
            continue
        prices = stock_data[stk].dropna()
        listing_times[stk] = len(prices)
        if len(prices) < 2:
            irr_dict[stk] = np.nan
            continue
        irr_val = calculate_irr_from_prices(prices)
        irr_dict[stk] = irr_val

    #---------------------------------------------------
    # 10-3. 計算不同期間的投資指標 (Sharpe, MDD, 年化收益)
    #---------------------------------------------------
    all_metrics_df = pd.DataFrame(index=stock_data.columns)
    all_metrics_df['IRR'] = [irr_dict.get(s, np.nan) for s in all_metrics_df.index]

    periods = [60, 120, 240, 480]
    for period in periods:
        subset = stock_data[-period:].dropna(how='all')
        if subset.empty:
            print(f"最近 {period} 天 subset 為空, 跳過...")
            continue

        corr_matrix = subset.corr()
        corr_csv = os.path.join(output_dir, f"correlation_matrix_{period}_days.csv")
        corr_matrix.to_csv(corr_csv, encoding='utf-8-sig')

        plot_correlation_heatmap(corr_matrix, period)

        for s in subset.columns:
            sh = calculate_sharpe(subset[s], period)
            mdd = calculate_max_drawdown(subset[s])
            ann_ret = calculate_annualized_return(subset[s], period)
            all_metrics_df.loc[s, f"{period}_Sharpe"] = sh
            all_metrics_df.loc[s, f"{period}_MDD"] = mdd
            all_metrics_df.loc[s, f"{period}_AnnualReturn"] = ann_ret

    # 輸出總表
    all_metrics_file = os.path.join(output_dir, "all_investment_metrics.csv")
    all_metrics_df.to_csv(all_metrics_file, encoding='utf-8-sig')
    print(f"投資指標(含 IRR、Sharpe、MDD、年化報酬)已儲存到 {all_metrics_file}")

    #---------------------------------------------------
    # 分析 USDTWD=X 上升 / 下降區段 (日線連漲連跌)
    #---------------------------------------------------
    usd_series = stock_data['USDTWD=X'].dropna()
    trend_periods = identify_usdtwd_trend_periods(usd_series)

    if not trend_periods:
        print("USDTWD=X 資料不足或無法分割出趨勢區段。")
    else:
        print("=== USDTWD=X 上升 / 下降區段 (日線) ===")
        for (sd, ed, ttype) in trend_periods:
            print(f"{sd.date()} ~ {ed.date()} => {ttype}")

    # 在每個 "上升期 / 下降期" 計算指標
    segment_analysis = []
    for (start_d, end_d, trend_type) in trend_periods:
        segment_data = stock_data.loc[start_d:end_d].dropna(how='all')
        if segment_data.empty:
            continue

        for stk in segment_data.columns:
            prices = segment_data[stk].dropna()
            if len(prices) < 2:
                continue

            # IRR
            irr_val = np.nan
            try:
                irr_val = calculate_irr_from_prices(prices)
            except:
                pass

            period_len = len(prices)
            sh = calculate_sharpe(prices, period_len)
            mdd = calculate_max_drawdown(prices)
            total_ret = (prices.iloc[-1] / prices.iloc[0]) - 1
            ann_ret = (1 + total_ret)**(252/period_len) - 1 if period_len>20 else np.nan

            # 與 USDTWD=X 相關
            if 'USDTWD=X' in segment_data.columns and stk != 'USDTWD=X':
                df_corr = segment_data[[stk, 'USDTWD=X']].dropna()
                corr_val = df_corr.corr().iloc[0,1] if len(df_corr)>2 else np.nan
            else:
                corr_val = np.nan

            segment_analysis.append({
                "TrendType": trend_type,
                "StartDate": start_d,
                "EndDate": end_d,
                "Symbol": stk,
                "SegmentDays": period_len,
                "IRR": irr_val,
                "Sharpe": sh,
                "MaxDrawdown": mdd,
                "TotalReturn": total_ret,
                "AnnualReturn": ann_ret,
                "CorrWithUSDTWD": corr_val
            })

    seg_df = pd.DataFrame(segment_analysis)
    seg_df.sort_values(by=["TrendType","StartDate","Symbol"], inplace=True)
    seg_csv = os.path.join(output_dir, "usdtwd_trend_segment_analysis.csv")
    seg_df.to_csv(seg_csv, index=False, encoding="utf-8-sig")
    print(f"各標的在 USDTWD=X 升/降 區段的投資指標分析 => {seg_csv}")
    print(seg_df.head(20))

    #-------------------------------------------------------
    # 分析兩個空頭市場 (2020 & 2022) - 亦可保留或移除
    #-------------------------------------------------------
    bear_markets = {
        "2020 新冠疫情": ("2020-01-01", "2020-05-01"),
        "2022 FED升息": ("2022-01-01", "2022-12-31")
    }
    for market, (start_dt, end_dt) in bear_markets.items():
        print(f"\n--- 分析空頭市場: {market} ---")
        bear_results = analyze_bear_market(stock_data, start_dt, end_dt)
        if bear_results.empty:
            print(f"{market}期間沒有有效數據，跳過...")
            continue
        out_bear = os.path.join(output_dir, f"bear_market_{market.replace(' ', '_')}.csv")
        bear_results.to_csv(out_bear, encoding='utf-8-sig')
        print(f"{market}分析結果 => {out_bear}")

        # 繪圖 (選擇性)
        if '最大回撤' in bear_results.columns:
            bear_results['最大回撤'].plot(kind='bar', title=f"最大回撤 ({market})")
            plt.ylabel("最大回撤")
            plt.tight_layout()
            plt.savefig(os.path.join(output_dir, f"max_drawdown_{market.replace(' ', '_')}.png"))
        if '波動率' in bear_results.columns:
            bear_results['波動率'].plot(kind='bar', title=f"波動率 ({market})")
            plt.ylabel("波動率")
            plt.tight_layout()
            plt.savefig(os.path.join(output_dir, f"volatility_{market.replace(' ', '_')}.png"))
        if '總收益率' in bear_results.columns:
            bear_results['總收益率'].plot(kind='bar', title=f"總收益率 ({market})")
            plt.ylabel("總收益率")
            plt.tight_layout()
            plt.savefig(os.path.join(output_dir, f"total_return_{market.replace(' ', '_')}.png"))

    #-------------------------------------------------------
    # (額外) 使用人工定義之「長周期」manual_periods 分析
    #-------------------------------------------------------
    # 你可以在這裡直接修改想要的區段與標記
    manual_periods = [
        ("2009-01-01", "2011-01-01", "Down"),
        ("2011-01-02", "2013-01-01", "Up"),
        ("2013-01-02", "2015-01-01", "Down"),
        ("2015-01-02", "2021-01-01", "Down"),  # 可自行標為 Up/Down
        ("2021-01-02", "2025-12-31", "Up"),
    ]

    manual_results = []
    for (start_d, end_d, trend_label) in manual_periods:
        # 篩出區間
        seg_data = stock_data.loc[start_d:end_d].dropna(how='all')
        if seg_data.empty:
            continue

        # 每個標的計算 IRR, Sharpe, MDD, AnnualReturn, 與 USDTWD=X 相關性
        for col in seg_data.columns:
            prices = seg_data[col].dropna()
            if len(prices) < 2:
                continue

            irr_val = calculate_irr_from_prices(prices)
            period_len = len(prices)
            sh = calculate_sharpe(prices, period_len)
            mdd = calculate_max_drawdown(prices)
            total_return = (prices.iloc[-1] / prices.iloc[0]) - 1

            ann_ret = (1 + total_return)**(252/period_len) - 1 if period_len>20 else np.nan

            # 與 USDTWD=X 相關係數
            if 'USDTWD=X' in seg_data.columns and col != 'USDTWD=X':
                df_corr = seg_data[[col, 'USDTWD=X']].dropna()
                corr_val = df_corr.corr().iloc[0,1] if len(df_corr)>2 else np.nan
            else:
                corr_val = np.nan

            manual_results.append({
                "TrendLabel": trend_label,
                "StartDate": start_d,
                "EndDate": end_d,
                "Symbol": col,
                "Days": period_len,
                "IRR": irr_val,
                "Sharpe": sh,
                "MaxDrawdown": mdd,
                "TotalReturn": total_return,
                "AnnualReturn": ann_ret,
                "CorrWithUSDTWD": corr_val
            })

    final_df = pd.DataFrame(manual_results)
    final_df.sort_values(["TrendLabel","StartDate","Symbol"], inplace=True)
    manual_csv = os.path.join(output_dir, "manual_trend_analysis.csv")
    final_df.to_csv(manual_csv, index=False, encoding="utf-8-sig")
    print(f"人工區段分析結果 => {manual_csv}")
    print(final_df.head(20))

    print("\n=== 程式執行完畢 ===")
