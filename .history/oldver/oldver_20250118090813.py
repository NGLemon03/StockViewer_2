import os
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

# 抓取並保存資料
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
    """
    stock_data = {}
    for stock in stock_list:
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
            continue

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
    prices = prices.resample('M').last().dropna()
    if len(prices) < 2:
        return np.nan

    # 模擬每月投資金額
    num_months = len(prices)
    monthly_investments = [investment_per_month] * num_months

    # 累積持股數
    shares_purchased = [-investment_per_month / price for price in prices]
    total_shares = sum(shares_purchased)

    # 最終現金流
    total_value = total_shares * prices.iloc[-1]  # 期末價值
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

        max_drawdown = calculate_max_drawdown(prices)
        vol = _volatility(prices)
        total_return = (prices.iloc[-1] / prices.iloc[0]) - 1

        results[stock] = {
            "最大回撤": max_drawdown,
            "波動率": vol,
            "總收益率": total_return
        }

    return pd.DataFrame(results).T

#-------------------------------------------------------
# 10. 主程式
#-------------------------------------------------------
if __name__ == "__main__":
    # 輸入股票代碼列表
    stock_list = [
        '00679B','00687B','00694B','00695B','00696B','00697B','00710B','00711B','00718B','00719B','00720B','00722B','00723B','00724B','00725B','00726B','00727B','00734B',
        '00740B','00741B','00746B','00749B','00750B','00751B','00754B','00755B','00756B','00758B','00759B','00760B','00761B','00764B','00768B','00772B','00773B','00775B',
        '00777B','00778B','00779B','00780B','00781B','00782B','00784B','00785B','00786B','00787B','00788B','00789B','00790B','00791B','00792B','00793B','00794B','00795B',
        '00799B','00834B','00836B','00840B','00841B','00842B','00844B','00845B','00846B','00847B','00848B','00849B','00853B','00856B','00857B','00859B','00860B','00862B',
        '00863B','00864B','00865B','00867B','00870B','00883B','00884B','00890B','00931B','00933B','00937B','00942B','00945B','00948B','00950B','00953B','00957B','00958B',
        '00959B','00966B','00967B','00968B','00969B','00970B','00980B','02001B'
    ]

    # 下載數據並處理
    stock_data = download_data(stock_list)

    #---------------------------------------------------
    # 10-1. 計算定期定額 IRR 和上市時間，存檔
    #---------------------------------------------------
    irr_dict = {}
    listing_times = {}
    for stock in stock_list:
        if stock not in stock_data.columns:
            print(f"{stock}沒有數據,跳過...")
            continue

        prices = stock_data[stock].dropna()
        if prices.empty:
            print(f"{stock}沒有有效數據,跳過...")
            continue

        listing_time = len(prices)  # 用於紀錄有多少天交易資料
        listing_times[stock] = listing_time

        if listing_time < 2:
            irr_dict[stock] = np.nan
            continue

        irr = calculate_irr_from_prices(prices)
        irr_dict[stock] = irr

    output_file_irr = os.path.join(output_dir, "irr_and_listing_times.csv")
    with open(output_file_irr, "w", encoding="utf-8-sig") as f:
        for stock in stock_list:
            irr = irr_dict.get(stock, np.nan)
            listing_time = listing_times.get(stock, 0)
            line = f"{stock}: IRR = {irr:.2%}, 上市時間 = {listing_time} 天\n"
            f.write(line)
    print(f"定期定額 IRR 和上市時間已儲存到 {output_file_irr}")

    #---------------------------------------------------
    # 10-2. 計算不同期間的投資指標 (Sharpe, MDD, 年化收益)
    #       a) 相關性矩陣各存各檔
    #       b) 其餘指標整合存到單一檔案
    #---------------------------------------------------
    periods = [60, 120, 240, 480]
    # 用來彙整多個期間指標的 DataFrame
    all_metrics_df = pd.DataFrame(index=stock_data.columns)

    for period in periods:
        # 取得最後 period 天的子集
        subset = stock_data[-period:].dropna(how='all')

        # ---------------------------
        # (a) 輸出相關性矩陣 + 熱力圖
        # ---------------------------
        correlation_matrix = subset.corr()
        # 儲存相關性矩陣
        corr_file = os.path.join(output_dir, f"correlation_matrix_{period}_days.csv")
        correlation_matrix.to_csv(corr_file, encoding='utf-8-sig')
        print(f"相關性矩陣已儲存到 {corr_file}")
        # 繪製熱力圖
        plot_correlation_heatmap(correlation_matrix, period)

        # ---------------------------
        # (b) 計算各項投資指標 (Sharpe, MDD, 年化收益)
        # ---------------------------
        sharpe_ratios = {}
        max_drawdowns = {}
        annualized_returns = {}

        for stock in subset.columns:
            sharpe_ratios[stock] = calculate_sharpe(subset[stock], period)
            max_drawdowns[stock] = calculate_max_drawdown(subset[stock])
            annualized_returns[stock] = calculate_annualized_return(subset[stock], period)

        # 將結果放入 all_metrics_df
        # 欄位名稱格式: {天數}_{指標}
        for stock in subset.columns:
            all_metrics_df.loc[stock, f"{period}_Sharpe"] = sharpe_ratios[stock]
            all_metrics_df.loc[stock, f"{period}_MDD"] = max_drawdowns[stock]
            all_metrics_df.loc[stock, f"{period}_AnnualReturn"] = annualized_returns[stock]

    # 最後將 all_metrics_df 輸出到一個 CSV
    all_metrics_file = os.path.join(output_dir, "all_investment_metrics.csv")
    all_metrics_df.to_csv(all_metrics_file, encoding='utf-8-sig')
    print(f"投資指標(Sharpe, MDD, 年化報酬)已整合儲存到 {all_metrics_file}")

    #---------------------------------------------------
    # 10-3. 分析兩個空頭市場
    #---------------------------------------------------
    bear_markets = {
        "2020 新冠疫情": ("2020-01-01", "2020-05-01"),
        "2022 FED升息": ("2022-01-01", "2022-12-31")
    }

    for market, (start_date, end_date) in bear_markets.items():
        print(f"\n--- 分析空頭市場: {market} ---")
        bear_results = analyze_bear_market(stock_data, start_date, end_date)

        if bear_results.empty:
            print(f"{market}期間沒有有效數據，跳過...")
            continue

        # 儲存分析結果到 CSV
        output_file_bear = os.path.join(output_dir, f"bear_market_{market.replace(' ', '_')}.csv")
        bear_results.to_csv(output_file_bear, encoding='utf-8-sig')
        print(f"{market}分析結果已儲存到 {output_file_bear}")

        # 可視化結果
        if '最大回撤' in bear_results.columns:
            bear_results['最大回撤'].plot(kind='bar', title=f"最大回撤 ({market})")
            plt.ylabel("最大回撤")
            plt.tight_layout()
            plt.savefig(os.path.join(output_dir, f"max_drawdown_{market.replace(' ', '_')}.png"))
            plt.show()

        if '波動率' in bear_results.columns:
            bear_results['波動率'].plot(kind='bar', title=f"波動率 ({market})")
            plt.ylabel("波動率")
            plt.tight_layout()
            plt.savefig(os.path.join(output_dir, f"volatility_{market.replace(' ', '_')}.png"))
            plt.show()

        if '總收益率' in bear_results.columns:
            bear_results['總收益率'].plot(kind='bar', title=f"總收益率 ({market})")
            plt.ylabel("總收益率")
            plt.tight_layout()
            plt.savefig(os.path.join(output_dir, f"total_return_{market.replace(' ', '_')}.png"))
            plt.show()
