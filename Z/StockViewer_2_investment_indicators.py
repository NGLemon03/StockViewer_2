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
                  "MDD": mdd, "AnnualVol": vol,
                  "Sharpe": shp, "Sortino": srt,
                  "AnnualReturn": ann_ret,
                  "DCA_IRR": irr,
                  "Alpha": alpha_val*100, 
                  "Beta": beta_val
                }
        results.append(row)
    return pd.DataFrame(results)
