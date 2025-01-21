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
