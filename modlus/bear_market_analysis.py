# bear_market_analysis.py
import pandas as pd
import numpy as np
from modlus.investment_indicators import calculate_max_drawdown

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
