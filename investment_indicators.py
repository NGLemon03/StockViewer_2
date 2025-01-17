# investment_indicators.py
import numpy as np
import numpy_financial as npf

def calculate_max_drawdown(prices_series):
    """計算最大回撤"""
    prices = prices_series.dropna()
    if len(prices) < 2:
        return np.nan
    cumulative = (1 + prices.pct_change()).cumprod()
    rolling_max = cumulative.cummax()
    drawdown = (cumulative - rolling_max) / rolling_max
    return drawdown.min()

def calculate_sharpe(prices_series, period, risk_free_rate=0.01):
    """計算年化 Sharpe"""
    returns = prices_series.pct_change().dropna()[-period:]
    if len(returns) < 2:
        return np.nan
    mean_return = returns.mean() * 252
    std_dev = returns.std() * np.sqrt(252)
    sharpe_ratio = (mean_return - risk_free_rate) / std_dev if std_dev != 0 else np.nan
    return sharpe_ratio

def calculate_annualized_return(prices_series, period):
    """計算年化報酬率"""
    prices = prices_series.dropna()[-period:]
    if len(prices) < 2:
        return np.nan
    total_return = (prices.iloc[-1] / prices.iloc[0]) - 1
    annualized_return = (1 + total_return) ** (252 / period) - 1
    return annualized_return

def calculate_irr_from_prices(prices, investment_per_month=-1000):
    """定期定額 IRR"""
    prices = prices.resample('M').last().dropna()
    if len(prices) < 2:
        return np.nan

    num_months = len(prices)
    monthly_investments = [investment_per_month] * num_months

    shares_purchased = [-investment_per_month / price for price in prices]
    total_shares = sum(shares_purchased)

    total_value = total_shares * prices.iloc[-1]
    cash_flows = monthly_investments + [total_value]

    irr = npf.irr(cash_flows)
    if irr is None or np.isnan(irr):
        return np.nan

    annualized_irr = (1 + irr) ** 12 - 1
    return annualized_irr
