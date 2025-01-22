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
