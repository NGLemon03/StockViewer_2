import os
import logging
import pandas as pd
from dash import dcc, html, callback, Input, Output, State
import plotly.express as px
from datetime import datetime
from modlus.config import DL_DIR
from modlus.investment_indicators import calc_multiple_period_metrics, get_risk_free_rate
# 設定 logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("app.log", mode='a', encoding='utf-8'),
        logging.StreamHandler()  # 保留控制台輸出
    ]
)

logger = logging.getLogger(__name__)
def render_page1_basic_info():
    return html.Div([
        html.Div([
            html.Label("主要股票"),
            dcc.Input(id='main_stock_id', type='text', value='2412', style={'width': '100px'}),
            html.Label("市場基準"),
            dcc.Input(id='market_id', type='text', value='^TWII', style={'width': '100px'}),
            html.Label("比較股票"),
            dcc.Input(id='other_ids', type='text', value='2330,00713,006208', style={'width': '200px'})
        ], style={'marginBottom': '20px'}),

        html.Div([
            html.Label("分析截止日期"),
            dcc.Input(id='analysis_end_date', type='text', value=datetime.today().strftime('%Y-%m-%d'),
                      style={'width': '120px'}),
            html.Label("使用價格類型"),
            dcc.RadioItems(id='price_type', options=[
                {'label': 'Close', 'value': 'Close'},
                {'label': 'Adj Close', 'value': 'Adj Close'}
            ], value='Close', inline=True)
        ], style={'marginBottom': '20px'}),

        html.Button("計算多期指標", id='calculate_button'),
        html.Div(id='multi_period_output', style={'marginTop': '20px'}),

        html.Div([
            html.Label("分析起始日期"),
            dcc.Input(id='analysis_start_date', type='text', value='2010-01-01', style={'width': '120px'}),
            html.Button("繪製累積報酬圖", id='plot_cumulative_button'),
            html.Div(id='cumulative_chart_output', style={'marginTop': '20px'})
        ])
    ])


@callback(
    Output('multi_period_output', 'children'),
    Input('calculate_button', 'n_clicks'),
    State('main_stock_id', 'value'),
    State('market_id', 'value'),
    State('other_ids', 'value'),
    State('analysis_end_date', 'value'),
    State('price_type', 'value')
)
def update_multi_period_metrics(n_clicks, stock_id, market_id, other_ids, analysis_end_date, price_type):
    if not n_clicks:
        return ""

    logger.info("開始計算多期指標")
    as_of_dt = pd.Timestamp(analysis_end_date)
    logger.info(f"分析截止日期: {as_of_dt}")

    tnxf = os.path.join(DL_DIR, "^TNX", "^TNX_price.csv")
    daily_rf = 0.0
    yearly_rf = 0.01

    if os.path.exists(tnxf):
        try:
            dftnx = pd.read_csv(tnxf, parse_dates=["Date"], index_col="Date")
            dftnx.index = dftnx.index.tz_localize(None)
            daily_rf, yearly_rf = get_risk_free_rate(dftnx.loc[:as_of_dt])
            logger.info(f"無風險利率: 日收益率={daily_rf}, 年收益率={yearly_rf}")
        except Exception as e:
            logger.error(f"讀取 ^TNX 資料失敗: {e}")

    all_syms = [stock_id]
    if market_id.strip():
        all_syms.append(market_id.strip())
    others = [x.strip() for x in other_ids.split(',') if x.strip()]
    all_syms.extend(others)

    logger.info(f"分析股票清單: {all_syms}")

    market_df = pd.DataFrame()
    if market_id.strip():
        market_csv = os.path.join(DL_DIR, market_id.strip(), f"{market_id.strip()}_price.csv")
        if os.path.exists(market_csv):
            market_df = pd.read_csv(market_csv, parse_dates=["Date"], index_col="Date")
            market_df.index = market_df.index.tz_localize(None)

    multi_data = []
    for sid in all_syms:
        cf = os.path.join(DL_DIR, sid, f"{sid}_price.csv")
        if not os.path.exists(cf):
            logger.warning(f"檔案不存在: {cf}")
            continue
        try:
            dfp = pd.read_csv(cf, parse_dates=["Date"], index_col="Date")
            dfp.index = dfp.index.tz_localize(None)

            df_metrics = calc_multiple_period_metrics(
                stock_df=dfp,
                as_of_date=as_of_dt,
                years_list=[3, 5, 10, 15, 20],
                market_df=market_df,
                daily_rf_return=daily_rf,
                use_adj_close=(price_type == "Adj Close"),
                freq_for_reg="W",
                rf_annual_rate=yearly_rf
            )
            df_metrics["股票"] = sid
            multi_data.append(df_metrics)
            logger.info(f"完成計算: {sid}")
        except Exception as e:
            logger.error(f"處理股票 {sid} 時發生錯誤: {e}")

    if not multi_data:
        logger.warning("無可用數據計算多期指標")
        return html.Div("無可顯示的多期指標")

    merged_df = pd.concat(multi_data, ignore_index=True)
    pivoted = merged_df.pivot(index="股票", columns="Years")

    def format_table(df, title, columns_to_format):
        formatted_df = df.copy()
        for col, fmt in columns_to_format.items():
            formatted_df[col] = formatted_df[col].map(lambda x: f"{x:.2f}" if pd.notnull(x) else "")
        return html.Div([
            html.H4(title),
            dcc.Markdown(formatted_df.to_markdown())
        ])

    return html.Div([
        format_table(pivoted[["Sharpe", "Sortino", "Alpha", "Beta"]], "回報及風險表現", {"Sharpe": ".2f", "Sortino": ".2f", "Alpha": ".2f", "Beta": ".2f"}),
        format_table(pivoted[["MDD", "AnnualVol", "DCA_IRR", "AnnualReturn"]], "資產表現", {"MDD": ".1f", "AnnualVol": ".1f", "DCA_IRR": ".1f", "AnnualReturn": ".1f"})
    ])


@callback(
    Output('cumulative_chart_output', 'children'),
    Input('plot_cumulative_button', 'n_clicks'),
    State('main_stock_id', 'value'),
    State('market_id', 'value'),
    State('other_ids', 'value'),
    State('analysis_start_date', 'value'),
    State('analysis_end_date', 'value')
)
def plot_cumulative_chart(n_clicks, stock_id, market_id, other_ids, analysis_start_date, analysis_end_date):
    if not n_clicks:
        return ""

    logger.info("開始繪製累積報酬圖")
    analysis_start_date = pd.Timestamp(analysis_start_date)
    analysis_end_date = pd.Timestamp(analysis_end_date)
    logger.info(f"分析期間: {analysis_start_date} 至 {analysis_end_date}")

    all_stocks = [stock_id]
    if market_id.strip():
        all_stocks.append(market_id.strip())
    others = [x.strip() for x in other_ids.split(',') if x.strip()]
    all_stocks.extend(others)

    logger.info(f"分析股票清單: {all_stocks}")

    cum_df = pd.DataFrame()
    for sid in all_stocks:
        csvf = os.path.join(DL_DIR, sid, f"{sid}_price.csv")
        if not os.path.exists(csvf):
            logger.warning(f"檔案不存在: {csvf}")
            continue
        try:
            dfp = pd.read_csv(csvf, parse_dates=["Date"], index_col="Date")
            dfp.index = dfp.index.tz_localize(None)
            subp = dfp.loc[analysis_start_date:analysis_end_date].copy()
            if subp.empty:
                logger.warning(f"股票 {sid} 無資料在指定期間內")
                continue
            subp['pct'] = subp['Close'].pct_change().fillna(0)
            subp['cum'] = (1 + subp['pct']).cumprod()
            base = subp['cum'].iloc[0] if len(subp) > 0 else 1
            subp['rebase'] = subp['cum'] / base
            cum_df[sid] = subp['rebase']
        except Exception as e:
            continue

    if cum_df.empty:
        return html.Div("無資料可繪製")

    fig = px.line(cum_df, x=cum_df.index, y=cum_df.columns, title="累積報酬比較",
                  labels={"value": "累積報酬", "variable": "股票"})
    fig.update_layout(legend=dict(orientation="h"))

    return dcc.Graph(figure=fig)
