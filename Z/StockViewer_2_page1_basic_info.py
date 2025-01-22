import os
import dash_core_components as dcc
import plotly.graph_objs as go
import pandas as pd
from modlus.DL_Y import download_stock_price, fetch_all_data
from modlus.config import DL_dir
# 1) 個股基本資訊 Tab
def render_page1_basic_info(stock_id, market_id, other_ids, start_date, end_date,
                           start_yq, end_yq):
    """
    原先 pages/1_個股基本資訊.py 邏輯:
      - fetch_all_data() 下載資料
      - 比較多股票 Rebase 圖
      - 多期(3/5/10/15/20年) 指標
    這裡簡化示範: 只做「多股票 Rebase 圖」(and partial 3/5/10/15/20).
    您可自行擴充。
    """
    container = []

    # 下載 or 讀取 data
    # 在 Dash 中，可能不會像 Streamlit 用 button 事件立即下載
    # 若您仍要做 "下載" 按鈕，可以在callback中執行 fetch_all_data()

    # 這裡示範：檢查 local CSV 有無，若無再下載
    # 主要股票
    main_dir = os.path.join(DL_dir, stock_id)
    os.makedirs(main_dir, exist_ok=True)
    # 假設您要自動下載 (若已存在就跳過)
    fetch_all_data(
        stockID=stock_id,
        base_dir=main_dir,
        start_date=str(start_date),
        end_date=str(end_date),
        start_year=int(start_yq.split("-")[0]),
        start_quarter=int(start_yq.split("-")[1]),
        end_year=int(end_yq.split("-")[0]),
        end_quarter=int(end_yq.split("-")[1])
    )

    # 下載市場基準
    all_stocks = [stock_id]
    if market_id.strip():
        all_stocks.append(market_id.strip())
        market_dir = os.path.join(DL_dir, market_id.strip())
        os.makedirs(market_dir, exist_ok=True)
        download_stock_price(
            stockID=market_id.strip(),
            base_dir=market_dir,
            start_date=str(start_date),
            end_date=str(end_date)
        )

    # 下載其他股票
    others = []
    if other_ids:
        others = [x.strip() for x in other_ids.split(",") if x.strip()]
        for sid in others:
            sid_dir = os.path.join(DL_dir, sid)
            os.makedirs(sid_dir, exist_ok=True)
            download_stock_price(
                stockID=sid,
                base_dir=sid_dir,
                start_date=str(start_date),
                end_date=str(end_date)
            )
    all_stocks.extend(others)

    # -------------------------------
    # 2) Rebase 圖
    # -------------------------------
    rebase_fig = go.Figure()
    rebase_fig.update_layout(
        title="多股票累積報酬(Rebase)比較",
        hovermode='x unified'
    )
    for sid in all_stocks:
        csvf = os.path.join(DL_dir, sid, f"{sid}_price.csv")
        if not os.path.exists(csvf):
            continue
        dfp = pd.read_csv(csvf, parse_dates=["Date"], index_col="Date")
        if dfp.empty:
            continue
        subp = dfp.loc[start_date:end_date].copy()
        if subp.empty:
            continue
        if 'Close' not in subp.columns:
            continue
        subp['pct'] = subp['Close'].pct_change().fillna(0)
        subp['cum'] = (1+subp['pct']).cumprod()
        base = subp['cum'].iloc[0] if len(subp)>0 else 1
        subp['rebase'] = subp['cum']/base
        rebase_fig.add_trace(
            go.Scatter(
                x=subp.index,
                y=subp['rebase'],
                mode='lines',
                name=sid
            )
        )

    container.append(dcc.Graph(figure=rebase_fig))

    return container