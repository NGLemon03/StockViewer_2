import os
import dash as dcc
import plotly.graph_objs as go
import pandas as pd
from modlus.config import DL_DIR
from plotly.subplots import make_subplots

# 2) 財報下載/預覽 Tab (含 4 合 1 圖: 單季EPS&YOY、毛利率/營業利益率/淨利率、4Q EPS合計、月營收)
#   原先 pages/2_財報.py
def render_page2_financial(stock_id):
    """
    讀取EPS_Quar.csv, MonthlyRevenue.csv, 產生 2x2 subplot.
    """
    base_dir = os.path.join(DL_DIR, stock_id)
    # --- 讀取EPS_Quar ---
    eps_f = os.path.join(base_dir, "EPS_Quar.csv")
    df_quarter = pd.DataFrame()
    if os.path.exists(eps_f):
        df_quarter = pd.read_csv(eps_f)
        # 簡化: 只抓 單季 EPS(元)_稅後EPS, 營業毛利, 營業收入, ...
        # 這裡可直接套您pages/2_財報.py的函式 => 先略寫
        # ... (與您pages/2_財報.py相同處理)...

    # (這裡為了示範，我們用簡化假資料)
    df_q = pd.DataFrame({
       'Date': pd.date_range("2023-01-01", periods=4, freq='Q'),
       'EPS': [1.2, 1.3, 1.1, 1.4],
       'YOY': [0.05,0.08,-0.1,0.15],
       'GrossMargin': [35,36,34,37],
       'OperatingMargin': [25,26,24,26],
       'NetMargin': [20,21,19,22],
    }).set_index('Date')

    # ---- 月營收 (示範用假資料) ----
    df_month = pd.DataFrame({
       'Date': pd.date_range("2023-01-01", periods=6, freq='MS'),
       'Revenue': [180,190,170,200,210,205]
    }).set_index('Date')

    # 建立 2x2 subplots
    fig = make_subplots(
        rows=2, cols=2,
        shared_xaxes=True,
        subplot_titles=["(A) 單季EPS & YOY","(B) 毛利/營業/淨利率",
                        "(C) 近四季EPS合計","(D) 月營收" ],
        vertical_spacing=0.12
    )

    # (A) 單季EPS (Bar) + YOY (Line,右軸)
    fig.add_trace(
        go.Bar(
            x=df_q.index, y=df_q['EPS'], name="EPS"
        ),
        row=1, col=1
    )
    fig.add_trace(
        go.Scatter(
            x=df_q.index,
            y=df_q['YOY']*100,
            name="EPSYOY(%)",
            yaxis='y2'
        ),
        row=1, col=1
    )
    # 設定右軸
    fig.update_layout(
        yaxis2=dict(
            overlaying='y',
            side='right'
        )
    )

    # (B) 三條線
    fig.add_trace(
        go.Scatter(
            x=df_q.index, y=df_q['GrossMargin'], name="毛利率(%)"
        ),
        row=1, col=2
    )
    fig.add_trace(
        go.Scatter(
            x=df_q.index, y=df_q['OperatingMargin'], name="營業利率(%)"
        ),
        row=1, col=2
    )
    fig.add_trace(
        go.Scatter(
            x=df_q.index, y=df_q['NetMargin'], name="淨利率(%)"
        ),
        row=1, col=2
    )

    # (C) 近四季 EPS 合計(假: 先把EPS累加)
    cumsum_val = df_q['EPS'].cumsum()
    fig.add_trace(
        go.Bar(
            x=cumsum_val.index, y=cumsum_val, name="近四季EPS合計"
        ),
        row=2, col=1
    )

    # (D) 月營收
    fig.add_trace(
        go.Scatter(
            x=df_month.index, y=df_month['Revenue'],
            name="月營收(億)"
        ),
        row=2, col=2
    )

    fig.update_layout(
        title=f"{stock_id} 財報分析 (示範)",
        hovermode='x unified',
        height=800
    )

    return dcc.Graph(figure=fig)