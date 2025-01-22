# 假設檔名: pages/2_財報.py

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objs as go
from plotly.subplots import make_subplots
import os

##########################
#   1) 輔助函式
##########################

def quarter_to_datetime(quarter_str):
    """
    '2024Q3' => pd.Timestamp('2024-09-30') 這樣可排序
    這裡簡化用每季末月做日期
    """
    y = int(quarter_str[:4])
    q = int(quarter_str[-1])
    month_map = {1:3, 2:6, 3:9, 4:12}
    return pd.to_datetime(f"{y}-{month_map[q]}-28")  # 取月底(略有彈性)

def datetime_to_quarter(dt):
    """反向: 2024-09-30 => '2024Q3'"""
    y = dt.year
    m = dt.month
    if m <= 3:
        q = 1
    elif m <= 6:
        q = 2
    elif m <= 9:
        q = 3
    else:
        q = 4
    return f"{y}Q{q}"

def to_float(val):
    """把CSV中帶逗號或空白的數字轉成 float"""
    if isinstance(val, str):
        val = val.replace(",", "").replace("--","").strip()
    try:
        return float(val)
    except:
        return np.nan

def compute_4Q_rolling(df: pd.DataFrame, cols=None):
    """
    將df中的 cols 欄位做 rolling(4) 累計.
    df的index必須是可排序的 datetime (或已經按季度順序)
    """
    if cols is None:
        cols = df.columns
    df_4q = df[cols].rolling(4).sum()
    # 復原季度索引
    df_4q.index = df_4q.index.map(datetime_to_quarter)
    # 欄位改名: 營業收入 -> 營業收入_4Q
    df_4q = df_4q.add_suffix("_4Q")
    return df_4q

def compute_yoy(df, col_name):
    """
    用相同季度做年增率: yoy = (本季 - 去年同季)/去年同季
    df index 為 datetime 或排序後的quarter，shift(4) 即可。
    """
    df[f"{col_name}_YOY"] = df[col_name].pct_change(4)
    return df

##########################
#   2) 讀取EPS_Quar.csv
##########################
def load_quarter_eps_csv(base_dir: str):
    """
    讀取 Goodinfo 抓下來的 EPS_Quar.csv，
    並將主要欄位轉成 float, index = quarter_str => 轉成 datetime => sort_index
    回傳 DataFrame（index 為 datetime）
    """
    fpath = os.path.join(base_dir, "EPS_Quar.csv")
    if not os.path.exists(fpath):
        return pd.DataFrame()

    df = pd.read_csv(fpath)
    # 假設 CSV 有欄位 '季度_季度' (ex: 2024Q3) 與 '獲利金額(億)_營業收入'、'獲利金額(億)_營業毛利'...
    if '季度_季度' not in df.columns:
        return pd.DataFrame()

    # 取出有用欄位 (可依您實際需要調整)
    df['季度']   = df['季度_季度'].astype(str).str.strip()
    df['營業收入'] = df['獲利金額(億)_營業收入'].apply(to_float)
    df['營業毛利'] = df['獲利金額(億)_營業毛利'].apply(to_float)
    df['營業利益'] = df['獲利金額(億)_營業利益'].apply(to_float)
    df['稅後淨利'] = df['獲利金額(億)_稅後淨利'].apply(to_float)
    df['EPS']   = df['EPS(元)_稅後EPS'].apply(to_float)

    # 只保留有效季度
    df = df.dropna(subset=['季度'])
    df = df[df['季度'].str.match(r'^\d{4}Q[1-4]$')]
    df = df.set_index('季度', drop=True)

    # 將index(季度)轉成 datetime 以便排序、rolling
    dt_index = df.index.map(quarter_to_datetime)
    df = df.set_index(dt_index).sort_index()

    # 清理出需要的欄位
    keep_cols = ['營業收入','營業毛利','營業利益','稅後淨利','EPS']
    df = df[keep_cols].dropna(how='all')
    return df

##########################
#   3) 做分析 + 畫圖
##########################
def analyze_and_plot_quarterly(df: pd.DataFrame, stock_id: str):
    """
    傳入「單季」財報df (index為datetime, columns=[營業收入,毛利,利益,EPS,...])，
    回傳 plotly figure + (df單季, df_4Q)
    """
    if df.empty:
        fig = go.Figure()
        fig.update_layout(title="無季度資料")
        return fig, pd.DataFrame(), pd.DataFrame()

    # 1) 計算近四季合計
    df_4q = compute_4Q_rolling(df, cols=['營業收入','營業毛利','營業利益','稅後淨利','EPS'])

    # 2) 單季年增率
    df_analysis = df.copy()
    for c in ['營業收入','稅後淨利','EPS']:
        df_analysis = compute_yoy(df_analysis, c)
    # 3) 毛利率 / 營業利益率 / 淨利率
    df_analysis['毛利率(%)']   = (df_analysis['營業毛利'] / df_analysis['營業收入'])*100
    df_analysis['營業利益率(%)'] = (df_analysis['營業利益'] / df_analysis['營業收入'])*100
    df_analysis['淨利率(%)']   = (df_analysis['稅後淨利'] / df_analysis['營業收入'])*100

    # 4) 繪圖: 用 subplot 分3區
    fig = make_subplots(
        rows=3, cols=1,
        shared_xaxes=True,
        subplot_titles=[
            f"{stock_id} 單季 EPS & 年增率",
            "毛利率 / 營業利益率 / 淨利率 (單季)",
            "近四季累計EPS"
        ],
        vertical_spacing=0.08
    )

    # (A) 單季 EPS
    fig.add_trace(
        go.Bar(
            x=df_analysis.index,  # datetime
            y=df_analysis['EPS'],
            name="單季EPS"
        ),
        row=1, col=1
    )
    # 右軸: EPS年增率
    fig.add_trace(
        go.Scatter(
            x=df_analysis.index,
            y=df_analysis['EPS_YOY']*100,
            mode='lines+markers',
            name="EPS年增率(%)",
            yaxis='y2'
        ),
        row=1, col=1
    )

    # 設定 row1 右軸
    fig.update_layout(
        yaxis2=dict(
            overlaying='y',
            side='right',
            showgrid=False
        )
    )

    # (B) 毛利率 / 營業利益率 / 淨利率
    fig.add_trace(
        go.Scatter(x=df_analysis.index, y=df_analysis['毛利率(%)'], name="毛利率(%)"),
        row=2, col=1
    )
    fig.add_trace(
        go.Scatter(x=df_analysis.index, y=df_analysis['營業利益率(%)'], name="營業利益率(%)"),
        row=2, col=1
    )
    fig.add_trace(
        go.Scatter(x=df_analysis.index, y=df_analysis['淨利率(%)'], name="淨利率(%)"),
        row=2, col=1
    )

    # (C) 近四季累計 EPS
    fig.add_trace(
        go.Bar(
            x=df_4q.index, 
            y=df_4q['EPS_4Q'],
            name="近四季累計EPS"
        ),
        row=3, col=1
    )

    fig.update_layout(
        title=f"{stock_id} 財報關鍵指標",
        hovermode='x unified',
        height=900
    )

    return fig, df_analysis, df_4q


##########################
#   4) Streamlit 頁面
##########################
def page_financial_analysis():
    # 輸入框(預設2412)
    colA, colB = st.columns([1,2])
    with colA:
        stock_id = st.text_input("請輸入需分析之股票代號:", value="2412")
    with colB:
        # 預留訊息顯示區域，避免訊息對齊問題
        with st.container():
            message_placeholder = st.empty()
            message_placeholder1 = st.empty()

    # 指定讀哪個資料夾
    base_dir = os.path.join("./DL", stock_id)

    if st.button("讀取與分析"):
        message_placeholder1.markdown(f"⚙️ **嘗試從 `{base_dir}` 讀取EPS_Quar.csv...**")
        df_q = load_quarter_eps_csv(base_dir)

        # 檢查資料是否存在
        if df_q.empty:
            message_placeholder.markdown(
                "<span style='color:red;'>找不到 EPS_Quar.csv 或檔案內容不符，無法分析。請先在『個股基本資訊』下載。</span>",
                unsafe_allow_html=True
            )
            return

        # 分析和繪圖
        fig, df_single, df_4q = analyze_and_plot_quarterly(df_q, stock_id)
        st.plotly_chart(fig, use_container_width=True)

        # 顯示數據表 (可收合)
        with st.expander("查看單季數據(含YOY & 利潤率)"):
            st.dataframe(df_single.style.format("{:.2f}"))

        with st.expander("查看近四季累計"):
            st.dataframe(df_4q.style.format("{:.2f}"))

        # 更新提示訊息
        message_placeholder.markdown(
            "<span style='color:green;'>✅ 分析完成！</span>",
            unsafe_allow_html=True
        )
    else:
        # 顯示初始提示
        message_placeholder.markdown(
            "📋 **請輸入股票代號後，按下『讀取與分析』。**"
        )



# 下方這段只在「直接執行 pages/2_財報.py」時才會觸發
# 若您是透過 streamlit_app.py 的多頁式導覽，則只會呼叫 page_financial_analysis()。
if __name__ == "__main__":
    page_financial_analysis()
