# 假設檔名: pages/2_財報.py
import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objs as go
from plotly.subplots import make_subplots
import os

##############################################################
# 1) 一些基礎函式: (quarter<->datetime, to_float, yoy, rolling4Q)
##############################################################
def quarter_to_datetime(quarter_str):
    """
    例如 '2024Q3' => pd.Timestamp('2024-09-28') (取季末月的 28 號)
    便於排序、繪圖能跟月/日資料對齊。
    """
    y = int(quarter_str[:4])
    q = int(quarter_str[-1])
    month_map = {1:3, 2:6, 3:9, 4:12}
    return pd.to_datetime(f"{y}-{month_map[q]}-28")

def datetime_to_quarter(dt):
    """反向: 2024-09-xx => '2024Q3'"""
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
    """把 CSV 中的字串(帶逗號、'-'等) 轉成 float"""
    if isinstance(val, str):
        val = val.replace(",", "").replace("--","").replace("-", "").strip()
    try:
        return float(val)
    except:
        return np.nan

def compute_yoy(df, col_name):
    """
    以【同一季度】做年增率: yoy = (本季 - 去年同季)/去年同季
    df 的 index 為『可排序』(datetime)，shift(4)表示往前 4 筆 => 同季
    """
    df[f"{col_name}_YOY"] = df[col_name].pct_change(4)
    return df

def compute_4Q_rolling(df: pd.DataFrame, cols=None):
    """
    將df中的 cols 欄位做 rolling(4) 累計 (近四季)。
    df的index必須是可排序的 datetime。
    """
    if cols is None:
        cols = df.columns
    df_4q = df[cols].rolling(4).sum()
    # 回復成 'YYYYQX' 當作index (非必要, 只是顯示好看)
    df_4q.index = df_4q.index.map(datetime_to_quarter)
    df_4q = df_4q.add_suffix("_4Q")
    return df_4q


##############################################################
# 2) 讀取 EPS_Quar.csv (單季財報)
##############################################################
def load_quarter_eps_csv(base_dir: str):
    """
    讀取 Goodinfo 抓下來的 EPS_Quar.csv。
     - 回傳 DataFrame(index=datetime, columns=[營業收入,毛利,EPS,...])
     - 若檔案不存在或欄位對不上，回傳空 DataFrame
    """
    fpath = os.path.join(base_dir, "EPS_Quar.csv")
    if not os.path.exists(fpath):
        return pd.DataFrame()

    df = pd.read_csv(fpath)
    if '季度_季度' not in df.columns:
        return pd.DataFrame()

    # 這裡示範抓「營業收入」「營業毛利」「營業利益」「稅後淨利」「EPS」五個欄位：
    df['季度'] = df['季度_季度'].astype(str).str.strip()

    # 注意：若 CSV 實際欄名不一樣，請自行對應
    df['營業收入'] = df.get('獲利金額(億)_營業收入', pd.Series([np.nan]*len(df))).apply(to_float)
    df['營業毛利'] = df.get('獲利金額(億)_營業毛利', pd.Series([np.nan]*len(df))).apply(to_float)
    df['營業利益'] = df.get('獲利金額(億)_營業利益', pd.Series([np.nan]*len(df))).apply(to_float)
    df['稅後淨利'] = df.get('獲利金額(億)_稅後淨利', pd.Series([np.nan]*len(df))).apply(to_float)

    # EPS 欄位可能是 'EPS(元)_稅後EPS' or 'EPS(元)_稅後EPS(元)'，請視實際情況修改
    eps_col_candidates = ["EPS(元)_稅後EPS", "EPS(元)_稅後EPS(元)"]
    found_eps_col = None
    for c in eps_col_candidates:
        if c in df.columns:
            found_eps_col = c
            break
    if found_eps_col:
        df['EPS'] = df[found_eps_col].apply(to_float)
    else:
        df['EPS'] = np.nan

    # 篩出有效季度
    df = df.dropna(subset=['季度'])
    df = df[df['季度'].str.match(r'^\d{4}Q[1-4]$')]
    df = df.set_index('季度', drop=True)

    # 將 index('2024Q3') -> datetime
    dt_index = df.index.map(quarter_to_datetime)
    df = df.set_index(dt_index).sort_index()

    # 只留關鍵欄
    keep_cols = ['營業收入','營業毛利','營業利益','稅後淨利','EPS']
    df = df[keep_cols]

    # 把全是 NaN 的行丟掉
    df = df.dropna(how='all')
    return df


##############################################################
# 3) 讀取 MonthlyRevenue.csv (月營收)
##############################################################
def load_monthly_revenue_csv(base_dir: str):
    """
    讀取 Goodinfo 抓下來的 MonthlyRevenue.csv，
     - 回傳 DataFrame(index=datetime, columns=['當月營收','月增(%)','年增(%)']等)
     - 若檔案不存在或欄位對不上，回傳空 DataFrame
    """
    fpath = os.path.join(base_dir, "MonthlyRevenue.csv")
    if not os.path.exists(fpath):
        return pd.DataFrame()

    df = pd.read_csv(fpath)
    if '月別_月別_月別' not in df.columns:
        return pd.DataFrame()

    # 假設 CSV 有 "YYYY/MM" 在 '月別_月別_月別'
    df['年月'] = df['月別_月別_月別'].astype(str).str.strip()
    df = df[df['年月'].str.match(r'^\d{4}/\d{2}$')]  # 篩掉空白列

    # 將 '2024/09' -> datetime(2024,9,1)
    def ym_to_dt(ym):
        y,m = ym.split('/')
        return pd.to_datetime(f"{y}-{m}-1")

    dt_index = df['年月'].map(ym_to_dt)
    df.set_index(dt_index, inplace=True)
    df.sort_index(inplace=True)

    # 取出營業收入 (單月)、月增、年增(若有) => 視實際欄位名而定
    df['單月營收(億)'] = df.get('營業收入_單月_營收(億)', pd.Series([np.nan]*len(df))).apply(to_float)
    df['月增(%)']   = df.get('營業收入_單月_月增(%)', pd.Series([np.nan]*len(df))).apply(to_float)
    df['年增(%)']   = df.get('營業收入_單月_年增(%)', pd.Series([np.nan]*len(df))).apply(to_float)

    # 同理: 把全是 NaN 的行刪除
    df = df[['單月營收(億)','月增(%)','年增(%)']].dropna(how='all')
    return df


##############################################################
# 4) 在同一張 Figure 內放「四個子圖」
#    (A) 單季EPS & EPS YOY
#    (B) 毛利率/營業利益率/淨利率
#    (C) 近四季累計EPS
#    (D) 月營收趨勢
#   => 有了 shared_xaxes=True + hovermode='x unified'，就能移動游標時彼此同步。
##############################################################
def make_four_subplots(df_quarter: pd.DataFrame, df_monthly: pd.DataFrame, stock_id: str):
    """
    df_quarter: 單季財報資料 (index=datetime)
    df_monthly: 月營收資料 (index=datetime)
    回傳 figure
    """

    # 若無資料，直接回傳空圖
    if df_quarter.empty and df_monthly.empty:
        fig = go.Figure()
        fig.update_layout(title="無可用資料")
        return fig

    # 先做一些計算:
    # 1) 單季年增率
    df_q = df_quarter.copy()
    for col in ['營業收入','稅後淨利','EPS']:
        df_q = compute_yoy(df_q, col)

    # 2) 毛利率/營業利益率/淨利率
    df_q['毛利率(%)']   = df_q['營業毛利'] / df_q['營業收入'] * 100
    df_q['營業利益率(%)'] = df_q['營業利益'] / df_q['營業收入'] * 100
    df_q['淨利率(%)']   = df_q['稅後淨利'] / df_q['營業收入'] * 100

    # 3) 近四季累計EPS
    df_4q = compute_4Q_rolling(df_q, cols=['EPS'])

    # 建立多子圖 (2x2)
    fig = make_subplots(
        rows=2, cols=2,
        shared_xaxes=True,    # 讓 X 軸同步
        vertical_spacing=0.15,
        horizontal_spacing=0.08,
        subplot_titles=[
            "(A) 單季 EPS & YOY",
            "(B) 毛利率 / 營業利益率 / 淨利率",
            "(C) 近四季累計EPS",
            "(D) 月營收"
        ]
    )

    # ====== (A) 單季 EPS(Bar) + YOY(%)Line => row=1, col=1 ======
    fig.add_trace(
        go.Bar(
            x=df_q.index,  # datetime
            y=df_q['EPS'],
            name="單季EPS",
            marker_color='rgb(31, 119, 180)'
        ),
        row=1, col=1
    )
    # 加一條 YOY(%) => 同一子圖的右軸
    fig.add_trace(
        go.Scatter(
            x=df_q.index,
            y=df_q['EPS_YOY']*100,
            mode='lines+markers',
            name="EPS年增率(%)",
            yaxis='y2'
        ),
        row=1, col=1
    )

    # 設定 row=1 col=1 的右軸
    fig.update_layout(
        yaxis2=dict(
            overlaying='y',
            side='right',
            showgrid=False,
        )
    )

    # ====== (B) 毛利率/營業利益率/淨利率 => row=1, col=2 ======
    fig.add_trace(
        go.Scatter(
            x=df_q.index,
            y=df_q['毛利率(%)'],
            mode='lines+markers',
            name="毛利率(%)",
        ),
        row=1, col=2
    )
    fig.add_trace(
        go.Scatter(
            x=df_q.index,
            y=df_q['營業利益率(%)'],
            mode='lines+markers',
            name="營業利益率(%)"
        ),
        row=1, col=2
    )
    fig.add_trace(
        go.Scatter(
            x=df_q.index,
            y=df_q['淨利率(%)'],
            mode='lines+markers',
            name="淨利率(%)"
        ),
        row=1, col=2
    )

    # ====== (C) 近四季累計EPS => row=2, col=1 ======
    if not df_4q.empty:
        fig.add_trace(
            go.Bar(
                x=df_4q.index,
                y=df_4q['EPS_4Q'],
                name="近四季EPS合計",
                marker_color='rgb(255, 127, 14)'
            ),
            row=2, col=1
        )

    # ====== (D) 月營收 => row=2, col=2 ======
    if not df_monthly.empty:
        fig.add_trace(
            go.Scatter(
                x=df_monthly.index,
                y=df_monthly['單月營收(億)'],
                mode='lines+markers',
                name="單月營收(億)",
                marker_color='rgb(44, 160, 44)'
            ),
            row=2, col=2
        )

    # 統一設定
    fig.update_layout(
        title=f"{stock_id} 財報與月營收多圖",
        hovermode='x unified',  # 滑鼠移動同一X值處可同時顯示
        height=900
    )

    return fig


##############################################################
# 5) Streamlit 頁面主函式
##############################################################
def page_financial_analysis():
    st.title("財報 + 月營收 多圖同步示範")

    # 欄位設計
    colA, colB = st.columns([1,2])
    with colA:
        stock_id = st.text_input("請輸入需分析之股票代號", value="2412")

    base_dir = os.path.join("./DL", stock_id)

    if st.button("讀取並繪圖"):
        st.info(f"正在嘗試從 {base_dir} 讀取 EPS_Quar.csv 以及 MonthlyRevenue.csv ...")

        # 讀取單季財報
        df_quarter = load_quarter_eps_csv(base_dir)
        # 讀取月營收
        df_monthly = load_monthly_revenue_csv(base_dir)

        if df_quarter.empty and df_monthly.empty:
            st.warning("找不到EPS_Quar.csv或MonthlyRevenue.csv (或內容皆空)，無法分析。請先下載。")
            return

        # 建立 2x2 子圖
        fig = make_four_subplots(df_quarter, df_monthly, stock_id)
        st.plotly_chart(fig, use_container_width=True)

        # 另外提供表格查看
        with st.expander("單季財報 (原始)"):
            st.dataframe(df_quarter.style.format("{:.2f}"))

        with st.expander("月營收 (原始)"):
            st.dataframe(df_monthly.style.format("{:.2f}"))

    else:
        st.write("請輸入代號後，點擊『讀取並繪圖』")


# 若直接執行本檔 (python 2_財報.py)，進入下方測試。
if __name__ == "__main__":
    page_financial_analysis()
