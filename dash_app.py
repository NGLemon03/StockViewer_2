import dash
from dash import dcc, html, Input, Output, State
import dash_bootstrap_components as dbc
from datetime import datetime
from pages.page1_basic_info import render_page1_basic_info
from pages.page2_financial import render_page2_financial
from pages.page3_backtest import render_page3_backtest
from pages.page4_compare import render_page4_compare
from pages.page5_bear import render_page5_bear
from modlus.DL_Y import download_stock_price  # 資料下載模組
from modlus.config import DL_DIR
# 建立 Dash App
app = dash.Dash(
    __name__,
    suppress_callback_exceptions=True,
    external_stylesheets=[dbc.themes.BOOTSTRAP]
)
app.title = "Dash 投資分析平台"

# 頁面導航
def render_navbar():
    return html.Div(
        id="navbar",
        children=[        
                dbc.Button("下載資料",id="toggle-collapse-button",className="mb-3",color="primary",outline=True,n_clicks=0,size="sm"),
                dbc.Collapse(
                    dbc.Card(
                        dbc.CardBody([
                            dbc.Row([
                                dbc.Col(html.Label("主要股票", className="fw-bold text-end"), width=4),
                                dbc.Col(dcc.Input(id="main_stock_id", type="text", value="2412", className="form-control"), width=8),
                            ], className="mb-3 align-items-center"),
                            dbc.Row([
                                dbc.Col(html.Label("市場基準", className="fw-bold text-end"), width=4),
                                dbc.Col(dcc.Input(id="market_id", type="text", value="^TWII", className="form-control"), width=8),
                            ], className="mb-3 align-items-center"),
                            dbc.Row([
                                dbc.Col(html.Label("比較股票", className="fw-bold text-end"), width=4),
                                dbc.Col(dcc.Input(id="other_ids", type="text", value="2330,00713,006208", className="form-control"), width=8),
                            ], className="mb-3 align-items-center"),
                            dbc.Row([
                                dbc.Col(html.Label("開始日期", className="fw-bold text-end"), width=4),
                                dbc.Col(dcc.Input(id="start_date", type="text", value="2000-01-01", className="form-control"), width=8),
                            ], className="mb-3 align-items-center"),
                            dbc.Row([
                                dbc.Col(html.Label("結束日期", className="fw-bold text-end"), width=4),
                                dbc.Col(dcc.Input(id="end_date", type="text", value=datetime.today().strftime("%Y-%m-%d"), className="form-control"), width=8),
                            ], className="mb-3 align-items-center"),
                            dbc.Button("下載資料", id="download_data_button", color="success", className="w-100"),
                            html.Div(id="download_status", className="mt-2 text-danger")
                        ]),),
                    id="collapse-area",
                    is_open=False
                ),
                # 導航欄的控制按鈕
                dbc.Button(
                    "⇆",  # 使用箭頭圖標
                    id="toggle-navbar", 
                    className="toggle-navbar",  # 初始類名稱
                    color="primary",
                    outline=True,
                    size="sm",
                    style={
                        "position": "absolute",  # 固定在視窗內，不隨內容滾動
                        "top": "50%",         # 垂直居中
                        "left": "15%",        # 與導航欄寬度一致
                        "transform": "translateY(-50%)",  # 垂直居中修正
                    }
                ),
                html.H2("導航欄",className="custom-heading"),
                dcc.Link("基本資訊", href="/page1", className="nav-link"), html.Br(),
                dcc.Link("財報圖表", href="/page2", className="nav-link"), html.Br(),
                dcc.Link("回測分析", href="/page3", className="nav-link"), html.Br(),
                dcc.Link("多標的比較", href="/page4", className="nav-link"), html.Br(),
                dcc.Link("空頭分析", href="/page5", className="nav-link"), html.Br(),
                html.Hr()
            ],         style={
            "width": "15%", 
            "padding": "10px", 
            "borderRight": "1px solid #ddd", 
            "fontSize": "14px",
            "transition": "width 0.3s ease",  # 添加過渡效果
            "overflow": "hidden"  # 隱藏溢出內容
        }
    )
app.layout = html.Div([
    dcc.Location(id="url"),
    html.Div(
        [
            render_navbar(),
            html.Div(
                id="page-content", 
                className="page-content"
            )
        ],
        className="main-container"  # Flexbox 容器
    )
])

'''
width: "75%"：設定頁面主要內容區域的寬度為 75%，與導航欄共同組成 100%。
float: "right"：使內容區域浮動到頁面右側，與左側的導航欄相鄰。
padding: "20px"：提供 20px 的內邊距，使內容與邊界保持距離。
'''


# 導航欄控制回調（使用 Flexbox 佈局）
@app.callback(
    [Output("navbar", "style"), Output("toggle-navbar", "style")],
    Input("toggle-navbar", "n_clicks"),
    [State("navbar", "style"), State("toggle-navbar", "style")]
)
def toggle_navbar(n_clicks, navbar_style, button_style):
    if n_clicks:
        if navbar_style.get("width", "15%") != "0%":  # 導航欄目前是顯示狀態
            navbar_style.update({
                "width": "0%",
                "padding": "0",
                "borderRight": "none"
            })
            button_style.update({
                "left": "0%"
            })
        else:  # 導航欄目前是隱藏狀態
            navbar_style.update({
                "width": "15%",
                "padding": "10px",
                "borderRight": "1px solid #ddd"
            })
            button_style.update({
                "left": "15%"
            })
    return navbar_style, button_style




# 頁面路由
@app.callback(
    Output("page-content", "children"),
    Input("url", "pathname")
)
def display_page(pathname):
    if pathname == "/page1":
        return render_page1_basic_info()
    elif pathname == "/page2":
        return render_page2_financial()
    elif pathname == "/page3":
        return render_page3_backtest()
    elif pathname == "/page4":
        return render_page4_compare()
    elif pathname == "/page5":
        return render_page5_bear()
    else:
        return html.Div("歡迎使用 Dash 投資分析平台！請從左側導航選擇一個頁面。")

# 資料下載 Callback
@app.callback(
    Output("download_status", "children"),
    Input("download_data_button", "n_clicks"),
    State("main_stock_id", "value"),
    State("market_id", "value"),
    State("other_ids", "value"),
    State("start_date", "value"),
    State("end_date", "value")
)
def download_data(n_clicks, main_stock_id, market_id, other_ids, start_date, end_date):
    if not n_clicks:
        return ""
    
    try:
        download_stock_price(
            stockID=main_stock_id,
            base_dir="./data",  # 下載資料的基準目錄
            start_date=start_date,
            end_date=end_date,
            start_year=int(start_date[:4]),
            start_quarter=1,
            end_year=int(end_date[:4]),
            end_quarter=4
        )
        return "資料下載成功！"
    except Exception as e:
        return f"下載失敗：{e}"

@app.callback(
    Output("collapse-area", "is_open"),
    [Input("toggle-collapse-button", "n_clicks")],
    [State("collapse-area", "is_open")]
)
def toggle_collapse(n_clicks, is_open):
    if n_clicks:
        return not is_open
    return is_open
# 啟動伺服器
if __name__ == "__main__":
    app.run_server(debug=True, port=27451)
