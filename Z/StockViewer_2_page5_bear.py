from dash import html, dcc
from modlus.bear_market_analysis import analyze_bear_market
from modlus.stock_data_processing import download_data
def render_page5_bear(stock_list):
    """
    analyze_bear_market(stock_data, start_date, end_date)
    """
    df = download_data(stock_list)
    if df.empty:
        return html.Div("無可用股價資料")

    # 假設預設兩段區間
    # 這裡不做互動輸入, 直接寫死
    period_map = {
        "疫情": ("2020-01-01","2020-05-01"),
        "FED升息":("2022-01-01","2022-12-31")
    }

    children_list = []
    for label,(sd,ed) in period_map.items():
        subres = analyze_bear_market(df, sd, ed)
        if subres.empty:
            children_list.append(html.Div(f"{label} 無資料"))
            continue
        # Convert to table
        subres_html = subres.style.format("{:.4f}").to_html()
        children_list.append(html.H4(f"{label} : {sd} ~ {ed}"))
        children_list.append(html.Div([
            dcc.Markdown(subres_html, dangerously_allow_html=True)
        ]))
    return html.Div(children_list)
