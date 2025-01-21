import streamlit as st



# ====== 主程式 ======
def streamlit_app():
    st.title("整合式多分頁介面")
    st.sidebar.title("功能導覽")
    page = st.sidebar.radio("選擇分頁", [
        "個股基本資訊", 
        "財報下載/預覽", 
        "回測", 
        "多標的比較", 
        "空頭分析"
    ])
