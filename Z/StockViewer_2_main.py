# main_app.py
import streamlit as st

def main():
    st.set_page_config(
        page_title="我的多頁版投資分析平台",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    st.title("歡迎使用投資分析平台 (多頁版)")
    st.write("""
        左側是多頁面的導覽列，可在「個股基本資訊」、「財報」、「回測」、「比較」、「空頭」之間切換。
        \n在每個分頁中，可各自使用您先前整合的邏輯及介面。
    """)

if __name__ == "__main__":
    main()
