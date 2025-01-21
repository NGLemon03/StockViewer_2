import os
import shutil

# 設定工作目錄和目標目錄
work_dir = os.path.dirname(os.path.abspath(__file__))
os.chdir(work_dir)
source_dir = work_dir
target_dir = os.path.join(work_dir, "Z")
integrated_file = os.path.join(target_dir, "@Integr.py")
ver_name = os.path.basename(work_dir)

# 文件清單
files_to_process = [
    "bear_market_analysis.py",
    "config.py",
    'DL_Y.py',
    "fetch_stock_list.py",
    "financial_statements_fetcher.py",
    "stock_data_processing.py",
    "investment_indicators.py",
    "main.py",
    "streamlit_app.py",
    "pages/1_個股基本資訊.py",
    "pages/2_財報.py",
    "pages/3_回測.py",
    "pages/4_比較.py",
    "pages/5_空頭.py"
]

# 確認目標資料夾是否存在
os.makedirs(target_dir, exist_ok=True)

# 整合檔案初始化
if os.path.exists(integrated_file):
    os.remove(integrated_file)

# 開始處理文件
with open(integrated_file, "w", encoding="utf-8") as output_file:
    for file_name in files_to_process:
        source_path = os.path.join(source_dir, file_name)
        target_file_name = f"{ver_name}_{os.path.basename(file_name)}"
        target_path = os.path.join(target_dir, target_file_name)

        if os.path.exists(source_path):
            # 複製文件到目標目錄(覆蓋舊文件）
            shutil.copy(source_path, target_path)
            print(f"已複製: {file_name} -> {target_file_name}")

            # 整合代碼
            with open(source_path, "r", encoding="utf-8") as input_file:
                output_file.write(f"# {'-' * 10} Start of {file_name} {'-' * 10}\n")
                output_file.write(input_file.read())
                output_file.write(f"\n# {'-' * 10} End of {file_name} {'-' * 10}\n\n")
            print(f"已整合: {file_name}")
        else:
            print(f"找不到文件: {file_name}")

print(f"文件處理完成！代碼已整合至: {integrated_file}")
