import os
import pandas as pd
work_dir = os.path.dirname(os.path.abspath(__file__))
os.chdir(work_dir)

# 設定目前資料夾
base_folder = os.path.join(os.getcwd(), "DL", "2412")

# 定義文件名稱
file_names = [
    "EquityDistribution.csv",
    "Dividend.csv",
    "MonthlyRevenue.csv",
    "PER_PBR.csv",
    "CashFlow.csv",
    "EPS_Year.csv",
    "EPS_Quar.csv",
]

# 定義文件路徑
file_paths = [os.path.join(base_folder, file) for file in file_names]

# 定義函數：讀取每個CSV文件的head並顯示
def show_csv_heads(file_paths):
    csv_heads = {}
    for path in file_paths:
        file_name = os.path.basename(path)  # 確保正確初始化
        if not os.path.exists(path):
            csv_heads[file_name] = f"Error: File not found at {path}"
            continue
        try:
            # 嘗試讀取CSV文件
            df = pd.read_csv(path)
            print(f"Reading file: {path}")
            print(f"df.head(10):\n{df.head(10)}")
        except Exception as e:
            csv_heads[file_name] = f"Error reading file: {e}"
    return csv_heads

# 執行函數並返回結果
csv_heads = show_csv_heads(file_paths)

