# config.py
import os
import matplotlib

# 設定工作資料夾（以執行該檔案所在路徑為基準）
work_dir = os.path.dirname(os.path.abspath(__file__))
os.chdir(work_dir)

# 建議使用者安裝「思源黑體」或其他中文字體，這裡示範使用微軟正黑體
matplotlib.rcParams['font.family'] = 'Microsoft YaHei'
matplotlib.rcParams['font.size'] = 10

# 資料夾結構
list_dir = os.path.join(work_dir, "list")
stock_dir = os.path.join(work_dir, "stock")
output_dir = os.path.join(work_dir, "output")
heatmap_dir = os.path.join(output_dir, "heatmaps")
DL_dir =  os.path.join(work_dir, "DL")

os.makedirs(list_dir, exist_ok=True)
os.makedirs(stock_dir, exist_ok=True)
os.makedirs(output_dir, exist_ok=True)
os.makedirs(heatmap_dir, exist_ok=True)
os.makedirs(DL_dir, exist_ok=True)
