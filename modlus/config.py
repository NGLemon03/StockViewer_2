# config.py
import os

# 主資料夾路徑
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))

# 定義其他子資料夾路徑
DL_DIR = os.path.join(BASE_DIR, "DL")
LIST_DIR = os.path.join(BASE_DIR, "list")
OUTPUT_DIR = os.path.join(BASE_DIR, "output")
HEATMAP_DIR = os.path.join(OUTPUT_DIR, "heatmaps")

# 確保資料夾存在
os.makedirs(DL_DIR, exist_ok=True)
os.makedirs(LIST_DIR, exist_ok=True)
os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(HEATMAP_DIR, exist_ok=True)
