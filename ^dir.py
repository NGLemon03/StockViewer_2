import os

# 獲取當前腳本的目錄並設置為工作目錄
work_dir = os.path.dirname(os.path.abspath(__file__))
os.chdir(work_dir)

def list_directory_structure(root_dir, output_file, level=0):
    """
    遞歸列出目錄結構並保存到文件
    :param root_dir: 起始目錄
    :param output_file: 保存結果的文件路徑
    :param level: 用於控制層次結構的縮排
    """
    try:
        # 獲取目錄中的所有項目
        items = os.listdir(root_dir)
        for item in items:
            # 排除 .git 目錄
            if item == ".git":
                continue

            # 拼接完整路徑
            full_path = os.path.join(root_dir, item)

            # 構造項目名稱，按層次結構縮排
            line = "  " * level + f"|- {item}\n"
            with open(output_file, "a", encoding="utf-8") as f:
                f.write(line)

            # 如果是目錄，遞歸處理
            if os.path.isdir(full_path):
                list_directory_structure(full_path, output_file, level + 1)
    except PermissionError:
        with open(output_file, "a", encoding="utf-8") as f:
            f.write("  " * level + "|- [Permission Denied]\n")
    except FileNotFoundError:
        with open(output_file, "a", encoding="utf-8") as f:
            f.write("  " * level + "|- [Path Not Found]\n")

# 設置起始資料夾路徑
root_directory = "./"  # 替換為目標資料夾的路徑
output_file = "directory_structure.txt"  # 輸出的文件路徑

# 清空輸出文件
with open(output_file, "w", encoding="utf-8") as f:
    f.write(f"目錄結構: {root_directory}\n\n")

# 列出目錄結構並保存到文件
list_directory_structure(root_directory, output_file)

print(f"目錄結構已保存到 {output_file}")
