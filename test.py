import pandas as pd
import os

# 原始 CSV 路径
csv_path = "/scratch/bdem/jcai1/bdem_usefulxai/code/CLAM/dataset_csv/tumor_vs_normal_dummy_just_test.csv"

# H5 文件所在目录
h5_dir = "/scratch/bdem/jcai1/bdem_usefulxai/data/camelyon16-clam/tumor_vs_normal_resnet_features/h5_files_labels"

# 读取 CSV
df = pd.read_csv(csv_path)

# 打印原始行数
print(f"[INFO] 原始总行数：{len(df)}")

# 检查每行对应的 h5 文件是否存在
def check_h5_exists(slide_id2):
    h5_path = os.path.join(h5_dir, f"{slide_id2}.h5")
    return os.path.exists(h5_path)

# 保留存在对应 h5 文件的行
df_filtered = df[df["slide_id2"].apply(check_h5_exists)].reset_index(drop=True)

# 打印剩余行数
print(f"[INFO] 过滤后剩余行数：{len(df_filtered)}")

# 保存为新 CSV 文件（也可以覆盖原始文件）
filtered_csv_path = csv_path.replace(".csv", "_filtered.csv")
df_filtered.to_csv(filtered_csv_path, index=False)
print(f"[INFO] 保存新文件：{filtered_csv_path}")
