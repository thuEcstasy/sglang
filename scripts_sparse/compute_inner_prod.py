import torch
import matplotlib.pyplot as plt
import seaborn as sns

# 加载 tensor 并转为 float32
tensor = torch.load("/home/haizhonz/Zhaofeng/sglang/scripts_sparse/layer_2_dense/kv_head_1.pt")
tensor = tensor.to(torch.float32)

# 计算内积矩阵
dot_product_matrix = tensor @ tensor.T
dot_product_matrix = dot_product_matrix.cpu().numpy()

# 画热力图，红色越深值越大
plt.figure(figsize=(10, 8))
sns.heatmap(dot_product_matrix, cmap="Reds")  # Reds 表示值越大越红
plt.title("Token-to-Token Inner Product (Reds Heatmap)")
plt.xlabel("Token Index")
plt.ylabel("Token Index")
plt.tight_layout()
plt.show()

plt.savefig("inner_product_heatmap.png")
