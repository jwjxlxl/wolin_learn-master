import numpy as np
import torch

a = np.random.rand(2, 3)
t = torch.rand(2, 3)

# print(a.shape, t.shape)     # (2,3), torch.Size([2,3])
# print(a.ndim, t.ndim)       # 维度数
# print(a.size, t.numel())    # 元素总数
# print(a.dtype, t.dtype)     # 数据类型
#
# # NumPy
# a = np.arange(9).reshape(3, 3)
# print(a[0])      # 第一行
# print(a[:, 1])   # 第二列
# print(a[1:, ::2])
#
# # PyTorch
# t = torch.arange(9).reshape(3, 3)
# print(t[0])
# print(t[:, 1])
# print(t[1:, ::2])

t = torch.rand(1000, 1000).to("cuda")
print(t.ndim)