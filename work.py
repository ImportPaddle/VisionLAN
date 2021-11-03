import paddle

x = paddle.randn([2, 3, 4])
print(x.shape)
x_transposed = paddle.transpose(x, perm=[1, 0, 2])
print(x_transposed.shape)
import torch

x = torch.randn([2, 3, 4])
print(x.shape)
x_transposed = x.permute(1, 0, 2)
print(x_transposed.shape)
# [3L, 2L, 4L]
