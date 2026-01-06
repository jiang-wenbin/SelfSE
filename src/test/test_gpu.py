import torch

m = k = n = 8192
a = torch.zeros(m, k, dtype=torch.float32).cuda("cuda:0")
b = torch.zeros(k, n, dtype=torch.float32).cuda("cuda:0")
for _ in range(10000):
    y = torch.matmul(a, b)
torch.cuda.synchronize("cuda:0")
