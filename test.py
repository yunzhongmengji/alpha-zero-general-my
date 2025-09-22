import torch, numpy as np
print("torch:", torch.__version__, "cuda:", torch.version.cuda)
print("CUDA available:", torch.cuda.is_available())
if torch.cuda.is_available():
    print("GPU:", torch.cuda.get_device_name(0))
# 简单跑一次GPU计算（若可用）
if torch.cuda.is_available():
    a = torch.randn(1024, 1024, device='cuda')
    b = torch.randn(1024, 1024, device='cuda')
    c = a @ b
    print("GPU matmul ok:", c.shape)
