import torch

# 检查PyTorch编译时使用的CUDA版本
print(f"PyTorch CUDA version: {torch.version.cuda}")

# 检查CUDA是否可用
print(f"CUDA available: {torch.cuda.is_available()}")

# 检查当前GPU数量
print(f"GPU count: {torch.cuda.device_count()}")

# 检查GPU名称
if torch.cuda.is_available():
    print(f"GPU name: {torch.cuda.get_device_name(0)}")
