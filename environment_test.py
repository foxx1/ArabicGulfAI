import sys
import platform
import torch
import numpy as np
import pandas as pd

print("=" * 45)
print("ARABIC GULF AI - ENVIRONMENT CHECK")
print("=" * 45)

print("Python:", sys.version)
print("Platform:", platform.platform())
print("PyTorch:", torch.__version__)
print("NumPy:", np.__version__)
print("Pandas:", pd.__version__)

print("\nDevice:", "CUDA" if torch.cuda.is_available() else "CPU")

# Create a tensor
x = torch.tensor([[1.0, 2.0], [3.0, 4.0]])

# Perform matrix multiplication
result = torch.matmul(x, x)

print("\nTensor:")
print(x)

print("\nMatrix multiplication:")
print(result)

print("\nEnvironment test completed.")