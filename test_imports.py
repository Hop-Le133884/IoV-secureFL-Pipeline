# test_imports.py
import pandas as pd
import numpy as np
import torch
import flwr as fl
from opacus import PrivacyEngine
from cryptography.fernet import Fernet

print("Imports OK!")
print(f"PyTorch: {torch.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")
print(f"Flower: {fl.__version__}")