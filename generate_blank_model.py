import torch
import torch.nn as nn
from mistral.quant_utils import activation_quant, weight_quant
from mistral.model import Transformer, BitLinear, ModelArgs
from tqdm import tqdm
import json
import os
import time
import re
import argparse
from pathlib import Path

# Define the desired model size
vocab_size = 50000  # Example vocabulary size
embedding_dim = 2048  # Increase embedding dimension
hidden_dim = 8192  # Increase hidden dimension
n_layers = 48  # Increase number of layers
n_heads = 16  # Increase number of attention heads
head_dim = 128  # Adjust head dimension accordingly
max_batch_size = 1

# Create model arguments
model_args = ModelArgs(
    dim=embedding_dim,
    n_layers=n_layers,
    head_dim=head_dim,
    hidden_dim=hidden_dim,
    n_heads=n_heads,
    n_kv_heads=4,  # Adjust based on n_heads
    norm_eps=1e-6,
    vocab_size=vocab_size,
    max_batch_size=max_batch_size
)

model = Transformer(model_args)

output_folder = Path("mistral_bitnet_7B")
output_folder.mkdir(parents=True, exist_ok=True)

torch.save(model.state_dict(), output_folder / "model.pth")
with open(output_folder / "params.json", "w") as f:
    json.dump(model_args.to_dict(), f)

open(output_folder / "tokenizer.model", "a").close()

print("generated model")
