import torch
import torch.nn as nn
from mistral.model import Transformer, ModelArgs
from mistral.bitlinear import BitLinear, weight_quant, activation_quant
import json
import os
import time
import re
import argparse
from pathlib import Path

# Define the desired model size
vocab_size = 32000  # Example vocabulary size
embedding_dim = 4096  # Increase embedding dimension
hidden_dim = 14336  # Increase hidden dimension
n_layers = 32  # Increase number of layers
n_heads = 32  # Increase number of attention heads
head_dim = 128  # Adjust head dimension accordingly
max_batch_size = 3

# Create model arguments
model_args = ModelArgs(
    dim=embedding_dim,
    n_layers=n_layers,
    head_dim=head_dim,
    hidden_dim=hidden_dim,
    n_heads=n_heads,
    n_kv_heads=8,  # Adjust based on n_heads
    norm_eps=1e-5,
    vocab_size=vocab_size,
    sliding_window=4096,
    max_batch_size=max_batch_size,
)

model = Transformer(model_args)

# Without training, generated model will not have
# ternarized weights and quantized activations.
# Ternarize and quantize after generation
for name, module in model.named_modules():
    if isinstance(module, BitLinear):
        module.weight = nn.Parameter(weight_quant(module.weight))

for name, module in model.named_modules():
    if isinstance(module, nn.Linear) and not isinstance(module, BitLinear):
        module.forward = lambda x: activation_quant(module(x))

output_folder = Path("models/mistral_bitnet_7B")
output_folder.mkdir(parents=True, exist_ok=True)

torch.save(model.state_dict(), output_folder / "consolidated.00.pth")
with open(output_folder / "params.json", "w") as f:
    json.dump(model_args.to_dict(), f)

print("generated model")
