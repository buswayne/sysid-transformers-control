from pathlib import Path
import time
import torch
import numpy as np
import math
from functools import partial
# from dataset import WHDataset, LinearDynamicalDataset
from torch.utils.data import DataLoader
from transformer_sim import Config, TSTransformer
from transformer_onestep import warmup_cosine_lr
import argparse
import wandb

if __name__ == '__main__':
    import torch

    out_dir = "out"
    cuda_device = "cuda:1"
    no_cuda = True
    threads = 10
    compile = False

    # Configure compute
    torch.set_num_threads(threads)
    use_cuda = not no_cuda and torch.cuda.is_available()
    device_name = cuda_device if use_cuda else "cpu"
    device = torch.device(device_name)
    device_type = 'cuda' if 'cuda' in device_name else 'cpu'  # for later use in torch.autocast
    torch.set_float32_matmul_precision("high")

    # Define the path to the checkpoint
    out_dir = "out"
    out_dir = Path(out_dir)
    checkpoint = torch.load(out_dir / "ckpt_wh1_prova.pt", map_location=device)

    # Print the keys of the checkpoint dictionary
    print(checkpoint.keys())

    # Inspect individual components
    print("Model state dict:", checkpoint['model'].keys())
    print("Optimizer state dict:", checkpoint['optimizer'].keys())
    print("Iteration number:", checkpoint['iter_num'])
    print("Training time:", checkpoint['train_time'])
    print("Loss history:", checkpoint['LOSS'])
    print("Validation loss history:", checkpoint['LOSS_VAL'])
    print("Best validation loss:", checkpoint['best_val_loss'])
    print("Configuration:", checkpoint['cfg'])

    model_state_dict = checkpoint['model']

    #for param_tensor in model_state_dict:
     #   print(param_tensor, "\t", model_state_dict[param_tensor].size)


