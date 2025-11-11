import os, re, json, argparse, math, random

from pathlib import Path
from typing import List, Tuple
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader


def main():
    print("Shell Next Command Prediction with LSTM")

    p = argparse.ArgumentParser(description="Train an LSTM model to predict the next shell command.")
    sub = p.add_subparsers(dest="cmd")

    t = sub.add_parser("train");
    t.add_argument("--epochs", type=int, default=3)

    args = p.parse_args()

    if (args.cmd == "train"):
        print("Start Training")
    elif (args.cmd == "suggest"):
        print("Start Suggesting")
    else:
        p.print_help()



if __name__ == "__main__":
    main()
