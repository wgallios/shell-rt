import os, re, json, argparse, math, random

from pathlib import Path
from typing import List, Tuple
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

from vocab.char_vocab import CharVocab
from dataset.char_dataset import CharDataset


def read_shell_history() -> str:
    paths = [Path("~/.zsh_history").expanduser(), Path("~/.bash_history").expanduser()]
    lines: List[str] = []

    for p in paths:
        if not p.exists(): continue

        with p.open("r", errors="ignore") as f:
            for line in f:
                line = line.rstrip("\n")

                #zsh format: ": 1627891234:0;git status"

                m = re.match(r"^: \d+:0;(.*)$", line)
                cmd = m.group(1) if m else line
                cmd = cmd.strip()

                if (cmd):
                    # print(f"Read command: {cmd}")
                    lines.append(cmd)

    return "\n".join(lines)





def train_model(args):
    print("Start Training Model")
    text = read_shell_history()
    print(f"Read {len(text)} characters from shell history.")

    if len(text) < 200:
        print("Not enough data to train. Please add more commands to your shell history.")
        return

    vocab = CharVocab(text)



def main():
    print("Shell Next Command Prediction with LSTM")

    p = argparse.ArgumentParser(description="Train an LSTM model to predict the next shell command.")
    sub = p.add_subparsers(dest="cmd")

    t = sub.add_parser("train");
    t.add_argument("--epochs", type=int, default=3)
    t.add_argument("--batch-size", type=int, default=64)
    t.add_argument("--seq-len", type=int, default=128)
    t.add_argument("--emb", type=int, default=128)
    t.add_argument("--hidden", type=int, default=256)
    t.add_argument("--layers", type=int, default=2)
    t.add_argument("--lr", type=float, default=3e-3)
    t.add_argument("--out-dir", type=str, default="./model")


    s = sub.add_parser("suggest");
    s.add_argument("--model", type=str, default="./model/checkpoint.pt")
    s.add_argument("--prompt", type=str, required=True, help="Seed text, e.g. 'git add .\\n'")
    s.add_argument("--max-new", type=int, default=120)
    s.add_argument("--temp", type=float, default=0.8)
    s.add_argument("--top-k", type=int, default=20)

    args = p.parse_args()

    if (args.cmd == "train"):
        train_model(args)
    elif (args.cmd == "suggest"):
        print("Start Suggesting")
    else:
        p.print_help()



if __name__ == "__main__":
    main()
