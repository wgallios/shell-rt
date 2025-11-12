import torch.nn as nn

class CharLSTM(nn.Module):
    def __init__(self, vocab_size: int, emb_dim=128, hidden=256, layers=2, dropout=0.1):
        super().__init__()
        print(f"Initializing CharLSTM: vocab_size={vocab_size}, emb_dim={emb_dim}, hidden={hidden}, layers={layers}, dropout={dropout}")
