import torch.nn as nn

class CharLSTM(nn.Module):
    def __init__(self, vocab_size: int, emb_dim=128, hidden=256, layers=2, dropout=0.1):
        super().__init__()
        print(f"Initializing CharLSTM: vocab_size={vocab_size}, emb_dim={emb_dim}, hidden={hidden}, layers={layers}, dropout={dropout}")

        self.emb = nn.Embedding(vocab_size, emb_dim)
        self.lstm = nn.LSTM(emb_dim, hidden, num_layers=layers, dropout=dropout, batch_first=True)
        self.head = nn.Linear(hidden, vocab_size)

    def foward(self, x, h=None):
        x = self.emb(x)
        out, h = self.lstm(x, h)
        logits = self.head(out)

        return logits, h
