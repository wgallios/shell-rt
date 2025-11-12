import torch
from torch.utils.data import Dataset, DataLoader
from vocab.char_vocab import CharVocab

class CharDataset(Dataset):
    def __init__(self, text: str, vocab: CharVocab, seq_len: int=128):
        self.vocab = vocab
        self.seq_len = seq_len

        data = vocab.encode(text)

        self.data = torch.tensor(data, dtype=torch.long)

        self.n = len(self.data - seq_len - 1)
        self.n = max(self.n, 0)

    def __len__(self):
        return self.n

    def __getitem__(self, idx):
        x = self.data[idx:idex+self.seq_len]
        y = self.data[idx+1:idx+self.seq_len+1]
        return x, y
