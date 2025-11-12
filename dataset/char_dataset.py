import torch
from torch.utils.data import Dataset, DataLoader
from vocab.char_vocab import CharVocab

class CharDataset(Dataset):
    def __init__(self, text: str, vocab: CharVocab, seq_len: int=128):
        self.vocab = vocab
        self.seq_len = seq_len

        data = vocab.encode(text)

        self.data = torch.tensor(data, dtype=torch.long)
