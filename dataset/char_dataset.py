from torch.utils.data import Dataset, DataLoader
from vocab.char_vocab import CharVocab
class CharDataset(Dataset):
    def __init__(self, text: str, vocab: CharVocab, seq_len: int=128):
        print("Preparing character dataset...")
