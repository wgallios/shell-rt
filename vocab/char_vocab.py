class CharVocab:
    def __init__(self, text: str):
        chars = sorted(list(set(text)))

        self.itos = ["<pad>"] + chars
        self.stoi = {ch:i for i,ch in enumerate(self.itos)}

    def encode(self, s: str) -> list[int]:
        return [self.stoi.get(ch, 0) for ch in s]

    def decode(self, ids: list[int]) -> str:
        return "".join(self.itos[i] for i in ids if i < len(self.itos) and i != 0)
