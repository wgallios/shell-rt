import torch

from dataset.char_dataset import CharDataset
from vocab.char_vocab import CharVocab


def test_dataset_len_counts_available_next_character_windows():
    vocab = CharVocab("abcdef")
    dataset = CharDataset("abcdef", vocab, seq_len=3)

    assert len(dataset) == 3


def test_dataset_returns_input_and_shifted_target_tensors():
    text = "abcdef"
    vocab = CharVocab(text)
    dataset = CharDataset(text, vocab, seq_len=3)

    x, y = dataset[1]

    assert torch.equal(x, torch.tensor(vocab.encode("bcd"), dtype=torch.long))
    assert torch.equal(y, torch.tensor(vocab.encode("cde"), dtype=torch.long))


def test_dataset_len_is_zero_when_text_is_too_short():
    vocab = CharVocab("abc")
    dataset = CharDataset("abc", vocab, seq_len=3)

    assert len(dataset) == 0
