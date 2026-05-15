import torch

from LSTM.CharLSTM import CharLSTM


def test_forward_returns_logits_for_each_batch_and_sequence_position():
    model = CharLSTM(vocab_size=7, emb_dim=4, hidden=5, layers=1, dropout=0.0)
    x = torch.tensor([[1, 2, 3], [4, 5, 6]], dtype=torch.long)

    logits, hidden = model(x)

    assert logits.shape == (2, 3, 7)
    assert hidden[0].shape == (1, 2, 5)
    assert hidden[1].shape == (1, 2, 5)
