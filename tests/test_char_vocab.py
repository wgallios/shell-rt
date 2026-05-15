from vocab.char_vocab import CharVocab


def test_encode_decode_round_trip_known_characters():
    vocab = CharVocab("git status")

    encoded = vocab.encode("git")

    assert encoded == [vocab.stoi["g"], vocab.stoi["i"], vocab.stoi["t"]]
    assert vocab.decode(encoded) == "git"


def test_unknown_characters_encode_as_padding_and_decode_skips_padding():
    vocab = CharVocab("abc")

    assert vocab.encode("az") == [vocab.stoi["a"], 0]
    assert vocab.decode([vocab.stoi["a"], 0, vocab.stoi["b"]]) == "ab"
