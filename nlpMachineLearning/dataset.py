# src/dataset.py
import os
import gzip
import pickle
from collections import Counter
from typing import List, Optional, Tuple

import spacy
import torch
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence

# =============================
# 0. Helpers: try to load spaCy models, fallback to blank tokenizers
# =============================
def load_spacy_tokenizer(model_name: str):
    try:
        return spacy.load(model_name)
    except Exception:
        # fallback to blank model (works but less accurate)
        return spacy.blank(model_name.split("_")[0] if "_" in model_name else model_name)

_spacy_en = load_spacy_tokenizer("en_core_web_sm")
_spacy_fr = load_spacy_tokenizer("fr_core_news_sm")

def tokenize_en(text: str) -> List[str]:
    return [tok.text.lower() for tok in _spacy_en.tokenizer(text.strip()) if tok.text.strip()]

def tokenize_fr(text: str) -> List[str]:
    return [tok.text.lower() for tok in _spacy_fr.tokenizer(text.strip()) if tok.text.strip()]

# =============================
# 1. Vocab class
# =============================
class Vocab:
    """
    Minimal vocabulary builder.
    - special tokens: <unk>, <pad>, <sos>, <eos>
    - max_size: limit most frequent tokens
    - freq_threshold: minimum freq to keep token
    """
    def __init__(self, freq_threshold: int = 1, max_size: Optional[int] = 10000):
        self.freq_threshold = freq_threshold
        self.max_size = max_size

        # Order chosen so pad idx = 1 (or you can change). We define explicitly.
        # We'll set indices as: <unk>=0, <pad>=1, <sos>=2, <eos>=3
        self.specials = ["<unk>", "<pad>", "<sos>", "<eos>"]
        self.itos = {i: tok for i, tok in enumerate(self.specials)}
        self.stoi = {tok: i for i, tok in self.itos.items()}
        self._frozen = False

    def build_vocabulary(self, sentence_list: List[List[str]]):
        if self._frozen:
            return
        freqs = Counter()
        for sent in sentence_list:
            freqs.update(sent)

        # filter by freq threshold and sort by frequency desc
        words = [w for w, f in freqs.items() if f >= self.freq_threshold]
        words_sorted = sorted(words, key=lambda w: freqs[w], reverse=True)

        if self.max_size:
            words_sorted = words_sorted[: max(0, self.max_size - len(self.specials))]

        idx = len(self.specials)
        for w in words_sorted:
            if w not in self.stoi:
                self.stoi[w] = idx
                self.itos[idx] = w
                idx += 1

        self._frozen = True

    def numericalize(self, tokens: List[str]) -> List[int]:
        return [self.stoi.get(t, self.stoi["<unk>"]) for t in tokens]

    @property
    def pad_idx(self) -> int:
        return self.stoi["<pad>"]

    @property
    def unk_idx(self) -> int:
        return self.stoi["<unk>"]

    def __len__(self):
        return len(self.stoi)

# =============================
# 2. Dataset
# =============================
class TranslationDataset(Dataset):
    """
    - src_path, trg_path: raw text files (one sentence per line), can be .gz
    - tokenizers: functions returning list[str]
    - src_vocab/trg_vocab: optional pre-built Vocab
    """
    def __init__(
        self,
        src_path: str,
        trg_path: str,
        src_tokenizer,
        trg_tokenizer,
        src_vocab: Optional[Vocab] = None,
        trg_vocab: Optional[Vocab] = None,
        max_lines: Optional[int] = None,
    ):
        self.src_lines = self._read_file(src_path, max_lines)
        self.trg_lines = self._read_file(trg_path, max_lines)
        assert len(self.src_lines) == len(self.trg_lines), "src/trg length mismatch"

        # tokenized (list[list[str]])
        self.src_tokens = [src_tokenizer(line) for line in self.src_lines]
        self.trg_tokens = [trg_tokenizer(line) for line in self.trg_lines]

        # build vocabularies if not provided
        if src_vocab is None:
            self.src_vocab = Vocab(max_size=10000)
            self.src_vocab.build_vocabulary(self.src_tokens)
        else:
            self.src_vocab = src_vocab

        if trg_vocab is None:
            self.trg_vocab = Vocab(max_size=10000)
            self.trg_vocab.build_vocabulary(self.trg_tokens)
        else:
            self.trg_vocab = trg_vocab

    @staticmethod
    def _read_file(path: str, max_lines: Optional[int] = None) -> List[str]:
        if path.endswith(".gz"):
            with gzip.open(path, mode="rt", encoding="utf-8") as f:
                lines = [l.rstrip("\n") for l in f]
        else:
            with open(path, "r", encoding="utf-8") as f:
                lines = [l.rstrip("\n") for l in f]
        if max_lines is not None:
            return lines[:max_lines]
        return lines

    def __len__(self):
        return len(self.src_tokens)

    def __getitem__(self, idx):
        # return raw numericalized sequence WITHOUT padding
        src_seq = [self.src_vocab.stoi["<sos>"]] + self.src_vocab.numericalize(self.src_tokens[idx]) + [self.src_vocab.stoi["<eos>"]]
        trg_seq = [self.trg_vocab.stoi["<sos>"]] + self.trg_vocab.numericalize(self.trg_tokens[idx]) + [self.trg_vocab.stoi["<eos>"]]

        return torch.tensor(src_seq, dtype=torch.long), torch.tensor(trg_seq, dtype=torch.long)

# =============================
# 3. Collate function (padding + sort by length desc)
# =============================
class Collate:
    """
    Pads src and trg separately, returns:
    src_padded, trg_padded, src_lengths, trg_lengths
    - all tensors are sorted by src_lengths descending (suitable for pack_padded_sequence(enforce_sorted=True))
    """
    def __init__(self, src_pad_idx: int, trg_pad_idx: int):
        self.src_pad_idx = src_pad_idx
        self.trg_pad_idx = trg_pad_idx

    def __call__(self, batch: List[Tuple[torch.Tensor, torch.Tensor]]):
        src_batch, trg_batch = zip(*batch)  # tuples of tensors (variable lengths)

        # lengths BEFORE padding
        src_lengths = torch.tensor([s.size(0) for s in src_batch], dtype=torch.long)
        trg_lengths = torch.tensor([t.size(0) for t in trg_batch], dtype=torch.long)

        # pad
        src_padded = pad_sequence(src_batch, batch_first=True, padding_value=self.src_pad_idx)
        trg_padded = pad_sequence(trg_batch, batch_first=True, padding_value=self.trg_pad_idx)

        # sort by src length descending
        src_lengths_sorted, perm_idx = src_lengths.sort(descending=True)
        src_padded = src_padded[perm_idx]
        trg_padded = trg_padded[perm_idx]
        trg_lengths_sorted = trg_lengths[perm_idx]

        return src_padded, trg_padded, src_lengths_sorted, trg_lengths_sorted

# =============================
# 4. DataLoader factory
# =============================
def get_loader(
    src_path: str,
    trg_path: str,
    batch_size: int = 64,
    shuffle: bool = True,
    src_tokenizer=tokenize_en,
    trg_tokenizer=tokenize_fr,
    src_vocab: Optional[Vocab] = None,
    trg_vocab: Optional[Vocab] = None,
    max_lines: Optional[int] = None,
) -> Tuple[DataLoader, TranslationDataset]:
    ds = TranslationDataset(
        src_path=src_path,
        trg_path=trg_path,
        src_tokenizer=src_tokenizer,
        trg_tokenizer=trg_tokenizer,
        src_vocab=src_vocab,
        trg_vocab=trg_vocab,
        max_lines=max_lines,
    )

    collate = Collate(src_pad_idx=ds.src_vocab.pad_idx, trg_pad_idx=ds.trg_vocab.pad_idx)

    loader = DataLoader(
        ds,
        batch_size=batch_size,
        shuffle=shuffle,
        collate_fn=collate,
        drop_last=False,
        num_workers=0,  # change if you want multiprocessing
    )

    return loader, ds

# =============================
# 5. Save / Load utilities for vocab & dataset
# =============================
def save_vocab(vocab: Vocab, path: str):
    with open(path, "wb") as f:
        pickle.dump(vocab, f)

def load_vocab(path: str) -> Vocab:
    with open(path, "rb") as f:
        return pickle.load(f)

def save_dataset(dataset: TranslationDataset, path: str):
    # save entire dataset object (contains token lists and vocabs)
    torch.save(dataset, path)

def load_dataset(path: str) -> TranslationDataset:
    # ensure same class available when loading
    return torch.load(path)

# =============================
# 6. Example usage (if run as script)
# =============================
if __name__ == "__main__":
    # example: replace with your Multi30K raw file paths
    TRAIN_EN = "/kaggle/input/englishfrance/train.en"
    TRAIN_FR = "/kaggle/input/englishfrance/train.fr"

    loader, dataset = get_loader(TRAIN_EN, TRAIN_FR, batch_size=32, shuffle=True)

    # inspect one batch
    for src_batch, trg_batch, src_lens, trg_lens in loader:
        print("src:", src_batch.shape, "trg:", trg_batch.shape)
        print("src_lens:", src_lens)
        print("trg_lens:", trg_lens)
        break

    os.makedirs("./data", exist_ok=True)
    save_vocab(dataset.src_vocab, "./data/vocab_en.pkl")
    save_vocab(dataset.trg_vocab, "./data/vocab_fr.pkl")
    save_dataset(dataset, "./data/train_dataset.pt")
    print("Saved vocab & dataset to ./data")
