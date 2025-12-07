# src/dataset.py
import torch
from torch.utils.data import Dataset, DataLoader
import spacy
from collections import Counter
from tqdm import tqdm
from torch.nn.utils.rnn import pad_sequence

import gzip, os, pickle

# đọc file thường hoặc file nén .gz
def read_file(path):
    if path.endswith(".gz"):
        with gzip.open(path, "rt", encoding="utf-8") as f:
            return f.read().strip().split("\n")
    else:
        with open(path, encoding="utf-8") as f:
            return f.read().strip().split("\n")


# --- 1. Tokenizer ---
# Sử dụng spaCy để tokenize tiếng Anh và tiếng Pháp
#  tokenizer trả về danh sách các token (lowercase)
spacy_en = spacy.load("en_core_web_sm")
spacy_fr = spacy.load("fr_core_news_sm")

def tokenize_en(text):
    return [tok.text.lower() for tok in spacy_en.tokenizer(text)]

def tokenize_fr(text):
    return [tok.text.lower() for tok in spacy_fr.tokenizer(text)]


# --- 2. Vocab Builder ---
class Vocab:
    def __init__(self, min_freq=2, max_size=10_000, specials=['<pad>', '<sos>', '<eos>', '<unk>']):
        self.min_freq = min_freq
        self.max_size = max_size
        self.specials = specials
        self.freqs = Counter()
        self.itos = []
        self.stoi = {}

    def build_vocabulary(self, sentence_list):
        for sentence in sentence_list:
            self.freqs.update(sentence)

        # Lọc theo tần suất & giới hạn top N
        words = [w for w, f in self.freqs.items() if f >= self.min_freq]
        most_common = [w for w, _ in Counter(words).most_common(self.max_size)]

        self.itos = list(self.specials) + most_common
        self.stoi = {tok: idx for idx, tok in enumerate(self.itos)}

    def numericalize(self, text):
        return [self.stoi.get(tok, self.stoi['<unk>']) for tok in text]


# --- 3. Custom Dataset ---
class TranslationDataset(Dataset):
    def __init__(self, src_path, trg_path, src_tokenizer, trg_tokenizer, src_vocab=None, trg_vocab=None, src_sentences=None, trg_sentences=None):
        # src_lines = read_file(src_path)
        # trg_lines = read_file(trg_path)
        # self.src_sentences = [src_tokenizer(line) for line in tqdm(src_lines)]
        # self.trg_sentences = [trg_tokenizer(line) for line in tqdm(trg_lines)]


        """
        Nếu src_sentences và trg_sentences đã được cung cấp => bỏ qua việc đọc file.
        """
        # --- Nếu có sẵn tokenized data ---
        if src_sentences is not None and trg_sentences is not None:
            self.src_sentences = src_sentences
            self.trg_sentences = trg_sentences
        else:
            src_lines = read_file(src_path)
            trg_lines = read_file(trg_path)
            self.src_sentences = [src_tokenizer(line) for line in tqdm(src_lines)]
            self.trg_sentences = [trg_tokenizer(line) for line in tqdm(trg_lines)]


        # Xây vocab nếu chưa có
        if src_vocab is None:
            self.src_vocab = Vocab()
            self.src_vocab.build_vocabulary(self.src_sentences)
        else:
            self.src_vocab = src_vocab

        if trg_vocab is None:
            self.trg_vocab = Vocab()
            self.trg_vocab.build_vocabulary(self.trg_sentences)
        else:
            self.trg_vocab = trg_vocab

        self.pad_idx = self.src_vocab.stoi["<pad>"]

    def __len__(self):
        return len(self.src_sentences)

    def __getitem__(self, idx):
        src = self.src_sentences[idx]
        trg = self.trg_sentences[idx]

        src_numerical = [self.src_vocab.stoi["<sos>"]] + \
                        self.src_vocab.numericalize(src) + \
                        [self.src_vocab.stoi["<eos>"]]

        trg_numerical = [self.trg_vocab.stoi["<sos>"]] + \
                        self.trg_vocab.numericalize(trg) + \
                        [self.trg_vocab.stoi["<eos>"]]

        return torch.tensor(src_numerical), torch.tensor(trg_numerical)


# --- 4. Collate Function (padding) ---
class MyCollate:
    def __init__(self, pad_idx):
        self.pad_idx = pad_idx

    def __call__(self, batch):
        src = [item[0] for item in batch]
        trg = [item[1] for item in batch]
        src = pad_sequence(src, batch_first=True, padding_value=self.pad_idx)
        trg = pad_sequence(trg, batch_first=True, padding_value=self.pad_idx)
        return src, trg


# --- 5. Helper to get DataLoader ---
def get_loader(src_path, trg_path, batch_size=32, shuffle=True):
    dataset = TranslationDataset(src_path, trg_path, tokenize_en, tokenize_fr)
    pad_idx = dataset.pad_idx
    loader = DataLoader(dataset=dataset,
                        batch_size=batch_size,
                        shuffle=shuffle,
                        collate_fn=MyCollate(pad_idx))
    return loader, dataset

# --- 6. Cached DataLoader (mỗi ngôn ngữ 1 file) ---
def get_loader_cached(src_path, trg_path, batch_size=32, shuffle=True, cache_dir="cache"):
    """
    Cache riêng cho từng ngôn ngữ:
    - cache/{src_name}.pkl
    - cache/{trg_name}.pkl
    """
    os.makedirs(cache_dir, exist_ok=True)

    src_name = os.path.basename(src_path).replace(".", "_")
    trg_name = os.path.basename(trg_path).replace(".", "_")
    src_cache = os.path.join(cache_dir, f"{src_name}.pt")
    trg_cache = os.path.join(cache_dir, f"{trg_name}.pt")

    # --- English side ---
    if os.path.exists(src_cache):
        print(f"✅ Loading cached English dataset: {src_cache}")
        with open(src_cache, "rb") as f:
            src_sentences, src_vocab = pickle.load(f)
    else:
        print(f"🚀 Building English dataset from scratch...")
        src_lines = read_file(src_path)
        src_sentences = [tokenize_en(line) for line in tqdm(src_lines)]
        src_vocab = Vocab()
        src_vocab.build_vocabulary(src_sentences)
        with open(src_cache, "wb") as f:
            pickle.dump((src_sentences, src_vocab), f)
        print(f"💾 Saved English cache: {src_cache}")

    # --- French side ---
    if os.path.exists(trg_cache):
        print(f"✅ Loading cached French dataset: {trg_cache}")
        with open(trg_cache, "rb") as f:
            trg_sentences, trg_vocab = pickle.load(f)
    else:
        print(f"🚀 Building French dataset from scratch...")
        trg_lines = read_file(trg_path)
        trg_sentences = [tokenize_fr(line) for line in tqdm(trg_lines)]
        trg_vocab = Vocab()
        trg_vocab.build_vocabulary(trg_sentences)
        with open(trg_cache, "wb") as f:
            pickle.dump((trg_sentences, trg_vocab), f)
        print(f"💾 Saved French cache: {trg_cache}")

    # --- Dataset ---
    dataset = TranslationDataset(
        src_path=None, trg_path=None,
        src_tokenizer=None, trg_tokenizer=None,
        src_vocab=src_vocab, trg_vocab=trg_vocab,
        src_sentences=src_sentences, trg_sentences=trg_sentences
    )

    pad_idx = dataset.pad_idx
    loader = DataLoader(dataset=dataset,
                        batch_size=batch_size,
                        shuffle=shuffle,
                        collate_fn=MyCollate(pad_idx))
    return loader, dataset

# --- 6. Cached DataLoader (phiên bản 1)---


# def get_loader_cached(src_path, trg_path, batch_size=32, shuffle=True, cache_dir="cache"):
#     os.makedirs(cache_dir, exist_ok=True)

#     src_name = os.path.basename(src_path).replace(".", "_")
#     trg_name = os.path.basename(trg_path).replace(".", "_")

#     src_cache = os.path.join(cache_dir, f"{src_name}.pkl")
#     trg_cache = os.path.join(cache_dir, f"{trg_name}.pkl")

#     # === English ===
#     if os.path.exists(src_cache):
#         print(f"Loading cached English dataset: {src_cache}")
#         with open(src_cache, "rb") as f:
#             src_sentences, src_vocab = pickle.load(f)
#     else:
#         print(f"Building English dataset from scratch...")
#         with gzip.open(src_path, "rt", encoding="utf-8") if src_path.endswith(".gz") else open(src_path, encoding="utf-8") as f:
#             src_lines = f.read().strip().split("\n")
#         src_sentences = [tokenize_en(line) for line in src_lines]
#         src_vocab = Vocab()
#         src_vocab.build_vocabulary(src_sentences)
#         with open(src_cache, "wb") as f:
#             pickle.dump((src_sentences, src_vocab), f)
#         print(f"Saved English cache: {src_cache}")

#     # === French ===
#     if os.path.exists(trg_cache):
#         print(f"Loading cached French dataset: {trg_cache}")
#         with open(trg_cache, "rb") as f:
#             trg_sentences, trg_vocab = pickle.load(f)
#     else:
#         print(f"Building French dataset from scratch...")
#         with gzip.open(trg_path, "rt", encoding="utf-8") if trg_path.endswith(".gz") else open(trg_path, encoding="utf-8") as f:
#             trg_lines = f.read().strip().split("\n")
#         trg_sentences = [tokenize_fr(line) for line in trg_lines]
#         trg_vocab = Vocab()
#         trg_vocab.build_vocabulary(trg_sentences)
#         with open(trg_cache, "wb") as f:
#             pickle.dump((trg_sentences, trg_vocab), f)
#         print(f"Saved French cache: {trg_cache}")

#     # === Dataset & DataLoader ===
#     dataset = TranslationDataset(src_path, trg_path, tokenize_en, tokenize_fr, src_vocab, trg_vocab)
#     pad_idx = dataset.pad_idx

#     loader = DataLoader(dataset=dataset,
#                         batch_size=batch_size,
#                         shuffle=shuffle,
#                         collate_fn=MyCollate(pad_idx))
#     return loader, dataset

# --- 6. Cached DataLoader (phiên bản 2-10:30PM) ---
# def get_loader_cached(src_path, trg_path, batch_size=32, shuffle=True, cache_dir="cache"):
#     """
#     Sinh DataLoader với cache thông minh:
#     - Sử dụng read_file() để đọc dữ liệu (hỗ trợ .gz)
#     - Lưu toàn bộ tokenized sentences + vocab để load nhanh
#     """
#     import os, pickle, torch
#     os.makedirs(cache_dir, exist_ok=True)

#     # === 1. Đặt tên cache duy nhất ===
#     src_name = os.path.basename(src_path).replace(".", "_")
#     trg_name = os.path.basename(trg_path).replace(".", "_")
#     cache_file = os.path.join(cache_dir, f"{src_name}_{trg_name}_dataset.pkl")

#     # === 2. Dùng cache nếu có ===
#     if os.path.exists(cache_file):
#         print(f"✅ Loading cached dataset: {cache_file}")
#         with open(cache_file, "rb") as f:
#             cache = pickle.load(f)
#         src_sentences = cache["src_sentences"]
#         trg_sentences = cache["trg_sentences"]
#         src_vocab = cache["src_vocab"]
#         trg_vocab = cache["trg_vocab"]

#     else:
#         print(f"🚀 Building dataset from scratch...")
#         # --- Đọc file bằng read_file() ---
#         src_lines = read_file(src_path)
#         trg_lines = read_file(trg_path)

#         # --- Tokenize ---
#         print("🔤 Tokenizing...")
#         src_sentences = [tokenize_en(line) for line in src_lines]
#         trg_sentences = [tokenize_fr(line) for line in trg_lines]

#         # --- Build vocab ---
#         print("📘 Building vocabularies...")
#         src_vocab = Vocab()
#         src_vocab.build_vocabulary(src_sentences)

#         trg_vocab = Vocab()
#         trg_vocab.build_vocabulary(trg_sentences)

#         # --- Lưu cache ---
#         with open(cache_file, "wb") as f:
#             pickle.dump({
#                 "src_sentences": src_sentences,
#                 "trg_sentences": trg_sentences,
#                 "src_vocab": src_vocab,
#                 "trg_vocab": trg_vocab
#             }, f)
#         print(f"💾 Saved cache to {cache_file}")

#     # === 3. Tạo TranslationDataset từ dữ liệu đã tokenized ===
#     dataset = TranslationDataset(
#         src_path=None, trg_path=None,
#         src_tokenizer=None, trg_tokenizer=None,
#         src_vocab=src_vocab, trg_vocab=trg_vocab
#     )
#     dataset.src_sentences = src_sentences
#     dataset.trg_sentences = trg_sentences
#     dataset.pad_idx = src_vocab.stoi["<pad>"]

#     # === 4. DataLoader ===
#     loader = DataLoader(
#         dataset,
#         batch_size=batch_size,
#         shuffle=shuffle,
#         collate_fn=MyCollate(dataset.pad_idx)
#     )

#     return loader, dataset


# # --- 6. Cached DataLoader (phiên bản lưu dạng .pt)---
# def get_loader_cached(src_path, trg_path, batch_size=32, shuffle=True, cache_dir="cache"):
#     """
#     Sinh DataLoader với cache riêng cho mỗi ngôn ngữ:
#     - cache/{src_name}_{trg_name}_dataset.pt
#     """
#     import os, gzip, torch
#     from src.dataset import TranslationDataset, tokenize_en, tokenize_fr, Vocab, MyCollate
#     os.makedirs(cache_dir, exist_ok=True)

#     # === 1. Tên file cache ===
#     src_name = os.path.basename(src_path).replace(".", "_")
#     trg_name = os.path.basename(trg_path).replace(".", "_")
#     cache_file = os.path.join(cache_dir, f"{src_name}_{trg_name}_dataset.pt")

#     # === 2. Load cache nếu có, hoặc build từ scratch ===
#     if os.path.exists(cache_file):
#         print(f"✅ Loading cached dataset: {cache_file}")
#         data = torch.load(cache_file)
#         src_sentences = data["src_sentences"]
#         trg_sentences = data["trg_sentences"]
#         src_vocab = data["src_vocab"]
#         trg_vocab = data["trg_vocab"]
#     else:
#         print(f"🚀 Creating dataset from scratch...")
#         # Load file
#         if src_path.endswith(".gz"):
#             with gzip.open(src_path, "rt", encoding="utf-8") as f:
#                 src_lines = f.read().strip().split("\n")
#         else:
#             with open(src_path, encoding="utf-8") as f:
#                 src_lines = f.read().strip().split("\n")

#         if trg_path.endswith(".gz"):
#             with gzip.open(trg_path, "rt", encoding="utf-8") as f:
#                 trg_lines = f.read().strip().split("\n")
#         else:
#             with open(trg_path, encoding="utf-8") as f:
#                 trg_lines = f.read().strip().split("\n")

#         # Tokenize
#         src_sentences = [tokenize_en(line) for line in src_lines]
#         trg_sentences = [tokenize_fr(line) for line in trg_lines]

#         # Build vocab
#         src_vocab = Vocab()
#         src_vocab.build_vocabulary(src_sentences)

#         trg_vocab = Vocab()
#         trg_vocab.build_vocabulary(trg_sentences)

#         # Lưu cache
#         torch.save({
#             "src_sentences": src_sentences,
#             "trg_sentences": trg_sentences,
#             "src_vocab": src_vocab,
#             "trg_vocab": trg_vocab
#         }, cache_file)
#         print(f"💾 Saved cached dataset to {cache_file}")

#     # === 3. Tạo TranslationDataset và DataLoader ===
#     dataset = TranslationDataset(
#         src_path=None, trg_path=None,
#         src_tokenizer=None, trg_tokenizer=None,
#         src_vocab=src_vocab, trg_vocab=trg_vocab
#     )
#     # Gán dữ liệu trực tiếp
#     dataset.src_sentences = src_sentences
#     dataset.trg_sentences = trg_sentences

#     pad_idx = dataset.pad_idx
#     loader = torch.utils.data.DataLoader(
#         dataset,
#         batch_size=batch_size,
#         shuffle=shuffle,
#         collate_fn=MyCollate(pad_idx)
#     )

#     return loader, dataset
