import os
import gzip
import torch
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence
from collections import Counter
import spacy
from tqdm import tqdm
import pickle

# =====================================================
# 1. Tokenizers
# =====================================================

# Nếu Kaggle không có model spaCy, dùng blank model
try:
    spacy_en = spacy.load("en_core_web_sm")
except:
    spacy_en = spacy.blank("en")

try:
    spacy_fr = spacy.load("fr_core_news_sm")
except:
    spacy_fr = spacy.blank("fr")

def tokenize_en(text):
    return [tok.text.lower() for tok in spacy_en.tokenizer(text)]

def tokenize_fr(text):
    return [tok.text.lower() for tok in spacy_fr.tokenizer(text)]


# =====================================================
# 2. Vocabulary
# =====================================================

class Vocab:
    def __init__(self, max_size=10000, freq_threshold=1):
        self.max_size = max_size
        self.freq_threshold = freq_threshold

        # token đặc biệt
        self.itos = {
            0: "<pad>",
            1: "<sos>",
            2: "<eos>",
            3: "<unk>"
        }
        self.stoi = {v: k for k, v in self.itos.items()}

    def build_vocabulary(self, sentences):
        freqs = Counter()

        for sent in sentences:
            for token in sent:
                freqs[token] += 1

        # lọc theo threshold
        filtered = [w for w, f in freqs.items() if f >= self.freq_threshold]

        # sắp xếp theo tần suất
        sorted_words = sorted(filtered, key=lambda w: freqs[w], reverse=True)

        # giới hạn vocab
        if self.max_size:
            sorted_words = sorted_words[: self.max_size]

        # thêm vào vocab
        idx = len(self.itos)
        for word in sorted_words:
            self.itos[idx] = word
            self.stoi[word] = idx
            idx += 1

    def numericalize(self, tokens):
        return [self.stoi.get(token, self.stoi["<unk>"]) for token in tokens]


# =====================================================
# 3. Translation Dataset
# =====================================================

class TranslationDataset(Dataset):
    def __init__(self, src_path, trg_path, src_tokenizer, trg_tokenizer,
                 src_vocab=None, trg_vocab=None):

        # đọc file (có thể là .gz)
        def read_file(path):
            if path.endswith(".gz"):
                with gzip.open(path, "rt", encoding="utf-8") as f:
                    return f.read().strip().split("\n")
            else:
                with open(path, "r", encoding="utf-8") as f:
                    return f.read().strip().split("\n")

        # load câu tiếng Anh/French
        src_lines = read_file(src_path)
        trg_lines = read_file(trg_path)

        # tokenize
        self.src_sentences = [src_tokenizer(line) for line in tqdm(src_lines, desc="Tokenizing EN")]
        self.trg_sentences = [trg_tokenizer(line) for line in tqdm(trg_lines, desc="Tokenizing FR")]

        # xây vocab nếu chưa có
        if src_vocab is None:
            self.src_vocab = Vocab(max_size=10000)
            self.src_vocab.build_vocabulary(self.src_sentences)
        else:
            self.src_vocab = src_vocab

        if trg_vocab is None:
            self.trg_vocab = Vocab(max_size=10000)
            self.trg_vocab.build_vocabulary(self.trg_sentences)
        else:
            self.trg_vocab = trg_vocab

        self.src_pad_idx = self.src_vocab.stoi["<pad>"]
        self.trg_pad_idx = self.trg_vocab.stoi["<pad>"]

    def __len__(self):
        return len(self.src_sentences)

    def __getitem__(self, idx):
        src = [self.src_vocab.stoi["<sos>"]] + \
              self.src_vocab.numericalize(self.src_sentences[idx]) + \
              [self.src_vocab.stoi["<eos>"]]

        trg = [self.trg_vocab.stoi["<sos>"]] + \
              self.trg_vocab.numericalize(self.trg_sentences[idx]) + \
              [self.trg_vocab.stoi["<eos>"]]

        return torch.tensor(src), torch.tensor(trg)


# =====================================================
# 4. Collate Function — chuẩn LSTM Seq2Seq
# =====================================================

class MyCollate:
    def __init__(self, src_pad_idx, trg_pad_idx):
        self.src_pad_idx = src_pad_idx
        self.trg_pad_idx = trg_pad_idx

    def __call__(self, batch):
        src_batch = [item[0] for item in batch]
        trg_batch = [item[1] for item in batch]

        # độ dài thật trước khi padding
        src_lengths = torch.tensor([len(s) for s in src_batch])
        trg_lengths = torch.tensor([len(t) for t in trg_batch])

        # sort theo độ dài giảm dần
        src_lengths_sorted, perm_idx = src_lengths.sort(descending=True)

        src_batch = [src_batch[i] for i in perm_idx]
        trg_batch = [trg_batch[i] for i in perm_idx]
        trg_lengths = trg_lengths[perm_idx]

        # padding
        src_padded = pad_sequence(src_batch, batch_first=True, padding_value=self.src_pad_idx)
        trg_padded = pad_sequence(trg_batch, batch_first=True, padding_value=self.trg_pad_idx)

        return src_padded, trg_padded, src_lengths_sorted, trg_lengths


# =====================================================
# 5. Helper: build DataLoader
# =====================================================

def get_loader(src_path, trg_path, batch_size=64,
               src_tokenizer=tokenize_en, trg_tokenizer=tokenize_fr,
               src_vocab=None, trg_vocab=None, shuffle=False):

    ds = TranslationDataset(src_path, trg_path,
                            src_tokenizer, trg_tokenizer,
                            src_vocab, trg_vocab)

    pad_src = ds.src_pad_idx
    pad_trg = ds.trg_pad_idx

    loader = DataLoader(
        ds,
        batch_size=batch_size,
        shuffle=shuffle,
        collate_fn=MyCollate(pad_src, pad_trg)
    )

    return loader, ds

# =====================================================
# 6. Save / Load Functions
# =====================================================

def save_vocab(vocab, path):
    with open(path, "wb") as f:
        pickle.dump(vocab, f)

def load_vocab(path):
    with open(path, "rb") as f:
        return pickle.load(f)

def save_dataset(dataset, path):
    torch.save(dataset, path)

def load_dataset(path):
    return torch.load(path)


# =====================================================
# 7. Example usage (Kaggle)
# =====================================================

if __name__ == "__main__":
    TRAIN_EN = "/kaggle/input/englishfrance/train.en"
    TRAIN_FR = "/kaggle/input/englishfrance/train.fr"

    loader, dataset = get_loader(TRAIN_EN, TRAIN_FR, batch_size=32)

    os.makedirs("/kaggle/working/data", exist_ok=True)

    save_vocab(dataset.src_vocab, "/kaggle/working/data/vocab_en.pkl")
    save_vocab(dataset.trg_vocab, "/kaggle/working/data/vocab_fr.pkl")

    save_dataset(dataset, "/kaggle/working/data/train_dataset.pt")

    print(" Saved vocab + dataset → /kaggle/working/data/")