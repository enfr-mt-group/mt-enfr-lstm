import os
import gzip
import torch
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence
from collections import Counter
import spacy
from tqdm import tqdm
import pickle
from tokenizers import Tokenizer, models, pre_tokenizers, trainers

# 1.1 Tokenizers
try:
    spacy_en = spacy.load("en_core_web_sm")
except:
    spacy_en = spacy.blank("en")

try:
    spacy_fr = spacy.load("fr_core_news_sm")
except:
    spacy_fr = spacy.blank("fr")
#nếu là Attention thì ko dùng reverse [::-1] 
def tokenize_en(text): 
    return [tok.text.lower() for tok in spacy_en.tokenizer(text)]

def tokenize_fr(text):
    return [tok.text.lower() for tok in spacy_fr.tokenizer(text)]

#1.2. dùng BPE subword
class BPESubword:
   # Huấn luyện và encode BPE tokenizer

    def __init__(self, vocab_size=10000, min_freq=2):
        self.vocab_size = vocab_size
        self.min_freq = min_freq
        self.tokenizer = None

    def train(self, sentences, save_path):
        # sentences: list[list[str]] (tokenized sentences)

        # Chuyển tokens thành chuỗi để BPE train
        lines = [" ".join(s) for s in sentences]

        # Model BPE rỗng
        bpe_model = models.BPE()
        self.tokenizer = Tokenizer(bpe_model)

        self.tokenizer.pre_tokenizer = pre_tokenizers.Whitespace()

        trainer = trainers.BpeTrainer(
            vocab_size=self.vocab_size,
            min_frequency=self.min_freq,
            special_tokens=["<pad>", "<sos>", "<eos>", "<unk>"]
        )

        self.tokenizer.train_from_iterator(lines, trainer=trainer)

        # Lưu lại để inference dùng chung
        self.tokenizer.save(save_path)
        print(f"Saved BPE tokenizer to {save_path}")

    def load(self, path):
        self.tokenizer = Tokenizer.from_file(path)

    def encode(self, tokens):
        #tokens: list[str] → return list[int]
        text = " ".join(tokens)
        ids = self.tokenizer.encode(text).ids
        return ids

    def pad_id(self):
        return self.tokenizer.token_to_id("<pad>")

# 2. Vocabulary
class Vocab:
    def __init__(self, max_size=10000, freq_threshold=2):
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

    # chuyển token thành chỉ số
    def numericalize(self, tokens):
        return [self.stoi.get(token, self.stoi["<unk>"]) for token in tokens]

# 3. Translation Dataset
class TranslationDataset(Dataset):
    def __init__(self, src_path, trg_path, src_tokenizer, trg_tokenizer,
                 src_vocab=None, trg_vocab=None, use_bpe=False, bpe_src=None, bpe_trg=None):
        
        self.use_bpe = use_bpe
        self.bpe_src = bpe_src
        self.bpe_trg = bpe_trg

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

        # nếu dùng bpe thì ko dùng class Vocab
        if use_bpe:
            self.src_pad_idx = bpe_src.pad_id()
            self.trg_pad_idx = bpe_trg.pad_id()
            self.src_vocab = None
            self.trg_vocab = None
            return

        # xây vocab nếu chưa có
        if src_vocab is None:
            self.src_vocab = Vocab(max_size=10000, freq_threshold=2)
            self.src_vocab.build_vocabulary(self.src_sentences)
        else:
            self.src_vocab = src_vocab

        if trg_vocab is None:
            self.trg_vocab = Vocab(max_size=10000, freq_threshold=2)
            self.trg_vocab.build_vocabulary(self.trg_sentences)
        else:
            self.trg_vocab = trg_vocab

        self.src_pad_idx = self.src_vocab.stoi["<pad>"]
        self.trg_pad_idx = self.trg_vocab.stoi["<pad>"]

    def __len__(self):
        return len(self.src_sentences)

    def __getitem__(self, idx):
        if self.use_bpe:
            src_ids = [self.bpe_src.tokenizer.token_to_id("<sos>")] + \
                      self.bpe_src.encode(self.src_sentences[idx]) + \
                      [self.bpe_src.tokenizer.token_to_id("<eos>")]

            trg_ids = [self.bpe_trg.tokenizer.token_to_id("<sos>")] + \
                      self.bpe_trg.encode(self.trg_sentences[idx]) + \
                      [self.bpe_trg.tokenizer.token_to_id("<eos>")]

            return torch.tensor(src_ids), torch.tensor(trg_ids)
    
        src = [self.src_vocab.stoi["<sos>"]] + \
              self.src_vocab.numericalize(self.src_sentences[idx]) + \
              [self.src_vocab.stoi["<eos>"]]

        trg = [self.trg_vocab.stoi["<sos>"]] + \
              self.trg_vocab.numericalize(self.trg_sentences[idx]) + \
              [self.trg_vocab.stoi["<eos>"]]

        return torch.tensor(src), torch.tensor(trg)

# 4. Collate Function — LSTM Seq2Seq
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

        # dùng pad_sequence để padding đồng bộ độ dài trong batch
        src_padded = pad_sequence(src_batch, batch_first=True, padding_value=self.src_pad_idx)
        trg_padded = pad_sequence(trg_batch, batch_first=True, padding_value=self.trg_pad_idx)

        return src_padded, trg_padded, src_lengths_sorted, trg_lengths


# 5. Build DataLoader
def get_loader(src_path, trg_path, batch_size=64,
               src_tokenizer=tokenize_en, trg_tokenizer=tokenize_fr,
               src_vocab=None, trg_vocab=None, shuffle=False, use_bpe=False,
               bpe_src=None, bpe_trg=None):

    ds = TranslationDataset(src_path, trg_path,
                            src_tokenizer, trg_tokenizer,
                            src_vocab, trg_vocab, use_bpe, bpe_src, bpe_trg)
    pad_src = ds.src_pad_idx
    pad_trg = ds.trg_pad_idx

    loader = DataLoader(
        ds,
        batch_size=batch_size,
        shuffle=shuffle,
        collate_fn=MyCollate(pad_src, pad_trg)
    )

    return loader, ds.src_vocab, ds.trg_vocab

# 6. Save / Load Functions
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