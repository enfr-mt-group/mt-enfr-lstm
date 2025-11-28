# src/dataset.py
import os
import gzip
import pickle
import torch
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence, pack_padded_sequence
from collections import Counter
from tqdm import tqdm
import spacy
#hihihihihi
# =============================
# 1. Tokenizer
# =============================
# Load spacy models cho tiếng Anh và tiếng Pháp
# Dùng blank models vì Kaggle không có en_core_web_sm và fr_core_news_sm
spacy_en = spacy.blank("en")
spacy_fr = spacy.blank("fr")

def tokenize_en(text):
    return [tok.text.lower() for tok in spacy_en.tokenizer(text)]

def tokenize_fr(text):
    return [tok.text.lower() for tok in spacy_fr.tokenizer(text)]

# =============================
# 2. Vocabulary
# =============================
class Vocab:
    """
    Xây dựng vocab từ danh sách câu.
    - freq_threshold: số lần xuất hiện tối thiểu để giữ từ
    - max_size: giới hạn số từ phổ biến nhất
    - stoi: từ -> index
    - itos: index -> từ
    """
    
    def __init__(self, freq_threshold=2, max_size=None):
        self.freq_threshold = freq_threshold
        self.max_size = max_size
        self.itos = {0:"<pad>", 1:"<sos>", 2:"<eos>", 3:"<unk>"}
        self.stoi = {v:k for k,v in self.itos.items()}

    def build_vocabulary(self, sentence_list):
        freqs = Counter()
        idx = 4
        for sentence in sentence_list:
            for word in sentence:
                freqs[word] += 1

        # Chọn từ theo tần suất >= freq_threshold, sắp xếp giảm dần
        sorted_words = sorted([w for w,f in freqs.items() if f>=self.freq_threshold], key=lambda x: freqs[x], reverse=True)
        if self.max_size:
            sorted_words = sorted_words[:self.max_size]

        for word in sorted_words:
            self.stoi[word] = idx
            self.itos[idx] = word
            idx += 1

    def numericalize(self, tokens):
        # Chuyển danh sách từ thành danh sách index
        return [self.stoi.get(token, self.stoi["<unk>"]) for token in tokens]

# =============================
# 3. Dataset
# =============================
class TranslationDataset(Dataset):
    """
    Custom Dataset cho English->French
    - src_path, trg_path: file raw (.en/.fr)
    - src_tokenizer, trg_tokenizer: hàm tokenize
    - src_vocab, trg_vocab: có thể truyền vào nếu đã build sẵn
    """
    def __init__(self, src_path, trg_path, src_tokenizer, trg_tokenizer, src_vocab=None, trg_vocab=None):
        def read_file(path):
            # Hỗ trợ đọc file .gz nếu cần
            if path.endswith(".gz"):
                with gzip.open(path, mode="rt", encoding="utf-8") as f:
                    lines = f.read().strip().split("\n")
            else:
                with open(path, encoding="utf-8") as f:
                    lines = f.read().strip().split("\n")
            return lines

        # load dữ liệu
        src_lines = read_file(src_path)
        trg_lines = read_file(trg_path)

        # tokenize sentences
        self.src_sentences = [src_tokenizer(line) for line in tqdm(src_lines)]
        self.trg_sentences = [trg_tokenizer(line) for line in tqdm(trg_lines)]

        # build vocab if not provided
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

        self.pad_idx = self.src_vocab.stoi["<pad>"]

    def __len__(self):
        return len(self.src_sentences)

    def __getitem__(self, idx):
        # Thêm token <sos> và <eos> cho mỗi câu - src( tiếng Anh) và trg( tiếng Pháp)
        src = [self.src_vocab.stoi["<sos>"]] + self.src_vocab.numericalize(self.src_sentences[idx]) + [self.src_vocab.stoi["<eos>"]]
        trg = [self.trg_vocab.stoi["<sos>"]] + self.trg_vocab.numericalize(self.trg_sentences[idx]) + [self.trg_vocab.stoi["<eos>"]]
        return torch.tensor(src), torch.tensor(trg)

# =============================
# 4. Collate function (padding)
# =============================
class MyCollate:
    """
    Padding batch để đồng bộ độ dài
    - dùng cho DataLoader
    """
    def __init__(self, pad_idx):
        self.pad_idx = pad_idx

    def __call__(self, batch):
        src = [item[0] for item in batch] # list of tensors (variable length)
        trg = [item[1] for item in batch]
        
        # padding
        src_padded = pad_sequence(src, batch_first=True, padding_value=self.pad_idx)
        trg_padded = pad_sequence(trg, batch_first=True, padding_value=self.pad_idx)
        
        # compute lengths BEFORE padding (true lengths)
        src_lengths = torch.tensor([s.size(0) for s in src], dtype=torch.long)
        trg_lengths = torch.tensor([t.size(0) for t in trg], dtype=torch.long)
        
         # sort by src length descending (so we can pack if needed)
        src_lengths_sorted, perm_idx = src_lengths.sort(0, descending=True)
        src_padded = src_padded[perm_idx]
        trg_padded = trg_padded[perm_idx]
        trg_lengths_sorted = trg_lengths[perm_idx]

        # trả về batch đã padding và độ dài
        return src_padded, trg_padded, src_lengths_sorted, trg_lengths_sorted

# =============================
# 5. Get DataLoader
# =============================
def get_loader(src_path, trg_path, batch_size=32, shuffle=True, src_vocab=None, trg_vocab=None):
    """
    Trả về DataLoader và Dataset
    - batch đã được padding sẵn
    - vocab riêng cho EN và FR
    """
    dataset = TranslationDataset(src_path, trg_path, tokenize_en, tokenize_fr, src_vocab, trg_vocab)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, collate_fn=MyCollate(dataset.pad_idx))
    return loader, dataset

# =============================
# 6. Save/Load utilities (pkl, pt - pickle, pytorch)
# =============================

def save_vocab(vocab, path):
    with open(path, "wb") as f:
        pickle.dump(vocab, f)

def load_vocab(path):
    with open(path, "rb") as f:
        return pickle.load(f)

def save_dataset(dataset, path):
    # Lưu toàn bộ object Dataset dưới dạng .pt
    torch.save(dataset, path)

# def load_dataset(path):
#     return torch.load(path)

def load_dataset(path):
    """
    Load dataset chứa class custom TranslationDataset
    Dùng PyTorch >=2.6
    """
  # chắc chắn import class trước khi load
    # weights_only=False để cho phép load cả object (không chỉ weights)
    dataset = torch.load(path, weights_only=False)
    return dataset

# =============================
# 7. Example usage (Kaggle)
# =============================
if __name__ == "__main__":
    # Dùng đúng đường dẫn Kaggle
    TRAIN_EN = "/kaggle/input/englishfrance/train.en"
    TRAIN_FR = "/kaggle/input/englishfrance/train.fr"

    # Tạo DataLoader và lưu dataset + vocab
    train_loader, train_dataset = get_loader(
        TRAIN_EN,
        TRAIN_FR,
        batch_size=32
    )
    
    # Tạo thư mục lưu nếu chưa có
    os.makedirs("/kaggle/working/data", exist_ok=True)

    # Lưu vocab
    save_vocab(train_dataset.src_vocab, "/kaggle/working/data/vocab_en.pkl")
    save_vocab(train_dataset.trg_vocab, "/kaggle/working/data/vocab_fr.pkl")
    
    # Lưu dataset đã tokenize và numericalize
    save_dataset(train_dataset, "/kaggle/working/data/train_dataset.pt")
    
    print("✅ DataLoader, vocab, dataset đã lưu vào /kaggle/working/data/")
