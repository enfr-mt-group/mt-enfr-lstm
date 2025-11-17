# src/model.py
import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import matplotlib.pyplot as plt
import argparse
import random
import spacy
import nltk

from dataset import TranslationDataset, Vocab, load_dataset, load_vocab, MyCollate

# =============================
# 1. Hyperparameters
# =============================
parser = argparse.ArgumentParser()
parser.add_argument("--num_epochs", type=int, default=10)
parser.add_argument("--batch_size", type=int, default=32)
parser.add_argument("--embed_size", type=int, default=256)
parser.add_argument("--hidden_size", type=int, default=512)
parser.add_argument("--num_layers", type=int, default=2)
parser.add_argument("--learning_rate", type=float, default=0.001)
parser.add_argument("--teacher_forcing_ratio", type=float, default=0.5)
parser.add_argument("--patience", type=int, default=3)  # early stopping
args = parser.parse_args()

BATCH_SIZE = args.batch_size
EMBED_SIZE = args.embed_size
HIDDEN_SIZE = args.hidden_size
NUM_LAYERS = args.num_layers
NUM_EPOCHS = args.num_epochs
LEARNING_RATE = args.learning_rate
TEACHER_FORCING_RATIO = args.teacher_forcing_ratio
PATIENCE = args.patience
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# =============================
# 2. Load dataset + vocab
# =============================
DATA_DIR = "/kaggle/working/data/"

src_vocab = load_vocab(os.path.join(DATA_DIR, "vocab_en.pkl"))
trg_vocab = load_vocab(os.path.join(DATA_DIR, "vocab_fr.pkl"))

# Try to load saved dataset; if it fails, you can recreate dataset with get_loader(...)
try:
    train_dataset = load_dataset(os.path.join(DATA_DIR, "train_dataset.pt"))
except Exception as e:
    print("⚠️ Warning loading saved dataset:", e)
    print("Bạn có thể rebuild dataset bằng get_loader(...) trong dataset.py nếu cần.")
    raise

PAD_IDX = src_vocab.stoi["<pad>"]

train_loader = DataLoader(
    train_dataset,
    batch_size=BATCH_SIZE,
    shuffle=True,
    collate_fn=MyCollate(PAD_IDX)
)

print(f"✅ Dataset, vocab loaded. PAD_IDX={PAD_IDX}")
print(f"Number of training examples: {len(train_dataset)}")

# =============================
# 2.1 Load Validation Set (raw files)
# =============================
VAL_SRC_PATH = "/kaggle/input/englishfrance/val.en"
VAL_TRG_PATH = "/kaggle/input/englishfrance/val.fr"

def load_raw_file(path):
    with open(path, "r", encoding="utf-8") as f:
        return [line.strip() for line in f.readlines()]

val_src_lines = load_raw_file(VAL_SRC_PATH)
val_trg_lines = load_raw_file(VAL_TRG_PATH)
assert len(val_src_lines) == len(val_trg_lines), "val.en và val.fr phải cùng số dòng"

# spacy tokenizer: try load models, fallback to blank
try:
    spacy_en = spacy.load("en_core_web_sm")
except Exception:
    spacy_en = spacy.blank("en")
try:
    spacy_fr = spacy.load("fr_core_news_sm")
except Exception:
    spacy_fr = spacy.blank("fr")

def tokenize_en(text):
    return [tok.text.lower() for tok in spacy_en(text)]

def tokenize_fr(text):
    return [tok.text.lower() for tok in spacy_fr(text)]

# tokenized lists
val_src_tokenized = [tokenize_en(s) for s in val_src_lines]
val_trg_tokenized = [tokenize_fr(s) for s in val_trg_lines]

# map tokens -> indices (use <unk> if missing)
UNK_SRC = src_vocab.stoi.get("<unk>", 3)
UNK_TRG = trg_vocab.stoi.get("<unk>", 3)
SOS_TRG = trg_vocab.stoi["<sos>"]
EOS_TRG = trg_vocab.stoi["<eos>"]
SOS_SRC = src_vocab.stoi.get("<sos>", 1)
EOS_SRC = src_vocab.stoi.get("<eos>", 2)

val_src_idx = [[src_vocab.stoi.get(tok, UNK_SRC) for tok in sent] for sent in val_src_tokenized]
val_trg_idx = [[trg_vocab.stoi.get(tok, UNK_TRG) for tok in sent] for sent in val_trg_tokenized]

# add <sos>/<eos> for both src and trg for consistency
val_src_idx = [[SOS_SRC] + s + [EOS_SRC] for s in val_src_idx]
val_trg_idx = [[SOS_TRG] + s + [EOS_TRG] for s in val_trg_idx]

# In-memory dataset wrapper
class InMemoryDataset(Dataset):
    def __init__(self, src_idx_list, trg_idx_list):
        assert len(src_idx_list) == len(trg_idx_list)
        self.src = [torch.tensor(x, dtype=torch.long) for x in src_idx_list]
        self.trg = [torch.tensor(x, dtype=torch.long) for x in trg_idx_list]
    def __len__(self):
        return len(self.src)
    def __getitem__(self, idx):
        return self.src[idx], self.trg[idx]

val_dataset = InMemoryDataset(val_src_idx, val_trg_idx)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, collate_fn=MyCollate(PAD_IDX))

print(f"✅ Validation set ready: {len(val_dataset)} samples")

# =============================
# 3. Encoder
# =============================
class Encoder(nn.Module):
    def __init__(self, input_dim, embed_size, hidden_size, num_layers, pad_idx=PAD_IDX, dropout=0.3):
        super().__init__()
        self.embedding = nn.Embedding(input_dim, embed_size, padding_idx=pad_idx)
        self.lstm = nn.LSTM(embed_size, hidden_size, num_layers, batch_first=True, dropout=dropout)

    def forward(self, x, lengths=None):
        embedded = self.embedding(x)
        if lengths is not None:
            packed = nn.utils.rnn.pack_padded_sequence(embedded, lengths.cpu(), batch_first=True, enforce_sorted=True)
            packed_outputs, (hidden, cell) = self.lstm(packed)
            outputs, _ = nn.utils.rnn.pad_packed_sequence(packed_outputs, batch_first=True)
        else:
            outputs, (hidden, cell) = self.lstm(embedded)
        return hidden, cell

# =============================
# 4. Decoder
# =============================
class Decoder(nn.Module):
    def __init__(self, output_dim, embed_size, hidden_size, num_layers, pad_idx=PAD_IDX, dropout=0.3):
        super().__init__()
        self.embedding = nn.Embedding(output_dim, embed_size, padding_idx=pad_idx)
        self.lstm = nn.LSTM(embed_size, hidden_size, num_layers, batch_first=True, dropout=dropout)
        self.fc = nn.Linear(hidden_size, output_dim)

    def forward(self, x, hidden, cell):
        x = x.unsqueeze(1)
        embedded = self.embedding(x)
        outputs, (hidden, cell) = self.lstm(embedded, (hidden, cell))
        predictions = self.fc(outputs.squeeze(1))
        return predictions, hidden, cell

# =============================
# 5. Seq2Seq
# =============================
class Seq2Seq(nn.Module):
    def __init__(self, encoder, decoder, device, trg_vocab_size):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.device = device
        self.trg_vocab_size = trg_vocab_size

    def forward(self, src, trg, src_lengths, teacher_forcing_ratio=0.5):
        # src: (batch, seq_len)
        batch_size = src.size(0)
        trg_len = trg.size(1)
        vocab_size = self.trg_vocab_size
        outputs = torch.zeros(batch_size, trg_len, vocab_size).to(self.device)

        hidden, cell = self.encoder(src, lengths=src_lengths)  # hidden,c: (num_layers, batch, hidden)
        input_tok = trg[:, 0]  # <sos>

        for t in range(1, trg_len):
            output, hidden, cell = self.decoder(input_tok, hidden, cell)  # output: (batch, vocab)
            outputs[:, t, :] = output
            teacher_force = random.random() < teacher_forcing_ratio
            top1 = output.argmax(1)
            input_tok = trg[:, t] if teacher_force else top1

        return outputs

# =============================
# 6. Init model/optim/loss
# =============================
encoder = Encoder(len(src_vocab.stoi), EMBED_SIZE, HIDDEN_SIZE, NUM_LAYERS).to(DEVICE)
decoder = Decoder(len(trg_vocab.stoi), EMBED_SIZE, HIDDEN_SIZE, NUM_LAYERS).to(DEVICE)
model = Seq2Seq(encoder, decoder, DEVICE, trg_vocab_size=len(trg_vocab.stoi)).to(DEVICE)

optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
criterion = nn.CrossEntropyLoss(ignore_index=PAD_IDX)

# optional LR scheduler
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=2, verbose=True)

os.makedirs("/kaggle/working/checkpoints", exist_ok=True)

# =============================
# 7. Train + Validate (with early stopping)
# =============================
train_losses = []
val_losses = []
best_val = float("inf")
no_improve = 0

for epoch in range(NUM_EPOCHS):
    model.train()
    epoch_loss = 0.0
    for src, trg, src_lengths, trg_lengths in train_loader:
        src, trg, src_lengths = src.to(DEVICE), trg.to(DEVICE), src_lengths.to(DEVICE)

        optimizer.zero_grad()
        outputs = model(src, trg, src_lengths, teacher_forcing_ratio=TEACHER_FORCING_RATIO)
        output_dim = outputs.shape[-1]

        # compute loss (ignore first token <sos>)
        logits = outputs[:, 1:, :].reshape(-1, output_dim)
        targets = trg[:, 1:].reshape(-1).to(DEVICE)

        loss = criterion(logits, targets)
        loss.backward()
        optimizer.step()
        epoch_loss += loss.item()

    train_loss = epoch_loss / len(train_loader)
    train_losses.append(train_loss)

    # validation
    model.eval()
    val_loss = 0.0
    with torch.no_grad():
        for src, trg, src_lengths, trg_lengths in val_loader:
            src, trg, src_lengths = src.to(DEVICE), trg.to(DEVICE), src_lengths.to(DEVICE)
            outputs = model(src, trg, src_lengths, teacher_forcing_ratio=0.0)  # no teacher forcing
            logits = outputs[:, 1:, :].reshape(-1, outputs.shape[-1])
            targets = trg[:, 1:].reshape(-1).to(DEVICE)
            loss = criterion(logits, targets)
            val_loss += loss.item()
    val_loss = val_loss / len(val_loader)
    val_losses.append(val_loss)

    # scheduler step
    scheduler.step(val_loss)

    print(f"Epoch {epoch+1}/{NUM_EPOCHS} — train_loss: {train_loss:.4f} | val_loss: {val_loss:.4f}")

    # early stopping & save best
    if val_loss < best_val - 1e-4:
        best_val = val_loss
        no_improve = 0
        torch.save(model.state_dict(), f"/kaggle/working/checkpoints/seq2seq_best.pth")
        print("Saved best model.")
    else:
        no_improve += 1
        if no_improve >= PATIENCE:
            print(f"Early stopping (no improvement for {PATIENCE} epochs).")
            break

# =============================
# 8. Plot losses
# =============================
plt.plot(range(1, len(train_losses)+1), train_losses, label="train")
plt.plot(range(1, len(val_losses)+1), val_losses, label="val")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.legend()
plt.grid(True)
plt.show()

# ----------------------------
# Simple detokenizer
# ----------------------------
def detokenize(tokens):
    """Xóa space trước dấu câu, tránh lỗi spacing."""
    sentence = " ".join(tokens)
    sentence = sentence.replace(" ,", ",")
    sentence = sentence.replace(" .", ".")
    sentence = sentence.replace(" !", "!")
    sentence = sentence.replace(" ?", "?")
    sentence = sentence.replace(" ;", ";")
    sentence = sentence.replace(" :", ":")
    return sentence

# =============================
# 9. Translate & BLEU utilities
# =============================
def translate(sentence, model, src_vocab, trg_vocab, device, max_len=50):
    model.eval()
    # ---- 1. Tokenize English sentence ----
    tokens = tokenize_en(sentence)
    indices = [src_vocab.stoi.get("<sos>", 1)] + \
              [src_vocab.stoi.get(t, src_vocab.stoi["<unk>"]) for t in tokens] + \
              [src_vocab.stoi.get("<eos>", 2)]
    # ---- 2. Convert to tensor ----
    src_tensor = torch.tensor(indices, dtype=torch.long, device=device).unsqueeze(0)
    src_len = torch.tensor([len(indices)], dtype=torch.long, device=device)
    # ---- 3. Encode ----
    with torch.no_grad():
        hidden, cell = model.encoder(src_tensor, lengths=src_len)
    # ---- 4. Decode ----
    trg_indices = []
    cur_tok = trg_vocab.stoi["<sos>"]
    for _ in range(max_len):
        cur_tensor = torch.tensor([cur_tok], dtype=torch.long, device=device)
        with torch.no_grad():
            output, hidden, cell = model.decoder(cur_tensor, hidden, cell)
            # greedy: pick highest prob
            top1 = output.argmax(1).item()
        if top1 == trg_vocab.stoi["<eos>"]:
            break
        trg_indices.append(top1)
        cur_tok = top1
    # ---- 5. Convert indices → words ----
    trg_tokens = [trg_vocab.itos[idx] for idx in trg_indices]
    return detokenize(trg_tokens)


def evaluate_bleu(model, dataset, src_vocab, trg_vocab, device, n_examples=5):
    model.eval()
    smoothie = SmoothingFunction().method4

    sentence_scores = []
    refs = []  # for corpus BLEU
    hyps = []

    for i in range(len(dataset)):
        src_tensor, trg_tensor = dataset[i]

        # Convert source to text
        src_sentence = " ".join([src_vocab.itos[idx.item()] for idx in src_tensor[1:-1]])

        # Predict
        pred_sentence = translate(src_sentence, model, src_vocab, trg_vocab, device)
        pred_tokens = pred_sentence.split()

        # Target tokens
        trg_tokens = [trg_vocab.itos[idx.item()] for idx in trg_tensor[1:-1]]

        # Store for corpus BLEU
        refs.append([trg_tokens])
        hyps.append(pred_tokens)

        # Sentence BLEU (smoothed, BLEU-4)
        score = sentence_bleu(
            [trg_tokens],
            pred_tokens,
            weights=(0.25, 0.25, 0.25, 0.25),
            smoothing_function=smoothie
        )
        sentence_scores.append(score)

    # ---- Average sentence BLEU ----
    avg_sentence_bleu = sum(sentence_scores) / len(sentence_scores)

    # ---- Corpus BLEU ----
    corpus_score = corpus_bleu(refs, hyps)

    print(f"\n==== BLEU Evaluation ====")
    print(f"Avg sentence BLEU-4 (smoothed): {avg_sentence_bleu:.4f}")
    print(f"Corpus BLEU-4: {corpus_score:.4f}\n")

    # ---- Print examples ----
    print("===== Examples =====")
    import random
    for _ in range(min(n_examples, len(dataset))):
        i = random.randint(0, len(dataset)-1)
        src_tensor, trg_tensor = dataset[i]
        src_sentence = " ".join([src_vocab.itos[idx.item()] for idx in src_tensor[1:-1]])

        print("EN:   ", src_sentence)
        print("GT:   ", " ".join([trg_vocab.itos[idx.item()] for idx in trg_tensor[1:-1]]))
        print("PRED: ", translate(src_sentence, model, src_vocab, trg_vocab, device))
        print("-"*40)

# optionally evaluate with best model
if os.path.exists("/kaggle/working/checkpoints/seq2seq_best.pth"):
    model.load_state_dict(torch.load("/kaggle/working/checkpoints/seq2seq_best.pth", map_location=DEVICE))
    print("Evaluating BLEU on validation set...")
    evaluate_bleu(model, val_dataset, src_vocab, trg_vocab, DEVICE, n_examples=5)
