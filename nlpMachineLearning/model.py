# src/train.py
import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import argparse
import random
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
args = parser.parse_args()

BATCH_SIZE = args.batch_size
EMBED_SIZE = args.embed_size
HIDDEN_SIZE = args.hidden_size
NUM_LAYERS = args.num_layers
NUM_EPOCHS = args.num_epochs
LEARNING_RATE = args.learning_rate
TEACHER_FORCING_RATIO = args.teacher_forcing_ratio
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# =============================
# 2. Load dataset + vocab
# =============================
# Load dataset + vocab
DATA_DIR = "/kaggle/working/data/"

src_vocab = load_vocab(os.path.join(DATA_DIR, "vocab_en.pkl"))
trg_vocab = load_vocab(os.path.join(DATA_DIR, "vocab_fr.pkl"))
train_dataset = load_dataset(os.path.join(DATA_DIR, "train_dataset.pt"))


PAD_IDX = src_vocab.stoi["<pad>"]

train_loader = DataLoader(
    train_dataset,
    batch_size=BATCH_SIZE,
    shuffle=True,
    collate_fn=MyCollate(PAD_IDX)
)

# Tạo thư mục checkpoints
os.makedirs("/kaggle/working/checkpoints", exist_ok=True)

print(f"✅ Dataset, vocab loaded. PAD_IDX={PAD_IDX}")
print(f"Number of training examples: {len(train_dataset)}")

# =============================
# 3. Encoder
# =============================
class Encoder(nn.Module):
    def __init__(self, input_dim, embed_size, hidden_size, num_layers, pad_idx=PAD_IDX):
        super().__init__()
        self.embedding = nn.Embedding(input_dim, embed_size, padding_idx=pad_idx)
        self.lstm = nn.LSTM(embed_size, hidden_size, num_layers, batch_first=True)

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
    def __init__(self, output_dim, embed_size, hidden_size, num_layers, pad_idx=PAD_IDX):
        super().__init__()
        self.embedding = nn.Embedding(output_dim, embed_size, padding_idx=pad_idx)
        self.lstm = nn.LSTM(embed_size, hidden_size, num_layers, batch_first=True)
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
        batch_size = src.shape[0]
        trg_len = trg.shape[1]
        trg_vocab_size = self.trg_vocab_size
        outputs = torch.zeros(batch_size, trg_len, trg_vocab_size).to(self.device)
        hidden, cell = self.encoder(src, lengths=src_lengths)
        input_tok = trg[:, 0]  # <sos>
        for t in range(1, trg_len):
            output, hidden, cell = self.decoder(input_tok, hidden, cell)
            outputs[:, t, :] = output
            teacher_force = random.random() < teacher_forcing_ratio
            top1 = output.argmax(1)
            input_tok = trg[:, t] if teacher_force else top1
        return outputs

# =============================
# 6. Initialize model + optimizer + loss
# =============================
encoder = Encoder(len(src_vocab.stoi), EMBED_SIZE, HIDDEN_SIZE, NUM_LAYERS, pad_idx=PAD_IDX).to(DEVICE)
decoder = Decoder(len(trg_vocab.stoi), EMBED_SIZE, HIDDEN_SIZE, NUM_LAYERS, pad_idx=PAD_IDX).to(DEVICE)
model = Seq2Seq(encoder, decoder, DEVICE, trg_vocab_size=len(trg_vocab.stoi)).to(DEVICE)

optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
criterion = nn.CrossEntropyLoss(ignore_index=PAD_IDX)

# =============================
# 7. Training loop
# =============================
loss_list = []

for epoch in range(NUM_EPOCHS):
    model.train()
    epoch_loss = 0
    for src, trg, src_lengths, trg_lengths in train_loader:
        src, trg, src_lengths = src.to(DEVICE), trg.to(DEVICE), src_lengths.to(DEVICE)
        optimizer.zero_grad()
        output = model(src, trg, src_lengths, teacher_forcing_ratio=TEACHER_FORCING_RATIO)
        output_dim = output.shape[-1]
        output = output[:,1:,:].reshape(-1, output_dim)
        trg_flat = trg[:,1:].reshape(-1)
        loss = criterion(output, trg_flat)
        loss.backward()
        optimizer.step()
        epoch_loss += loss.item()
    avg_loss = epoch_loss / len(train_loader)
    loss_list.append(avg_loss)
    print(f"Epoch [{epoch+1}/{NUM_EPOCHS}] Loss: {avg_loss:.4f}")
    # save checkpoint
    torch.save(model.state_dict(), f"/kaggle/working/checkpoints/seq2seq_epoch{epoch+1}.pth")

# =============================
# 8. Plot loss
# =============================
plt.plot(range(1, NUM_EPOCHS+1), loss_list, marker='o')
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("Training Loss")
plt.grid(True)
plt.show()
