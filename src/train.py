import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
from dataset import get_loader  # file dataset.py từ bước 1
from model import Encoder, Decoder, Seq2Seq

# =========================
# 1. Cấu hình
# =========================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

BATCH_SIZE = 32
EMBED_DIM = 256
HIDDEN_DIM = 512
NUM_LAYERS = 2
DROPOUT = 0.5
NUM_EPOCHS = 10
TEACHER_FORCING_RATIO = 0.5
LEARNING_RATE = 1e-3

# =========================
# 2. DataLoader
# =========================
train_loader, train_dataset = get_loader("data/train.en", "data/train.fr", batch_size=BATCH_SIZE, shuffle=True)
val_loader, val_dataset = get_loader("data/val.en", "data/val.fr", batch_size=BATCH_SIZE, shuffle=False)

SRC_VOCAB_SIZE = len(train_dataset.src_vocab.stoi)
TRG_VOCAB_SIZE = len(train_dataset.trg_vocab.stoi)
PAD_IDX = train_dataset.trg_vocab.stoi["<pad>"]

# =========================
# 3. Mô hình
# =========================
encoder = Encoder(SRC_VOCAB_SIZE, EMBED_DIM, HIDDEN_DIM, NUM_LAYERS, DROPOUT)
decoder = Decoder(TRG_VOCAB_SIZE, EMBED_DIM, HIDDEN_DIM, NUM_LAYERS, DROPOUT)
model = Seq2Seq(encoder, decoder, device).to(device)

# =========================
# 4. Loss & Optimizer
# =========================
criterion = nn.CrossEntropyLoss(ignore_index=PAD_IDX)
optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

# =========================
# 5. Training loop
# =========================
train_losses = []

for epoch in range(NUM_EPOCHS):
    model.train()
    epoch_loss = 0

    for src, trg in train_loader:
        src = src.to(device)
        trg = trg.to(device)

        optimizer.zero_grad()
        output = model(src, trg, teacher_forcing_ratio=TEACHER_FORCING_RATIO)
        # output: [batch, trg_len, trg_vocab_size]
        output_dim = output.shape[-1]

        # flatten output & target để tính loss
        output = output[:,1:,:].reshape(-1, output_dim)  # bỏ <sos>
        trg = trg[:,1:].reshape(-1)

        loss = criterion(output, trg)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1)  # tránh gradient explosion
        optimizer.step()

        epoch_loss += loss.item()

    avg_loss = epoch_loss / len(train_loader)
    train_losses.append(avg_loss)
    print(f"Epoch [{epoch+1}/{NUM_EPOCHS}] - Loss: {avg_loss:.4f}")

    # Lưu checkpoint sau mỗi epoch
    torch.save(model.state_dict(), f"checkpoint_epoch{epoch+1}.pth")

# =========================
# 6. Vẽ biểu đồ loss
# =========================
plt.figure(figsize=(8,5))
plt.plot(range(1, NUM_EPOCHS+1), train_losses, marker='o')
plt.title("Training Loss")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.grid()
plt.show()
