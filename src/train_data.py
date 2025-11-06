"""
Load dataset + vocab
Khởi tạo Encoder–Decoder LSTM
Teacher Forcing
CrossEntropyLoss (ignore <pad>)
Lưu checkpoint .pth
In loss + plot biểu đồ

✅ Điểm nổi bật:
Dataset & vocab được load từ .pt và .pkl → tiết kiệm thời gian tokenization và build vocab
Encoder–Decoder LSTM cơ bản, không dùng attention
Teacher Forcing tùy chỉnh với TEACHER_FORCING_RATIO
CrossEntropyLoss ignore <pad>
Checkpoint .pth lưu mỗi epoch
Plot loss sau khi train
"""

# src/train.py
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
from data import load_dataset, load_vocab, MyCollate
from data import TranslationDataset  # dùng khi tạo DataLoader

# =============================
# 1. Hyperparameters
# =============================
BATCH_SIZE = 32
EMBED_SIZE = 256
HIDDEN_SIZE = 512
NUM_LAYERS = 1
LEARNING_RATE = 0.001
NUM_EPOCHS = 10
TEACHER_FORCING_RATIO = 0.5
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# =============================
# 2. Load dataset + vocab
# =============================
train_dataset = load_dataset("data/train_dataset.pt")
src_vocab = load_vocab("data/vocab_en.pkl")
trg_vocab = load_vocab("data/vocab_fr.pkl")

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True,
                          collate_fn=MyCollate(train_dataset.pad_idx))

PAD_IDX = src_vocab.stoi["<pad>"]

# =============================
# 3. Encoder
# =============================
class Encoder(nn.Module):
    def __init__(self, input_dim, embed_size, hidden_size, num_layers):
        super().__init__()
        self.embedding = nn.Embedding(input_dim, embed_size, padding_idx=PAD_IDX)
        self.lstm = nn.LSTM(embed_size, hidden_size, num_layers, batch_first=True)

    def forward(self, x):
        embedded = self.embedding(x)
        outputs, (hidden, cell) = self.lstm(embedded)
        return hidden, cell

# =============================
# 4. Decoder
# =============================
class Decoder(nn.Module):
    def __init__(self, output_dim, embed_size, hidden_size, num_layers):
        super().__init__()
        self.embedding = nn.Embedding(output_dim, embed_size, padding_idx=PAD_IDX)
        self.lstm = nn.LSTM(embed_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_dim)

    def forward(self, x, hidden, cell):
        x = x.unsqueeze(1)  # (batch, 1)
        embedded = self.embedding(x)
        outputs, (hidden, cell) = self.lstm(embedded, (hidden, cell))
        predictions = self.fc(outputs.squeeze(1))
        return predictions, hidden, cell

# =============================
# 5. Seq2Seq
# =============================
class Seq2Seq(nn.Module):
    def __init__(self, encoder, decoder, device):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.device = device

    def forward(self, src, trg, teacher_forcing_ratio=0.5):
        batch_size = src.shape[0]
        trg_len = trg.shape[1]
        trg_vocab_size = len(trg_vocab.stoi)

        outputs = torch.zeros(batch_size, trg_len, trg_vocab_size).to(self.device)

        hidden, cell = self.encoder(src)
        input = trg[:,0]  # <sos>

        for t in range(1, trg_len):
            output, hidden, cell = self.decoder(input, hidden, cell)
            outputs[:,t,:] = output
            teacher_force = torch.rand(1).item() < teacher_forcing_ratio
            top1 = output.argmax(1)
            input = trg[:,t] if teacher_force else top1

        return outputs

# =============================
# 6. Initialize model + optimizer + loss
# =============================
encoder = Encoder(len(src_vocab.stoi), EMBED_SIZE, HIDDEN_SIZE, NUM_LAYERS).to(DEVICE)
decoder = Decoder(len(trg_vocab.stoi), EMBED_SIZE, HIDDEN_SIZE, NUM_LAYERS).to(DEVICE)
model = Seq2Seq(encoder, decoder, DEVICE).to(DEVICE)

optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
criterion = nn.CrossEntropyLoss(ignore_index=PAD_IDX)

# =============================
# 7. Training loop
# =============================
loss_list = []

for epoch in range(NUM_EPOCHS):
    model.train()
    epoch_loss = 0
    for src, trg in train_loader:
        src, trg = src.to(DEVICE), trg.to(DEVICE)
        optimizer.zero_grad()
        output = model(src, trg, teacher_forcing_ratio=TEACHER_FORCING_RATIO)

        # reshape for loss: (batch*seq_len, vocab_size)
        output_dim = output.shape[-1]
        output = output[:,1:,:].reshape(-1, output_dim)
        trg = trg[:,1:].reshape(-1)
        loss = criterion(output, trg)
        loss.backward()
        optimizer.step()
        epoch_loss += loss.item()

    avg_loss = epoch_loss / len(train_loader)
    loss_list.append(avg_loss)
    print(f"Epoch [{epoch+1}/{NUM_EPOCHS}] Loss: {avg_loss:.4f}")

    # save checkpoint
    torch.save(model.state_dict(), f"checkpoints/seq2seq_epoch{epoch+1}.pth")

# =============================
# 8. Plot loss
# =============================
plt.plot(range(1, NUM_EPOCHS+1), loss_list, marker='o')
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("Training Loss")
plt.grid(True)
plt.show()
