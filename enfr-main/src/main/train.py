import torch
import torch.nn as nn
import torch.optim as optim
import random
import os
from torch.utils.data import DataLoader
from torch.nn.utils.rnn import pad_sequence

# ===============================================
# 1. Device
# ===============================================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ===============================================
# 2. Hàm huấn luyện 1 epoch
# ===============================================
def train_epoch(model, dataloader, optimizer, criterion, pad_idx, teacher_forcing_ratio=0.5):
    model.train()
    epoch_loss = 0

    for src, trg, src_len, trg_len in dataloader:
        src = src.to(device)
        trg = trg.to(device)
        src_len = src_len.to(device)

        optimizer.zero_grad()

        # Forward pass
        output = model(src, src_len, trg)  # [batch, trg_len, vocab_size]

        # Flatten để tính loss
        output_dim = output.shape[-1]
        output = output[:, 1:].reshape(-1, output_dim)  # bỏ <sos>
        trg = trg[:, 1:].reshape(-1)

        loss = criterion(output, trg)
        loss.backward()

        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

        optimizer.step()
        epoch_loss += loss.item()

    return epoch_loss / len(dataloader)


# ===============================================
# 3. Hàm validation
# ===============================================
def evaluate(model, dataloader, criterion, pad_idx):
    model.eval()
    epoch_loss = 0

    with torch.no_grad():
        for src, trg, src_len, trg_len in dataloader:
            src = src.to(device)
            trg = trg.to(device)
            src_len = src_len.to(device)

            output = model(src, src_len, trg)

            output_dim = output.shape[-1]
            output = output[:, 1:].reshape(-1, output_dim)
            trg = trg[:, 1:].reshape(-1)

            loss = criterion(output, trg)
            epoch_loss += loss.item()

    return epoch_loss / len(dataloader)


# ===============================================
# 4. Huấn luyện đầy đủ + Early stopping + Save best
# ===============================================
def train_model(model, train_loader, val_loader, pad_idx,
                n_epochs=20, lr=0.001, teacher_forcing_ratio=0.5,
                save_path="best_model.pt"):

    optimizer = optim.Adam(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss(ignore_index=pad_idx)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=0.5, patience=2, verbose=True)

    best_val_loss = float("inf")
    epochs_no_improve = 0
    early_stop_patience = 3

    for epoch in range(1, n_epochs+1):
        train_loss = train_epoch(model, train_loader, optimizer, criterion, pad_idx, teacher_forcing_ratio)
        val_loss = evaluate(model, val_loader, criterion, pad_idx)

        print(f"Epoch [{epoch}/{n_epochs}] | Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f}")

        scheduler.step(val_loss)

        # Early stopping + save best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), save_path)
            epochs_no_improve = 0
            print(" Best model saved")
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= early_stop_patience:
                print(f"⚠ Early stopping at epoch {epoch}")
                break

    print("Training complete. Best Val Loss: {:.4f}".format(best_val_loss))

# Giả sử bạn đã có:
# train_loader, val_loader (DataLoader)
# pad_idx = dataset.src_vocab.stoi["<pad>"] hoặc trg_vocab

model.to(device)

train_model(
    model,
    train_loader=train_loader,
    val_loader=val_loader,
    pad_idx=dataset.trg_vocab.stoi["<pad>"],
    n_epochs=20,
    lr=0.001,
    teacher_forcing_ratio=0.5,
    save_path="best_seq2seq.pt"
)

