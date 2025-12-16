import torch
import torch.nn as nn
import torch.optim as optim
import random
import os
from torch.utils.data import DataLoader
from torch.nn.utils.rnn import pad_sequence

# 1. Device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 2. Hàm huấn luyện 1 epoch
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

# 3. Hàm validation
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

# 4. Huấn luyện đầy đủ + Early stopping + Save best
def train_model(model, train_loader, val_loader, src_vocab, trg_vocab, pad_idx,
                n_epochs=20, lr=0.001, teacher_forcing_ratio=0.5,
                save_path="best_model.pth"):

    optimizer = optim.Adam(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss(ignore_index=pad_idx)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=0.5, patience=2, verbose=True)

    best_val_loss = float("inf")
    epochs_no_improve = 0
    early_stop_patience = 3

    train_losses = []
    val_losses = []

    for epoch in range(1, n_epochs+1):
        #Train loss
        train_loss = train_epoch(model, train_loader, optimizer, criterion, pad_idx, teacher_forcing_ratio)
        train_losses.append(train_loss)

        #Validation loss
        val_loss = evaluate(model, val_loader, criterion, pad_idx)
        val_losses.append(val_loss)

        print(f"Epoch [{epoch}/{n_epochs}] | Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f}")

        scheduler.step(val_loss)

        # Early stopping + save best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            torch.save({
                "model_state": model.state_dict(),
                "src_vocab": src_vocab,
                "trg_vocab": trg_vocab,
                "config": {
                    "embed_dim": model.encoder.embedding.embedding_dim,
                    "hidden_dim": model.encoder.lstm.hidden_size,
                    "num_layers": model.encoder.lstm.num_layers,
                    "dropout": model.encoder.dropout.p
                }
            }, save_path)
            epochs_no_improve = 0
            print(" Best model saved")
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= early_stop_patience:
                print(f"Early stopping at epoch {epoch}")
                break

    print("Training hoàn thành. Best Val Loss: {:.4f}".format(best_val_loss))

    return train_losses, val_losses
