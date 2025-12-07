import torch
import matplotlib.pyplot as plt
from nltk.translate.bleu_score import corpus_bleu
import random
import numpy as np

# -----------------------------
# 1. Set seed reproducibility
# -----------------------------
def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)

# -----------------------------
# 2. Lưu/Load checkpoint
# -----------------------------
def save_checkpoint(model, optimizer, epoch, path):
    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict()
    }, path)

def load_checkpoint(path, model, optimizer=None):
    checkpoint = torch.load(path)
    model.load_state_dict(checkpoint['model_state_dict'])
    if optimizer:
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    return checkpoint['epoch']

# -----------------------------
# 3. Tính BLEU
# -----------------------------
def compute_bleu(references, hypotheses):
    return corpus_bleu(references, hypotheses)

# -----------------------------
# 4. Vẽ đồ thị loss
# -----------------------------
def plot_loss(train_losses, val_losses=None):
    plt.figure(figsize=(8,6))
    plt.plot(train_losses, label="Train loss")
    if val_losses:
        plt.plot(val_losses, label="Validation loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.show()

# -----------------------------
# 5. Convert indices -> tokens
# -----------------------------
def idx2tokens(indices, vocab, ignore_tokens=None):
    if ignore_tokens is None:
        ignore_tokens = []
    return [vocab.itos[i] for i in indices if i not in ignore_tokens]
