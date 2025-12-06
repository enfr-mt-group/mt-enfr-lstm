import os
import csv
import torch
import argparse

from dataset import get_loader, tokenize_en, tokenize_fr
from model import Encoder, Decoder, Seq2Seq
from train import train_model
from inference import translate
from evaluate import evaluate_with_metrics

# Argument Parser
parser = argparse.ArgumentParser(description="Seq2Seq EN->FR Translation")

parser.add_argument("--batch_size", type=int, default=32)
parser.add_argument("--n_epochs", type=int, default=1)
parser.add_argument("--lr", type=float, default=0.001)
parser.add_argument("--teacher_forcing_ratio", type=float, default=0.5)
parser.add_argument("--save_path", type=str, default="best_seq2seq.pt")

parser.add_argument("--embed_dim", type=int, default=256)
parser.add_argument("--hidden_dim", type=int, default=512)
parser.add_argument("--num_layers", type=int, default=2)
parser.add_argument("--dropout", type=float, default=0.3)

parser.add_argument("--exp", type=str, default=None)

args = parser.parse_args()

# Experiment presets
experiment_presets = {
    "A1": {"embed_dim": 256, "hidden_dim": 256, "num_layers": 1, "dropout": 0.2,
           "batch_size": 32, "lr": 0.001, "teacher_forcing_ratio": 0.5, "n_epochs": 1},

    "A2": {"embed_dim": 256, "hidden_dim": 512, "num_layers": 2, "dropout": 0.3,
           "batch_size": 32, "lr": 0.001, "teacher_forcing_ratio": 0.5, "n_epochs": 20},
}

# Apply preset
if args.exp:
    preset = experiment_presets[args.exp]
    print(f"Using preset {args.exp}: {preset}")

    for k, v in preset.items():
        setattr(args, k, v)


# ============================================================
# Hyperparameters
# ============================================================
BATCH_SIZE = args.batch_size
N_EPOCHS = args.n_epochs
LR = args.lr
TEACHER_FORCING_RATIO = args.teacher_forcing_ratio
SAVE_PATH = args.save_path

EMBED_DIM = args.embed_dim
HIDDEN_DIM = args.hidden_dim
NUM_LAYERS = args.num_layers
DROPOUT = args.dropout

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)


# Data paths
TRAIN_EN = "/kaggle/input/englishfrance/train.en"
TRAIN_FR = "/kaggle/input/englishfrance/train.fr"

VAL_EN = "/kaggle/input/englishfrance/val.en"
VAL_FR = "/kaggle/input/englishfrance/val.fr"

TEST_EN = "/kaggle/input/englishfrance/test_2018_flickr.en"
TEST_FR = "/kaggle/input/englishfrance/test_2018_flickr.fr"

# Load Data
print("Building DataLoaders...")

train_loader, src_vocab, trg_vocab = get_loader(
    TRAIN_EN, TRAIN_FR, batch_size=BATCH_SIZE, shuffle=True
)
val_loader, _, _ = get_loader(
    VAL_EN, VAL_FR, batch_size=BATCH_SIZE, src_vocab=src_vocab, trg_vocab=trg_vocab
)
test_loader, _, _ = get_loader(
    TEST_EN, TEST_FR, batch_size=BATCH_SIZE, src_vocab=src_vocab, trg_vocab=trg_vocab
)

print("Train size:", len(train_loader.dataset))
print("Val size:", len(val_loader.dataset))
print("Test size:", len(test_loader.dataset))

# Build Model
enc = Encoder(len(src_vocab.itos), EMBED_DIM, HIDDEN_DIM, NUM_LAYERS, DROPOUT)
dec = Decoder(len(trg_vocab.itos), EMBED_DIM, HIDDEN_DIM, NUM_LAYERS, DROPOUT)

model = Seq2Seq(enc, dec, device, TEACHER_FORCING_RATIO).to(device)
print("Model initialized.")

# Train
train_model(
    model, train_loader, val_loader,
    pad_idx=trg_vocab.stoi["<pad>"],
    n_epochs=N_EPOCHS, lr=LR,
    teacher_forcing_ratio=TEACHER_FORCING_RATIO,
    save_path=SAVE_PATH
)

model.load_state_dict(torch.load(SAVE_PATH))
model.to(device)

# Test Example Predictions
examples = [
    "I love natural language processing.",
    "Machine learning is amazing.",
]

for s in examples:
    print("EN:", s)
    print("FR:", translate(s, model, src_vocab, trg_vocab, tokenize_en), "\n")

# Evaluate Test Set
avg_bleu, ppl, bleu_scores, examples = evaluate_with_metrics(
    model,
    test_loader,
    src_vocab,
    trg_vocab,
    tokenize_en,
    trg_vocab.stoi["<pad>"],
    device
)

print("Final BLEU:", avg_bleu)
print("Final Perplexity:", ppl)