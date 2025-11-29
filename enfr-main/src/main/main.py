import os
import torch
import argparse
from torch.utils.data import random_split
from dataset import get_loader, tokenize_en, tokenize_fr
from model import Encoder, Decoder, Seq2Seq
from train import train_model
from inference import translate
from evaluate import evaluate_with_metrics

# ===========================================
# 1. ARGPARSE
# ===========================================
parser = argparse.ArgumentParser(description="Seq2Seq EN->FR Translation Training")

# Basic args
parser.add_argument("--batch_size", type=int, default=32, help="Batch size")
parser.add_argument("--n_epochs", type=int, default=1, help="Number of epochs")
parser.add_argument("--lr", type=float, default=0.001, help="Learning rate")
parser.add_argument("--teacher_forcing_ratio", type=float, default=0.5, help="Teacher forcing ratio")
parser.add_argument("--save_path", type=str, default="best_seq2seq.pt", help="Path to save model")

# Model args
parser.add_argument("--embed_dim", type=int, default=256, help="Embedding dimension")
parser.add_argument("--hidden_dim", type=int, default=512, help="LSTM hidden size")
parser.add_argument("--num_layers", type=int, default=2, help="Number of LSTM layers")
parser.add_argument("--dropout", type=float, default=0.3, help="Dropout rate")

# Experiment preset
parser.add_argument("--exp", type=str, default=None,
                    help="Experiment code: A1, A2, A3, A4, B1, B2, C1, C3")

args = parser.parse_args()

# ===========================================
# 1.1. EXPERIMENT PRESETS
# ===========================================
experiment_presets = {
    "A1": {"embed_dim": 256, "hidden_dim": 256, "num_layers": 1, "dropout": 0.2,
           "batch_size": 32, "lr": 0.001, "teacher_forcing_ratio": 0.5, "n_epochs": 10},

    "A2": {"embed_dim": 256, "hidden_dim": 512, "num_layers": 2, "dropout": 0.3,
           "batch_size": 32, "lr": 0.001, "teacher_forcing_ratio": 0.5, "n_epochs": 20},

    "A3": {"embed_dim": 512, "hidden_dim": 512, "num_layers": 2, "dropout": 0.3,
           "batch_size": 32, "lr": 0.001, "teacher_forcing_ratio": 0.5, "n_epochs": 20},

    "A4": {"embed_dim": 512, "hidden_dim": 1024, "num_layers": 2, "dropout": 0.4,
           "batch_size": 32, "lr": 0.001, "teacher_forcing_ratio": 0.5, "n_epochs": 20},

    "B1": {"embed_dim": 256, "hidden_dim": 512, "num_layers": 2, "dropout": 0.3,
           "batch_size": 32, "lr": 0.001, "teacher_forcing_ratio": 1.0, "n_epochs": 20},

    "B2": {"embed_dim": 256, "hidden_dim": 512, "num_layers": 2, "dropout": 0.3,
           "batch_size": 32, "lr": 0.001, "teacher_forcing_ratio": 0.2, "n_epochs": 20},

    "C1": {"embed_dim": 256, "hidden_dim": 512, "num_layers": 2, "dropout": 0.3,
           "batch_size": 32, "lr": 0.0005, "teacher_forcing_ratio": 0.5, "n_epochs": 20},

    "C3": {"embed_dim": 256, "hidden_dim": 512, "num_layers": 2, "dropout": 0.3,
           "batch_size": 32, "lr": 0.002, "teacher_forcing_ratio": 0.5, "n_epochs": 20},
}

# ===========================================
# 1.2. APPLY EXPERIMENT PRESET IF PROVIDED
# ===========================================
if args.exp is not None:
    if args.exp not in experiment_presets:
        raise ValueError(f"Experiment {args.exp} không tồn tại!")

    preset = experiment_presets[args.exp]

    print(f"\n⚙️ Using experiment preset: {args.exp}")
    print(preset)

    # Override args
    args.embed_dim = preset["embed_dim"]
    args.hidden_dim = preset["hidden_dim"]
    args.num_layers = preset["num_layers"]
    args.dropout = preset["dropout"]
    args.batch_size = preset["batch_size"]
    args.lr = preset["lr"]
    args.teacher_forcing_ratio = preset["teacher_forcing_ratio"]
    args.n_epochs = preset["n_epochs"]


# ===========================================
# 1.3. Assign final variables
# ===========================================
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

# ===========================================
# 2. DATA PATHS
# ===========================================
TRAIN_EN = "/kaggle/input/englishfrance/train.en"
TRAIN_FR = "/kaggle/input/englishfrance/train.fr"

VAL_EN = "/kaggle/input/englishfrance/val.en"
VAL_FR = "/kaggle/input/englishfrance/val.fr"

TEST_EN = "/kaggle/input/englishfrance/test_2018_flickr.en"
TEST_FR = "/kaggle/input/englishfrance/test_2018_flickr.fr"

# ===========================================
# 3. LOAD DATA
# ===========================================
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

SRC_VOCAB_SIZE = len(src_vocab.itos)
TRG_VOCAB_SIZE = len(trg_vocab.itos)

print(f"Dataset sizes: Train={len(train_loader.dataset)}, Val={len(val_loader.dataset)}, Test={len(test_loader.dataset)}")
print(f"Vocab sizes: EN={SRC_VOCAB_SIZE}, FR={TRG_VOCAB_SIZE}")

# ===========================================
# 4. INIT MODEL
# ===========================================
enc = Encoder(
    input_dim=SRC_VOCAB_SIZE,
    embed_dim=EMBED_DIM,
    hidden_dim=HIDDEN_DIM,
    num_layers=NUM_LAYERS,
    dropout=DROPOUT
)
dec = Decoder(
    output_dim=TRG_VOCAB_SIZE,
    embed_dim=EMBED_DIM,
    hidden_dim=HIDDEN_DIM,
    num_layers=NUM_LAYERS,
    dropout=DROPOUT
)

model = Seq2Seq(enc, dec, device, teacher_forcing_ratio=TEACHER_FORCING_RATIO).to(device)
print("Model initialized")

# ===========================================
# 5. TRAINING
# ===========================================
print("Start training...")

train_model(
    model,
    train_loader=train_loader,
    val_loader=val_loader,
    pad_idx=trg_vocab.stoi["<pad>"],
    n_epochs=N_EPOCHS,
    lr=LR,
    teacher_forcing_ratio=TEACHER_FORCING_RATIO,
    save_path=SAVE_PATH
)

model.load_state_dict(torch.load(SAVE_PATH))
model.to(device)
print("Best model loaded")

# ===========================================
# 6. INFERENCE EXAMPLES
# ===========================================
example_sentences = [
    "I love natural language processing.",
    "Machine learning is amazing.",
    "This is a simple test sentence."
]

print("\n Translation examples:")
for s in example_sentences:
    pred = translate(s, model, src_vocab, trg_vocab, tokenize_en)
    print(f"EN: {s}")
    print(f"FR(pred): {pred}\n")

# ===========================================
# 7. EVALUATION
# ===========================================
print("Evaluating on test set...")
avg_bleu, ppl, bleu_scores, examples = evaluate_with_metrics(
    model=model,
    dataloader=test_loader,
    src_vocab=src_vocab,
    trg_vocab=trg_vocab,
    src_tokenizer=tokenize_en,
    pad_idx=trg_vocab.stoi["<pad>"],
    device=device
)