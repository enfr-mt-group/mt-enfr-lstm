import os
import csv
import torch
import argparse
from torch.utils.data import random_split
from dataset import get_loader, tokenize_en, tokenize_fr
from model import Encoder, Decoder, Seq2Seq
from train import train_model
from inference import translate
from evaluate import evaluate_with_metrics
import matplotlib.pyplot as plt
import random
import numpy as np
from loadCheckpoint import load_checkpoint
# Đặt seed để kiểm soát tính ngẫu nhiên
def setup_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

setup_seed(42)

# 1. Tham số thực nghiệm
parser = argparse.ArgumentParser(description="EN FR Translation Training")

# Basic args
parser.add_argument("--batch_size", type=int, default=32)
parser.add_argument("--n_epochs", type=int, default=1)
parser.add_argument("--lr", type=float, default=0.001)
parser.add_argument("--teacher_forcing_ratio", type=float, default=0.5)
parser.add_argument("--save_path", type=str, default="enfr-main/src/checkpoint/best_seq2seq.pth")

# Model args
parser.add_argument("--embed_dim", type=int, default=256)
parser.add_argument("--hidden_dim", type=int, default=512)
parser.add_argument("--num_layers", type=int, default=2)
parser.add_argument("--dropout", type=float, default=0.3)

parser.add_argument("--use_attention", action="store_true",
                    help="Enable Luong Attention in Decoder")

# Experiment preset
parser.add_argument("--exp", type=str, default=None,
                    help="Experiment code: A1, A2, A3, A4, A5")

args = parser.parse_args()

# 1.1 Kịch bản thực nghiệm
experiment_presets = {
    "A1": {"embed_dim": 256, "hidden_dim": 512, "num_layers": 2, "dropout": 0.3, "batch_size": 32, "lr": 0.001,
           "teacher_forcing_ratio": 0.5, "n_epochs": 20, "use_attention": True},

    "A2": {"embed_dim": 512, "hidden_dim": 512, "num_layers": 2, "dropout": 0.3, "batch_size": 32, "lr": 0.001,
           "teacher_forcing_ratio": 0.5, "n_epochs": 20, "use_attention": True},

    "A3": {"embed_dim": 512, "hidden_dim": 512, "num_layers": 2, "dropout": 0.5, "batch_size": 32, "lr": 0.001,
           "teacher_forcing_ratio": 0.5, "n_epochs": 20, "use_attention": True},

    "A4": {"embed_dim": 512, "hidden_dim": 512, "num_layers": 2, "dropout": 0.3, "batch_size": 128, "lr": 0.001,
           "teacher_forcing_ratio": 0.5, "n_epochs": 20, "use_attention": True},

    "A5": {"embed_dim": 512, "hidden_dim": 512, "num_layers": 2, "dropout": 0.3, "batch_size": 64, "lr": 0.001,
           "teacher_forcing_ratio": 0.5, "n_epochs": 20, "use_attention": True},
}

# 1.2 áp dụng giá trị preset
if args.exp is not None:
    if args.exp not in experiment_presets:
        raise ValueError(f"Preset {args.exp} does not exist!")

    preset = experiment_presets[args.exp]

    print(f"\nUsing preset: {args.exp}")
    print(preset)

    args.embed_dim = preset["embed_dim"]
    args.hidden_dim = preset["hidden_dim"]
    args.num_layers = preset["num_layers"]
    args.dropout = preset["dropout"]
    args.batch_size = preset["batch_size"]
    args.lr = preset["lr"]
    args.teacher_forcing_ratio = preset["teacher_forcing_ratio"]
    args.n_epochs = preset["n_epochs"]
    args.use_attention = preset["use_attention"]

# 1.3 Tham số đưa vào mô hình
BATCH_SIZE = args.batch_size
N_EPOCHS = args.n_epochs
LR = args.lr
TEACHER_FORCING_RATIO = args.teacher_forcing_ratio
SAVE_PATH = args.save_path
EMBED_DIM = args.embed_dim
HIDDEN_DIM = args.hidden_dim
NUM_LAYERS = args.num_layers
DROPOUT = args.dropout
USE_ATTENTION = args.use_attention

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# hàm lưu thực nghiệm vào CSV
def log_experiment(csv_path, row):
    file_exists = os.path.isfile(csv_path)
    with open(csv_path, "a", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        if not file_exists:
            writer.writerow([
                "exp", "embed", "hidden", "layers", "dropout",
                "batch", "lr", "teacher_forcing", "use_attention",
                "epochs", "BLEU", "PPL"
            ])
        writer.writerow(row)

# 2. data paths
TRAIN_EN = "/kaggle/input/englishfrance/train.en"
TRAIN_FR = "/kaggle/input/englishfrance/train.fr"

VAL_EN = "/kaggle/input/englishfrance/val.en"
VAL_FR = "/kaggle/input/englishfrance/val.fr"

TEST_EN = "/kaggle/input/englishfrance/test_2016_flickr.en"
TEST_FR = "/kaggle/input/englishfrance/test_2016_flickr.fr"

# 3. load DataLoaderss
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

# 4. Xây dựng mô hình
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
    dropout=DROPOUT,
    use_attention=USE_ATTENTION
)

model = Seq2Seq(
    encoder=enc, decoder=dec, device=device, teacher_forcing_ratio=TEACHER_FORCING_RATIO
).to(device)
print("Model initialized")

# 5. Huấn luyện mô hình
print("Start training...")

train_losses, val_losses = train_model(
    model,
    train_loader=train_loader,
    val_loader=val_loader,
    pad_idx=trg_vocab.stoi["<pad>"],
    n_epochs=N_EPOCHS,
    lr=LR,
    teacher_forcing_ratio=TEACHER_FORCING_RATIO,
    save_path=SAVE_PATH
)

plt.figure(figsize=(10,6))
plt.plot(train_losses, label="Train Loss", marker="o")
plt.plot(val_losses, label="Val Loss", marker="x")

plt.title("Training Curve (Loss per Epoch)")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.savefig("loss_curve.png")
plt.show()

model.load_state_dict(torch.load(SAVE_PATH))
model.to(device)
print("Best model loaded")

# 6. Ví dụ dự đoán dịch câu từ Anh sang Pháp
example_sentences = [
    "a man wearing a hat is standing in front of a book store while looking at a window .",
    "a girl in a pink coat is looking for a book while standing in a building .",
    "a person wearing goggles is sledding down a hill in front of a building ."
]

print("\n Translation examples:")
for s in example_sentences:
    pred = translate(s, model, src_vocab, trg_vocab, tokenize_en, method="beam", beam_sizes=5) #
    print(f"EN: {s}")
    print(f"FR(pred): {pred}\n")

# 7. Đánh giá tập testt
print("Evaluating on test set...")
avg_bleu, ppl, bleu_scores, examples = evaluate_with_metrics(
    model=model,
    dataloader=test_loader,
    src_vocab=src_vocab,
    trg_vocab=trg_vocab,
    src_tokenizer=tokenize_en,
    pad_idx=trg_vocab.stoi["<pad>"],
    device=device,
    method="beam",
    beam_sizes=5
)

print(f"\nFinal BLEU: {avg_bleu:.4f}")
print(f"Final Perplexity: {ppl:.4f}")

load_checkpoint(SAVE_PATH)

# 8. Lưu kết quả thực nghiệm vào csv
log_experiment(
    "experiment_results.csv",
    [
        args.exp if args.exp else "Custom",
        EMBED_DIM,
        HIDDEN_DIM,
        NUM_LAYERS,
        DROPOUT,
        BATCH_SIZE,
        LR,
        TEACHER_FORCING_RATIO,
        USE_ATTENTION,
        N_EPOCHS,
        avg_bleu,
        ppl
    ]
)

print("\n Kết quả được lưu vào experiment_results.csv")