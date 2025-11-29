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
# 1. THAM SỐ
# ===========================================
parser = argparse.ArgumentParser(description="Seq2Seq EN->FR Translation Training")
parser.add_argument("--batch_size", type=int, default=32, help="Batch size")
parser.add_argument("--n_epochs", type=int, default=1, help="Number of epochs")
parser.add_argument("--lr", type=float, default=0.001, help="Learning rate")
parser.add_argument("--teacher_forcing_ratio", type=float, default=0.5, help="Teacher forcing ratio")
parser.add_argument("--save_path", type=str, default="best_seq2seq.pt", help="Path to save model")
args = parser.parse_args()

BATCH_SIZE = args.batch_size
N_EPOCHS = args.n_epochs
LR = args.lr
TEACHER_FORCING_RATIO = args.teacher_forcing_ratio
SAVE_PATH = args.save_path

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ===========================================
# 2. Đường dẫn dữ liệu
# ===========================================
TRAIN_EN = "/kaggle/input/englishfrance/train.en"
TRAIN_FR = "/kaggle/input/englishfrance/train.fr"

VAL_EN = "/kaggle/input/englishfrance/val.en"
VAL_FR = "/kaggle/input/englishfrance/val.fr"

TEST_EN = "/kaggle/input/englishfrance/test_2018_flickr.en"
TEST_FR = "/kaggle/input/englishfrance/test_2018_flickr.fr"

# ===========================================
# 3. Load dataset và DataLoader
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
# 4. Khởi tạo Model
# ===========================================
enc = Encoder(SRC_VOCAB_SIZE, embed_dim=256, hidden_dim=512, num_layers=2, dropout=0.3)
dec = Decoder(TRG_VOCAB_SIZE, embed_dim=256, hidden_dim=512, num_layers=2, dropout=0.3)

model = Seq2Seq(enc, dec, device, teacher_forcing_ratio=TEACHER_FORCING_RATIO).to(device)
print("Model initialized")

# ===========================================
# 5. Huấn luyện
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

# Load best model
model.load_state_dict(torch.load(SAVE_PATH))
model.to(device)
print("Best model loaded")

# ===========================================
# 6. Dự đoán ví dụ
# ===========================================
example_sentences = [
    "I love natural language processing.",
    "Machine learning is amazing.",
    "This is a simple test sentence."
]

print("\n🔹 Translation examples:")
for s in example_sentences:
    pred = translate(s, model, src_vocab, trg_vocab, tokenize_en)
    print(f"EN: {s}")
    print(f"FR(pred): {pred}\n")

# ===========================================
# 7. Đánh giá trên tập test
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
