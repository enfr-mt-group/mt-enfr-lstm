import torch
import torch.nn as nn
from nltk.translate.bleu_score import corpus_bleu
from dataset import get_loader
from model import Encoder, Decoder, Seq2Seq

# =========================
# 1. Cấu hình
# =========================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
BATCH_SIZE = 32

# =========================
# 2. Load dataset
# =========================
test_loader, test_dataset = get_loader("data/test.en", "data/test.fr", batch_size=BATCH_SIZE, shuffle=False)
SRC_VOCAB_SIZE = len(test_dataset.src_vocab.stoi)
TRG_VOCAB_SIZE = len(test_dataset.trg_vocab.stoi)
PAD_IDX = test_dataset.trg_vocab.stoi["<pad>"]
SOS_IDX = test_dataset.trg_vocab.stoi["<sos>"]
EOS_IDX = test_dataset.trg_vocab.stoi["<eos>"]

# =========================
# 3. Load mô hình
# =========================
# Phải khớp với thông số train.py
EMBED_DIM = 256
HIDDEN_DIM = 512
NUM_LAYERS = 2
DROPOUT = 0.5

encoder = Encoder(SRC_VOCAB_SIZE, EMBED_DIM, HIDDEN_DIM, NUM_LAYERS, DROPOUT)
decoder = Decoder(TRG_VOCAB_SIZE, EMBED_DIM, HIDDEN_DIM, NUM_LAYERS, DROPOUT)
model = Seq2Seq(encoder, decoder, device).to(device)
model.load_state_dict(torch.load("checkpoint_epoch10.pth", map_location=device))
model.eval()

# =========================
# 4. Hàm sinh câu
# =========================
def translate_sentence(src_sentence, src_vocab, trg_vocab, max_len=20):
    model.eval()
    
    # Chuyển sentence -> indices
    src_indices = [src_vocab.stoi.get(tok, src_vocab.stoi["<unk>"]) for tok in src_sentence]
    src_tensor = torch.LongTensor([src_indices]).to(device)
    
    with torch.no_grad():
        hidden, cell = model.encoder(src_tensor)
    
    # Khởi tạo decoder input với <sos>
    trg_indices = [SOS_IDX]
    
    for _ in range(max_len):
        input_tensor = torch.LongTensor([trg_indices[-1]]).to(device)
        with torch.no_grad():
            output, hidden, cell = model.decoder(input_tensor, hidden, cell)
            pred_token = output.argmax(1).item()
        if pred_token == EOS_IDX:
            break
        trg_indices.append(pred_token)
    
    # Chuyển indices -> tokens
    trg_tokens = [trg_vocab.itos[idx] for idx in trg_indices[1:]]  # bỏ <sos>
    return trg_tokens

# =========================
# 5. Tính BLEU score
# =========================
all_references = []
all_hypotheses = []

for src_batch, trg_batch in test_loader:
    for i in range(src_batch.size(0)):
        src_indices = src_batch[i].tolist()
        trg_indices = trg_batch[i].tolist()
        
        # Loại bỏ padding & special tokens
        src_tokens = [test_dataset.src_vocab.itos[idx] for idx in src_indices if idx not in [PAD_IDX, SOS_IDX, EOS_IDX]]
        trg_tokens = [test_dataset.trg_vocab.itos[idx] for idx in trg_indices if idx not in [PAD_IDX, SOS_IDX, EOS_IDX]]
        
        pred_tokens = translate_sentence(src_tokens, test_dataset.src_vocab, test_dataset.trg_vocab)
        
        all_references.append([trg_tokens])  # BLEU yêu cầu list of list
        all_hypotheses.append(pred_tokens)

# BLEU score
bleu_score = corpus_bleu(all_references, all_hypotheses)
print(f"BLEU score: {bleu_score*100:.2f}")

# =========================
# 6. In vài câu ví dụ
# =========================
print("\n=== Ví dụ dịch ===")
for i in range(5):
    print(f"Src: {' '.join([test_dataset.src_vocab.itos[idx] for idx in src_batch[i].tolist() if idx not in [PAD_IDX, SOS_IDX, EOS_IDX]])}")
    print(f"Trg: {' '.join([test_dataset.trg_vocab.itos[idx] for idx in trg_batch[i].tolist() if idx not in [PAD_IDX, SOS_IDX, EOS_IDX]])}")
    print(f"Pred: {' '.join(translate_sentence([test_dataset.src_vocab.itos[idx] for idx in src_batch[i].tolist() if idx not in [PAD_IDX, SOS_IDX, EOS_IDX]], test_dataset.src_vocab, test_dataset.trg_vocab))}")
    print("--------------------------")
