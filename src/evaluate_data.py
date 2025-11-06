import torch
from torch.utils.data import DataLoader
from src.data import load_dataset, load_vocab, MyCollate
from src.train import Encoder, Decoder, Seq2Seq  # import model classes
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

"""
Giải thích
- Batch size = 1 để dịch từng câu.
- tensor_to_sentence loại bỏ <sos> và <pad>, dừng khi gặp <eos>.
- translate_sentence là greedy decoding (không dùng beam search) để sinh câu dịch.
- BLEU score dùng nltk với smoothing function để tránh lỗi khi câu quá ngắn.
- Kết quả in ra là BLEU trung bình trên toàn bộ tập test.
"""

# =============================
# 1. Load dataset + vocab + model
# =============================
test_dataset = load_dataset("data/test_dataset.pt")  # bạn có thể chuẩn bị dataset test
src_vocab = load_vocab("data/vocab_en.pkl")
trg_vocab = load_vocab("data/vocab_fr.pkl")

PAD_IDX = src_vocab.stoi["<pad>"]

test_loader = DataLoader(
    test_dataset, batch_size=1, shuffle=False,
    collate_fn=MyCollate(test_dataset.pad_idx)
)

# load model
encoder = Encoder(len(src_vocab.stoi), 256, 512, 1).to(DEVICE)
decoder = Decoder(len(trg_vocab.stoi), 256, 512, 1).to(DEVICE)
model = Seq2Seq(encoder, decoder, DEVICE).to(DEVICE)

model.load_state_dict(torch.load("checkpoints/seq2seq_epoch10.pth", map_location=DEVICE))
model.eval()

# =============================
# 2. Helper: decode tensor -> sentence
# =============================
def tensor_to_sentence(tensor, vocab):
    tokens = []
    for idx in tensor:
        word = vocab.itos.get(idx.item(), "<unk>")
        if word in ["<sos>", "<pad>"]:
            continue
        if word == "<eos>":
            break
        tokens.append(word)
    return tokens

# =============================
# 3. Inference function
# =============================
def translate_sentence(model, src_tensor, max_len=50):
    model.eval()
    src_tensor = src_tensor.to(DEVICE).unsqueeze(0)  # add batch dim
    with torch.no_grad():
        hidden, cell = model.encoder(src_tensor)
        input = torch.tensor([trg_vocab.stoi["<sos>"]], device=DEVICE)
        outputs = []
        for _ in range(max_len):
            output, hidden, cell = model.decoder(input, hidden, cell)
            top1 = output.argmax(1)
            word_idx = top1.item()
            if word_idx == trg_vocab.stoi["<eos>"]:
                break
            outputs.append(word_idx)
            input = top1
    return outputs

# =============================
# 4. Compute BLEU score
# =============================
smooth_fn = SmoothingFunction().method1
bleu_scores = []

for src, trg in test_loader:
    src = src[0].to(DEVICE)
    trg = trg[0]
    pred_idxs = translate_sentence(model, src)
    pred_tokens = [trg_vocab.itos[idx] for idx in pred_idxs]
    trg_tokens = tensor_to_sentence(trg, trg_vocab)
    bleu = sentence_bleu([trg_tokens], pred_tokens, smoothing_function=smooth_fn)
    bleu_scores.append(bleu)

avg_bleu = sum(bleu_scores)/len(bleu_scores)
print(f"Average BLEU score: {avg_bleu*100:.2f}")
