import math
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
import matplotlib.pyplot as plt
import torch.nn as nn
import torch.optim as optim
from inference import translate
import torch

def calculate_perplexity(model, dataloader, pad_idx, device="cuda"):
    # Tính Perplexity cho Seq2Seq LSTM trên 1 dataloader

    model.eval()
    criterion = nn.CrossEntropyLoss(ignore_index=pad_idx)
    total_loss = 0
    n_batches = 0

    with torch.no_grad():
        for src, trg, src_len, trg_len in dataloader:
            src = src.to(device)
            trg = trg.to(device)
            src_len = src_len.to(device)

            output = model(src, src_len, trg)
            output_dim = output.shape[-1]

            output = output[:, 1:].reshape(-1, output_dim)
            trg_flat = trg[:, 1:].reshape(-1)

            loss = criterion(output, trg_flat)
            total_loss += loss.item()
            n_batches += 1

    avg_loss = total_loss / n_batches
    ppl = math.exp(avg_loss)
    return ppl

def evaluate_with_metrics(model, dataloader, src_vocab, trg_vocab, src_tokenizer, pad_idx, device="cuda"):

    # Trả về: trung bình BLEU, danh sách BLEU từng câu, Perplexity
    model.eval()
    bleu_scores = []
    examples = []

    smooth_fn = SmoothingFunction().method1

    for i, (src, trg, src_len, trg_len) in enumerate(dataloader):
        src = src.to(device)
        trg = trg.to(device)
        src_len = src_len.to(device)

        for j in range(src.size(0)):
            src_seq = src[j, :src_len[j]].cpu()
            trg_seq = trg[j, :trg_len[j]].cpu()

            src_sentence = " ".join([src_vocab.itos[idx.item()] for idx in src_seq if idx.item() not in [src_vocab.stoi["<sos>"], src_vocab.stoi["<eos>"], src_vocab.stoi["<pad>"]]])
            trg_sentence = [trg_vocab.itos[idx.item()] for idx in trg_seq if idx.item() not in [trg_vocab.stoi["<sos>"], trg_vocab.stoi["<eos>"], trg_vocab.stoi["<pad>"]]]

            pred_sentence = translate(src_sentence, model, src_vocab, trg_vocab, src_tokenizer).split()
            bleu = sentence_bleu([trg_sentence], pred_sentence, smoothing_function=smooth_fn)
            bleu_scores.append(bleu)

            if len(examples) < 5:
                examples.append({
                    "src": src_sentence,
                    "pred": " ".join(pred_sentence),
                    "trg": " ".join(trg_sentence),
                    "bleu": bleu
                })

    avg_bleu = sum(bleu_scores) / len(bleu_scores)
    ppl = calculate_perplexity(model, dataloader, pad_idx, device)

    print(f"Average BLEU score: {avg_bleu:.4f}")
    print(f"Perplexity: {ppl:.4f}\n")

    # In 5 ví dụ
    print("Examples:")
    for ex in examples:
        print(f"EN: {ex['src']}")
        print(f"FR(pred): {ex['pred']}")
        print(f"FR(true): {ex['trg']}")
        print(f"BLEU: {ex['bleu']:.4f}\n")

    # Biểu đồ phân phối BLEU
    plt.figure(figsize=(10,4))
    plt.subplot(1,2,1)
    plt.hist(bleu_scores, bins=20, color="skyblue", edgecolor="black")
    plt.title("BLEU score distribution")
    plt.xlabel("BLEU")
    plt.ylabel("Number of sentences")

    # Biểu đồ Perplexity (chỉ 1 giá trị PPL -> vẽ bar)
    plt.subplot(1,2,2)
    plt.bar(["Perplexity"], [ppl], color="salmon")
    plt.title("Perplexity")
    plt.ylabel("PPL")
    plt.show()

    return avg_bleu, ppl, bleu_scores, examples