import math
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from inference import translate

# 1. PERPLEXITY
def calculate_perplexity(model, dataloader, pad_idx, device="cuda"):
    # Tính Perplexity = exp(average_loss)
    model.eval()
    criterion = nn.CrossEntropyLoss(ignore_index=pad_idx)
    total_loss = 0
    n_batches = 0

    with torch.no_grad():
        for src, trg, src_len, trg_len in dataloader:

            src = src.to(device)
            trg = trg.to(device)
            src_len = src_len.to(device)

            # forward
            output = model(src, src_len, trg)
            output_dim = output.shape[-1]

            # bỏ <sos>
            output = output[:, 1:].reshape(-1, output_dim)
            trg_flat = trg[:, 1:].reshape(-1)

            loss = criterion(output, trg_flat)
            total_loss += loss.item()
            n_batches += 1

    avg_loss = total_loss / n_batches
    ppl = math.exp(avg_loss)
    return ppl

# 2. BLEU + Example
def evaluate_with_metrics(model, dataloader, src_vocab, trg_vocab,
                          src_tokenizer, pad_idx, device="cuda", method="greedy", beam_sizes=5, use_bpe=False, bpe_src=None, bpe_trg=None):

    model.eval()
    bleu_scores = []
    examples = []

    smooth_fn = SmoothingFunction().method1

    for i, (src, trg, src_len, trg_len) in enumerate(dataloader):

        src = src.to(device)
        trg = trg.to(device)
        src_len = src_len.to(device)

        batch_size = src.size(0)

        # duyệt từng câu trong batch
        for j in range(batch_size):
            if not use_bpe:
                # 1. Chuyển tensor -> câu English
                src_seq = src[j, :src_len[j]].cpu().tolist()
                clean_src = [
                    src_vocab.itos[idx]
                    for idx in src_seq
                    if idx not in [
                        src_vocab.stoi["<sos>"],
                        src_vocab.stoi["<eos>"],
                        src_vocab.stoi["<pad>"],
                    ]
                ]
                src_sentence = " ".join(clean_src)
            else:
                # BPE subword
                src_ids = src[j, :src_len[j]].cpu().tolist()
                # remove special tokens
                removed = [
                    id for id in src_ids
                    if id not in {
                        bpe_src.tokenizer.token_to_id("<sos>"),
                        bpe_src.tokenizer.token_to_id("<eos>"),
                        bpe_src.tokenizer.token_to_id("<pad>")
                    }
                ]
                src_sentence = bpe_src.tokenizer.decode(removed, skip_special_tokens=True)
            # 2. Ground truth French
            if not use_bpe:
                trg_seq = trg[j, :trg_len[j]].cpu().tolist()
                trg_sentence = [
                    trg_vocab.itos[idx]
                    for idx in trg_seq
                    if idx not in [
                        trg_vocab.stoi["<sos>"],
                        trg_vocab.stoi["<eos>"],
                        trg_vocab.stoi["<pad>"],
                    ]
                ]
            else:
                trg_ids = trg[j, :trg_len[j]].cpu().tolist()

                cleaned = [
                    id for id in trg_ids
                    if id not in {
                        bpe_trg.tokenizer.token_to_id("<sos>"),
                        bpe_trg.tokenizer.token_to_id("<eos>"),
                        bpe_trg.tokenizer.token_to_id("<pad>")
                    }
                ]
                trg_sentence_text = bpe_trg.tokenizer.decode(cleaned, skip_special_tokens=True)
                trg_sentence = trg_sentence_text.split()
            # 3. Predicted French
            pred_sentence = translate(
                src_sentence,
                model,
                src_vocab,
                trg_vocab,
                src_tokenizer,
                method=method, # "greedy" hoặc "beam"
                beam_sizes=beam_sizes,
                use_bpe=use_bpe,
                bpe_src=bpe_src,
                bpe_trg=bpe_trg
            ).split()

            # 4. Tính BLEU Score
            bleu = sentence_bleu(
                [trg_sentence],
                pred_sentence,
                smoothing_function=smooth_fn
            )
            bleu_scores.append(bleu)

            # Lưu 5 ví dụ đầu
            if len(examples) < 5:
                examples.append({
                    "src": src_sentence,
                    "pred": " ".join(pred_sentence),
                    "trg": " ".join(trg_sentence),
                    "bleu": bleu
                })

    # BLEU trung bình
    avg_bleu = sum(bleu_scores) / len(bleu_scores)

    # Perplexity
    ppl = calculate_perplexity(model, dataloader, pad_idx, device)

    print(f"\n==============================")
    print(f"Average BLEU score: {avg_bleu:.4f}")
    print(f"Perplexity: {ppl:.4f}")
    print("==============================\n")

    # In 5 ví dụ dịch
    print("----- EXAMPLES -----")
    for ex in examples:
        print(f"EN: {ex['src']}")
        print(f"FR(pred): {ex['pred']}")
        print(f"FR(true): {ex['trg']}")
        print(f"BLEU: {ex['bleu']:.4f}\n")

    return avg_bleu, ppl, bleu_scores, examples
