import torch
import torch.nn.functional as F

def translate(sentence, model, src_vocab, trg_vocab, src_tokenizer, max_len=50, method="greedy", beam_sizes=5):

    model.eval()
    device = next(model.parameters()).device

    # 1. Tokenize + numericalize
    tokens = [src_vocab.stoi["<sos>"]] \
             + src_vocab.numericalize(src_tokenizer(sentence)) \
             + [src_vocab.stoi["<eos>"]]

    src_tensor = torch.tensor(tokens).unsqueeze(0).to(device)
    src_len = torch.tensor([len(tokens)]).to(device)

    # 2. Encoder (đúng 3 giá trị)
    with torch.no_grad():
        encoder_outputs, hidden, cell = model.encoder(src_tensor, src_len)

    use_attention = getattr(model.decoder, "use_attention", False)

    # Greedy decoding
    if method == "greedy":
    # 3. Decoder init với <sos>
        trg_indexes = [trg_vocab.stoi["<sos>"]]
        input_token = torch.tensor([trg_vocab.stoi["<sos>"]]).to(src_tensor.device)
        for _ in range(max_len):
            with torch.no_grad():
                if use_attention:
                    # Decoder trả về 4 giá trị
                    output, hidden, cell, attn_weights = model.decoder(
                        input_token, hidden, cell, encoder_outputs
                    )
                else:
                    # Decoder trả về 3 giá trị
                    output, hidden, cell, _ = model.decoder(
                        input_token, hidden, cell, None
                    )
            pred_token = output.argmax(1).item()
            trg_indexes.append(pred_token)
            if pred_token == trg_vocab.stoi["<eos>"]:
                break
            input_token = torch.tensor([pred_token]).to(device)
        final_indexes = trg_indexes

    # Beam Search decoding
    elif method == "beam":
         # mỗi beam = (log_prob, sequence, hidden, cell)
        beams = [(0.0, [trg_vocab.stoi["<sos>"]], hidden, cell)]
        completed = []

        for _ in range(max_len):
            new_beams = []

            for log_prob, seq, h, c in beams:

                last_token = seq[-1]

                # nếu đã kết thúc câu → archive
                if last_token == trg_vocab.stoi["<eos>"]:
                    completed.append((log_prob, seq))
                    continue

                input_token = torch.tensor([last_token]).to(device)

                with torch.no_grad():
                    if use_attention:
                        output, h_new, c_new, attn = model.decoder(
                            input_token, h, c, encoder_outputs
                        )
                    else:
                        output, h_new, c_new, _ = model.decoder(
                            input_token, h, c, None
                        )

                log_probs = F.log_softmax(output, dim=1).squeeze(0)
                topk = torch.topk(log_probs, beam_sizes)

                for prob, idx in zip(topk.values, topk.indices):
                    new_seq = seq + [idx.item()]
                    new_log_prob = log_prob + prob.item()
                    new_beams.append((new_log_prob, new_seq, h_new, c_new))

            # Giữ beam_size chuỗi tốt nhất
            beams = sorted(new_beams, key=lambda x: x[0], reverse=True)[:beam_sizes]

            if len(completed) >= beam_sizes:
                break

        # chọn câu tốt nhất
        if completed:
            final_indexes = max(completed, key=lambda x: x[0])[1]
        else:
            final_indexes = beams[0][1]

    else:
        raise ValueError("Chọn beam hoặc greedy")

    # 4. Convert indices -> tokens
    trg_tokens = [
        trg_vocab.itos[idx]
        for idx in final_indexes[1:]
        if idx != trg_vocab.stoi["<eos>"]
    ]

    return " ".join(trg_tokens)
