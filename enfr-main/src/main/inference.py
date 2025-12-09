import torch
import torch.nn.functional as F

def translate(sentence, model, src_vocab, trg_vocab, src_tokenizer, max_len=50, method="greedy", beam_sizes=5, use_bpe=False, bpe_src=None, bpe_trg=None):

    model.eval()
    device = next(model.parameters()).device

    # 1. Tokenize + numericalize
    if not use_bpe:
        tokens = [src_vocab.stoi["<sos>"]] \
                + src_vocab.numericalize(src_tokenizer(sentence)) \
                + [src_vocab.stoi["<eos>"]]

        src_tensor = torch.tensor(tokens).unsqueeze(0).to(device)
        src_len = torch.tensor([len(tokens)]).to(device)
    else:
        # ------- BPE MODE -------
        # spaCy tokenize sang BPE encode
        tok = src_tokenizer(sentence)
        ids = bpe_src.encode(tok)

        sos = bpe_src.tokenizer.token_to_id("<sos>")
        eos = bpe_src.tokenizer.token_to_id("<eos>")

        ids = [sos] + ids + [eos]

        src_tensor = torch.tensor(ids).unsqueeze(0).to(device)
        src_len = torch.tensor([len(ids)]).to(device)

    # 2. Encoder (đúng 3 giá trị)
    with torch.no_grad():
        encoder_outputs, hidden, cell = model.encoder(src_tensor, src_len)

    use_attention = getattr(model.decoder, "use_attention", False)
    # 3. Decoder init với <sos>
    # Greedy decoding
    if method == "greedy":
   
        if not use_bpe:
            input_token = torch.tensor([trg_vocab.stoi["<sos>"]]).to(device)
            trg_sos = trg_vocab.stoi["<sos>"]
            trg_eos = trg_vocab.stoi["<eos>"]
        else:
            trg_sos = bpe_trg.tokenizer.token_to_id("<sos>")
            trg_eos = bpe_trg.tokenizer.token_to_id("<eos>")
            input_token = torch.tensor([trg_sos]).to(device)

        final_indexes = [trg_sos]

        for _ in range(max_len):
            with torch.no_grad():
                if use_attention:
                    # Decoder trả về 4 giá trị
                    output, hidden, cell, attn_weights = model.decoder(
                        input_token, hidden, cell, encoder_outputs
                    )
                else:
                    # Decoder trả về 3 giá trị
                    output, hidden, cell, _ = model.decoder(input_token, hidden, cell, None)
            
            pred_token = output.argmax(1).item()
            final_indexes.append(pred_token)
            if pred_token == trg_eos:
                break
            input_token = torch.tensor([pred_token]).to(device)

    # Beam Search decoding
    elif method == "beam":
         # mỗi beam = (log_prob, sequence, hidden, cell)
        if not use_bpe:
            sos = trg_vocab.stoi["<sos>"]
            eos = trg_vocab.stoi["<eos>"]
        else:
            eos = bpe_trg.tokenizer.token_to_id("<eos>")
            sos = bpe_trg.tokenizer.token_to_id("<sos>")

        beams = [(0.0, [sos], hidden, cell)]
        completed = []

        for _ in range(max_len):
            new_beams = []

            for log_prob, seq, h, c in beams:

                last_token = seq[-1]

                if last_token == eos:
                    completed.append((log_prob, seq))
                    continue

                input_token = torch.tensor([last_token]).to(device)

                with torch.no_grad():
                    if use_attention:
                        output, h_new, c_new, attn = model.decoder(input_token, h, c, encoder_outputs)
                    else:
                        output, h_new, c_new, _ = model.decoder(input_token, h, c, None)

                log_probs = F.log_softmax(output, dim=1).squeeze(0)
                topk = torch.topk(log_probs, beam_sizes)

                for p, idx in zip(topk.values, topk.indices):
                    new_seq = seq + [idx.item()]
                    new_beams.append((log_prob + p.item(), new_seq, h_new, c_new))

            beams = sorted(new_beams, key=lambda x: x[0], reverse=True)[:beam_sizes]

        final_indexes = beams[0][1]

    else:
        raise ValueError("method phải là greedy hoặc beam")

    # 4. Convert indices -> tokens
    if not use_bpe:
        trg_tokens = [
            trg_vocab.itos[idx]
            for idx in final_indexes[1:]
            if idx != trg_vocab.stoi["<eos>"]
        ]

        return " ".join(trg_tokens)
    else:
        clean_ids = [
            idx for idx in final_indexes
            if idx not in {
                bpe_trg.tokenizer.token_to_id("<sos>"),
                bpe_trg.tokenizer.token_to_id("<eos>")
            }
        ]

        # Decode subword → text
        text = bpe_trg.tokenizer.decode(clean_ids, skip_special_tokens=True)
        return text.strip()
