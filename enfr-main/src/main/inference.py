import torch
import torch.nn.functional as F

def translate(sentence, model, src_vocab, trg_vocab, src_tokenizer, max_len=50, method="greedy", beem_sizes=5):

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
        input_token = torch.tensor([pred_token]).to(src_tensor.device)

    # 4. Convert indices -> tokens
    trg_tokens = [
        trg_vocab.itos[idx]
        for idx in trg_indexes[1:]
        if idx != trg_vocab.stoi["<eos>"]
    ]

    return " ".join(trg_tokens)
