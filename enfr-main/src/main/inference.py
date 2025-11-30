import torch

def translate(sentence, model, src_vocab, trg_vocab, src_tokenizer, max_len=50):
    """
    sentence: input sentence
    model: Seq2Seq with Attention
    """

    model.eval()
    device = next(model.parameters()).device

    # 1. Tokenize + numericalize
    tokens = (
        [src_vocab.stoi["<sos>"]] +
        src_vocab.numericalize(src_tokenizer(sentence)) +
        [src_vocab.stoi["<eos>"]]
    )

    src_tensor = torch.tensor(tokens).unsqueeze(0).to(device)
    src_len = torch.tensor([len(tokens)]).to(device)

    #2. Encoder
    with torch.no_grad():
        hidden, cell, encoder_outputs = model.encoder(src_tensor, src_len)
        # encoder_outputs: [1, src_len, hidden_dim]

    # 3. Decoder init
    trg_indexes = [trg_vocab.stoi["<sos>"]]
    input_token = torch.tensor([trg_vocab.stoi["<sos>"]]).to(device)

    # 4. Decode Loop
    for _ in range(max_len):
        with torch.no_grad():
            output, hidden, cell, attn_weights = model.decoder(
                input_token, hidden, cell, encoder_outputs
            )
            # output: [1, trg_vocab_size]

            pred_token = output.argmax(1).item()

        trg_indexes.append(pred_token)

        if pred_token == trg_vocab.stoi["<eos>"]:
            break

        input_token = torch.tensor([pred_token]).to(device)

    # 5. Convert idx sang tokens
    trg_tokens = [
        trg_vocab.itos[idx]
        for idx in trg_indexes[1:]   # bỏ <sos>
        if idx != trg_vocab.stoi["<eos>"]
    ]

    return " ".join(trg_tokens)
