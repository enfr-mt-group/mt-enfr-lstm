import torch

def translate(sentence, model, src_vocab, trg_vocab, src_tokenizer, max_len=50):
    """
    sentence: str tiếng Anh
    model: Seq2Seq LSTM đã train
    src_vocab: vocab_en
    trg_vocab: vocab_fr
    src_tokenizer: function tokenize_en
    max_len: độ dài tối đa
    """

    model.eval()

    # 1 Tokenize và numericalize
    tokens = [src_vocab.stoi["<sos>"]] + src_vocab.numericalize(src_tokenizer(sentence)) + [src_vocab.stoi["<eos>"]]
    src_tensor = torch.tensor(tokens).unsqueeze(0).to(next(model.parameters()).device)  # [1, seq_len]
    src_len = torch.tensor([len(tokens)]).to(src_tensor.device)

    # 2 Encoder
    with torch.no_grad():
        hidden, cell = model.encoder(src_tensor, src_len)

    # 3 Decoder init với <sos>
    trg_indexes = [trg_vocab.stoi["<sos>"]]
    input_token = torch.tensor([trg_vocab.stoi["<sos>"]]).to(src_tensor.device)

    for _ in range(max_len):
        with torch.no_grad():
            output, hidden, cell = model.decoder(input_token, hidden, cell)
            # output: [1, vocab_size]
            pred_token = output.argmax(1).item()

        trg_indexes.append(pred_token)

        if pred_token == trg_vocab.stoi["<eos>"]:
            break

        input_token = torch.tensor([pred_token]).to(src_tensor.device)

    # 4 Convert indices -> words
    trg_tokens = [trg_vocab.itos[idx] for idx in trg_indexes[1:] if idx != trg_vocab.stoi["<eos>"]]

    translated_sentence = " ".join(trg_tokens)
    return translated_sentence


sentence_en = "I love natural language processing."
translated_fr = translate(sentence_en, model, dataset.src_vocab, dataset.trg_vocab, tokenize_en)
print(translated_fr)

