# translate_example.py
import torch
from model import Seq2Seq, Encoder, Decoder, translate, src_vocab, trg_vocab, DEVICE

# 1️⃣ Tạo model giống lúc train
encoder = Encoder(len(src_vocab.stoi), 256, 512, 2).to(DEVICE)
decoder = Decoder(len(trg_vocab.stoi), 256, 512, 2).to(DEVICE)
model = Seq2Seq(encoder, decoder, DEVICE, trg_vocab_size=len(trg_vocab.stoi)).to(DEVICE)

# 2️⃣ Load weights
model.load_state_dict(torch.load("/kaggle/working/checkpoints/seq2seq_best.pth", map_location=DEVICE))
model.eval()

# 3️⃣ Hàm translate đơn giản
def translate_sentence(sentence):
    return translate(sentence, model, src_vocab, trg_vocab, DEVICE)

# 4️⃣ Test
if __name__ == "__main__":
    while True:
        text = input("Enter English sentence (or 'quit'): ")
        if text.lower() == "quit":
            break
        fr = translate_sentence(text)
        print("French:", fr)
