import torch
import torch.nn as nn

class Encoder(nn.Module):
    def __init__(self, input_dim, embed_dim=256, hidden_dim=512, num_layers=2, dropout=0.3):
        super().__init__()

        self.embedding = nn.Embedding(input_dim, embed_dim)

        self.lstm = nn.LSTM(
            embed_dim,
            hidden_dim,
            num_layers=num_layers,
            dropout=dropout,
            batch_first=True
        )

    def forward(self, src, src_lengths):
        # src = [batch, src_len]
        embedded = self.embedding(src)

        # pack cho LSTM
        packed = nn.utils.rnn.pack_padded_sequence(embedded, src_lengths.cpu(), batch_first=True)

        outputs, (hidden, cell) = self.lstm(packed)

        # Encoder trả về (h_n, c_n)
        return hidden, cell

class Decoder(nn.Module):
    def __init__(self, output_dim, embed_dim=256, hidden_dim=512, num_layers=2, dropout=0.3):
        super().__init__()

        self.embedding = nn.Embedding(output_dim, embed_dim)

        self.lstm = nn.LSTM(
            embed_dim,
            hidden_dim,
            num_layers=num_layers,
            dropout=dropout,
            batch_first=True
        )

        self.fc_out = nn.Linear(hidden_dim, output_dim)

    def forward(self, input_token, hidden, cell):
        # input_token: [batch] (1 step)
        input_token = input_token.unsqueeze(1)  # -> [batch, 1]

        embedded = self.embedding(input_token)  # -> [batch, 1, embed_dim]

        outputs, (hidden, cell) = self.lstm(embedded, (hidden, cell))

        # outputs = [batch, 1, hidden_dim]
        prediction = self.fc_out(outputs.squeeze(1))  # -> [batch, output_dim]

        return prediction, hidden, cell

class Seq2Seq(nn.Module):
    def __init__(self, encoder, decoder, device, teacher_forcing_ratio=0.5):
        super().__init__()

        self.encoder = encoder
        self.decoder = decoder
        self.device = device
        self.teacher_forcing_ratio = teacher_forcing_ratio

    def forward(self, src, src_lengths, trg):
        batch_size = src.size(0)
        trg_len = trg.size(1)
        trg_vocab_size = self.decoder.fc_out.out_features

        outputs = torch.zeros(batch_size, trg_len, trg_vocab_size).to(self.device)

        # === 1. Encoder ===
        hidden, cell = self.encoder(src, src_lengths)

        # === 2. Decoder step đầu: dùng <sos> ===
        input_token = trg[:, 0]  # <sos>

        # === 3. Loop qua từng timestep ===
        for t in range(1, trg_len):
            output, hidden, cell = self.decoder(input_token, hidden, cell)

            outputs[:, t] = output

            # Teacher forcing
            use_teacher = torch.rand(1).item() < self.teacher_forcing_ratio

            input_token = trg[:, t] if use_teacher else output.argmax(1)

        return outputs

INPUT_DIM = len(vocab_en)     # số từ điển EN
OUTPUT_DIM = len(vocab_fr)    # số từ điển FR

enc = Encoder(INPUT_DIM, embed_dim=256, hidden_dim=512, num_layers=2, dropout=0.3)
dec = Decoder(OUTPUT_DIM, embed_dim=256, hidden_dim=512, num_layers=2, dropout=0.3)

model = Seq2Seq(enc, dec, device="cuda", teacher_forcing_ratio=0.5)
