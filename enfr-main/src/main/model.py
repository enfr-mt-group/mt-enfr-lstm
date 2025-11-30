import torch
import torch.nn as nn
import torch.nn.functional as F

class Attention(nn.Module):
    def __init__(self, hidden_dim):
        super().__init__()
        self.attn = nn.Linear(hidden_dim, hidden_dim)

    def forward(self, decoder_hidden, encoder_outputs):
        # decoder_hidden: [batch, hidden_dim]
        # encoder_outputs: [batch, src_len, hidden_dim]

        # transform decoder hidden state
        dec_hidden = self.attn(decoder_hidden).unsqueeze(2)  # [batch, hidden_dim, 1]

        # compute energy scores
        # energy = encoder_outputs • dec_hidden
        energy = torch.bmm(encoder_outputs, dec_hidden).squeeze(2)  # [batch, src_len]

        # attention weights
        attn_weights = F.softmax(energy, dim=1)  # [batch, src_len]

        # compute context vector
        context = torch.bmm(attn_weights.unsqueeze(1), encoder_outputs)  # [batch, 1, hidden_dim]

        return context.squeeze(1), attn_weights  # [batch, hidden_dim]

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
        packed_outputs, (hidden, cell) = self.lstm(packed)
        outputs, _ = nn.utils.rnn.pad_packed_sequence(packed_outputs, batch_first=True)

        # Encoder trả về (h_n, c_n)
        return outputs, hidden, cell

class Decoder(nn.Module):
    def __init__(self, output_dim, embed_dim=256, hidden_dim=512, num_layers=2, dropout=0.3):
        super().__init__()

        self.output_dim = output_dim
        self.embedding = nn.Embedding(output_dim, embed_dim)

        #Thêm attention
        self.attention = Attention(hidden_dim)

        self.lstm = nn.LSTM(
            embed_dim + hidden_dim,  # input gồm embedding + context vector
            hidden_dim,
            num_layers=num_layers,
            dropout=dropout,
            batch_first=True
        )

        self.fc_out = nn.Linear(hidden_dim + hidden_dim + embed_dim, output_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, input_token, hidden, cell, encoder_outputs):
        # input_token: [batch] (1 step)
        input_token = input_token.unsqueeze(1)  # -> [batch, 1]

        embedded = self.dropout(self.embedding(input_token))  # -> [batch, 1, embed_dim]
        
        dec_hidden = hidden[-1]
        
        # attention
        context, attn_weights = self.attention(dec_hidden, encoder_outputs)  # [batch, hidden_dim]

        # chuẩn hóa shape để concat
        context = context.unsqueeze(1)  # [batch, 1, hidden_dim]

        # input vào LSTM
        lstm_input = torch.cat((embedded, context), dim=2)

        output, (hidden, cell) = self.lstm(lstm_input, (hidden, cell))

        output = output.squeeze(1)
        context = context.squeeze(1)
        embedded = embedded.squeeze(1)

        prediction = self.fc_out(torch.cat((output, context, embedded), dim=1))

        return prediction, hidden, cell, attn_weights

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
        trg_vocab_size = self.decoder.output_dim

        outputs = torch.zeros(batch_size, trg_len, trg_vocab_size).to(self.device)

        # === 1. Encoder ===
        encoder_outputs, hidden, cell = self.encoder(src, src_lengths)

        # === 2. Decoder step đầu: dùng <sos> ===
        input_token = trg[:, 0]  # <sos>

        # === 3. Loop qua từng timestep ===
        for t in range(1, trg_len):
            output, hidden, cell, attn = self.decoder(input_token, hidden, cell, encoder_outputs)

            outputs[:, t] = output

            # Teacher forcing
            teacher_force = torch.rand(1).item() < self.teacher_forcing_ratio
            top1 = output.argmax(1)

            input_token = trg[:, t] if teacher_force else top1

        return outputs
