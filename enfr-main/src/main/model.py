import torch
import torch.nn as nn
import torch.nn.functional as F

#thêm attention 
class LuongAttention(nn.Module):
    def __init__(self, hidden_dim):
        super(LuongAttention, self).__init__()
        self.attn = nn.Linear(hidden_dim, hidden_dim)

    def forward(self, decoder_hidden, encoder_outputs):
        # decoder_hidden: [batch, hidden]
        # encoder_outputs: [batch, src_len, hidden]

        # [batch, hidden] -> [batch, 1, hidden]
        decoder_hidden = decoder_hidden.unsqueeze(1)

        # energy = decoder_hidden * W * encoder_outputs^T
        energy = torch.bmm(decoder_hidden, self.attn(encoder_outputs).transpose(1, 2))
        # [batch, 1, src_len]

        attn_weights = F.softmax(energy, dim=-1)
        # [batch, 1, src_len]

        # Weighted sum
        context = torch.bmm(attn_weights, encoder_outputs)
        # [batch, 1, hidden]

        context = context.squeeze(1)
        attn_weights = attn_weights.squeeze(1)

        return context, attn_weights

class Encoder(nn.Module):
    #luồng Encoder : TokenId -> Embedding -> LSTM (Packed) -> Unpack -> Outputs, Hidden, Cell -> đưa sang Decoder
    def __init__(self, input_dim, embed_dim=256, hidden_dim=512, num_layers=2, dropout=0.3):
        super().__init__()

        self.embedding = nn.Embedding(input_dim, embed_dim)
        self.dropout = nn.Dropout(dropout)

        self.lstm = nn.LSTM(
            embed_dim,
            hidden_dim,
            num_layers=num_layers,
            dropout=dropout,
            batch_first=True
        )

    def forward(self, src, src_lengths):
        # src = [batch, src_len]
        # src_lengths là đo độ dài thật của mỗi câu trong batch
        embedded = self.dropout(self.embedding(src))

        # pack cho LSTM
        packed = nn.utils.rnn.pack_padded_sequence(embedded, src_lengths.cpu(), batch_first=True, enforce_sorted=True)

        packed_outputs, (hidden, cell) = self.lstm(packed)

        # Lấy chuỗi hidden states (cần cho attention — optional)
        outputs, _ = nn.utils.rnn.pad_packed_sequence(
            packed_outputs,
            batch_first=True
        )
        # outputs = [batch, src_len, hidden_dim]

        # hidden, cell = [num_layers, batch, hidden_dim]

        return outputs, hidden, cell

class Decoder(nn.Module):
    def __init__(self, output_dim, embed_dim, hidden_dim, num_layers=2, dropout=0.3, use_attention=False):
        super().__init__()

        self.output_dim = output_dim
        self.hidden_dim = hidden_dim
        self.use_attention = use_attention

        self.embedding = nn.Embedding(output_dim, embed_dim)
        self.dropout = nn.Dropout(dropout)

        self.lstm = nn.LSTM(
            embed_dim,
            hidden_dim,
            num_layers=num_layers,
            dropout=dropout,
            batch_first=True
        )

        if use_attention:
            self.attention = LuongAttention(hidden_dim)
            self.fc_out = nn.Linear(hidden_dim * 2, output_dim)
        else:
            self.fc_out = nn.Linear(hidden_dim, output_dim)

    def forward(self, input_token, hidden, cell, encoder_outputs=None):
        # input_token: [batch] (1 step)
        input_token = input_token.unsqueeze(1)  # -> [batch, 1]

        embedded = self.dropout(self.embedding(input_token))  # -> [batch, 1, embed_dim]

        output, (hidden, cell) = self.lstm(embedded, (hidden, cell))
        output = output.squeeze(1)

        if self.use_attention:
            context, attn_weights = self.attention(output, encoder_outputs)
            combined = torch.cat((output, context), dim=1)
            prediction = self.fc_out(combined)
            return prediction, hidden, cell, attn_weights
        else:
            prediction = self.fc_out(output)
            return prediction, hidden, cell, None

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

        # 1. Encoder
        encoder_outputs, hidden, cell = self.encoder(src, src_lengths)

        # 2. Decoder step đầu: dùng <sos>
        input_token = trg[:, 0]  # <sos>

        # 3. Loop qua từng timestep
        for t in range(1, trg_len):
            # output = [batch, vocab_size]
            output, hidden, cell, _ = self.decoder(input_token, hidden, cell, encoder_outputs)

            outputs[:, t] = output

            # Teacher forcing cho từng sample trong batch
            teacher_force = torch.rand(batch_size).to(self.device) < self.teacher_forcing_ratio

            next_input = output.argmax(1)

            # nếu teacher_force[k] == True dùng trg[k, t]
            input_token = torch.where(teacher_force, trg[:, t], next_input)

        return outputs