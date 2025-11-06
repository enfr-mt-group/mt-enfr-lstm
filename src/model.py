import torch
import torch.nn as nn

# ============================================================
# 1. ENCODER
# ============================================================

class Encoder(nn.Module):
    def __init__(self, input_dim, embed_dim, hidden_dim, num_layers, dropout):
        super().__init__()

        self.embedding = nn.Embedding(input_dim, embed_dim)
        self.lstm = nn.LSTM(embed_dim, hidden_dim, num_layers, dropout=dropout, batch_first=True)
        self.dropout = nn.Dropout(dropout)

    def forward(self, src):
        """
        src: [batch_size, src_len]
        """
        embedded = self.dropout(self.embedding(src))  # [batch_size, src_len, embed_dim]
        outputs, (hidden, cell) = self.lstm(embedded)  # hidden/cell: [num_layers, batch, hidden_dim]

        # outputs (tất cả bước thời gian) có thể bỏ qua vì ta chỉ dùng context vector (hidden, cell)
        return hidden, cell


# ============================================================
# 2. DECODER
# ============================================================

class Decoder(nn.Module):
    def __init__(self, output_dim, embed_dim, hidden_dim, num_layers, dropout):
        super().__init__()

        self.output_dim = output_dim
        self.embedding = nn.Embedding(output_dim, embed_dim)
        self.lstm = nn.LSTM(embed_dim, hidden_dim, num_layers, dropout=dropout, batch_first=True)
        self.fc_out = nn.Linear(hidden_dim, output_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, input, hidden, cell):
        """
        input: [batch_size]
        hidden, cell: từ Encoder
        """
        input = input.unsqueeze(1)  # [batch, 1]
        embedded = self.dropout(self.embedding(input))  # [batch, 1, embed_dim]
        output, (hidden, cell) = self.lstm(embedded, (hidden, cell))
        prediction = self.fc_out(output.squeeze(1))  # [batch, output_dim]
        return prediction, hidden, cell


# ============================================================
# 3. SEQ2SEQ (ENCODER + DECODER)
# ============================================================

class Seq2Seq(nn.Module):
    def __init__(self, encoder, decoder, device):
        super().__init__()

        self.encoder = encoder
        self.decoder = decoder
        self.device = device

        # Kiểm tra xem hidden_dim của Encoder và Decoder có khớp không
        assert encoder.lstm.hidden_size == decoder.lstm.hidden_size, \
            "Hidden dimensions of encoder and decoder must be equal!"

        assert encoder.lstm.num_layers == decoder.lstm.num_layers, \
            "Encoder and decoder must have equal number of layers!"

    def forward(self, src, trg, teacher_forcing_ratio=0.5):
        """
        src: [batch, src_len]
        trg: [batch, trg_len]
        """
        batch_size = src.shape[0]
        trg_len = trg.shape[1]
        trg_vocab_size = self.decoder.output_dim

        # tensor lưu tất cả kết quả dự đoán
        outputs = torch.zeros(batch_size, trg_len, trg_vocab_size).to(self.device)

        # 1. Truyền src qua encoder
        hidden, cell = self.encoder(src)

        # 2. Lấy token đầu tiên (<sos>) của decoder
        input = trg[:, 0]

        # 3. Chạy qua từng bước thời gian
        for t in range(1, trg_len):
            output, hidden, cell = self.decoder(input, hidden, cell)
            outputs[:, t, :] = output

            # chọn token có xác suất cao nhất
            top1 = output.argmax(1)

            # quyết định có dùng teacher forcing không
            teacher_force = torch.rand(1).item() < teacher_forcing_ratio
            input = trg[:, t] if teacher_force else top1

        return outputs
