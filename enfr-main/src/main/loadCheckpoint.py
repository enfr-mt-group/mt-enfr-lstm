import torch
from tabulate import tabulate

def load_checkpoint(path):
    checkpoint = torch.load(path, map_location="cpu")

    # checkpoint mới
    if isinstance(checkpoint, dict) and "model_state" in checkpoint:
        state_dict = checkpoint["model_state"]
        print("Checkpoint type: FULL (model_state + vocab + config)")
    else:
        state_dict = checkpoint
        print("Checkpoint type: STATE_DICT only")

    print("Số lượng trọng số:", len(state_dict))

    # 1. BẢNG TỔNG HỢP TẤT CẢ WEIGHTS 
    table_all = []
    for name, tensor in state_dict.items():
        table_all.append([name, list(tensor.shape), tensor.numel()])

    print("\nBẢNG 1: DANH SÁCH TOÀN BỘ THAM SỐ")
    print(tabulate(
        table_all,
        headers=["Layer Name", "Tensor Shape", "Số Parameter"],
        tablefmt="grid"
    ))

    # 2. BẢNG EMBEDDING 
    embedding_table = []
    for name, tensor in state_dict.items():
        if "embedding" in name:
            embedding_table.append([name, list(tensor.shape), tensor.numel()])

    print("\nBẢNG 2: EMBEDDING")
    print(tabulate(
        embedding_table,
        headers=["Layer", "Shape", "Params"],
        tablefmt="grid"
    ))

    # 3. BẢNG LSTM ENCODER 
    encoder_lstm = []
    for name, tensor in state_dict.items():
        if name.startswith("encoder.lstm"):
            encoder_lstm.append([name, list(tensor.shape), tensor.numel()])

    print("\nBẢNG 3: LSTM ENCODER")
    print(tabulate(
        encoder_lstm,
        headers=["Layer", "Shape", "Params"],
        tablefmt="grid"
    ))

    # 4. BẢNG LSTM DECODER 
    decoder_lstm = []
    for name, tensor in state_dict.items():
        if name.startswith("decoder.lstm"):
            decoder_lstm.append([name, list(tensor.shape), tensor.numel()])

    print("\nBẢNG 4: LSTM DECODER")
    print(tabulate(
        decoder_lstm,
        headers=["Layer", "Shape", "Params"],
        tablefmt="grid"
    ))

    # 5. BẢNG ATTENTION 
    attention_table = []
    for name, tensor in state_dict.items():
        if "attention" in name:
            attention_table.append([name, list(tensor.shape), tensor.numel()])

    print("\nBẢNG 5: ATTENTION")
    print(tabulate(
        attention_table,
        headers=["Layer", "Shape", "Params"],
        tablefmt="grid"
    ))

    # 6. BẢNG FC OUTPUT 
    fc_table = []
    for name, tensor in state_dict.items():
        if "fc_out" in name:
            fc_table.append([name, list(tensor.shape), tensor.numel()])

    print("\nBẢNG 6: FULLY CONNECTED OUTPUT")
    print(tabulate(
        fc_table,
        headers=["Layer", "Shape", "Params"],
        tablefmt="grid"
    ))

    # 7. TỔNG SỐ PARAMETER 
    total_params = sum(t.numel() for t in state_dict.values())
    print("\nTỔNG SỐ PARAMETER CỦA MÔ HÌNH:", f"{total_params:,}")
