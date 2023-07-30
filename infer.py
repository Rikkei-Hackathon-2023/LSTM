from models.lstm import LSTM
import torch
import pandas as pd

model_config = {
    "num_classes": 5,
    "input_size": 5,
    "hidden_size": 64,
    "num_layers": 2,
    "bi": False
}
data_config = {
    'csv_file': 'data_sol.csv',
    'time_span': 50,
    'features': ['Open', 'High', 'Low', 'Close', 'Volume'],
    'scale': True,
}
train_config = {
    'batch_size': 16,
    'num_workers': 8,
    'device': 'cuda:0',
    'lr': 1e-4,
    'wd': 1e-2,
    'epoch': 100,
}

model = LSTM(**model_config)
data = pd.read_csv("data_sol.csv")

train_split = int(len(data) * 0.8)
val_split = int(len(data) * 0.9)
checkpoint = torch.load("best.pt")
