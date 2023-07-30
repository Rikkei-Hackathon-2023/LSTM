import torch.nn as nn
import torch


# 9 14 26 34 50 200
class LSTM(nn.Module):
    def __init__(self, num_classes, input_size, hidden_size, num_layers, bi=True, **kwargs):
        super().__init__()
        self.num_classes = num_classes
        self.num_Layers = num_layers
        self.input_size = input_size

        self.lstm = nn.LSTM(input_size=input_size, hidden_size=hidden_size,
                            num_layers=num_layers, batch_first=True, dropout=0.2, bidirectional=bi)
        self.bi = bi
        self.fc_1 = nn.Linear(hidden_size * (bi + 1), 128)
        self.fc_2 = nn.Linear(128, num_classes)
        self.act = nn.ReLU()
        self.drop = nn.Dropout(p=0.1)

    def forward(self, x):
        lstm_out, lstm_hidden = self.lstm(x)
        embedding_dim = lstm_out.shape[2] // 2
        if self.bi:
            forward_feature = lstm_out[:, -1, :embedding_dim]
            backward_feature = lstm_out[:, 0, embedding_dim:]
            feature = torch.cat([forward_feature, backward_feature], dim=-1)
        else:
            feature = lstm_out[:, -1, :]
        output1 = self.fc_1(self.drop(feature))
        output2 = self.fc_2(self.drop(self.act(output1)))
        return self.act(output2)


if __name__ == "__main__":
    lstm = LSTM(6, 6, 128, num_layers=2, bi=False)
    input_test = torch.randn(1, 100, 6)
    output = lstm(input_test)
    print(output.shape)
