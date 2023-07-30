import time

import numpy as np
from torch.utils.data.dataset import Dataset
import pandas as pd
from sklearn.preprocessing import MinMaxScaler


class SolanaDataset(Dataset):
    def __init__(self, data, time_span=200,
                 features=['Open', 'High', 'Low', 'Close', 'Volume'],
                 scale=True, index=None, max=None, min=None, only_last=False, **kwargs):
        self.data = data
        self.data = self.data[features]
        if index is not None:
            self.data = self.data.iloc[index]
        self.raw_data = self.data.copy()
        self.time_span = time_span
        self.max = 1e9
        self.min = 0
        if scale:
            if min is None or max is None:
                scaler = MinMaxScaler()
                self.data = scaler.fit_transform(self.data)
                self.max = scaler.data_max_
                self.min = scaler.data_min_
            else:
                self.data = self.data.to_numpy()
                self.max = max
                self.min = min
                self.data = (self.data - self.min) / (self.max - self.min)
        self.only_last = only_last
        # print(type(self.data))
        # time.sleep(5)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, item):
        if item + self.time_span >= len(self):  # over limit
            item = len(self) - self.time_span - 1

        source_span = self.data[item:item + self.time_span, :].astype(np.float32)  # take out the span
        target_span_ = self.data[item + 1:item + self.time_span + 1, :].astype(np.float32)
        # target_span_ = target_span.copy()
        # print(source_span.shape, target_span.shape)
        # print(item, item + self.time_span, item + 1, item + self.time_span + 1, len(self))
        last = None
        target_span = target_span_.copy()
        if self.only_last:
            target_span = target_span_[-1, :]
            last = target_span_[-2, :]
        return source_span, target_span, last
        # min max scaler


if __name__ == "__main__":
    data = pd.read_csv('../data_sol.csv')
    train_split = int(len(data) * 0.8)

    dataset = SolanaDataset(data[:train_split], time_span=3,
                            )
    i = len(dataset) - 3
    a = dataset[i]
    data_max = dataset.max
    data_min = dataset.min
    print(data_max)
    print(data_min)
    print("norm source: ", a[0])
    print("norm target: ", a[1])
    print("unorm source: ", (a[0] * (data_max - data_min) + data_min))
    print("unorm target: ", (a[1] * (data_max - data_min) + data_min))

    print("true: ", dataset.raw_data.iloc[i:i + 4])
