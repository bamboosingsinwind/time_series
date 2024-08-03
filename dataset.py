import pandas as pd
import torch
from torch.utils.data import Dataset

class TimeSeriesDataset(Dataset):
    def __init__(self, data, window_size=3*24, step=1):
        self.data = data
        self.window_size = window_size
        self.step = step
        self.label_col = 2 #"click_count"
        self.features_num = data.shape[1]
        # self.features.remove('click_count')

    def __len__(self):
        return (len(self.data) - self.window_size) // self.step

    def __getitem__(self, idx):
        start = idx * self.step
        end = start + self.window_size
        x = self.data[start:end,:]
        y = self.data[end,self.label_col]
        return torch.tensor(x, dtype=torch.float32), torch.tensor(y, dtype=torch.float32)

# Example usage:
# data = pd.read_excel('../docs/feat.xlsx')
# dataset = TimeSeriesDataset(data)
# print(len(dataset))
# train_loader = torch.utils.data.DataLoader(dataset, batch_size=64, shuffle=True)