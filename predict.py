# %%
import requests
import os
from sklearn.metrics import accuracy_score
from torch_geometric.data import Dataset
import torch
import pandas as pd
from torch.nn import Linear
from torch_geometric.nn import GCNConv
import numpy as np


class HW3Dataset(Dataset):
    url = 'https://technionmail-my.sharepoint.com/:u:/g/personal/ploznik_campus_technion_ac_il/EUHUDSoVnitIrEA6ALsAK1QBpphP5jX3OmGyZAgnbUFo0A?download=1'

    def __init__(self, root, transform=None, pre_transform=None):
        super(HW3Dataset, self).__init__(root, transform, pre_transform)

    @property
    def raw_file_names(self):
        return ['data.pt']

    @property
    def processed_file_names(self):
        return ['data.pt']

    def download(self):
        file_url = self.url.replace(' ', '%20')
        response = requests.get(file_url)

        if response.status_code != 200:
            raise Exception(f"Failed to download the file, status code: {response.status_code}")

        with open(os.path.join(self.raw_dir, self.raw_file_names[0]), 'wb') as f:
            f.write(response.content)

    def process(self):
        raw_path = os.path.join(self.raw_dir, self.raw_file_names[0])
        data = torch.load(raw_path)
        torch.save(data, self.processed_paths[0])

    def len(self):
        return 1

    def get(self, idx):
        return torch.load(self.processed_paths[0])


class GCN(torch.nn.Module):
    def __init__(self, hidden_channels):
        super().__init__()
        torch.manual_seed(1234)
        self.conv1 = GCNConv(dataset.num_features, hidden_channels)
        self.conv2 = GCNConv(hidden_channels, hidden_channels)
        self.conv3 = GCNConv(hidden_channels, hidden_channels // 2)
        self.classifier = Linear(hidden_channels // 2, dataset.num_classes)

    def forward(self, x, edge_index):
        h = self.conv1(x, edge_index)
        h = h.tanh()
        h = self.conv2(h, edge_index)
        h = h.tanh()
        h = self.conv3(h, edge_index)
        h = h.tanh()

        out = self.classifier(h)

        return out


dataset = HW3Dataset(root='data/hw3/')
data = dataset[0]

# loading the trained model
loaded_model = torch.load("GCN_best_model.pkl")
out_loaded = loaded_model(data.x, data.edge_index)
pred_loaded = out_loaded.argmax(dim=1)



# creating the dataframe
idx = [i for i in range(len(data.x))]
real_full = data.y.resize_(len(data.x))
real_labels = real_full.numpy()
predicted_labels = pred_loaded.numpy()
print(accuracy_score(real_labels, predicted_labels))

results = pd.DataFrame(list(zip(idx, list(predicted_labels))), columns=['idx', 'prediction'])

results.to_csv('prediction.csv', index=False)
print('Your new prediction.csv file is available :)')

# %%
