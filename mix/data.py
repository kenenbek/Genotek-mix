import os.path as osp

from torch_geometric.data import Dataset

import torch
from torch_geometric.data import Data
import pandas as pd
from tqdm import tqdm, trange

import numpy as np
from collections import defaultdict

import os.path as osp

import torch
from torch_geometric.data import Dataset, download_url


class MyOwnDataset(Dataset):

    def __init__(self, root, transform=None, pre_transform=None, pre_filter=None):
        super().__init__(root, transform, pre_transform, pre_filter)

    @property
    def raw_file_names(self):
        return ["ru-jw.csv"]

    @property
    def processed_file_names(self):
        return ['data_0.pt']

    def download(self):
        # Download to `self.raw_dir`.
        pass

    def process(self):

        idx = 0
        for raw_path in self.raw_paths:
            edge_index = []
            edge_attr = []

            y_labels = {}

            x_data_ = defaultdict(lambda: 10 * [0])
            ibd_list_ = defaultdict(lambda: [[], [], [], [], [],
                                             [], [], [], [], []])
            x_data = defaultdict(lambda: (20 * [0]))

            dataset_csv = pd.read_csv(raw_path)

            for index, row in tqdm(dataset_csv.iterrows()):
                node1 = row["node_id1"]
                node2 = row["node_id2"]
                ru1 = row["ru1"]
                jw1 = row["jw1"]
                ru2 = row["ru2"]
                jw2 = row["jw2"]

                ibd_sum = row["ibd"]

                id1 = int(node1[5:])
                id2 = int(node2[5:])

                if jw1 == 1:
                    jw1 = 0.99
                if jw2 == 1:
                    jw2 = 0.99

                ind = int(np.floor(jw1 * 100 / 10))
                y_labels[id1] = ind
                x_data_[id2][ind] += 1
                ibd_list_[id2][ind].append(ibd_sum)

                ind = int(np.floor(jw2 * 100 / 10))
                y_labels[id2] = ind
                x_data_[id1][ind] += 1
                ibd_list_[id1][ind].append(ibd_sum)

                edge_index.append([id1, id2])
                edge_index.append([id2, id1])
                edge_attr.append([ibd_sum])
                edge_attr.append([ibd_sum])

            for i in x_data_.keys():
                for j in range(10):
                    x_data[i][2 * j] = x_data_[i][j]

                    x_data[i][2 * j + 1] = np.nan_to_num(np.sum(ibd_list_[i][j]))

            y_labels = dict(sorted(y_labels.items()))
            y = torch.Tensor(list(y_labels.values())).type(torch.long).contiguous()

            x_data = dict(sorted(x_data.items()))
            x = torch.Tensor(list(x_data.values())).contiguous()

            edge_attr = torch.Tensor(edge_attr).type(torch.float).contiguous()
            edge_index = torch.Tensor(edge_index).type(torch.long).t().contiguous()

            data = Data(x=x, edge_index=edge_index, edge_attr=edge_attr, y=y)
            torch.save(data, osp.join(self.processed_dir, f'data_{idx}.pt'))
            idx += 1

    def len(self):
        return len(self.processed_file_names)

    def get(self, idx):
        data = torch.load(osp.join(self.processed_dir, f'data_{idx}.pt'))
        return data