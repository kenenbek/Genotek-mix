import os.path as osp

from torch_geometric.data import Dataset

import torch
from torch_geometric.data import Data
import pandas as pd
from tqdm import tqdm, trange

import numpy as np
from collections import defaultdict


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

            x_data_ = defaultdict(lambda: (3 * [0], [], [], []))
            x_data = defaultdict(lambda: (9 * [0]))

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

                if np.isclose(ru1, 1):
                    x_data_[id2][0][0] += 1
                    x_data_[id2][1].append(ibd_sum)
                elif np.isclose(jw1, 1):
                    x_data_[id2][0][1] += 1
                    x_data_[id2][2].append(ibd_sum)
                else:
                    x_data_[id2][0][2] += 1
                    x_data_[id2][3].append(ibd_sum)

                if np.isclose(ru2, 1):
                    x_data_[id1][0][0] += 1
                    x_data_[id1][1].append(ibd_sum)
                elif np.isclose(jw2, 1):
                    x_data_[id1][0][1] += 1
                    x_data_[id1][2].append(ibd_sum)
                else:
                    x_data_[id1][0][2] += 1
                    x_data_[id1][3].append(ibd_sum)

                edge_index.append([id1, id2])
                edge_index.append([id2, id1])
                edge_attr.append([ibd_sum])
                edge_attr.append([ibd_sum])

                if id1 not in y_labels:
                    y_labels[id1] = [ru1, jw1]
                if id2 not in y_labels:
                    y_labels[id2] = [ru2, jw2]

            for i in x_data_.keys():
                x_data[i][0] = x_data_[i][0][0]
                x_data[i][3] = x_data_[i][0][1]
                x_data[i][6] = x_data_[i][0][2]

                x_data[i][1] = np.nan_to_num(np.mean(x_data_[i][1]))
                x_data[i][2] = np.nan_to_num(np.std(x_data_[i][1]))

                x_data[i][4] = np.nan_to_num(np.mean(x_data_[i][2]))
                x_data[i][5] = np.nan_to_num(np.std(x_data_[i][2]))

                x_data[i][7] = np.nan_to_num(np.mean(x_data_[i][3]))
                x_data[i][8] = np.nan_to_num(np.std(x_data_[i][3]))

            y_labels = dict(sorted(y_labels.items()))
            y = torch.Tensor(list(y_labels.values()))

            x_data = dict(sorted(x_data.items()))
            x = torch.Tensor(list(x_data.values()))
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


class MyOwnDataset22Class(MyOwnDataset):

    def process(self):

        idx = 0
        for raw_path in self.raw_paths:
            edge_index = []
            edge_attr = []

            y_labels = {}

            x_data_ = defaultdict(lambda: 22 * [0])
            ibd_list_ = defaultdict(lambda: 22 * [[], [], [], [], [], [], [], [], [], [],
                                                  [], [], [], [], [], [], [], [], [], [],
                                                  [], []])
            x_data = defaultdict(lambda: (66 * [0]))

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

                if np.isclose(ru1, 1):
                    y_labels[id1] = 0
                    x_data_[id2][0] += 1
                    ibd_list_[id2][0].append(ibd_sum)
                elif np.isclose(jw1, 1):
                    y_labels[id1] = 21
                    x_data_[id2][21] += 1
                    ibd_list_[id2][21].append(ibd_sum)
                else:
                    ind = int(np.ceil(jw1 * 100 / 5))
                    y_labels[id1] = ind
                    x_data_[id2][ind] += 1
                    ibd_list_[id2][ind].append(ibd_sum)

                if np.isclose(ru2, 1):
                    y_labels[id2] = 0
                    x_data_[id1][0] += 1
                    ibd_list_[id1][0].append(ibd_sum)
                elif np.isclose(jw2, 1):
                    y_labels[id2] = 21
                    x_data_[id1][21] += 1
                    ibd_list_[id1][21].append(ibd_sum)
                else:
                    ind = int(np.ceil(jw2 * 100 / 5))
                    y_labels[id2] = ind
                    x_data_[id1][ind] += 1
                    ibd_list_[id1][ind].append(ibd_sum)

                edge_index.append([id1, id2])
                edge_index.append([id2, id1])
                edge_attr.append([ibd_sum])
                edge_attr.append([ibd_sum])

            for i in x_data_.keys():
                for j in range(22):
                    x_data[i][3 * j] = x_data_[i][j]

                    x_data[i][3 * j + 1] = np.nan_to_num(np.mean(ibd_list_[i][j]))
                    x_data[i][3 * j + 2] = np.nan_to_num(np.std(ibd_list_[i][j]))

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
