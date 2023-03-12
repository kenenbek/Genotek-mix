import torch
from torch_geometric.nn import GATConv
from torch.nn import Linear, BatchNorm1d
from torch_geometric.nn import GCNConv


class AttnGCN(torch.nn.Module):
    def __init__(self):
        super().__init__()
        torch.manual_seed(1234)
        self.norm = BatchNorm1d(30)
        self.conv1 = GATConv(in_channels=30,
                             out_channels=30,
                             heads=4,
                             concat=False,
                             add_self_loops=True,
                             edge_dim=1)
        self.fc1 = Linear(10, 10)
        self.fc2 = Linear(10, 10)
        self.fc3 = Linear(10, 10)

    def forward(self, h, edge_index, edge_weight):
        h = self.norm(h)
        h = self.conv1(h, edge_index, edge_weight)
        h = h.relu()
        h = self.fc1(h)
        h = h.relu()
        h = self.fc2(h)
        h = h.relu()
        h = self.fc3(h)
        return h


class SimpleNN(torch.nn.Module):
    def __init__(self):
        super().__init__()
        torch.manual_seed(1234)
        self.norm = BatchNorm1d(66)
        self.fc1 = Linear(66, 66)
        self.fc2 = Linear(66, 66)
        self.fc3 = Linear(66, 66)
        self.fc4 = Linear(66, 66)
        self.fc5 = Linear(66, 22)

    def forward(self, h, edge_index, edge_weight):
        h = self.norm(h)
        h = self.fc1(h)
        h = h.relu()
        h = self.fc2(h)
        h = h.relu()
        h = self.fc3(h)
        h = h.relu()
        h = self.fc4(h)
        h = h.relu()
        h = self.fc5(h)
        return h


class GCN(torch.nn.Module):
    def __init__(self):
        super().__init__()
        torch.manual_seed(1234)
        self.conv1 = GCNConv(30, 30,
                             add_self_loops=True,
                             normalize=True)
        self.fc1 = Linear(30, 10)
        self.fc2 = Linear(10, 10)
        self.fc3 = Linear(10, 10)

    def forward(self, h, edge_index, edge_weight):
        h = self.conv1(h, edge_index, edge_weight)
        h = h.relu()
        h = self.fc1(h)
        h = h.relu()
        h = self.fc2(h)
        h = h.relu()
        h = self.fc3(h)
        return h
