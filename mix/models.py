import torch
from torch_geometric.nn import GATConv
from torch.nn import Linear, BatchNorm1d
from torch_geometric.nn import GCNConv


class AttnGCN(torch.nn.Module):
    def __init__(self):
        super().__init__()
        torch.manual_seed(1234)
        self.norm = BatchNorm1d(66)
        self.conv1 = GATConv(in_channels=66,
                             out_channels=66,
                             heads=1,
                             add_self_loops=False,
                             edge_dim=1)
        self.conv2 = GATConv(in_channels=66,
                             out_channels=66,
                             heads=1,
                             add_self_loops=False,
                             edge_dim=1)
        self.conv3 = GATConv(in_channels=66,
                             out_channels=66,
                             heads=1,
                             add_self_loops=False,
                             edge_dim=1)
        self.fc1 = Linear(66, 66)
        self.fc2 = Linear(66, 66)
        self.fc3 = Linear(66, 66)
        self.fc4 = Linear(66, 66)
        self.fc5 = Linear(66, 22)

    def forward(self, h, edge_index, edge_weight):
        h = self.norm(h)
        h = self.conv1(h, edge_index, edge_weight)
        h = h.relu()
        h = self.conv2(h, edge_index, edge_weight)
        h = h.relu()
        h = self.conv3(h, edge_index, edge_weight)
        h = h.relu()
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
        self.norm = BatchNorm1d(66)
        self.conv1 = GCNConv(66, 66,
                             add_self_loops=False,
                             normalize=True)
        self.conv2 = GCNConv(66, 66,
                             add_self_loops=False,
                             normalize=True)
        self.conv3 = GCNConv(66, 66,
                             add_self_loops=False,
                             normalize=True)
        self.fc1 = Linear(66, 66)
        self.fc2 = Linear(66, 66)
        self.fc3 = Linear(66, 66)
        self.fc4 = Linear(66, 22)

    def forward(self, h, edge_index, edge_weight):
        h = self.norm(h)
        h = self.conv1(h, edge_index, edge_weight)
        h = h.relu()
        h = self.conv2(h, edge_index, edge_weight)
        h = h.relu()
        h = self.conv3(h, edge_index, edge_weight)
        h = h.relu()
        h = self.fc1(h)
        h = h.relu()
        h = self.fc2(h)
        h = h.relu()
        h = self.fc3(h)
        h = h.relu()
        h = self.fc4(h)
        return h
