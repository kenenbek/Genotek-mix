import torch
import torch.nn.functional as F
from torch_geometric.nn import GATConv
from torch.nn import Linear, BatchNorm1d
from torch_geometric.nn import GCNConv


import torch
from torch.nn import Linear
from torch_geometric.nn import GATConv


class AttnGCN(torch.nn.Module):
    def __init__(self):
        super().__init__()
        torch.manual_seed(1234)
        self.norm = BatchNorm1d(20)
        self.conv1 = GATConv(in_channels=20,
                             out_channels=10,
                             heads=1,
                             add_self_loops=True,
                             edge_dim=1)
        # self.conv2 = GATConv(in_channels=10,
        #                      out_channels=10,
        #                      heads=1,
        #                      add_self_loops=True,
        #                      edge_dim=1)
        # self.conv3 = GATConv(in_channels=66,
        #                      out_channels=66,
        #                      heads=1,
        #                      add_self_loops=False,
        #                      edge_dim=1)
        self.fc1 = Linear(10, 10)
        self.fc2 = Linear(10, 10)
        self.fc3 = Linear(10, 10)
        #self.fc4 = Linear(10, 10)
        #self.fc5 = Linear(10, 10)

    def forward(self, h, edge_index, edge_weight):
        h = self.norm(h)
        h = self.conv1(h, edge_index, edge_weight)
        embeddings = h.relu()
        #h = self.conv2(embeddings, edge_index, edge_weight)
        #h = h.relu()
        # h = self.conv3(h, edge_index, edge_weight)
        # h = h.relu()
        h = self.fc1(h)
        h = h.relu()
        h = self.fc2(h)
        h = h.relu()
        h = self.fc3(h)
#         h = h.relu()
#         h = self.fc4(h)
#         h = h.relu()
#         h = self.fc5(h)
        return embeddings, h


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
        self.fc4 = Linear(10, 1)

    def forward(self, h, edge_index, edge_weight):
        h = self.conv1(h, edge_index, edge_weight)
        h = h.relu()
        h = self.fc1(h)
        h = h.relu()
        h = self.fc2(h)
        h = h.relu()
        h = self.fc3(h)
        h = h.relu()
        h = self.fc4(h)
        h = h.sigmoid()
        h = h.squeeze(1)
        return h


class AttnMDN(torch.nn.Module):
    def __init__(self):
        super().__init__()
        torch.manual_seed(1234)
        self.norm = BatchNorm1d(30)
        self.conv1 = GATConv(in_channels=30,
                             out_channels=30,
                             heads=2,
                             add_self_loops=True,
                             edge_dim=1)
        self.fc1 = Linear(60, 10)
        self.fc2 = Linear(10, 10)
        self.fc3 = Linear(10, 10)
        self.fc4 = Linear(10, 1)
        self.fc5 = Linear(10, 1)

    def forward(self, h, edge_index, edge_weight):
        h = self.norm(h)
        h = self.conv1(h, edge_index, edge_weight)
        h = h.relu()
        h = self.fc1(h)
        h = h.relu()
        h = self.fc2(h)
        h = h.relu()
        h = self.fc3(h)

        alpha = F.elu(self.fc4(h)) + 1
        beta = F.elu(self.fc5(h)) + 1

        return alpha, beta


def mdn_gamma_loss(out, y):
    alpha, beta = out
    # dist = torch.distributions.Gamma(concentration=alpha, rate=beta)
    dist = torch.distributions.Normal(alpha, beta)
    loss = -dist.log_prob(y)
    return torch.mean(loss)
