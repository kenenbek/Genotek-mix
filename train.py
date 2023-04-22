import comet_ml
from comet_ml import Experiment

import torch

from mix import AttnGCN, AttnMDN, GCN
from mix import MyOwnDataset
from mix import mdn_gamma_loss

from tqdm import trange


def train(epoch):
    model.train()
    optimizer.zero_grad()  # Clear gradients.
    embeddings, out = model(train_data.x, train_data.edge_index, train_data.edge_attr)  # Perform a single forward pass.
    loss = criterion(out, train_data.y) #+ custom_loss(embeddings, train_data.edge_index, train_data.edge_attr, train_data.y)  # Compute the loss solely based on the training nodes.
    loss.backward()  # Derive gradients.
    optimizer.step()  # Update parameters based on gradients.
    return loss


class FocalLoss(torch.nn.modules.loss._WeightedLoss):
    def __init__(self, weight=None, gamma=2, reduction='mean'):
        super(FocalLoss, self).__init__(weight, reduction=reduction)
        self.gamma = gamma
        self.weight = weight  # weight parameter will act as the alpha parameter to balance class weights

    def forward(self, input, target):
        ce_loss = torch.nn.functional.cross_entropy(input, target, reduction=self.reduction, weight=self.weight)
        pt = torch.exp(-ce_loss)
        focal_loss = ((1 - pt) ** self.gamma * ce_loss).mean()
        return focal_loss


def custom_loss(embeddings, edge_index, edge_weights, labels):
    src, dest = edge_index
    embeddings_dot = torch.sum(embeddings[src] * embeddings[dest], dim=-1).unsqueeze(1)

    same_class_mask = (labels[src] == labels[dest])
    diff_class_mask = ~same_class_mask

    same_class_loss = torch.sigmoid(edge_weights[same_class_mask] * embeddings_dot[same_class_mask])
    diff_class_loss = torch.sigmoid(edge_weights[diff_class_mask] * embeddings_dot[diff_class_mask])

    loss = - torch.mean(same_class_loss) + torch.mean(diff_class_loss)

    return loss

if __name__ == '__main__':

    experiment = Experiment(
        project_name="genotek",
        workspace="kenenbek",
    )

    if torch.cuda.is_available():
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')

    model = AttnGCN().to(device)
    criterion = torch.nn.CrossEntropyLoss()
    # criterion = torch.nn.MSELoss()
    # criterion = FocalLoss(
    #    weight=1. / torch.tensor([3449, 1021, 1001,  469,  189,  826,   73,   69,   67,  546], dtype=torch.float).to(device))
    # criterion = mdn_gamma_loss

    train_dataset = MyOwnDataset(root="train_data_10_class/")
    train_data = train_dataset.get(0).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=5e-4)

    t = trange(10000, leave=True)
    losses = []
    for epoch in t:
        loss = train(epoch)
        losses.append(loss)
        t.set_description(str(round(loss.item(), 6)))
        experiment.log_metric("loss", loss, epoch=epoch)

    torch.save(model, "attn.pt")
