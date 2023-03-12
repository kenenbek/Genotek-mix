import comet_ml
from comet_ml import Experiment

import torch

from mix import AttnGCN, GCN
from mix import MyOwnDataset22Class
from tqdm import trange


def train(epoch):
    model.train()
    optimizer.zero_grad()  # Clear gradients.
    out = model(train_data.x, train_data.edge_index, train_data.edge_attr)  # Perform a single forward pass.
    loss = criterion(out, train_data.y)  # Compute the loss solely based on the training nodes.
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
    # criterion = FocalLoss(
    #     weight=1. / torch.tensor([2815, 1, 635, 620, 403, 432, 573, 353, 109, 100, 107, 625, 183, 55, 19, 20,
    #                               48, 39, 28, 11, 1, 535], dtype=torch.float).to(device))

    train_dataset = MyOwnDataset22Class(root="train_data_10_class/")
    train_data = train_dataset.get(0).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=5e-4)

    t = trange(20000, leave=True)
    losses = []
    for epoch in t:
        loss = train(epoch)
        losses.append(loss)
        t.set_description(str(round(loss.item(), 6)))
        experiment.log_metric("loss", loss, epoch=epoch)

    torch.save(model, "attn.pt")
