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
    loss = criterion(out, train_data.y) + custom_loss(out, train_data.edge_index, train_data.edge_attr, train_data.y)  # Compute the loss solely based on the training nodes.
    loss.backward()  # Derive gradients.
    optimizer.step()  # Update parameters based on gradients.
    return loss


def in_between_loss(z, y, proportions):
    # z is hidden states
    # Compute the "in-betweenness" loss for mixed objects
    mixed_mask = (y > 0) & (y < 10)
    mixed_z = z[mixed_mask]
    A = z[y == 0].mean(dim=0)  # class 0 objects
    B = z[y == 10].mean(dim=0)  # class 10 objects

    # Compute the projection of each mixed representation onto the line from A to B,
    # as a fraction of the length of the line from A to B
    projection = ((mixed_z - A).matmul(B - A) / (B - A).dot(B - A))

    # Compute the absolute difference between the projection and the mixture proportion
    dist_loss = (projection - proportions[mixed_mask]).abs().mean()

    return dist_loss

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
    train_data = torch.load('Raw/train_data.pt', map_location=torch.device('cpu')).to(device)
    weight = 1000 / torch.unique(train_data.y, return_counts=True)[1].type(torch.float).to(device)
    criterion = torch.nn.CrossEntropyLoss(weight=weight)

    optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=5e-4)

    t = trange(6000, leave=True)
    losses = []

    for epoch in t:
        model.train()
        optimizer.zero_grad()
        z, out = model(train_data.x, train_data.edge_index)
        loss = criterion(out, train_data.y) + in_between_loss(z, train_data.y, train_data.y_floats)
        loss.backward()
        optimizer.step()

        losses.append(loss)
        t.set_description(str(round(loss.item(), 6)))
        experiment.log_metric("loss", loss, epoch=epoch)

    torch.save(model, "attn.pt")
