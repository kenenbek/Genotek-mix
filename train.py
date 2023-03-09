import comet_ml
from comet_ml import Experiment

import torch

from mix import AttnGCN
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

    train_dataset = MyOwnDataset22Class(root="train_data_22_class/")
    train_data = train_dataset.get(0).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=5e-4)

    t = trange(10000, leave=True)
    losses = []
    for epoch in t:
        loss = train(epoch)
        losses.append(loss)
        t.set_description(str(round(loss.item(), 6)))
        experiment.log_metric("accuracy", loss, epoch=epoch)
