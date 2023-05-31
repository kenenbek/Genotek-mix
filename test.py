import torch
import torch_geometric
from matplotlib import pyplot as plt

test_data = torch.load('Raw/test_data.pt', map_location=torch.device('cpu'))
model = torch.load("attn.pt", map_location=torch.device('cpu'))
model.eval()

with torch.no_grad():
    z, out = model(test_data.x, test_data.edge_index)
pred = out.argmax(dim=1)  # Use the class with highest probability.

from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

fig, ax = plt.subplots(figsize=(10, 10))
ConfusionMatrixDisplay.from_predictions(test_data.y, pred, ax=ax,
                                       )
plt.savefig("df.png")