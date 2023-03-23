import torch
import torch.nn as nn
from torch.utils.data import TensorDataset
from torch.utils.data import DataLoader


import numpy as np
import pandas as pd


from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split

import pennylane as qml
from pennylane import numpy as np

import pennylane as qml
from pennylane import numpy as np






dev = qml.device("default.qubit", wires=4)
n_qubits = 4
n_layers = 1


def qnode(inputs, weights):
    # デフォルトではRx回転角が使用される
    qml.AngleEmbedding(inputs, wires=range(n_qubits))
    qml.StronglyEntanglingLayers(weights, wires=range(n_qubits))
    #qml.BasicEntanglerLayers(weights, wires=range(n_qubits))
    #qml.AngleEmbedding(inputs, wires=range(n_qubits))
    #qml.StronglyEntanglingLayers(weights, wires=range(n_qubits))
    
    return [qml.expval(qml.PauliZ(wires=i)) for i in range(n_qubits)]



class HybridModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.clayer_1 = torch.nn.Linear(4, 4)
        self.qlayer_1 = qml.qnn.TorchLayer(qnode, weight_shapes)
        self.clayer_2 = torch.nn.Linear(4, 3)
        self.softmax = torch.nn.Softmax(dim=1)

    def forward(self, x):
        x = self.clayer_1(x)
        x = self.qlayer_1(x)
        x = self.clayer_2(x)
        return self.softmax(x)



@qml.qnode(dev)


model = HybridModel()

opt = torch.optim.SGD(model.parameters(), lr=0.2)
loss = torch.nn.L1Loss()
criterion = nn.CrossEntropyLoss()

train_dataset = TensorDataset(x_train, y_train)
valid_dataset = TensorDataset(x_valid, y_valid)

batch_size = 32
train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
valid_dataloader = DataLoader(valid_dataset, batch_size=batch_size, shuffle=False)

batch_size = 32
batches = len(x_train) // batch_size



criterion = nn.CrossEntropyLoss()


epochs = 20

for epoch in range(epochs):

    running_loss = 0

    for xs, ys in train_dataloader:
        opt.zero_grad()
        #print(ys)
        outputs = model(xs)
        #print(outputs)
        #y_valid = torch.tensor(y_valid, dtype=torch.int32)

        loss_evaluated = criterion(outputs, ys)
        loss_evaluated.backward()

        opt.step()

        running_loss += loss_evaluated

    avg_loss = running_loss / batches
    print("Average loss over epoch {}: {:.4f}".format(epoch + 1, avg_loss))

y_pred = model(x_valid)
predictions = torch.argmax(y_pred, axis=1).detach().numpy()

correct = [1 if p == p_true else 0 for p, p_true in zip(predictions, y_valid)]
accuracy = sum(correct) / len(correct)
print(f"Accuracy: {accuracy * 100}%")