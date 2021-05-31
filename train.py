import numpy as np
import sklearn.metrics as metrics
import torch
import torch.optim as optim

from data import Data
from FM import *
from NAC import *


device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')


n_epochs = 50
batch_size = 256
learning_rate = 0.001
hidden_dim = 20

path = './data/dota2018.csv' 
team_size = 5

dataset = Data(path, team_size=team_size)
n_individual = dataset.n_individual


# model = HOI(n_individual, team_size, hidden_dim, device=device)
model = NAC(n_individual, hidden_dim=hidden_dim, need_att=True, device=device)

model = model.to(device)
optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=0.0001)


def evaluate(pred, label):
    if type(pred) != np.ndarray:
        pred = pred.cpu().detach().numpy()

    pred = pred.reshape(-1)
    label = label.reshape(-1)
    auc = metrics.roc_auc_score(label, pred)
    pred = np.clip(pred, 0.001, 0.999)
    logloss = metrics.log_loss(label, pred)
    pred = (pred > 0.5) * 1
    acc = (label == pred).sum() / len(label)
    return auc, acc, logloss


criterion = nn.BCELoss()
total_step = len(dataset.train) // batch_size + 1

print('training begin')
for epoch in range(n_epochs):
    model.train()
    batch_gen = dataset.get_batch(batch_size)
    for i, (X, y) in enumerate(batch_gen):
        y_tensor = torch.Tensor(y).to(device)
        pred = model(X)
        loss = criterion(pred, y_tensor)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if (i + 1) % 1000 == 0:
            print('Epoch [{}/{}], Step [{}/{}]'.format(epoch + 1, n_epochs, i + 1, total_step))
            
    model.eval()
    preds = []
    y = dataset.train[:, -1]
    batch_gen = dataset.get_batch(30000, 'train', shuffle=False)
    for X, _ in batch_gen:
        with torch.no_grad():
            pred = model(X)
            preds.append(pred.reshape(-1))

    auc, acc, logloss = evaluate(torch.cat(preds), y)
    print('Epoch [{}/{}], train set auc: {:.4f}, acc: {:.4f}, logloss: {:.4f}'.format(epoch + 1, n_epochs, auc, acc,
                                                                                      logloss))

    preds = []
    y = dataset.valid[:, -1]
    batch_gen = dataset.get_batch(30000, 'valid', shuffle=False)
    for X, _ in batch_gen:
        with torch.no_grad():
            pred = model(X)
            preds.append(pred.reshape(-1))

    auc, acc, logloss = evaluate(torch.cat(preds), y)
    print('Epoch [{}/{}], valid set auc: {:.4f}, acc: {:.4f}, logloss: {:.4f}'.format(epoch + 1, n_epochs, auc, acc,
                                                                                      logloss))

    preds = []
    y = dataset.test[:, -1]
    batch_gen = dataset.get_batch(30000, 'test', shuffle=False)
    for X, _ in batch_gen:
        with torch.no_grad():
            pred = model(X)
            preds.append(pred.reshape(-1))

    auc, acc, logloss = evaluate(torch.cat(preds), y)
    print('Epoch [{}/{}], test set auc: {:.4f}, acc: {:.4f}, logloss: {:.4f}'.format(epoch + 1, n_epochs, auc, acc,
                                                                                         logloss))
