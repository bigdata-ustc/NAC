import numpy as np
import torch
import torch.optim as optim
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn import manifold
import pandas as pd
import random

from data import Data
from NAC import *
from utils import Dota_hero


# plt.rcParams['font.family'] = ['sans-serif']
# plt.rcParams['font.sans-serif'] = ['SimHei']
# plt.rcParams['axes.unicode_minus'] = False

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
path = './data/dota2018.csv' 
team_size = 5


dataset = Data(path, team_size=team_size)
tool = Dota_hero(dataset)
n_individual = dataset.n_individual

model = NAC(n_individual, hidden_dim=20, need_att=True, device=device)
model_name = 'NAC'
model.load_state_dict(torch.load(f'./param/{model_name}'))
model.to(device)
_ = model.eval()


def plot_hotmap(result, pic_name='temp', cmap='Oranges', annot=True):
    vmax = result.max()
    vmin = result[result != 0].min()
    result = pd.DataFrame(result)
    result.index = [names[i] for i in range(num)]
    result.columns = [names[i] for i in range(num)]
    fig, ax = plt.subplots(figsize=(10, 10))
    sns.heatmap(result.round(2), annot=annot, vmax=vmax+0.02, vmin=vmin-0.02, square=True, cbar=True,
                xticklabels=True, yticklabels=True, cmap=cmap)   # Blues
    #
    # ax.set_title(pic_name, fontsize = 18)
    # ax.set_ylabel('A1', fontsize = 18)
    # ax.set_xlabel('A2', fontsize = 18)
    #
    plt.savefig(f'./{pic_name}.svg')
    plt.show()


# ================ individual effects ===================
names = tool.query(range(dataset.n_individual), name='en_name')
indiviual_score = model.BT.skill.weight.data.cpu().numpy().reshape(-1)

# ================ prepare ===================

num = 10
embed_ids, names = tool.most_common(num, name='en_name')
names = ['Pudge', 'SF', 'Zeus', 'Wind', 'Jugg', 'Invoker', 'FV', 'Lion', 'Rubick', 'Shaman']
embed_ids = torch.LongTensor(embed_ids).to(device)
team_A = embed_ids.unsqueeze(0)
team_B = team_A

coop_idx1, coop_idx2 = combine(num)
comp_idx1 = np.repeat([i for i in range(num)], num)
comp_idx2 = np.tile([i for i in range(num)], num)

# ================ cooperation effects ===================
a = team_A[:, coop_idx1]
b = team_A[:, coop_idx2]
a = model.Coop.embedding(a)
b = model.Coop.embedding(b)
order2 = model.Coop.MLP(a * b).squeeze()
attenM_W = model.Coop.attenM.W
score = (attenM_W(a) * b).sum(dim=2)    # [batch_size, 25]

result = np.zeros((num, num))
result[coop_idx1, coop_idx2] = order2.view(-1).cpu().detach().numpy()

att = np.ones((num, num)) * score.view(-1).cpu().detach().numpy().min().round(1) - 0.1
att[coop_idx1, coop_idx2] = score.view(-1).cpu().detach().numpy()

plot_hotmap(result, f'{model_name}_coop', 'Blues', annot=True)
plot_hotmap(att, f'{model_name}_coop_att', annot=True)


# ================ competition effects ===================
a = team_A[:, coop_idx1]
b = team_B[:, coop_idx2]
a = model.Comp.blade(a)   # (batch_size, 25, hidden_dim)
b = model.Comp.chest(b)

a_beat_b = model.Comp.MLP(a * b).squeeze()    # a_beat_b: [batch_size, 25]
attenM_W = model.Comp.attenM.W
score = (attenM_W(a) * b).sum(dim=2)    # [batch_size, 25]

result = np.zeros((num, num))
result[coop_idx1, coop_idx2] = a_beat_b.view(-1).cpu().detach().numpy()

att = np.ones((num, num)) * score.view(-1).cpu().detach().numpy().min().round(1) - 0.1
att[coop_idx1, coop_idx2] = score.view(-1).cpu().detach().numpy()

plot_hotmap(result, f'{model_name}_beat', 'Blues')
plot_hotmap(att, f'{model_name}_beat_att')
