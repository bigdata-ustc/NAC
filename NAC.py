import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parameter import Parameter
from itertools import combinations
import numpy as np


def combine(team_size=5):
    index1, index2 = [], []
    for i in range(team_size):
        for j in range(team_size):
            if i == j:
                continue
            index1.append(i)
            index2.append(j)
    #
    return index1, index2


class BT(nn.Module):
    def __init__(self, n_hero):
        super(BT, self).__init__()
        assert n_hero > 1
        self.skill = nn.Embedding(n_hero, 1)

    def forward(self, team):
        n_match = len(team)
        hero_skill = self.skill(team).view(n_match, -1)
        team_skill = hero_skill.sum(dim=1, keepdim=True)
        return team_skill


class ANFM(nn.Module):
    def __init__(self, n_hero, team_size, hidden_dim, need_att=True):
        super(ANFM, self).__init__()
        assert(n_hero > 1 and team_size>1)
        self.n_hero = n_hero
        self.team_size = team_size
        self.hidden_dim = hidden_dim
        self.embedding = nn.Embedding(n_hero, hidden_dim)
        self.index1, self.index2 = combine(5)
        self.need_att = need_att

        self.attenM = AttM(n_hero, team_size, hidden_dim, reduce=True)
        dropout = nn.Dropout(0.2)
        self.MLP = nn.Sequential(
            nn.Linear(hidden_dim, 50), nn.ReLU(), dropout, # nn.BatchNorm1d(50),
            nn.Linear(50, 1, bias=True), nn.ReLU(),
        )

    def forward(self, team):
        n_match = len(team)
        a = team[:, self.index1]
        b = team[:, self.index2]
        a = self.embedding(a)
        b = self.embedding(b)
        order2 = self.MLP(a * b).squeeze()  # [batch_size, hidden_dim]
        if self.need_att:
            normal = self.attenM(a, b, dim=2)
            order2 = order2 * normal

        order2 = order2.sum(dim=1, keepdim=True)  # [batch_size, 1]
        return order2


class AttM(nn.Module):
    def __init__(self, n_hero, length=5, hidden_dim=10, reduce=False):
        super(AttM, self).__init__()
        assert (n_hero > 1 and length > 1)
        self.n_hero = n_hero
        self.hidden_dim = hidden_dim
        self.length1 = length
        self.length2 = length if not reduce else length - 1
        self.W = nn.Linear(hidden_dim, hidden_dim)

    def forward(self, team1, team2, dim=2):     # team1: [batch_size, 25, hidden_dim]
        assert team1.shape == team2.shape
        length1, length2 = self.length1, self.length2
        team1 = team1.view(-1, length1, length2, self.hidden_dim)
        team2 = team2.view(-1, length1, length2, self.hidden_dim)
        score = (self.W(team1) * team2).sum(dim=3)     # [batch_size, 5, 5]
        score = F.softmax(score, dim=dim)
        return score.view(-1, length1*length2)      # [batch_size, 25]


class Blade_chest(nn.Module):
    def __init__(self, n_hero, team_size, hidden_dim, method='inner', need_att=True):
        super(Blade_chest, self).__init__()
        assert(n_hero > 1 and team_size>1)
        assert method in ['inner', 'dist']
        self.team_size = team_size
        self.blade = nn.Embedding(n_hero, hidden_dim)
        self.chest = nn.Embedding(n_hero, hidden_dim)

        self.index1 = np.repeat([i for i in range(team_size)], team_size)   # [0, 0, 0, 1, 1, 1, 2, 2, 2]
        self.index2 = np.tile([i for i in range(team_size)], team_size)     # [0, 1, 2, 0, 1, 2, 0, 1, 2]
        self.attenM = AttM(n_hero, team_size, hidden_dim)
        self.method = method
        self.need_att = need_att
        dropout = nn.Dropout(0.2)
        self.MLP = nn.Sequential(
            nn.Linear(hidden_dim, 50), nn.ReLU(), dropout, # nn.BatchNorm1d(50),
            nn.Linear(50, 1, bias=True), nn.ReLU(),
        )

    def forward(self, team_A, team_B):
        a = team_A[:, self.index1]
        b = team_B[:, self.index2]
        a_blade = self.blade(a)   # (batch_size, 25, hidden_dim)
        b_chest = self.chest(b)
        if self.method == 'inner':
            a_beat_b = self.inner(a_blade, b_chest)
        else:
            a_beat_b = self.dist(a_blade, b_chest)

        return a_beat_b

    def inner(self, a_blade, b_chest):
        interact = a_blade * b_chest
        a_beat_b = self.MLP(interact).squeeze()

        normal = self.attenM(a_blade, b_chest, dim=2) if self.need_att else 1
        a_beat_b = a_beat_b * normal
        return a_beat_b.sum(dim=1, keepdim=True)

    def dist(self, a_blade, b_chest):     # (batch_size, 25, hidden_dim)
        interact = (a_blade - b_chest)**2
        a_beat_b = self.MLP(interact).squeeze()  # (batch_size, 25)

        normal = self.attenM(a_blade, b_chest, dim=2) if self.need_att else 1
        a_beat_b = a_beat_b * normal
        return a_beat_b.sum(dim=1, keepdim=True)


class NAC(nn.Module):
    def __init__(self, n_hero, team_size=5, hidden_dim=10, need_att=True, device=torch.device('cpu')):
        super(NAC, self).__init__()
        assert(n_hero > 1 and team_size>1)
        self.n_hero = n_hero
        self.team_size = team_size
        self.hidden_dim = hidden_dim
        self.device = device
        self.BT = BT(n_hero)
        self.Coop = ANFM(n_hero, team_size, hidden_dim, need_att)
        self.Comp = Blade_chest(n_hero, team_size, hidden_dim, need_att=need_att)
        # self.bias = Parameter(torch.Tensor(1))  # add a bias term when necessary

    def forward(self, data):
        assert data.shape[1] == 2 * self.team_size
        data = torch.LongTensor(data).to(self.device)
        team_A = data[:, :self.team_size]
        team_B = data[:, self.team_size:]
        A_coop = self.Coop(team_A) + self.BT(team_A)
        B_coop = self.Coop(team_B) + self.BT(team_B)
        
        A_comp = self.Comp(team_A, team_B)
        B_comp = self.Comp(team_B, team_A)
        
        adv = A_comp - B_comp
        probs = torch.sigmoid(A_coop - B_coop + adv).view(-1)
        return probs


if __name__ == '__main__':
    n_hero = 2000
    n_match = 15
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    data = np.random.randint(0, n_hero, (n_match, 10))

    model = NAC(n_hero, device=device).to(device)
    preds = model(data)
    print('preds:', preds.shape)
