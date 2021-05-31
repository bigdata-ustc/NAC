import torch
import torch.nn as nn
from itertools import combinations
import numpy as np


class FM(nn.Module):
    def __init__(self, n_hero, team_size=5, hidden_dim=10, device=torch.device('cpu')):
        super(FM, self).__init__()
        assert(n_hero > 1 and team_size>1)
        self.n_hero = n_hero
        self.team_size = team_size
        self.skill = nn.Embedding(n_hero, 1)
        self.embedding = nn.Embedding(n_hero, hidden_dim)
        # self.embedding.weight.data = self.embedding.weight /1000 + 1
        self.hidden_dim = hidden_dim
        self.device = device
        gen = combinations(range(team_size), 2)
        self.index1, self.index2 = list(zip(*gen))

    def forward(self, team):
        assert team.max() < self.n_hero
        assert team.shape[1] == self.team_size
        n_match = len(team)
        team = torch.LongTensor(team).to(self.device)
        hero_skill = self.skill(team).view(n_match, -1)
        team_skill = hero_skill.sum(dim=1, keepdim=True)
        order2 = self.interact(team)
        return team_skill + order2

    def interact(self, team):
        a = team[:, self.index1]
        b = team[:, self.index2]
        a = self.embedding(a)
        b = self.embedding(b)
        order2 = (a * b).sum(dim=(1, 2)).view(-1, 1)
        return order2


class HOI(nn.Module):
    def __init__(self, n_hero, team_size=5, hidden_dim=10, device=torch.device('cpu')):
        super(HOI, self).__init__()
        assert(n_hero > 1 and team_size>1)
        self.team_size = team_size
        self.component = FM(n_hero, team_size, hidden_dim, device=device)

    def forward(self, data):
        assert data.shape[1] == 2*self.team_size
        team_A = data[:, :self.team_size]
        team_B = data[:, self.team_size:]
        Aability = self.component(team_A)
        Bability = self.component(team_B)
        probs = torch.sigmoid(Aability - Bability).view(-1)
        return probs



