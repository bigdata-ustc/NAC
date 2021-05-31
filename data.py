import numpy as np
import pandas as pd
import random
from sklearn.model_selection import StratifiedKFold, KFold, train_test_split
from scipy.sparse import csr_matrix, hstack, vstack
from collections import Counter


class Data:
    def __init__(self, path=None, symmetry=True, team_size=5, test_size=0.1, seed=None):
        if path is None:
            path = '../data/dota2018.csv'  #dota2018

        self.seed = seed
        self.team_size = team_size
        self.test_size = test_size
        self.symmetry = symmetry
        df = pd.read_csv(path)
        self.n_player = -1
        self.n_hero = -1

        player_set, hero_set = set(), set()

        hero_cols = [i for i in df.columns if 'hero' in i]
        assert len(hero_cols) == team_size * 2
        for col in hero_cols:
            hero_set = hero_set | set(df[col])

        hero2int = {name: i for i, name in enumerate(hero_set)}
        int2hero = {hero2int[name]: name for name in hero2int}
        self.hero2int, self.int2hero = hero2int, int2hero
        for col in hero_cols:
            df[col] = df[col].map(hero2int)
        self.n_hero = len(self.hero2int)

        self.n_individual = self.n_hero
        self.time_cols = 'date'
        target_col = ['target'] if 'target' in df.columns else ['radiant_win']
        self.data = df.loc[:, hero_cols + target_col].to_numpy().astype(int)[:800000]
        print('whole dataset size', self.data.shape)
        self.split()

    def split(self):
        self.data, self.test = train_test_split(self.data, test_size=0.10, random_state=self.seed)
        self.train, self.valid = train_test_split(self.data, test_size=0.11, random_state=self.seed)
        train_set = set(self.select(self.train).reshape(-1))
        valid_set = set(self.select(self.valid).reshape(-1))
        test_set = set(self.select(self.test).reshape(-1))
        print('individual in valid not in train', len(valid_set - train_set))
        print('individual in test not in train', len(test_set - train_set))
        train_cnt = Counter(self.select(self.train).reshape(-1))
        valid_cnt = Counter(self.select(self.valid).reshape(-1))
        test_cnt = Counter(self.select(self.test).reshape(-1))
        self.hero_cnt = train_cnt + valid_cnt + test_cnt

        print('train shape: ', self.train.shape)
        print('valid shape: ', self.valid.shape)
        print('test shape: ', self.test.shape)

    def encode(self, data):
        t = self.team_size
        A = self.sparse(data[:, :t], self.n_hero)
        B = self.sparse(data[:, t:], self.n_hero)

        if self.symmetry:
            return A + B * -1
        else:
            return hstack([A, B])

    def sparse(self, dense, n_individual):
        t = self.team_size
        n_match = len(dense)
        values = np.ones(n_match * t)
        rowptr = np.array([i * t for i in range(n_match + 1)])
        col_index = dense.reshape(-1)
        A = csr_matrix((values, col_index, rowptr), shape=(n_match, n_individual))
        return A

    def get_all(self, type='train', encoding=False):
        if type == 'train':
            data = self.train
        elif type == 'valid':
            data = self.valid
        elif type == 'test':
            data = self.test
        else:
            raise Warning('wrong type!')

        y = data[:, -1]
        data = self.select(data)

        if encoding:  # BT/LR
            return self.encode(data), y
        else:
            return data, y

    def get_batch(self, batch_size=32, type='train', shuffle=True):
        if type == 'train':
            data = self.train
        elif type == 'valid':
            data = self.valid
        elif type == 'test':
            data = self.test
        else:
            raise Warning('wrong type!')

        y = data[:, -1]
        data = self.select(data)
        length = len(data)
        index = np.arange(length)
        if shuffle:
            random.shuffle(index)

        start_idx = 0
        while start_idx < length:
            end_idx = min(length, start_idx + batch_size)
            excerpt = index[start_idx:end_idx]
            X = data[excerpt]
            yield X, y[excerpt]
            start_idx += batch_size

    def select(self, data):
        t = self.team_size
        data = data[:, :t*2]
        return data


if __name__ == '__main__':
    dataset = Data()
    print('*'*10)
    x, y = dataset.get_all()
    print(x.shape)
    print(y.shape)
    x, y = dataset.get_all(type='valid', encoding=True)
    print(x.shape)

    for x, y in dataset.get_batch():
        print('batch')
        print(x, y)
        break
