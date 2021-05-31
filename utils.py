import numpy as np
import pandas as pd
import random


class Logger:
    def __init__(self, name, dir='./log/', record=False):
        self.name = dir + name
        self.record = record
        
    def printf(self, string):
        print(string)
        if self.record:
            file = open(self.name, 'a')
            file.write(string)
            file.write('\n')
            file.close()


class Dota_hero:
    def __init__(self, dataset):
        hero_info = pd.read_csv('./vis/dota_hero.csv')
        hero_info['embed_id'] = hero_info['hero_id'].map(dataset.hero2int)
        self.hero_info = hero_info[hero_info.embed_id.notna()]
        print(hero_info.head())
        
        self.hero_cnt = dataset.hero_cnt
        self.hero_set = set(self.hero_cnt)
    #
    def query(self, x, name='cn_name'):
        names = []
        if type(x) == np.ndarray:
            x = x.reshape(-1)
        for i in x:
            get = self.hero_info[self.hero_info['embed_id'] == i][name].values
            assert len(get) == 1
            names.append(get[0])
        return names
    
    def most_common(self, n, need_name=True, name='cn_name'):
        temp = self.hero_cnt.most_common(n)
        embed_ids = [i[0] for i in temp]
        print('most common and least common', temp[0][1], temp[-1][1])
        if need_name:
            return embed_ids, self.query(embed_ids, name)
        return embed_ids






