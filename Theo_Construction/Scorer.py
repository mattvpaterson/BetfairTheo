import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
import statsmodels.api as sm
from sklearn.model_selection import train_test_split
import datetime as datetime
import time


class Scorer:

    def __init__(self, base_frame):
        self.base = base_frame

    def simulate(self,no_runs):

        predictions_dict = {}
        for idno in self.base['Id'].unique():
            frame = self.base[self.base['Id'] == idno]

            storage_dict = {}
            for name, speed in zip(frame['HorseName'], frame['predictions']):
                storage_dict[name] = {'speed': speed}
            for name, std in zip(frame['HorseName'], frame['std']):
                storage_dict[name]['std'] = std

            predictions_dict[idno] = storage_dict

        results_dict = {}
        for key in predictions_dict.keys():
            results_dict[key] = Scorer.simulator(predictions_dict[key],no_runs)
        self.results_dict = results_dict

        self.base['Theo'] = self.base.apply(lambda x: Scorer.theo_associater(x['Id'], x['HorseName'], results_dict), axis=1)

    def score(self):
        self.score2 =  ((1 / self.base['PPWAP'] - 1 / self.base['Theo']) ** 2).sum()
        self.score =  ((1 / self.base['PPWAP'] - 1 / self.base['Theo'])).sum()

    @staticmethod
    def simulator(combo_dict, no_runs):
        counts = [0 for i in combo_dict.keys()]
        sims = [np.random.normal(combo_dict[horse]['speed'], combo_dict[horse]['std'], no_runs) for horse in combo_dict]

        for row in range(0, len(sims[0])):
            blank = []
            for i in sims:
                blank.append(i[row])
            counts[blank.index(max(blank))] += 1
        counts = [i + 1 for i in counts]  # to avoid division by 0 below for cases which don't win one in 100000

        return dict(zip(combo_dict.keys(), [np.round(no_runs / i, decimals=2) for i in counts]))

    @staticmethod
    def theo_associater(id_number,horse_name,results_dict):
        return results_dict[id_number][horse_name]
