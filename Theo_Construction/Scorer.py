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

    def simulate(self , no_runs):

        predictions_dict = {}
        for idno in self.base['Id'].unique():
            frame = self.base[self.base['Id'] == idno]

            storage_dict = {}
            for name, speed in zip(frame['HorseName'], frame['prediction']):
                storage_dict[name] = {'speed': speed}
            for name, std in zip(frame['HorseName'], frame['residual^2_prediction']**0.5):
                storage_dict[name]['std'] = std

            predictions_dict[idno] = storage_dict

        results_dict = {}
        for key in predictions_dict.keys():
            results_dict[key] = Scorer.simulator(predictions_dict[key],no_runs)
        self.results_dict = results_dict

        self.base['Theo'] = self.base.apply(lambda x: Scorer.theo_associater(x['Id'], x['HorseName'], results_dict), axis=1)

    def score(self, print_results = True):
        self.score_theo = (-np.log(1/self.base['Theo'])*self.base['WIN_LOSE'] - np.log(1-1/self.base['Theo'])*(1-self.base['WIN_LOSE'])).sum()
        self.score_PPWAP = (-np.log(1/self.base['PPWAP'])*self.base['WIN_LOSE'] - np.log(1-1/self.base['PPWAP'])*(1-self.base['WIN_LOSE'])).sum()
        self.score_BSP = (-np.log(1/self.base['BSP'])*self.base['WIN_LOSE'] - np.log(1-1/self.base['BSP'])*(1-self.base['WIN_LOSE'])).sum()

        if print_results:
            print('Score from our theoretical odds: ' + str(np.round(self.score_theo, decimals = 2)))
            print('Score from PPWAP implied odds: ' + str(np.round(self.score_PPWAP, decimals = 2)))
            print('Score from BSP implied odds: ' + str(np.round(self.score_BSP, decimals = 2)))

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
    def perfect_score(frame):
        storage_dict = {}
        
        for race_id in frame['Id'].unique():
            race_frame = frame[frame['Id'] == race_id]
            storage_dict[race_id] = {i:0 for i in race_frame['HorseName']}
            odds_list = [1/i for i in race_frame['PPWAP']]
            odds_list_normalised = [j/sum(odds_list) for j in odds_list]
            
            storage_dict[race_id][np.random.choice(list(race_frame['HorseName']), 1, p = odds_list_normalised)[0]] = 1
        frame['sim_result'] = frame.apply(lambda x: Scorer.theo_associater(x['Id'],x['HorseName'],storage_dict), axis = 1)
             
        return (-np.log(1/frame['PPWAP'])*frame['sim_result'] - np.log(1-1/frame['PPWAP'])*(1-frame['sim_result'])).sum()
    
    @staticmethod
    def theo_associater(id_number,horse_name,results_dict):
        return results_dict[id_number][horse_name]
