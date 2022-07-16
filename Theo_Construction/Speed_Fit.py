import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
import statsmodels.api as sm
from sklearn.model_selection import train_test_split
import datetime as datetime
import time


class Speed_Fit:

    def __init__(self, base_frame):
        self.base = base_frame[['Id', 'Course', 'RaceDate', 'RaceTime', 'Yards', 'Going', 'Seconds',
       'Ran', 'TotalBtn', 'HorseName', 'Age', 'WeightLBS', 'Jockey', 'Type',
       'horse_race_no', 'win_speed', 'individual_speed']]
        self.trade_data = base_frame[['Id','HorseName','PPWAP', 'PPMAX',
       'PPMIN', 'PPTRADEDVOL', 'WIN_LOSE']]


    def linear_race_fit(self):
        # Fitting for communal race features, to determine how much of the R^2 is not horse-specific
        new_frame = pd.concat([self.base,
                               pd.get_dummies(self.base["Going"]),
                               pd.get_dummies(self.base["Course"]),
                               pd.get_dummies(self.base["Type"]),
                               ], axis=1).drop(
            columns=['Id', 'RaceDate', 'RaceTime', 'Seconds', 'Course', 'Going', 'TotalBtn', 'win_speed',
                     'Ran', 'HorseName', 'Age', 'Jockey', 'HorseName', 'horse_race_no', 'Yards', 'WeightLBS','Type'])
        new_frame['constant'] = 1

        y = new_frame['individual_speed']
        x = new_frame.drop(columns=['individual_speed'])

        x, y = np.array(x), np.array(y)
        self.race_results = sm.OLS(y, x).fit()

        print(str(np.round(self.race_results.rsquared, decimals=4)) + ' R^2 Race General features ')

    def linear_horse_fit(self):
        # Fitting for all features, including horse-specific
        # Iterating to propagate predictions for calculation of difference vs previous prediction feature
        for i in range(0, 5):
            self.final = self.base
            self.final['unfinished_flag'] = 1 - self.final['individual_speed'].notnull().astype("int")
            self.final['unfinished_flag'] = self.final.groupby('HorseName')['unfinished_flag'].shift().fillna(0)

            new_frame = pd.concat([self.final,
                                   pd.get_dummies(self.final["Ran"]).rename(columns=lambda x: 'Ran_' + str(x)),
                                   pd.get_dummies(self.final["Age"]).rename(columns=lambda x: 'Age_' + str(x)),
                                   pd.get_dummies(self.final["horse_race_no"]).rename(
                                       columns=lambda x: 'Race_no_' + str(x)),
                                   pd.get_dummies(self.final["Going"]),
                                   pd.get_dummies(self.final["Course"]),
                                   pd.get_dummies(self.final["Type"]),
                                   pd.get_dummies(self.final["Jockey"])
                                   ], axis=1).drop(
                columns=['Id', 'RaceDate', 'RaceTime', 'Seconds', 'Course', 'Going', 'TotalBtn', 'win_speed',
                         'Ran', 'HorseName', 'Age', 'Jockey', 'HorseName', 'horse_race_no', 'Type'])
            new_frame['constant'] = 1

            train_frame = new_frame.dropna()
            y = train_frame['individual_speed']
            x = train_frame.drop(columns=['individual_speed'])
            x, y = np.array(x), np.array(y)
            self.results = sm.OLS(y, x).fit()

            self.base['predictions'] = self.results.predict(np.array(new_frame.drop(columns=['individual_speed'])))
            self.base['diff'] = self.base['individual_speed'] - self.base['predictions']
            self.base['diff'] = self.base['diff'].fillna(0)
            self.base['diff'] = self.base.groupby('HorseName')['diff'].shift()
            self.base['diff'] = self.base['diff'].fillna(0)
            del self.base['predictions']

        self.base['predictions'] = self.results.predict(np.array(new_frame.drop(columns=['individual_speed'])))
        print(str(np.round(self.results.rsquared, decimals=4)) + ' R^2 All')
        self.final = self.base.merge(self.trade_data, how='left', on=['Id', 'HorseName'])
        print(str(np.round(self.results.rsquared - self.race_results.rsquared, decimals=4)) + ' Horse Specific R^2 difference')




