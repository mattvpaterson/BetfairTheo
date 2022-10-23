import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
import statsmodels.api as sm
from sklearn.model_selection import train_test_split
import datetime as datetime
import time
from sklearn.metrics import r2_score

class Variance_Fit:
    
    #This trains and attaches predicted variance for each runner's individual speed performance vs prediction (the residual)
    #The speed data is heteroskedastic
    #The idea is that we can use this prediction to approximate the distribution of natural variance which each horses speed has. 
    #I would expect that more experienced & better performing horses have lower performance variance.
    
    def __init__(self, base_frame, trade_data, min_cap = 0.09, max_cap = 0.48):
        self.base = base_frame
        self.min_cap = min_cap
        self.max_cap = max_cap
        self.trade_data = trade_data
        
    def clip_prediction(self):
        try:
            self.base['residual^2_prediction'] = self.base['residual^2_prediction'].clip(self.min_cap,self.max_cap)
        except:
            print('No prediction column in base frame, please try again.')
    
    def associate_constant(self, chosen_variance):
        self.base['residual^2'] = (self.base['prediction'] - self.base['individual_speed'])**2
        self.base['residual^2_prediction'] = chosen_variance
        self.final = self.base.merge(self.trade_data, how='left', on=['Id', 'HorseName'])
    
    def linear_fit(self, train_split = 0.8, print_results = True):
        base = self.base
        base['residual^2'] = (base['prediction'] - base['individual_speed'])**2

        new_frame = pd.concat([base,
                                   pd.get_dummies(base["Ran"]).rename(columns=lambda x: 'Ran_' + str(x)),
                                   pd.get_dummies(base["Age"]).rename(columns=lambda x: 'Age_' + str(x)),
                                   pd.get_dummies(base["WeightLBS"]).rename(columns=lambda x: 'WeightLBS' + str(x)),
                                   pd.get_dummies(base["horse_race_no"]).rename(
                                       columns=lambda x: 'Race_no_' + str(x)),
                                   pd.get_dummies(base["Going"]),
                                   pd.get_dummies(base["Course"]),
                                   pd.get_dummies(base["Type"]),
                                   pd.get_dummies(base["Jockey"])
                                   ], axis=1).drop(
                          columns=['Id', 'RaceDate', 'RaceTime', 'Seconds', 'Course', 'Going', 'TotalBtn', 'win_speed', 'individual_speed',
                         'Ran', 'HorseName', 'Age', 'Jockey', 'HorseName', 'horse_race_no', 'Type','diff','prediction','WeightLBS'])
        
        new_frame['constant'] = 1

        y = new_frame['residual^2']
        x = new_frame.drop(columns=['residual^2'])

        y_train = y[:int(len(new_frame)*train_split)]
        x_train = x[:int(len(new_frame)*train_split)]
        y_test = y[int(len(new_frame)*train_split):]
        x_test = x[int(len(new_frame)*train_split):]

        x, y = np.array(x_train), np.array(y_train)
        self.results = sm.OLS(y,x).fit()

        base['residual^2_prediction'] = self.results.predict(np.array(new_frame.drop(columns=['residual^2'])))

        self.r2_train = np.round(self.results.rsquared, decimals=4)
        self.r2_test = np.round(r2_score(y_test, self.results.predict(x_test)) ,decimals=4)
        
        if print_results:
            print('Linear-Fit ' + str(self.r2_train) + ' R^2 train ')
            print('Linear-Fit ' + str(self.r2_test) + ' R^2 test ')       

        self.base = base
        self.final = self.base.merge(self.trade_data, how='left', on=['Id', 'HorseName'])
    
    def decision_tree_fit(self, train_split = 0.8, print_results = True, depth = 5):
        base = self.base
        base['residual^2'] = (base['prediction'] - base['individual_speed'])**2

        new_frame = pd.concat([base,
                                   pd.get_dummies(base["Ran"]).rename(columns=lambda x: 'Ran_' + str(x)),
                                   pd.get_dummies(base["Age"]).rename(columns=lambda x: 'Age_' + str(x)),
                                   pd.get_dummies(base["WeightLBS"]).rename(columns=lambda x: 'WeightLBS' + str(x)),
                                   pd.get_dummies(base["horse_race_no"]).rename(
                                       columns=lambda x: 'Race_no_' + str(x)),
                                   pd.get_dummies(base["Going"]),
                                   pd.get_dummies(base["Course"]),
                                   pd.get_dummies(base["Type"]),
                                   pd.get_dummies(base["Jockey"])
                                   ], axis=1).drop(
                          columns=['Id', 'RaceDate', 'RaceTime', 'Seconds', 'Course', 'Going', 'TotalBtn', 'win_speed', 'individual_speed',
                         'Ran', 'HorseName', 'Age', 'Jockey', 'HorseName', 'horse_race_no', 'Type','diff','prediction','WeightLBS'])
        
        new_frame['constant'] = 1

        y = new_frame['residual^2']
        x = new_frame.drop(columns=['residual^2'])

        y_train = y[:int(len(new_frame)*train_split)]
        x_train = x[:int(len(new_frame)*train_split)]
        y_test = y[int(len(new_frame)*train_split):]
        x_test = x[int(len(new_frame)*train_split):]

        x, y = np.array(x_train), np.array(y_train)
        
        regr = DecisionTreeRegressor(max_depth = depth)
        regr.fit(x_train, y_train)
        
        base['residual^2_prediction'] = regr.predict(np.array(new_frame.drop(columns=['residual^2'])))

        self.r2_train = np.round(r2_score(y_train, regr.predict(x_train)), decimals=4)
        self.r2_test = np.round(r2_score(y_test, regr.predict(x_test)), decimals=4)
        
        if print_results:
            print('Decision-Tree-Fit ' + str(self.r2_train) + ' R^2 train ')
            print('Decision-Tree-Fit ' + str(self.r2_test) + ' R^2 test ')       

        self.base = base
        self.final = self.base.merge(self.trade_data, how='left', on=['Id', 'HorseName'])
