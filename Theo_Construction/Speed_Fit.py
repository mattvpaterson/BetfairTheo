import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
import statsmodels.api as sm
from sklearn.model_selection import train_test_split
import datetime as datetime
from sklearn.metrics import r2_score
import xgboost as xgb
import time

class Speed_Fit:

    def __init__(self, base_frame):
        self.original = base_frame[['Id', 'Course', 'RaceDate', 'RaceTime', 'Yards', 'Going', 'Seconds',
       'Ran', 'TotalBtn', 'HorseName', 'Age', 'WeightLBS', 'Jockey', 'Type',
       'horse_race_no', 'win_speed', 'individual_speed']]
        self.trade_data = base_frame[['Id','HorseName','PPWAP', 'PPMAX',
       'PPMIN', 'PPTRADEDVOL', 'WIN_LOSE','BSP']]

    def linear_fit(self,train_split = 0.8, print_results = True):
        # Fitting for communal race features, to determine how much of the R^2 is not horse-specific
        self.race_base = self.original
        
        new_frame = pd.concat([self.race_base,
                               pd.get_dummies(self.race_base["Going"]),
                               pd.get_dummies(self.race_base["Course"]),
                               pd.get_dummies(self.race_base["Type"]),
                               ], axis=1).drop(
            columns=['Id', 'RaceDate', 'RaceTime', 'Seconds', 'Course', 'Going', 'TotalBtn', 'win_speed',
                     'Ran', 'HorseName', 'Age', 'Jockey', 'HorseName', 'horse_race_no', 'Yards', 'WeightLBS','Type'])
        new_frame['constant'] = 1
        
        y = new_frame['individual_speed']
        x = new_frame.drop(columns=['individual_speed'])
       
        y_train = y[:int(len(new_frame)*train_split)]
        x_train = x[:int(len(new_frame)*train_split)]
        y_test = y[int(len(new_frame)*train_split):]
        x_test = x[int(len(new_frame)*train_split):]
            
        x, y = np.array(x_train), np.array(y_train)
        self.race_results = sm.OLS(y, x).fit()

        self.r2_general = np.round(self.race_results.rsquared, decimals=4)
        self.r2_general_test = np.round(r2_score(y_test, self.race_results.predict(x_test)) ,decimals=4)
                    
        # Fitting for all features, including horse-specific
        # Iterating to propagate predictions for calculation of difference vs previous prediction feature
        
        base = self.original
        for i in range(0, 5):
            self.final = base

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

            
            y1 = new_frame['individual_speed']
            x1 = new_frame.drop(columns=['individual_speed'])

            y1_train = y1[:int(len(new_frame)*train_split)]
            x1_train = x1[:int(len(new_frame)*train_split)]
            y1_test = y1[int(len(new_frame)*train_split):]
            x1_test = x1[int(len(new_frame)*train_split):]
            
            x2, y2 = np.array(x1_train), np.array(y1_train)
            self.results = sm.OLS(y2, x2).fit()

            base['prediction'] = self.results.predict(np.array(new_frame.drop(columns=['individual_speed'])))
            base['diff'] = base['individual_speed'] - base['prediction']
            base['diff'] = base['diff'].fillna(0)
            base['diff'] = base.groupby('HorseName')['diff'].shift()
            base['diff'] = base['diff'].fillna(0)
            del base['prediction']

        base['prediction'] = self.results.predict(np.array(new_frame.drop(columns=['individual_speed'])))
        
        self.base = base
        
        self.r2_all = np.round(self.results.rsquared, decimals=4)
        self.r2_all_test = np.round(r2_score(y1_test, self.results.predict(x1_test)) ,decimals=4)
        self.r2_diff = np.round(self.results.rsquared - self.race_results.rsquared, decimals=4)

        if print_results:
            print('Linear-Fit ' + str(self.r2_general) + ' R^2 Race General ')
            print('Linear-Fit ' + str(self.r2_general_test) + ' R^2 Race General test ')       
            print('----------')
            print('Linear-Fit ' + str(self.r2_all) + ' R^2 All')
            print('Linear-Fit ' + str(self.r2_all_test) + ' R^2 All test')
            print('----------')
            print('Linear-Fit ' + str(self.r2_diff) + ' Horse Specific R^2 diff')
        
        self.final = self.base.merge(self.trade_data, how = 'left', on = ['Id','HorseName'])

    def xgboost_fit(self,train_split = 0.8, print_results = True):
        # Fitting for communal race features, to determine how much of the R^2 is not horse-specific
        self.race_base = self.original
        
        new_frame = pd.concat([self.race_base,
                               pd.get_dummies(self.race_base["Going"]),
                               pd.get_dummies(self.race_base["Course"]),
                               pd.get_dummies(self.race_base["Type"]),
                               ], axis=1).drop(
            columns=['Id', 'RaceDate', 'RaceTime', 'Seconds', 'Course', 'Going', 'TotalBtn', 'win_speed',
                     'Ran', 'HorseName', 'Age', 'Jockey', 'HorseName', 'horse_race_no', 'Yards', 'WeightLBS','Type'])
        new_frame['constant'] = 1
        
        y = new_frame['individual_speed']
        x = new_frame.drop(columns=['individual_speed'])
       
        y_train = y[:int(len(new_frame)*train_split)]
        x_train = x[:int(len(new_frame)*train_split)]
        y_test = y[int(len(new_frame)*train_split):]
        x_test = x[int(len(new_frame)*train_split):]
            
        x, y = np.array(x_train), np.array(y_train)

        xg_reg = xgb.XGBRegressor(objective ='reg:squarederror', colsample_bytree = 0.1, learning_rate = 0.2,
                max_depth = 5, n_estimators = 100)
        xg_reg.fit(x,y)
        
        self.r2_general = np.round(r2_score(y_train, xg_reg.predict(x_train)) ,decimals=4)
        self.r2_general_test = np.round(r2_score(y_test, xg_reg.predict(x_test)) ,decimals=4)
                    
        # Fitting for all features, including horse-specific
        # Iterating to propagate predictions for calculation of difference vs previous prediction feature
        
        base = self.original
    
        for i in range(0, 5):
            self.final = base

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
            
            y1 = new_frame['individual_speed']
            x1 = new_frame.drop(columns=['individual_speed'])

            y1_train = y1[:int(len(new_frame)*train_split)]
            x1_train = x1[:int(len(new_frame)*train_split)]
            y1_test = y1[int(len(new_frame)*train_split):]
            x1_test = x1[int(len(new_frame)*train_split):]
            
            x2, y2 = np.array(x1_train), np.array(y1_train)

            xg_reg = xgb.XGBRegressor(objective ='reg:squarederror', colsample_bytree = .1 ,learning_rate = 0.5,
                max_depth = 5, n_estimators = 50)
            xg_reg.fit(x2,y2)
            
            
            base['prediction'] = xg_reg.predict(np.array(new_frame.drop(columns=['individual_speed'])))
            base['diff'] = base['individual_speed'] - base['prediction']
            base['diff'] = base['diff'].fillna(0)
            base['diff'] = base.groupby('HorseName')['diff'].shift()
            base['diff'] = base['diff'].fillna(0)
            del base['prediction']

        base['prediction'] = xg_reg.predict(np.array(new_frame.drop(columns=['individual_speed'])))
        
        self.base = base

        self.r2_all = np.round(r2_score(y1_train, xg_reg.predict(x1_train)) ,decimals=4)
        self.r2_all_test = np.round(r2_score(y1_test, xg_reg.predict(x1_test)) ,decimals=4)
        self.r2_diff = np.round(self.r2_all_test - self.r2_general_test , decimals=4)

        if print_results:
            print('xgboost ' + str(self.r2_general) + ' R^2 Race General ')
            print('xgboost ' + str(self.r2_general_test) + ' R^2 Race General test ')       
            print('----------')
            print('xgboost ' + str(self.r2_all) + ' R^2 All')
            print('xgboost ' + str(self.r2_all_test) + ' R^2 All test')
            print('----------')
            print('xgboost ' + str(self.r2_diff) + ' Horse Specific R^2 diff')
        
        self.final = self.base.merge(self.trade_data, how='left', on=['Id', 'HorseName'])


    def decision_tree_fit(self, train_split = 0.8, depth = 5, all_depth = 6, print_results = True):
        # Fitting for communal race features, to determine how much of the R^2 is not horse-specific
        self.race_base = self.original
        
        new_frame = pd.concat([self.race_base,
                               pd.get_dummies(self.race_base["Going"]),
                               pd.get_dummies(self.race_base["Course"]),
                               pd.get_dummies(self.race_base["Type"]),
                               ], axis=1).drop(
            columns=['Id', 'RaceDate', 'RaceTime', 'Seconds', 'Course', 'Going', 'TotalBtn', 'win_speed',
                     'Ran', 'HorseName', 'Age', 'Jockey', 'HorseName', 'horse_race_no', 'Yards', 'WeightLBS','Type'])
        new_frame['constant'] = 1
        
        y = new_frame['individual_speed']
        x = new_frame.drop(columns=['individual_speed'])
       
        y_train = y[:int(len(new_frame)*train_split)]
        x_train = x[:int(len(new_frame)*train_split)]
        y_test = y[int(len(new_frame)*train_split):]
        x_test = x[int(len(new_frame)*train_split):]
            
        x, y = np.array(x_train), np.array(y_train)
                
        # Fit regression model
        regr_1 = DecisionTreeRegressor(max_depth = depth)
        regr_1.fit(x, y)

        y_pred_train = regr_1.predict(x_train)
        y_pred_test = regr_1.predict(x_test)

        self.r2_general = np.round(r2_score(y_train, y_pred_train) ,decimals=4)
        self.r2_general_test = np.round(r2_score(y_test, y_pred_test) ,decimals=4)
             
        # Fitting for all features, including horse-specific
        # Iterating to propagate predictions for calculation of difference vs previous prediction feature
        
        base = self.original
        for i in range(0, 5):
            self.final = base

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

            
            y1 = new_frame['individual_speed']
            x1 = new_frame.drop(columns=['individual_speed'])

            y1_train = y1[:int(len(new_frame)*train_split)]
            x1_train = x1[:int(len(new_frame)*train_split)]
            y1_test = y1[int(len(new_frame)*train_split):]
            x1_test = x1[int(len(new_frame)*train_split):]
            
            x2, y2 = np.array(x1_train), np.array(y1_train)
            
            # Fit regression model
            regr_2 = DecisionTreeRegressor(max_depth = all_depth)
            regr_2.fit(x2, y2)

            y1_pred_train = regr_2.predict(x1_train)
            y1_pred_test = regr_2.predict(x1_test)

            base['prediction'] = regr_2.predict(np.array(new_frame.drop(columns=['individual_speed'])))
                        
            base['diff'] = base['individual_speed'] - base['prediction']
            base['diff'] = base['diff'].fillna(0)
            base['diff'] = base.groupby('HorseName')['diff'].shift()
            base['diff'] = base['diff'].fillna(0)
            del base['prediction']

        base['prediction'] = regr_2.predict(np.array(new_frame.drop(columns=['individual_speed'])))
        
        self.base = base

        self.r2_all = np.round(r2_score(y1_train, y1_pred_train) ,decimals=4)
        self.r2_all_test = np.round(r2_score(y1_test, y1_pred_test) ,decimals=4)
        self.r2_diff = np.round(r2_score(y1_train, y1_pred_train) - r2_score(y_train, y_pred_train), decimals=4)

        if print_results:
            print('Decision-Tree-Fit ' + str(self.r2_general) + ' R^2 Race General ')
            print('Decision-Tree-Fit ' + str(self.r2_general_test) + ' R^2 Race General test ')       
            print('----------')
            print('Decision-Tree-Fit ' + str(self.r2_all) + ' R^2 All')
            print('Decision-Tree-Fit ' + str(self.r2_all_test) + ' R^2 All test')
            print('----------')
            print('Decision-Tree-Fit ' + str(self.r2_diff) + ' Horse Specific R^2 diff')
        
        self.final = self.base.merge(self.trade_data, how='left', on=['Id', 'HorseName'])
       

