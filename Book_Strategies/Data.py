import json
import datetime
import dateutil.parser
import pandas as pd
import numpy as np
import os
from sklearn.linear_model import LinearRegression
import statsmodels.api as sm
from sklearn.model_selection import train_test_split


class Data:
    
    #This class takes pandas book frames and fits for returns on specific features and prediction timescales

    def __init__(self, path, prediction_time, features):
        self.path = path
        self.target = prediction_time
        self.features = features
        self.frame = pd.DataFrame()

        for file in os.scandir(self.path):
            book = pd.read_csv(file.path, index_col=0)
            new = Data.prepare(book,self.target)
            self.frame = pd.concat([self.frame,new])
   
    def fit(self):
        self.fit_frame = self.frame[self.features]
        self.fit_frame['constant'] = 1
        y = self.frame['target']
        x = self.fit_frame
        
        x, y = np.array(x), np.array(y)
        self.results = sm.OLS(y, x).fit()
        
        self.fit_frame['prediction'] = self.results.predict(np.array(self.fit_frame))
        self.fit_frame['target'] = self.frame['target']
        
    def signal(self,buy_threshold,sell_threshold):
        self.fit_frame.apply(lambda x: config_flag(x,buy_threshold,sell_threshold))
        
    
    @staticmethod
    def config_flag(prediction,buy_threshold,sell_threshold):
        if prediction > buy_threshold:
            return 1
        elif prediction < sell_threshold:
            return -1
        else:
            return 0

    
    @staticmethod
    def prepare(frame,target):
        frame['mid'] = (frame['BB'] + frame['BL'])/2

        #OBI
        frame['TBI'] = 1/2 - (frame['QL0']/(frame['QL0']+frame['QB0']))
        frame['OBI'] = 1/2 - (frame['QL0']+frame['QL1']+frame['QL2']+frame['QL3'])/((frame['QL0']+frame['QL1']+frame['QL2']+frame['QL3']) + (frame['QB0']+frame['QB1']+frame['QB2']+frame['QB3']))

        #Return features
        frame['return_5s'] = frame['mid']/frame['mid'].shift(int(5/0.2))-1
        frame['return_20s'] = frame['mid']/frame['mid'].shift(int(20/0.2))-1
        frame['return_60s'] = frame['mid']/frame['mid'].shift(int(60/0.2))-1
        frame['return_180s'] = frame['mid']/frame['mid'].shift(int(180/0.2))-1
    
        #Target prediction
        frame['target'] = frame['mid'].shift(-int(target/0.2))/frame['mid']-1


        output = frame.dropna()
        
        return output

    
    

