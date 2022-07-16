import numpy as np
import pandas as pd
import datetime as datetime
import time


class Data:

    def __init__(self, race_data_path):
        self.original = pd.read_csv(race_data_path)[
            ['Id', 'Course', 'RaceDate', 'RaceTime', 'Race', 'Type', 'Class', 'Prize', 'Ran', 'Yards', 'Going',
             'Seconds', 'FPos', 'TotalBtn', 'HorseName', 'Draw', 'Sp', 'Age', 'WeightLBS', 'Aid', 'Trainer', 'Jockey',
             'Allow', 'OR']]

    def filter(self, race_type):
        self.base = self.original
        print(str(len(self.base['Id'].unique())) + ' races before filtering')

        #Picking only GB & IRE races
        target_courses_list = []
        for i in self.base['Course'].unique():
            if Data.is_GB_or_IRE(i):
                target_courses_list.append(i)
        self.base = self.base[self.base['Course'].isin(target_courses_list)]

        #Cleaning of labels
        self.base['Going'] = self.base['Going'].apply(lambda x: Data.remove_brackets(str(x)).strip())
        self.base['HorseName'] = self.base['HorseName'].apply(lambda x: Data.remove_brackets(str(x)).strip())
        self.base['HorseName'] = self.base['HorseName'].str.lower()
        self.base['Type'] = self.base['Type'].replace(np.nan, 'n', regex=True)

        #Remaining filters
        self.base = self.base[self.base['Going'].isin(self.base['Going'].value_counts()[:6].keys())]
        self.base = self.base[~self.base['Race'].str.contains("andicap")]
        self.base = self.base[self.base['Type'].isin(race_type)]


        #Feature/target calculations
        self.base['horse_race_no'] = self.base.groupby(['HorseName'])['Age'].cumcount() + 1
        self.base['win_speed'] = self.base['Yards'] / self.base['Seconds']
        self.base['individual_speed'] = self.base['win_speed'] * (1 - self.base['TotalBtn'] * 2.62 / self.base['Yards'])

        self.base['Jockey'] = self.base['Jockey'].apply(
            lambda x: Data.jockey_selector(5, x, self.base['Jockey'].value_counts().to_dict()))

        self.base = self.base.reset_index()
        self.base = self.base[
            ['Id', 'Course', 'RaceDate', 'RaceTime', 'Yards', 'Going', 'Seconds', 'Ran', 'TotalBtn', 'HorseName', 'Age',
             'WeightLBS', 'Jockey','Type','horse_race_no','win_speed','individual_speed']]


        print(str(len(self.base['Id'].unique())) + ' races after filtering')


    def match(self, files_loc):

        self.base['match'] = self.base.apply(lambda x: Data.race_finder(x['RaceDate'], x['HorseName'],files_loc), axis=1)
        self.base['PPWAP'] = self.base.apply(lambda x: x['match'][0], axis=1)
        self.base['PPMAX'] = self.base.apply(lambda x: x['match'][1], axis=1)
        self.base['PPMIN'] = self.base.apply(lambda x: x['match'][2], axis=1)
        self.base['PPTRADEDVOL'] = self.base.apply(lambda x: x['match'][3], axis=1)
        self.base['WIN_LOSE'] = self.base.apply(lambda x: x['match'][4], axis=1)
        self.base = self.base.dropna()
        del self.base['match']

        print(str(len(self.base['Id'].unique()))+' races after matching')

    @staticmethod
    def remove_brackets(string):
        result = ''
        for i in string:
            if i == '(':
                result = result[:-1]
                break
            result = result + i
        return result

    @staticmethod
    def is_GB_or_IRE(string):
        if 'IRE' in string:
            return True
        elif not '(' in string:
            return True
        else:
            return False

    @staticmethod
    def jockey_selector(threshold,name,jockey_list):
        if jockey_list[name] >= threshold:
            return name
        else:
            return 'Unexperienced'

    @staticmethod
    def race_finder(date,horse_name,files_loc):
        #returns PPWAP PPMAX PPMIN PPTRADEDVOL WIN_LOSE
        path = files_loc + datetime.datetime.strftime(datetime.datetime.strptime(date,'%Y-%m-%d'),'%d%m%Y') + '.csv'
        #Change this path
        try:
            frame = pd.read_csv(path)
            frame['SELECTION_NAME'] = frame['SELECTION_NAME'].str.lower()
            newframe = frame[(frame['SELECTION_NAME'] == horse_name) & (frame['PPWAP']!=1)]
            return [newframe['PPWAP'].values[0],newframe['PPMAX'].values[0],newframe['PPMIN'].values[0],newframe['PPTRADEDVOL'].values[0],newframe['WIN_LOSE'].values[0]]
        except:
            return [np.nan,np.nan,np.nan,np.nan,np.nan]
