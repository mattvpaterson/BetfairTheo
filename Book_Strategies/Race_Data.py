import json
import datetime
import dateutil.parser
import pandas as pd
import numpy as np

class Race_Data:

    #This class takes .txt packet data, cleans and outputs/saves into a manageable pandas csv form
    
    def __init__(self, path, period, length, save_path, race_no):
        self.path = path 
        self.period = period #book storage frequency in ms
        self.length = length #minutes before billed inplay that we compile from
        self.save_path = save_path #where to save the filtered compilations
        self.race_no = race_no
                    
        #Converting JSON file to readable list of updates 
        with open(self.path) as f:
            self.unpack = [json.loads(jsonObj) for jsonObj in f]

        #Compiling market changes from unpacked list
        self.updates_list = [i for i in self.unpack if "marketDefinition" in i['mc'][0]]
        
    
    def prepare(self):
        #Finding the time market first goes in play
        self.actual_inplay = min([i['pt'] for i in self.updates_list if i['mc'][0]['marketDefinition']['inPlay']])
    
        #Finding billed start time
        t = self.updates_list[0]['mc'][0]['marketDefinition']['marketTime']
        self.billed_inplay = dateutil.parser.parse(t).timestamp()*1000 #convert to ms

        self.storage_dict = {str(i['id']) : { 'packets':[] } for i in self.updates_list[0]['mc'][0]['marketDefinition']['runners']}
        self.runners = list(self.storage_dict.keys())
        for i in self.unpack:
            for k in i['mc'][0]['rc']:
                self.storage_dict[str(k['id'])]['packets'].append({**k,**{'pt':i['pt']}}) # adding packet time into the update dictionary
    
    def compile_runner(self,runner):
        #Dictionaries to represent current book for each side & cumulative represenatations
        cBack={'book':{}}#,'trd':0}
        cLay={'book':{}}#,'trd':0}
        backs={}
        lays={}
        
        for i in self.storage_dict[runner]['packets']:
            if i['pt'] > self.billed_inplay:
                break
    
            if 'atb' in i:
                for k in i['atb']:
                    cBack['book'][k[0]] = k[1]
                    cBack['book'] = cBack['book'].copy()

            if 'atl' in i:
                for k in i['atl']:
                    cLay['book'][k[0]]=k[1]
                    cLay['book']=cLay['book'].copy()
    

            backs[i['pt']] = cBack.copy()
            lays[i['pt']] = cLay.copy()
            
        tableBack = {key:Race_Data.bblyr(value['book']) for key,value in backs.items() if len(Race_Data.bblyr(value['book'])) == 8}
        tableLay = {key:Race_Data.balyr(value['book']) for key,value in lays.items() if len(Race_Data.balyr(value['book'])) == 8}

        
        lays_frame = pd.DataFrame.from_dict(tableLay).transpose()
        lays_frame = lays_frame.reset_index()
        lays_frame = lays_frame.rename(columns={"index":"pt",0:'BL',1:'BL+1',2:'BL+2',3:'BL+3',4:'QL0',5:'QL1',6:'QL2',7:'QL3'})
        lays_frame['type'] = 'lays'

        backs_frame = pd.DataFrame.from_dict(tableBack).transpose()
        backs_frame = backs_frame.reset_index()
        backs_frame = backs_frame.rename(columns={"index":"pt",0:'BB-3',1:'BB-2',2:'BB-1',3:'BB',4:'QB3',5:'QB2',6:'QB1',7:'QB0'})
        backs_frame['type'] = 'backs'

        joined_df = lays_frame.merge(backs_frame, how='outer').sort_values(by=['pt']).fillna(method='ffill').dropna()
        
        #### making empty frame every period to forward fill into
        rng = range(int(self.billed_inplay)-self.length*60*1000,int(self.billed_inplay),self.period)
        fill_frame = pd.DataFrame(list(rng), columns=['pt'])
        for col_name in joined_df.columns.drop(['type','pt']):
            fill_frame[col_name] = np.nan
        fill_frame['type'] = 'filler'
        
        final_frame = pd.concat([fill_frame,joined_df]).sort_values(by=['pt']).fillna(method='ffill')
        final_frame = final_frame[final_frame['type'] == 'filler']
        del final_frame['type']
        final_frame['pt'] = (final_frame['pt']-self.billed_inplay)/1000
        
        return final_frame
    
    def compile_all(self):
        self.final_books = {runner: self.compile_runner(runner) for runner in self.runners}
        
        #filtering out horses where there are no lays in the book at reference startpoint
        self.runners = [runner for runner in self.runners if 'BL' in self.final_books[runner].columns]
        
        #filtering only favourite and runners with best_lay at reference startpoint < 5 odds
        self.final_runners = [runner for runner in self.runners if (self.final_books[runner]['BL'].values[0]<5) or (self.final_books[runner]['BL'].values[0] == min([self.final_books[runner]['BL'].values[0] for runner in self.runners]))]
        self.results = {runner: self.final_books[runner] for runner in self.final_runners}
        
    def save(self):
        for runner in self.final_runners:
            self.results[runner].to_csv(self.save_path + r'/' + self.race_no + '-' + runner + '.csv')
            
    @staticmethod
    def bblyr(d):
        updt = {key:value for key,value in d.items() if value != 0}
        return sorted(updt)[-4:] + [updt[i] for i in sorted(updt)[-4:]]
        #returns ['BB-3','BB-2','BB-1','BB','QB3','QB2','QB1','QB0']
    
    @staticmethod     
    def balyr(d):
        updt = {key:value for key,value in d.items() if value != 0}
        return sorted(updt)[:4] + [updt[i] for i in sorted(updt)[:4]]
        #returns ['BL','BL+1','BL+2','BL+3','QL0','QL1','QL2','QL3']

