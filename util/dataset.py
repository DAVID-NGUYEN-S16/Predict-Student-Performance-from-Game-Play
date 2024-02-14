import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import json
import pandas as pd
import torch.nn as nn
from torch.optim import AdamW
import torch
from tqdm.notebook import tqdm
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report, ConfusionMatrixDisplay, confusion_matrix, roc_curve, auc
import numpy as np
def get_questions(level_group):
    return (['q1', 'q2', 'q3'] if level_group == '0-4' 
        else ['q4', 'q5', 'q6', 'q7', 'q8', 'q9', 'q10', 'q11', 'q12', 'q13'] if level_group == '5-12' 
        else ['q14', 'q15', 'q16', 'q17', 'q18'])
class Data:
    def __init__(self, level, data,  max_length = 600):
        
        self.level = level
        self.data = data.reset_index()
        self.question = get_questions(level)
        self.max_length = max_length
        with open('./util/encode_data.json', 'r') as file:
            self.encode_data = json.load(file)
        self.data['event_name_name'] = self.data['event_name'] + " " + self.data['name']
        self.data.text   = self.data.text.fillna(" ")
        self.data.fqid   = self.data.fqid.fillna(" ")
        self.data.text_fqid   = self.data.text_fqid.fillna(" ")
        self.data.elapsed_time   = self.data.elapsed_time.fillna(0.)
        for i in range(len(self.data['event_name_name'])):
            try:
                
                self.data.loc[i, 'event_name_name'] = int(self.encode_data['event_name_name'][self.data.loc[i, 'event_name_name']])
            except:
                self.encode_data['event_name_name'][self.data.loc[i, 'event_name_name']] = len(self.encode_data['event_name_name'])
                self.data.loc[i, 'event_name_name'] = int(self.encode_data['event_name_name'][self.data.loc[i, 'event_name_name']])
            
            try:
                
                self.data.loc[i, 'text'] = int(self.encode_data['text'][self.data.loc[i, 'text']])
            except:
                self.encode_data['text'][self.data.loc[i, 'text']] = len(self.encode_data['text'])
                self.data.loc[i, 'text'] = int(self.encode_data['text'][self.data.loc[i, 'text']])
                
            try:
                
                self.data.loc[i, 'fqid'] = int(self.encode_data['fqid'][self.data.loc[i, 'fqid']])
            except:
                self.encode_data['fqid'][self.data.loc[i, 'fqid']] = len(self.encode_data['fqid'])
                self.data.loc[i, 'fqid'] = int(self.encode_data['fqid'][self.data.loc[i, 'fqid']])
                
            try:
                
                self.data.loc[i, 'room_fqid'] = int(self.encode_data['room_fqid'][self.data.loc[i, 'room_fqid']])
            except:
                self.encode_data['room_fqid'][self.data.loc[i, 'room_fqid']] = len(self.encode_data['room_fqid'])
                self.data.loc[i, 'room_fqid'] = int(self.encode_data['room_fqid'][self.data.loc[i, 'room_fqid']])
                
            
    
    def __getitem__(self, session):
        # print(len(self.data), session)
        sample = self.data[self.data.session_id == session]
        
        time = list(sample['elapsed_time'])
        data = {
            "event_name_name": list(sample['event_name_name']),
            "room_fqid": list(sample['room_fqid']),
            "text": list(sample['text']),
            "fqid": list(sample['fqid']),
            "duration": np.max(time) - np.min(time),
        }
        
        if len(data['event_name_name']) > self.max_length:
            for i in ['event_name_name', 'room_fqid', 'text', 'fqid']:
                data[i] = data[i][-self.max_length: ]
        
        elif len(data['event_name_name']) < self.max_length:
            # missing data
            nb_miss = self.max_length -  len(data['event_name_name']) 
            ls_miss = [-1 for i in range(nb_miss)]

            for i in ['event_name_name', 'room_fqid', 'text', 'fqid']:
                data[i] = ls_miss + data[i]
    
        for i in range(len(data['event_name_name'])):
            data['event_name_name'][i] +=1
            data['room_fqid'][i] +=1
            data['text'][i] +=1
            data['fqid'][i] +=1
        return {
                'event_name_name': torch.tensor([data['event_name_name']]) ,
                'room_fqid': torch.tensor([data['room_fqid']]) , 
                'text': torch.tensor([data['text']]) , 
                'fqid': torch.tensor([data['fqid']]) , 
                'duration': torch.tensor([data['duration']]), 
                "session_id": torch.tensor(session)
        }