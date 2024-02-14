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
from models.convnet import ConvNet, SimpleHead
from models.model import PreTrainingModel
import streamlit as st
from util.dataset import Data, get_questions
device = "cpu"
# Kiểm tra xem đã load model chưa
@st.cache_resource  
def load_model():
    print("Loading model")
    D_MODEL = 24
    input_dims = {
        "event_name_name": 20, 
        "room_fqid": 20, 
        "text": 599, 
        "fqid": 130
    }
    N_UNIT = [300, 512, 512, 512]
    convnet04 = ConvNet( input_dims = input_dims, d_model=D_MODEL, n_blocks = 11).to(device)
    head04 = SimpleHead(n_units = N_UNIT, n_outputs = 3).to(device)
    model04 = PreTrainingModel(convnet04, head04).to(device)
    model04.load_state_dict(torch.load(f'./weight_models/best_acc_0-4.pth'))
    
    N_UNIT = [1000, 512, 512, 512]
    convnet512 = ConvNet( input_dims = input_dims, d_model=D_MODEL, n_blocks = 11).to(device)
    head512 = SimpleHead(n_units = N_UNIT, n_outputs = 10).to(device)
    model512 = PreTrainingModel(convnet512, head512).to(device)
    model512.load_state_dict(torch.load(f'./weight_models/best_acc_5-12.pth'))
    
    N_UNIT = [1500, 512, 512, 512]
    convnet1322 = ConvNet( input_dims = input_dims, d_model=D_MODEL, n_blocks = 11).to(device)
    head1322 = SimpleHead(n_units = N_UNIT, n_outputs = 5).to(device)
    model1322 = PreTrainingModel(convnet1322, head1322).to(device)
    model1322.load_state_dict(torch.load(f'./weight_models/best_acc_13-22.pth'))
    
    return [model04.eval(), model512.eval(), model1322.eval()]
level = ['0-4', '5-12', '13-22']
max_lengths = {
    '0-4':300, 
    '5-12':1000, 
    '13-22':1500
}
models = load_model()

# Thêm tiện ích để tải tệp tin
uploaded_file = st.file_uploader('Tải lên tệp tin dữ liệu của session bạn muốn dự đoán: ', type=['csv'])
thres = 0.6
submit_button = st.button('Submit')
if submit_button:
    result = pd.DataFrame(columns=['session_id', 'q1', 'q2', 'q3', 'q4', 'q5', 'q6', 'q7', 'q8', 'q9', 'q10', 'q11', 'q12', 'q13', 'q14', 'q15', 'q16', 'q17', 'q18'])
    data = pd.read_csv(uploaded_file)
    
    for i in range(3):
        sample = data[data.level_group == level[i]]
        dataset = Data(level=level[i], data=sample, max_length=max_lengths[level[i]])

        sessions = sample.session_id.unique()
        questions = get_questions(level[i])
        
        for id, se in enumerate(sessions):
            result.at[id, 'session_id'] = str(se)  # Use .at to set values in a DataFrame
            batch = dataset.__getitem__(se)
            
            for key in batch.keys():
                batch[key] = batch[key].to(dtype=torch.float32, device=device)
                
            output = models[i](batch)
            output = output[0].view(1, -1).tolist()[0]
            
            for k, q in enumerate(questions):
                result.at[id, q] = int(output[k] >= thres)

    st.write(result)
    csv_file = result.to_csv(index=False).encode('utf-8')
    st.download_button(label="Tải xuống kết quả", data=csv_file, file_name='result.csv', mime='text/csv')