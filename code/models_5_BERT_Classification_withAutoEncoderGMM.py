import torch
from torch import nn
from d2l import torch as d2l
import os
import pandas as pd
import numpy as np
from torch import optim
from sklearn.mixture import GaussianMixture
torch.cuda.empty_cache()
from tqdm import tqdm
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
from collections import Counter
import numpy as np
import torch
import torch.utils.data
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable
from torchvision import datasets, transforms
from torchvision.utils import make_grid , save_image
from sklearn import metrics
from transformers import BertTokenizer
from transformers import BertModel
from torch.optim import Adam

def load_data():
    
    df_train_all = pd.read_csv('../dataset/train_gmm.csv')
    df_test_all = pd.read_csv('../dataset/test_gmm.csv')
    df_val_all = pd.read_csv('../dataset/val_gmm.csv')
    
    return df_train_all, df_test_all, df_val_all

# data preprocessing for training model
class Dataset(torch.utils.data.Dataset):

    def __init__(self, df):
        
        tokenizer = BertTokenizer.from_pretrained('bert-base-cased')
        self.labels = [label for label in df['label']]
        self.texts = [tokenizer(text, padding='max_length', max_length = 512, truncation=True,
                                return_tensors="pt") for text in df['full_text']]
    def classes(self):
        return self.labels

    def __len__(self):
        return len(self.labels)

    def get_batch_labels(self, idx):
        return np.array(self.labels[idx])

    def get_batch_texts(self, idx):
        return self.texts[idx]

    def __getitem__(self, idx):
        batch_texts = self.get_batch_texts(idx)
        batch_y = self.get_batch_labels(idx)

        return batch_texts, batch_y

# BERT Classification Model
class BertClassifier(nn.Module):

    def __init__(self, dropout=0.5):

        super(BertClassifier, self).__init__()
        self.bert = BertModel.from_pretrained('bert-base-cased')
        self.dropout = nn.Dropout(dropout)
        self.linear = nn.Linear(768, 2)
        self.relu = nn.ReLU()

    def forward(self, input_id, mask):

        _, pooled_output = self.bert(input_ids= input_id, attention_mask=mask, return_dict=False)
        dropout_output = self.dropout(pooled_output)
        linear_output = self.linear(dropout_output)
        final_output = self.relu(linear_output)

        return final_output

# training
def train(net, df_train, df_valid, learning_rate, epochs, shuffle=False, cat=None):
    # construct dataLoader for train, validation dataset
    train_dataSet, valid_dataSet = Dataset(df_train), Dataset(df_valid)
    train_dataLoader = torch.utils.data.DataLoader(train_dataSet, batch_size=2, shuffle=True)
    valid_dataLoader = torch.utils.data.DataLoader(valid_dataSet, batch_size=2, shuffle=True)

    # use cross entropy as the loss funciton and adam as the optimizator
    criterion = nn.CrossEntropyLoss()
    optimizer = Adam(net.parameters(), lr = learning_rate)

    # use cuda
    net = net.cuda()
    criterion = criterion.cuda()
    device = torch.device("cuda")

    # training 
    for cur_epoch in range(epochs):
        total_acc_train = 0
        total_loss_train = 0

        for train_input, train_label in tqdm(train_dataLoader):
            # use cuda
            cur_train_label = train_label.to(device)
            cur_mask = train_input['attention_mask'].to(device)
            cur_input_id = train_input['input_ids'].squeeze(1).to(device)

            # forward calculation
            cur_output = net(cur_input_id, cur_mask)
            cur_loss = criterion(cur_output, cur_train_label)
            ## loss value
            total_loss_train += cur_loss.item()
            ## accuracy score
            cur_acc = (cur_output.argmax(dim=1)==cur_train_label).sum().item()
            total_acc_train += cur_acc
        
            # back propagation
            net.zero_grad()
            cur_loss.backward()
            optimizer.step()

        if not shuffle:
            # restore the parameters after each iteration in training
            torch.save(net.state_dict(), 'bert_{}.params'.format(cur_epoch))

        # validation
        total_acc_val = 0
        total_pred_pos_val = 0
        total_real_pos_val = 0
        total_pos_val = 0
        total_loss_val = 0

        with torch.no_grad():

            for val_input, val_label in valid_dataLoader:
                # use cuda
                cur_val_label = val_label.to(device)
                cur_mask = val_input['attention_mask'].to(device)
                cur_input_id = val_input['input_ids'].squeeze(1).to(device)

                # forward calculation
                cur_output = net(cur_input_id, cur_mask)
                ## loss value
                cur_loss = criterion(cur_output, cur_val_label)
                total_loss_val += cur_loss.item()
                ## accuracy score
                cur_acc = (cur_output.argmax(dim=1)==cur_val_label).sum().item()
                total_acc_val += cur_acc
                ## number of predicted as positive 
                pred_val_pos = (cur_output.argmax(dim=1) == 1).sum().item()
                total_pred_pos_val += pred_val_pos
                ## number of predicted as positive and the true prediction
                bool1 = cur_output.argmax(dim=1) == cur_val_label
                bool2 = cur_val_label == (torch.ones(len(cur_val_label)).to(device))
                cur_pos = bool1[bool2].sum().item()
                total_real_pos_val += cur_pos
                ## total amount of true positive samples in validation set
                total_pos_val += (cur_val_label==1).sum().item()

            recall_val = total_real_pos_val / total_pos_val
            precision_val = 0
            if total_pred_pos_val != 0:  
                precision_val = total_real_pos_val / total_pred_pos_val
        
        print(
                    f'Epochs: {cur_epoch + 1} | Train Loss: {total_loss_train / len(df_train): .3f} \
                    | Train Accuracy: {total_acc_train / len(df_train): .3f} \
                    | Val Loss: {total_loss_val / len(df_train): .3f} \
                    | Val Accuracy: {total_acc_val / len(df_train): .3f} \
                    | Val Recall: {recall_val: .3f} \
                    | Val Precision: {precision_val: .3f}'
            )


def train_BERT_Classification_With_ClusteringLabel(df_train_all, df_val_all):
    total_gmm_cat = len(df_train_all.gmm_label.unique())

    EPOCHS = 10
    LR = 1e-6

    for cur_cat in range(total_gmm_cat):
        net = BertClassifier()
        df_train = df_train_all[df_train_all.gmm_label==cur_cat].loc[:,['label','full_text']]
        df_val = df_val_all[df_val_all.gmm_label==cur_cat].loc[:,['label','full_text']]
        train(net,df_train,df_val,LR,EPOCHS,cat=cur_cat)
