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

# load and clean dataset
def load_clean_data(realFile, fakeFile):
    df_politiReal = pd.read_csv(realFile)
    df_politiFake = pd.read_csv(fakeFile)

    # data cleaning
    df_politiReal = df_politiReal[~df_politiReal.full_text.isna()]
    df_politiFake = df_politiFake[~df_politiFake.full_text.isna()]
    df_politiFake.full_text = df_politiFake.full_text.str.replace("\n{1,}","").str.replace("(http.+)\s{0,}","").to_list()
    df_politiReal.full_text = df_politiReal.full_text.str.replace("\n{1,}","").str.replace("(http.+)\s{0,}","").to_list()
    df_politiReal['label'] = 0
    df_politiFake['label'] = 1
    
    # combine real data and fake data 
    df_politi = pd.concat([df_politiReal.loc[:,['label','full_text']], df_politiFake.loc[:,['label','full_text']]])
    df_cleaned = df_politi 

    return df_cleaned

# split the dataset into train, test and validation
def split_data(df_cleaned):
    df_negative = df_cleaned[df_cleaned.label==0]
    df_nonNegative = df_cleaned[df_cleaned.label!=0]

    df_train_neg, df_val_neg, df_test_neg = np.split(df_negative.sample(frac=1, random_state=42),
                        [int(.8*len(df_negative)), int(.9*len(df_negative))])
    df_train_nonNeg, df_val_nonNeg, df_test_nonNeg = np.split(df_nonNegative.sample(frac=1, random_state=42),
                        [int(.8*len(df_nonNegative)), int(.9*len(df_nonNegative))])

    df_train = pd.concat([df_train_neg, df_train_nonNeg])
    df_val = pd.concat([df_val_neg, df_val_nonNeg])
    df_test = pd.concat([df_test_neg, df_test_nonNeg])

    return df_train, df_val, df_test

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

def train_BERT_classification(df_train, df_val, epochs=20, LR=1E-6):
    net = BertClassifier()
    train(net , df_train, df_val, LR, epochs)

def train_BERT_classification_withRandomLabeled(df_train, df_val, epochs=20, LR=1E-6):
    net = BertClassifier()
    
    # shuffle the label
    df_train_shuffle = df_train.copy()
    df_val_shuffle = df_val.copy()
    df_train_shuffle['label'] = df_train_shuffle.label.sample(frac=1,replace=False, random_state=43).to_list()
    df_val_shuffle['label'] = df_val_shuffle.label.sample(frac=1,replace=False, random_state=43).to_list() 

    # train BERT Classification model with shuffled label data
    train(net, df_train_shuffle, df_val_shuffle, LR, epochs, shuffle=True)


# evaluate model on test data              
def get_metrics(net, df_test):
    test = Dataset(df_test)

    test_dataloader = torch.utils.data.DataLoader(test, batch_size=2)
    
    test_prob = None
    test_y = None

    net = net.cpu()
    with torch.no_grad():

      for test_input, test_label in tqdm(test_dataloader):

        test_label = test_label
        mask = test_input['attention_mask']
        input_id = test_input['input_ids'].squeeze(1)
        output = net(input_id, mask)

        if test_prob is None:
            test_prob = torch.softmax(input=output.data,dim=1)
        else:
            test_prob = torch.vstack([test_prob,torch.softmax(input=output.data,dim=1)])
        if test_y is None:
            test_y  = test_label
        else:
            test_y = torch.hstack([test_y,test_label])

    pred_y = test_prob.argmax(dim=1)

    fpr, tpr, thresholds = metrics.roc_curve(test_y, test_prob[:,1], pos_label=1)
    # get the classification report
    print(metrics.classification_report(test_y,pred_y))
    # get the ROC plot and AUC score
    auc = metrics.roc_auc_score(test_y, test_prob[:,1])
    plt.plot([0,1],[0,1])
    plt.plot(fpr,tpr,label='AUC:{}'.format(np.round(auc,3)))
    plt.title('ROC: TextCNN')
    plt.xlabel('TPR')
    plt.ylabel('FPR')
    plt.legend()


