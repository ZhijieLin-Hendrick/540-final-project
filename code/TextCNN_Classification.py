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

# data preprocessing for TextCNN model: tokenization
def load_data(df_train, df_val, df_test, batch_size, num_steps=500, shuffle=False):
    
    train_data = (df_train.full_text.to_list(), df_train.label.to_list())
    val_data = (df_val.full_text.to_list(), df_val.label.to_list())
    test_data = (df_test.full_text.to_list(), df_test.label.to_list())
    if shuffle:
        train_data = (df_train.full_text.to_list(), df_train.label.sample(frac=1,replace=False).to_list())
        val_data = (df_val.full_text.to_list(), df_val.label.sample(frac=1,replace=False).to_list())
        test_data = (df_test.full_text.to_list(), df_test.label.sample(frac=1,replace=False).to_list())

    
    train_tokens = d2l.tokenize(train_data[0], token='word')
    val_tokens = d2l.tokenize(val_data[0], token='word')
    test_tokens = d2l.tokenize(test_data[0], token='word')
    
    vocab = d2l.Vocab(train_tokens, min_freq=1)
    train_features = torch.tensor([d2l.truncate_pad(
        vocab[line], num_steps, vocab['<pad>']) for line in train_tokens])
    val_features = torch.tensor([d2l.truncate_pad(
        vocab[line], num_steps, vocab['<pad>']) for line in val_tokens])
    test_features = torch.tensor([d2l.truncate_pad(
        vocab[line], num_steps, vocab['<pad>']) for line in test_tokens])
    
    train_iter = d2l.load_array((train_features, torch.tensor(train_data[1])),
                                batch_size)
    val_iter = d2l.load_array((val_features, torch.tensor(val_data[1])),
                                batch_size)
    test_iter = d2l.load_array((test_features, torch.tensor(test_data[1])),
                               batch_size,
                               is_train=False)
    
    return train_iter, val_iter, test_iter, vocab

# embedding the tokenized data
class TokenEmbedding:

    def __init__(self, embedding_name):
        self.idx_to_token, self.idx_to_vec = self._load_embedding(
            embedding_name)
        self.unknown_idx = 0
        self.token_to_idx = {token: idx for idx, token in
                             enumerate(self.idx_to_token)}

    def _load_embedding(self, embedding_name):
        idx_to_token, idx_to_vec = ['<unk>'], []
        data_dir = d2l.download_extract(embedding_name)
        with open(os.path.join(data_dir, 'vec.txt'), 'r', encoding='utf-8') as f:
            for line in f:
                elems = line.rstrip().split(' ')
                token, elems = elems[0], [float(elem) for elem in elems[1:]]
                # Skip header information, such as the top row in fastText
                if len(elems) > 1:
                    idx_to_token.append(token)
                    idx_to_vec.append(elems)
        idx_to_vec = [[0] * len(idx_to_vec[0])] + idx_to_vec
        return idx_to_token, torch.tensor(idx_to_vec)

    def __getitem__(self, tokens):
        indices = [self.token_to_idx.get(token, self.unknown_idx)
                   for token in tokens]
        vecs = self.idx_to_vec[torch.tensor(indices)]
        return vecs

    def __len__(self):
        return len(self.idx_to_token)

# TextCNN model
class TextCNN(nn.Module):
    def __init__(self, vocab_size, embed_size, kernel_sizes, num_channels,
                 **kwargs):
        super(TextCNN, self).__init__(**kwargs)
        self.embedding = nn.Embedding(vocab_size, embed_size)
        # The embedding layer not to be trained
        self.constant_embedding = nn.Embedding(vocab_size, embed_size)
        self.dropout = nn.Dropout(0.5)
        self.decoder = nn.Linear(sum(num_channels), 2)
        # The max-over-time pooling layer has no parameters, so this instance
        # can be shared
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.relu = nn.ReLU()
        # Create multiple one-dimensional convolutional layers
        self.convs = nn.ModuleList()
        for c, k in zip(num_channels, kernel_sizes):
            self.convs.append(nn.Conv1d(2 * embed_size, c, k))

    def forward(self, inputs):
        # Concatenate two embedding layer outputs with shape (batch size, no.
        # of tokens, token vector dimension) along vectors
        embeddings = torch.cat(
            (self.embedding(inputs), self.constant_embedding(inputs)), dim=2)
        # Per the input format of one-dimensional convolutional layers,
        # rearrange the tensor so that the second dimension stores channels
        embeddings = embeddings.permute(0, 2, 1)
        # For each one-dimensional convolutional layer, after max-over-time
        # pooling, a tensor of shape (batch size, no. of channels, 1) is
        # obtained. Remove the last dimension and concatenate along channels
        encoding = torch.cat([
            torch.squeeze(self.relu(self.pool(conv(embeddings))), dim=-1)
            for conv in self.convs], dim=1)
        outputs = self.decoder(self.dropout(encoding))
        return outputs

# initializing model
def init_weights(m):
    if type(m) in (nn.Linear, nn.Conv1d):
        nn.init.xavier_uniform_(m.weight)

# training and validating model
def train_valid_TextCNN(df_train, df_val, df_test, get_model=False):
    # load data loader and vocabulary
    train_iter, val_iter, _, vocab = load_data(df_train, df_val, df_test,batch_size=64)

    # model initialization
    embed_size, kernel_sizes, nums_channels = 100, [3, 4, 5], [100, 100, 100]
    devices = d2l.try_all_gpus()
    net = TextCNN(len(vocab), embed_size, kernel_sizes, nums_channels)
    net.apply(init_weights)

    # use pre-trained GloVe as the embedding layer
    glove_embedding = TokenEmbedding('glove.6b.100d')
    embeds = glove_embedding[vocab.idx_to_token]
    net.embedding.weight.data.copy_(embeds)
    net.constant_embedding.weight.data.copy_(embeds)
    net.constant_embedding.weight.requires_grad = False

    # training and testing model
    lr, num_epochs = 0.001, 50
    optimizer = torch.optim.Adam(net.parameters(), lr=lr)
    loss = nn.CrossEntropyLoss(reduction="none")
    d2l.train_ch13(net, train_iter, val_iter, loss, optimizer, num_epochs, devices)
    if get_model:
        return net

def train_valid_TextCNN_withRandomLabeled(df_train, df_val, df_test):
    # load data loader and vocabulary
    train_iter, val_iter, _, vocab = load_data(df_train, df_val, df_test,batch_size=64, shuffle=True)

    # model initialization
    embed_size, kernel_sizes, nums_channels = 100, [3, 4, 5], [100, 100, 100]
    devices = d2l.try_all_gpus()
    net = TextCNN(len(vocab), embed_size, kernel_sizes, nums_channels)
    net.apply(init_weights)

    # use pre-trained GloVe as the embedding layer
    glove_embedding = TokenEmbedding('glove.6b.100d')
    embeds = glove_embedding[vocab.idx_to_token]
    net.embedding.weight.data.copy_(embeds)
    net.constant_embedding.weight.data.copy_(embeds)
    net.constant_embedding.weight.requires_grad = False

    # training and testing model
    lr, num_epochs = 0.001, 50
    optimizer = torch.optim.Adam(net.parameters(), lr=lr)
    loss = nn.CrossEntropyLoss(reduction="none")
    d2l.train_ch13(net, train_iter, val_iter, loss, optimizer, num_epochs, devices)

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


