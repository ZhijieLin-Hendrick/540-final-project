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

# data preprocessing for AutoEncoder model: tokenization
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

# AutoEncoder Model
class autoencoder(nn.Module):
    def __init__(self, vocab_size, embed_size,
                 **kwargs):
        super().__init__(**kwargs)
        self.embedding = nn.Embedding(vocab_size, embed_size)
        self.flatten = nn.Flatten()
        self.encoder = nn.Sequential(
            nn.Linear(50000, 1000),
            nn.ReLU(),
            nn.Linear(1000, 500),
            nn.ReLU(),
            nn.Linear(500, 100),
            nn.ReLU()
        )
        self.decoder = nn.Sequential(
            nn.Linear(100, 500),
            nn.ReLU(),
            nn.Linear(500, 1000),
            nn.ReLU(),
            nn.Linear(1000, 50000), 
            nn.ReLU()
        )
    
    def forward(self, inputs):
        embed_output = self.embedding(inputs)
        flatten_output = self.flatten(embed_output)
        encoder_output = self.encoder(flatten_output)
        decoder_output = self.decoder(encoder_output)
        return decoder_output
    
    def encode(self, inputs):
        embed_output = self.embedding(inputs)
        flatten_output = self.flatten(embed_output)
        encoder_output = self.encoder(flatten_output)
        return encoder_output
    
    def get_embeded(self, inputs):
        embed_output = self.embedding(inputs)
        flatten_output = self.flatten(embed_output)
        return flatten_output

def init_train_AutoEncoder(df_train, df_val, df_test,embed_size=100, outprint=True):
    # loading data loader
    train_iter, _, _, vocab = load_data(df_train, df_val, df_test, batch_size=64, num_steps=500, shuffle=False)
    
    # model initialization
    autoencoder_net = autoencoder(len(vocab), embed_size)
    glove_embedding = TokenEmbedding('glove.6b.100d')
    embeds = glove_embedding[vocab.idx_to_token]
    autoencoder_net.embedding.weight.data.copy_(embeds)
    autoencoder_net.embedding.weight.requires_grad = False

    # use cuda
    device = torch.device("cuda")
    autoencoder_net.to(device)
    optimizer = optim.Adam(autoencoder_net.parameters(), lr=1e-3)
    criterion = nn.MSELoss()

    # training model
    epochs = 200
    for epoch in range(epochs):
        loss=0
        for X, y in train_iter:
            X = X.to(device)

            optimizer.zero_grad()
            outputs = autoencoder_net(X)
            embeded_X = autoencoder_net.get_embeded(X)
            train_loss = criterion(outputs, embeded_X)
            
            train_loss.backward()
            optimizer.step()
            
            loss += train_loss.item()

        if outprint:
            print("epoch: {}/{}, loss={:.6f}".format(epoch+1, epochs, loss))

    return autoencoder_net

# represent data with autoencoder_net
def encodeData(iter_, autoencoder_net):
    encoded_X = None
    Y = None
    for x, y in iter_:
        autoencoder_net.cpu()
        x = x
        if encoded_X is None:
            encoded_X = autoencoder_net.encode(x)
        else:
            encoded_X = torch.vstack([encoded_X, autoencoder_net.encode(x)])
        if Y is None:
            Y = y
        else:
            Y = torch.hstack([Y,y])
    encoded_X = encoded_X.cpu().detach().numpy()    
    return encoded_X

def cluster_withAutoEncoder_GMM(df_train, df_val, df_test):
    # loading data loader
    train_iter, val_iter, test_iter, vocab = load_data(df_train, df_val, df_test, batch_size=64, num_steps=500, shuffle=False)
    # get the model
    autoencoder_net = init_train_AutoEncoder(df_train,df_val,df_test,outprint=False)
    # encoding data
    encoded_train = encodeData(train_iter)
    encoded_val = encodeData(val_iter)
    encoded_test = encodeData(test_iter)

    # BIC score for GMM clustering result
    bic = []
    lowest_bic = np.infty
    best_gmm = None

    for i in range(10):
        estimator = GaussianMixture(n_components=i+1)
        estimator.fit(encoded_train)
        bic.append(estimator.bic(encoded_train))
        if bic[-1]<lowest_bic:
            lowest_bic = bic[-1]
            best_gmm = estimator
        
    train_gmm_label = best_gmm.predict(encoded_train)
    val_gmm_label = best_gmm.predict(encoded_val)
    test_gmm_label = best_gmm.predict(encoded_test)

    df_train['gmm_label'] = train_gmm_label
    df_val['gmm_label'] = val_gmm_label
    df_test['gmm_label'] = test_gmm_label

    # save the data
    df_train.to_csv('../dataset/train_gmm.csv',encoding='utf_8_sig')
    df_test.to_csv('../dataset/test_gmm.csv',encoding='utf_8_sig')
    df_val.to_csv('../dataset/val_gmm.csv',encoding='utf_8_sig')