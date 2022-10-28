import os
import time
import sys
import math
import functools as ft
import random as rnd
import glob
#import itertools
#from collections.abc import Iterable
from tqdm import tqdm
import h5py
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split as tts
import torch
import torch.nn as nn
import torch.utils.data as data
import torch.nn.functional as torF
import torch.optim as optim
import sklearn.metrics as sklF
import matplotlib.pyplot as plt
import seaborn as sns
import multiprocessing
from transformers import AutoTokenizer, AutoModel
from transformers import get_linear_schedule_with_warmup
from torchtext.vocab import build_vocab_from_iterator as buildV
from torchtext.data import get_tokenizer

from src.pre import clean_tweet

# g(f(x)) -> F([x, f, g...])
# g(f([x1, x2...])) -> FF([x1, x2...], f, g...)
F = lambda z: [*ft.reduce(lambda x, y: map(y, x), [z[:1]] + z[1:])][0]
FF = lambda *z: [*ft.reduce(lambda x, y: map(y, x), z)]
# f(x1, x2..., y1, y2...) -> fxy(f, x1, x2...)(y1, y2...)
# f(x1, x2..., y1, y2...) -> fyx(f, y1, y2...)(x1, x2...)
# f(x1, x2..., y1 = y1, y2 = y2...) -> fYx(f, y1 = y1, y2 = y2...)(x1, x2...)
fxy = lambda f, *x: lambda *y: f(*x, *y)
fyx = lambda f, *x: lambda *y: f(*y, *x)
fYx = lambda f, **x: lambda *y: f(*y, **x)

time_start = time.time()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
train_df = pd.read_csv("../data/nlp1/train.csv", sep = ',')
test_df = pd.read_csv("../data/nlp1/test.csv", sep = ',')
#print(train_df.head())
#print(test_df.head())
#exit()

train_df.text = train_df.text.apply(clean_tweet)
test_df.text = test_df.text.apply(clean_tweet)

tokenizer = get_tokenizer("basic_english")
train_df.text = train_df.text.apply(tokenizer)
test_df.text = test_df.text.apply(tokenizer)

vocab = buildV(train_df.text)
#print(type(vocab))
#print(vocab["test"])
#print(len(vocab))
#print(vocab.get_itos())
#print(vocab.get_stoi())
#exit()

def s2ipad(length):
    def s2i(sentence):
        sentence = torch.tensor([
            vocab[word] if word in vocab else len(vocab)
            for word in sentence
        ])
        return torF.pad(sentence, (0, length - len(sentence)), value = len(vocab))
    return s2i
def collate_batch(batch):
    length = max([len(data[0]) for data in batch])
    out = list(zip(*batch))
    return torch.stack(FF(out[0], s2ipad(length))), torch.tensor(out[1])

length = 0
train_df, val_df = tts(
    train_df, test_size = 0.1, random_state = 0, stratify = train_df.target
)
if length:
    vPos = s2ipad(length)
    train_df.text = train_df.text.apply(vPos)
    val_df.text = val_df.text.apply(vPos)
    test_df.text = test_df.text.apply(vPos)
    collate_fn = None
    trainDF = data.TensorDataset(
        torch.stack(train_df.text.to_list()),
        torch.tensor(train_df.target.to_list())
    )
    valDF = data.TensorDataset(
        torch.stack(val_df.text.to_list()),
        torch.tensor(val_df.target.to_list())
    )
else:
    maxlen = max([len(text) for text in test_df.text.to_list()])
    test_df.text = test_df.text.apply(s2ipad(maxlen))
    collate_fn = collate_batch
    DF2List = lambda x: list(zip(x.text.to_list(), x.target.to_list()))
    trainDF = DF2List(train_df)
    valDF = DF2List(val_df)
testDF = torch.stack(test_df.text.to_list())
trainDL = data.DataLoader(
    trainDF, batch_size = 64, collate_fn = collate_fn,
    pin_memory = True, shuffle = True
)
valDL = data.DataLoader(
    valDF, batch_size = 32, collate_fn = collate_fn,
    pin_memory = True, shuffle = True
)

#trainIter = iter(trainDL)
#print(next(trainIter))
#print(test_df.head())
#exit()
    
class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, num_heads, d_input = None):
        super().__init__()
        self.num_heads = num_heads
        self.d_model = d_model
        if d_input is None:
            d_xq = d_xk = d_xv = d_model
        else:
            d_xq, d_xk, d_xv = d_input
        assert d_model % self.num_heads == 0
        self.d_k = d_model // self.num_heads
        self.W_q = nn.Linear(d_xq, d_model, bias = False)
        self.W_k = nn.Linear(d_xk, d_model, bias = False)
        self.W_v = nn.Linear(d_xv, d_model, bias = False)
        self.W_h = nn.Linear(d_model, d_model)
    def attention(self, Q, K, V):
        # QK^T
        A = torch.matmul(Q, K.transpose(2, 3))
        # (bs, #heads, length, length)
        # Softmax(QK^T/sqrt(dk))
        A = nn.Softmax(dim = -1)(A / np.sqrt(self.d_k))
        # (bs, #heads, length, length)
        # H = AV
        H = torch.matmul(A, V)
        # (bs, #heads, length, dim_per_head)
        return H, A 
    def split_heads(self, x):
        return x.reshape(*x.shape[:2], self.num_heads, self.d_k).transpose(1, 2)
#       return x.reshape(*x.shape[:2], self.num_heads, self.d_k)
    def group_heads(self, x):
        return x.transpose(1, 2).reshape(*x.shape[::2], -1)
#       return x.reshape(x.size(0), x.size(1), -1)
    def forward(self, x_q, x_k, x_v):
        Q = self.split_heads(self.W_q(x_q))
        K = self.split_heads(self.W_k(x_k))
        V = self.split_heads(self.W_v(x_v))
        # (bs, #heads, length, dim_per_head)
        H, A = self.attention(Q, K, V)
        H = F([H, self.group_heads, self.W_h])
        # (bs, length, dim)
        return H, A

#temp_mha = MultiHeadAttention(d_model = 512, num_heads = 8)
#def print_out(Q, K, V):
#    temp_out, temp_attn = temp_mha.scaled_dot_product_attention(Q, K, V)
#    print("Attention weights are:", temp_attn.squeeze())
#    print("Output is:", temp_out.squeeze())
#test_K = torch.tensor(
#    [[10,  0,  0   ],
#     [0,   10, 0   ],
#     [0,   0,  10  ],
#     [0,   0,  10  ]]
#).float()[None, None]
#test_V = torch.tensor(
#    [[1,       0,  0   ],
#     [10,      0,  0   ],
#     [100,     5,  0   ],
#     [1000,    6,  0   ]]
#).float()[None, None]
#test_Q = torch.tensor(
#    [[0, 10, 0]]
#).float()[None, None]
#print_out(test_Q, test_K, test_V)
#test_Q = torch.tensor(
#    [[0, 0, 10], [0, 10, 0], [10, 10, 0]]
#).float()[None, None]
#print_out(test_Q, test_K, test_V)
#exit()

class FFN(nn.Module):
    def __init__(self, d_model, hidden_dim, ker = 0, pad = None):
        super().__init__()
        if ker:
            self.ffL1 = nn.Conv1d(d_model, hidden_dim, ker, padding = pad)
            self.ffL2 = nn.Conv1d(hidden_dim, d_model, ker, padding = pad)
        else:
            self.ffL1 = nn.Linear(d_model, hidden_dim)
            self.ffL2 = nn.Linear(hidden_dim, d_model)
        self.kernel = ker
    def forward(self, x):
        if self.kernel:
            transpose = fyx(torch.transpose, 1, 2)
            network = [transpose, self.ffL1, torF.relu,
                        self.ffL2, transpose]
        else:
            network = [self.ffL1, torF.relu, self.ffL2]
        return F([x] + network)

class EncoderLayer(nn.Module):
    def __init__(self, d_model, num_heads, hidden_dim, p):
        super().__init__()
        self.mha = MultiHeadAttention(d_model, num_heads)
        self.ffn = FFN(d_model, hidden_dim)
#       self.ffn = FFN(d_model, hidden_dim, 5, 2)
#       FFN with kernel and pad means CNN, while w/o means Linear.
        self.layernorm1 = nn.LayerNorm(normalized_shape = d_model, eps = 1E-6)
        self.layernorm2 = nn.LayerNorm(normalized_shape = d_model, eps = 1E-6)
        self.dropout1 = nn.Dropout(p)
        self.dropout2 = nn.Dropout(p)
    def forward(self, x):
        # Multi-head attention 
        attn_output, _ = self.mha(x, x, x)
        attn_output = self.dropout1(attn_output)
        # (batch_size, input_seq_len, d_model)
        # Layer norm after adding the residual connection 
        out1 = self.layernorm1(x + attn_output)
        # (batch_size, input_seq_len, d_model)
        # Feed forward 
        ffn_output = self.ffn(out1)
        ffn_output = self.dropout2(ffn_output)
        # (batch_size, input_seq_len, d_model)
        #Second layer norm after adding residual connection 
        out2 = self.layernorm2(out1 + ffn_output)
        # (batch_size, input_seq_len, d_model)
        return out2

class Embeddings(nn.Module):
    def __init__(self, d_model, vocab_size, max_position_embeddings):
        super().__init__()
        self.word_embeddings = nn.Embedding(
            vocab_size, d_model, padding_idx = vocab_size - 1
        )
        self.position_embeddings = nn.Embedding(
            max_position_embeddings, d_model
        )
        Embeddings.pos(
            max_position_embeddings, d_model, self.position_embeddings.weight
        )
        self.LayerNorm = nn.LayerNorm(d_model, eps = 1E-12)

    @staticmethod
    def pos(size, dim, E):
        theta = np.array([
            [i / np.power(10000, 2 * (j // 2) / dim) for j in range(dim)]
            for i in range(size)
        ])
        with torch.no_grad():
            E[:, 0::2] = torch.from_numpy(np.sin(theta[:, 0::2]))
            E[:, 1::2] = torch.from_numpy(np.cos(theta[:, 1::2]))
        E.detach_()

    def forward(self, x):
        position = torch.arange(
            x.size(1), dtype = torch.long, device = x.device
        ).expand_as(x) # (max_seq_length)
        # (bs, max_seq_length)
        # Get word embeddings for each input id
        word_embeddings = self.word_embeddings(x)
        # (bs, max_seq_length, dim)
        # Get position embeddings for each position id 
        position_embeddings = self.position_embeddings(position)
        # (bs, max_seq_length, dim)
        # Add them both 
        embeddings = word_embeddings + position_embeddings
        # (bs, max_seq_length, dim)
        # Layer norm 
        embeddings = self.LayerNorm(embeddings)
        # (bs, max_seq_length, dim)
        return embeddings

class Encoder(nn.Module):
    def __init__(self, num_layers, d_model, num_heads, ff_hidden_dim, 
                    input_vocab_size, maximum_position_encoding, p):
        super().__init__()
        self.d_model = d_model
        self.num_layers = num_layers
        self.embedding = Embeddings(
            d_model, input_vocab_size, maximum_position_encoding
        )
        layers = []
        for _ in range(num_layers):
            layers.append(EncoderLayer(d_model, num_heads, ff_hidden_dim, p))
            layers.append(nn.Dropout(p))
        self.enc_layers = nn.Sequential(*layers)
    def forward(self, x):
        network = [self.embedding, self.enc_layers]
        # Transform to (batch_size, input_seq_length, d_model)
        return F([x] + network)
        # (batch_size, input_seq_len, d_model)

class TransformerClassifier(nn.Module):
    def __init__(self, num_layers, d_model, num_heads,
                    hidden_dim, input_vocab_size, d_out,
                    maximum_position_encoding = 10000, p = 0):
        super().__init__()
        self.encoder = Encoder(
            num_layers, d_model, num_heads, hidden_dim,
            input_vocab_size, maximum_position_encoding, p
        )
        self.linear = nn.Linear(d_model, d_out)
    def forward(self, x):
        maxdim1 = lambda x: torch.max(x, dim = 1)[0]
        network = [self.encoder, maxdim1, self.linear, torch.sigmoid]
        return F([x] + network)

model = TransformerClassifier(
    num_layers = 5, d_model = 64, num_heads = 8, hidden_dim = 128, 
    input_vocab_size = len(vocab) + 1, d_out = 2,#p = 0.2,
#   maximum_position_encoding = 1000
).to(device)
print(device)

epochs = 50
estop = 10
lossFn = nn.CrossEntropyLoss()
optimizer = optim.AdamW(model.parameters(), lr = 1E-3)
#scheduler = get_linear_schedule_with_warmup(
#    optimizer, num_warmup_steps = 0,
#    num_training_steps = len(trainDL) * epochs
#)
#def train(model, dataloader, lossFn, optimizer, scheduler):
def train(model, dataloader, lossFn, optimizer):
    model = model.train()
    loss = []
    correct = 0
    for text, label in tqdm(dataloader):
        x = text.to(device)
        y = label.to(device)
        optimizer.zero_grad()
        y_pred = model(x)
        train_loss = lossFn(y_pred, y)
        train_loss.backward()
        loss.append(train_loss.item())
        correct += torch.sum(y == torch.max(y_pred, 1)[1])
#       nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
#       scheduler.step()
    return np.mean(loss), correct / len(dataloader.dataset)

def validate(model, dataloader, lossFn):
    model = model.eval()
    loss = []
    correct = 0
    with torch.no_grad():
        for text, label in tqdm(dataloader):
            x = text.to(device)
            y = label.to(device)
            y_pred = model(x)
            val_loss = lossFn(y_pred, y)
            loss.append(val_loss.item())
            correct += torch.sum(y == torch.max(y_pred, 1)[1])
    return np.mean(loss), correct / len(dataloader.dataset),

trainFlag = 1
if trainFlag:
    checkpoint = []
    for epoch in range(epochs):
        print(f"Epoch: {epoch + 1} / {epochs}")
        train_loss, train_accuracy = train(
            model, trainDL, lossFn, optimizer#, scheduler
        )
        val_loss, val_accuracy = validate(model, valDL, lossFn)
        print(f"Training Loss: {train_loss:.5f}\t\t"\
                + f"| Training Accuracy: {train_accuracy:.5f}")
        print(f"Validation Loss: {val_loss:.5f}\t"\
                + f"| Validation Accuracy: {val_accuracy:.5f}")
        if estop:
            if len(checkpoint) and checkpoint[0][1] > val_accuracy:
                if len(checkpoint) == estop or epoch + 1 == epochs:
                    print("Early Stopping at Epoch: %s"
                            % (epoch - len(checkpoint) + 1))
                    print(f"Validation Loss: {checkpoint[0][0]:.5f}\t"\
                            + f"| Validation Accuracy: {checkpoint[0][1]:.5f}")
                    break
                checkpoint.append(None)
            elif epoch + 1 != epochs:
                checkpoint = [[val_loss, val_accuracy]]
                torch.save(model, "best.model")

model = torch.load("best.model", torch.device("cpu")).eval()
y_pred = [torch.max(model(x[None]), 1)[1].item() for x in testDF]
testDF = pd.DataFrame({"id": test_df.id, "target": y_pred})
testDF.to_csv("submission.csv", index = False)

print(time.time() - time_start)

