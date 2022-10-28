import os
import time
import sys
import math
import functools as ft
import random as rnd
import glob
import itertools
from collections.abc import Iterable
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

from src.pre import clean_tweet

# g(f(x)) -> F([x, f, g...])
# g(f([x1, x2...])) -> FF([x1, x2...], f, g...)
F = lambda z: [*ft.reduce(lambda x, y: map(y, x), [z[:1]] + z[1:])][0]
FF = lambda *z: [*ft.reduce(lambda x, y: map(y, x), z)]
# f(x1, x2...) -> ff1(f)((x1, x2...))
# f((x1, x2...)) -> ff2(f)(x1, x2...)
# f(x1, x2..., y1, y2...) -> ff3(f, (y1, y2...))((x1, x2...))
# f(x1, x2..., y1, y2...) -> ff4(f, (y1, y2...))(x1, x2...)
# f(x1, x2..., y1, y2...) -> ff5(f, (x1, x2...))((y1, y2...))
# f(x1, x2..., y1, y2...) -> ff6(f, (x1, x2...))(y1, y2...)
ff1 = lambda f: lambda x: f(*x)
ff2 = lambda f: lambda *x: f(x)
ff3 = lambda f, x: lambda y: f(*y, *x)
ff4 = lambda f, x: ff2(ff3(f, x))
ff5 = lambda f, x: lambda y: f(*x, *y)
ff6 = lambda f, x: ff2(ff5(f, x))

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model_name = "sentence-transformers/bert-base-nli-mean-tokens"
# model_name = "cardiffnlp/twitter-roberta-base"

time_start = time.time()

train_df = pd.read_csv("../data/nlp1/train.csv", sep = ',')
test_df = pd.read_csv("../data/nlp1/test.csv", sep = ',')
#print(train_df.head())
#print(test_df.head())
train_df.text = train_df.text.apply(clean_tweet)

tokenizer = AutoTokenizer.from_pretrained(model_name)
encode_sentences = lambda x: tokenizer(
    x, return_tensors = "pt",
    padding = True, return_attention_mask = True
)
#print(
#    tokenizer(
#        "test token",
#        return_attention_mask = True,
#        return_tensors = "pt", padding = True,
#    )
#)
#exit()

def collate_batch(batch):
    sentences, targets = list(zip(*batch))
    return encode_sentences(list(sentences)), torch.tensor(targets)
    
class DF2List(torch.utils.data.Dataset):
    def __init__(self, df):
        self.text = df.text.to_list()
        self.targets = df.target.to_list()
    def __getitem__(self, x):
        return self.text[x], self.targets[x]
    def __len__(self):
        return len(self.targets)

trainDF, valDF = tts(
    train_df, test_size = 0.1, random_state = 0, stratify = train_df.target
)
trainDL = data.DataLoader(
    DF2List(trainDF), batch_size = 64, pin_memory = True,
    collate_fn = collate_batch, num_workers = 0
)
valDL = data.DataLoader(
    DF2List(valDF), batch_size = 64, pin_memory = True,
    collate_fn = collate_batch, num_workers = 0
)

#print(trainDF)
#print(len(DF2List(trainDF)))
#print(len(DF2List(trainDF).text))
#exit()

class Pretrained_Model(nn.Module):
    def __init__(self, model_name, num_cla):
        super().__init__()
        self.encoder = AutoModel.from_pretrained(model_name)
        hidden_size = self.encoder.config.hidden_size
        self.classify = nn.Sequential(
            nn.LayerNorm(hidden_size),
            nn.Dropout(0.3),
            nn.Linear(hidden_size, num_cla)
        )
    def forward(self, inputs):
        return self.classify(self.encoder(**inputs)[1])
model = Pretrained_Model(model_name, 2).to(device)
print(device)

with torch.no_grad():
    test = "shit, fuck, junk"
    test_encode = encode_sentences([test]).to(device)
    test_output = model(test_encode)
    print(test_output)

epochs = 5
lossFn = nn.CrossEntropyLoss()
optimizer = optim.AdamW(model.parameters(), betas = (0.99, 0.98), lr = 2E-5)
scheduler = get_linear_schedule_with_warmup(
    optimizer, num_warmup_steps = 0,
    num_training_steps = len(trainDL) * epochs
)

def train(model, dataloader, lossFn, optimizer, scheduler):
    model = model.train()
    losses = []
    correct_predictions = 0
    for sentences, targets in tqdm(dataloader):
#       print(sentences)
#       print(targets)
#       exit()
        input_ids = sentences["input_ids"].to(device)
        attention_mask = sentences["attention_mask"].to(device)
        targets = targets.to(device)
        optimizer.zero_grad()        
        outputs = model(
            dict(input_ids = input_ids,
                    attention_mask = attention_mask)
        )
        train_loss = lossFn(outputs, targets)
        train_loss.backward()
        losses.append(train_loss.item())
        _, preds = torch.max(outputs, dim = 1)
        correct_predictions += torch.sum(preds == targets)
#       nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        scheduler.step()
    return correct_predictions / len(dataloader.dataset), np.mean(losses)

def validate(model, dataloader, lossFn):
    model = model.eval()
    losses = []
    correct_predictions = 0
    with torch.no_grad():
        for sentences, targets in tqdm(dataloader):
            input_ids = sentences["input_ids"].to(device)
            attention_mask = sentences["attention_mask"].to(device)
            targets = targets.to(device)
            outputs = model(
                dict(input_ids = input_ids,
                        attention_mask = attention_mask)
            )
            val_loss = lossFn(outputs, targets)
            losses.append(val_loss.item())
            _, preds = torch.max(outputs, dim = 1)
            correct_predictions += torch.sum(preds == targets)
    return correct_predictions / len(dataloader.dataset), np.mean(losses)

for epoch in range(epochs):
    print(f"Epoch: {epoch + 1} / {epochs}")
    train_accuracy, train_loss = train(
        model, trainDL, lossFn, optimizer, scheduler
    )
    val_accuracy, val_loss = validate(model, valDL, lossFn)
    print(f"Training Loss: {train_loss}\
            | Training Accuracy: {train_accuracy}")
    print(f"Validation Loss: {val_loss}\
            | Validation Accuracy: {val_accuracy}")

print(time.time() - time_start)

