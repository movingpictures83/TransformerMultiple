#!/usr/bin/env python3
# -*- coding: UTF-8 -*-
"""
@ Project : ConvTransformerTS
@ FileName: transformer_singlestep.py
@ IDE     : PyCharm
@ Author  : Jimeng Shi
@ Time    : 1/3/22 11:12
"""

import torch
import torch.nn as nn
import numpy as np
import time
import math
import pandas as pd
import sys
from matplotlib import pyplot
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error
from helper import series_to_supervised, stage_series_to_supervised
import PyPluMA
torch.manual_seed(0)
np.random.seed(0)

# This concept is also called teacher forcing.
# The flag decides if the loss will be calculated over all or just the predicted values.
calculate_loss_over_all_values = False

# S is the source sequence length
# T is the target sequence length
# N is the batch size
# E is the feature number

# src = torch.rand((10, 32, 512)) # (S,N,E)
# tgt = torch.rand((20, 32, 512)) # (T,N,E)
# out = transformer_model(src, tgt)
# print(out)

input_window = 84
output_window = 12
batch_size = 512       # batch size
# args = [arg for arg in sys.argv[1:] if not arg.startswith("-")]
# input_window = int(args[0])
# output_window = int(args[1])
# batch_size = int(args[2])
# print("Answer:", int(a)+int(b))

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
# device = ("cpu")


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super(PositionalEncoding, self).__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)

        # pe.requires_grad = False
        self.register_buffer('pe', pe)
    
    def forward(self, x):
        return x + self.pe[:x.size(0), :]


class TransAm(nn.Module):
    def __init__(self, feature_size=16, num_layers=3, dropout=0.1):
        super(TransAm, self).__init__()
        self.model_type = 'Transformer'
        
        self.src_mask = None
        self.pos_encoder = PositionalEncoding(feature_size)
        self.encoder_layer = nn.TransformerEncoderLayer(d_model=feature_size, nhead=16, dropout=dropout)
        self.transformer_encoder = nn.TransformerEncoder(self.encoder_layer, num_layers=num_layers)
        self.decoder = nn.Linear(feature_size, feature_size)
        self.init_weights()
    
    def init_weights(self):
        initrange = 0.1
        self.decoder.bias.data.zero_()
        self.decoder.weight.data.uniform_(-initrange, initrange)
    
    def forward(self, src):
        if self.src_mask is None or self.src_mask.size(0) != len(src):
            device = src.device
            # print('a',src.size())
            mask = self._generate_square_subsequent_mask(len(src)).to(device)
            self.src_mask = mask
        
        src = self.pos_encoder(src)
        # print('j',src.size(),self.src_mask.size())
        output = self.transformer_encoder(src, self.src_mask)  # , self.src_mask)
        
        output = self.decoder(output)
        
        return output
    
    def _generate_square_subsequent_mask(self, sz):
        mask = (torch.triu(torch.ones(sz, sz)) == 1).transpose(0, 1)
        mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
        return mask


def create_inout_sequences(input_data, tw):
    inout_seq = []
    L = len(input_data)
    for i in range(L - tw):
        train_seq = np.append(input_data[i:i + tw, :][:-output_window, :], np.zeros((output_window, 16)), axis=0)
        train_label = input_data[i:i + tw, :]
        inout_seq.append((train_seq, train_label))
    return torch.FloatTensor(np.array(inout_seq))


def get_data(data_, target):
    from sklearn.preprocessing import MinMaxScaler
    scaler = MinMaxScaler(feature_range=(-1, 1))
    # WS_S1 WS_S4 FLOW_S25A GATE_S25A HWS_S25A TWS_S25A FLOW_S25B GATE_S25B HWS_S25B TWS_S25B FLOW_S26 GATE_S26 HWS_S26 TWS_S26 PUMP_S26 mean
    data = data_.loc[:, target:'mean']
    series = data.to_numpy()
    amplitude = scaler.fit_transform(series)
    sampels = int(len(data) * 0.8)
    train_data = amplitude[:sampels]
    test_data = amplitude[sampels:]
    
    train_sequence = create_inout_sequences(train_data, input_window)
    train_sequence = train_sequence[:-output_window]  # todo: fix hack?

    # test_data = torch.FloatTensor(test_data).view(-1)
    test_data = create_inout_sequences(test_data, input_window)
    test_data = test_data[:-output_window]  # todo: fix hack?
    
    return train_sequence.to(device), test_data.to(device), scaler


def get_batch(source, i, batch_size):
    seq_len = min(batch_size, len(source) - 1 - i)
    data = source[i:i + seq_len]
    input = torch.stack(torch.stack([item[0] for item in data]).chunk(input_window, 1)).squeeze()  # 1 is feature size
    target = torch.stack(torch.stack([item[1] for item in data]).chunk(input_window, 1)).squeeze()
    return input, target


## train models
def train_model(train_data, epoch):
    model.train()  # Turn on the train mode
    total_loss = 0.
    start_time = time.time()
    
    for batch, i in enumerate(range(0, len(train_data) - 1, batch_size)):
        data, targets = get_batch(train_data, i, batch_size)
        optimizer.zero_grad()
        output = model(data)
        
        if calculate_loss_over_all_values:
            loss = criterion(output, targets)
        else:
            loss = criterion(output[-output_window:], targets[-output_window:])
        
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 0.5)
        optimizer.step()
        
        total_loss += loss.item()
        log_interval = int(len(train_data) / batch_size / 5)
        if batch % log_interval == 0 and batch > 0:
            cur_loss = total_loss / log_interval
            elapsed = time.time() - start_time
            print('| epoch {:3d} | {:5d}/{:5d} batches | '
                  'lr {:02.6f} | {:5.2f} ms | '
                  'loss {:5.5f} | ppl {:8.2f}'.format(
                epoch, batch, len(train_data) // batch_size, scheduler.get_lr()[0],
                              elapsed * 1000 / log_interval,
                cur_loss, math.exp(cur_loss)))
            total_loss = 0
            start_time = time.time()


## get errors and plot curve
def plot_and_loss(eval_model, data_source, epoch, scaler):
    eval_model.eval()
    total_loss = 0.
    test_result = torch.Tensor(0)
    truth = torch.Tensor(0)
    with torch.no_grad():
        for i in range(0, len(data_source) - 1):
            data, target = get_batch(data_source, i, 1)
            data = data.unsqueeze(1)
            target = target.unsqueeze(1)
            
            # look like the model returns static values for the output window
            output = eval_model(data)
            
            if calculate_loss_over_all_values:
                total_loss += criterion(output, target).item()
            else:
                total_loss += criterion(output[-output_window:], target[-output_window:]).item()
            
            test_result = torch.cat((test_result, output[-1, :].squeeze(1).cpu()), 0)  # todo: check this. -> looks good to me
            truth = torch.cat((truth, target[-1, :].squeeze(1).cpu()), 0)

    # test_result = test_result.cpu().numpy()
    len(test_result)
    
    print(test_result.size(), truth.size())
    test_result = scaler.inverse_transform(test_result.reshape(-1, 1)).reshape(-1)
    truth = scaler.inverse_transform(truth.reshape(-1, 1)).reshape(-1)
    
    pyplot.plot(test_result, color="red")
    pyplot.plot(truth[:500], color="blue")
    pyplot.axhline(y=0, color='k')
    pyplot.xlabel("Periods")
    pyplot.ylabel("Y")
    #     pyplot.savefig('graph/transformer-epoch%d.png'%epoch)
    pyplot.close()
    return total_loss / i


def predict_future(eval_model, data_source, steps, epoch, scaler):
    eval_model.eval()
    total_loss = 0.
    test_result = torch.Tensor(0)
    truth = torch.Tensor(0)
    _, data = get_batch(data_source, 0, 1)
    with torch.no_grad():
        for i in range(0, steps, 1):
            input = torch.clone(data[-input_window:])
            input[-output_window:] = 0
            output = eval_model(data[-input_window:])
            data = torch.cat((data, output[-1:]))
    
    data = data.cpu().view(-1)
    
    data = scaler.inverse_transform(data.reshape(-1, 1)).reshape(-1)
    pyplot.plot(data, color="red")
    pyplot.plot(data[:input_window], color="blue")
    pyplot.grid(True, which='both')
    pyplot.axhline(y=0, color='k')
    #     pyplot.savefig('graph/transformer-future%d.png'%epoch)
    pyplot.close()


def evaluate(eval_model, data_source):
    eval_model.eval()  # Turn on the evaluation mode
    total_loss = 0.
    eval_batch_size = 1000
    with torch.no_grad():
        for i in range(0, len(data_source) - 1, eval_batch_size):
            data, targets = get_batch(data_source, i, eval_batch_size)
            output = eval_model(data)
            print(output[-output_window:].size(), targets[-output_window:].size())
            if calculate_loss_over_all_values:
                total_loss += len(data[0]) * criterion(output, targets).cpu().item()
            else:
                total_loss += len(data[0]) * criterion(output[-output_window:], targets[-output_window:]).cpu().item()
    return total_loss / len(data_source)


def plot(eval_model, data_source, epoch, scaler, stations, indices):
    eval_model.eval()
    total_loss = 0.
    test_result = torch.Tensor(0)
    truth = torch.Tensor(0)
    with torch.no_grad():
        for i in range(0, len(data_source) - 1):
            data, target = get_batch(data_source, i, 1)
            data = data.unsqueeze(1)
            target = target.unsqueeze(1)
            # look like the model returns static values for the output window
            output = eval_model(data)
            if calculate_loss_over_all_values:
                total_loss += criterion(output, target).item()
            else:
                total_loss += criterion(output[-output_window:], target[-output_window:]).item()
            
            test_result = torch.cat((test_result, output[-1, :].squeeze(1).cpu()), 0)  # todo: check this. -> looks good to me
            truth = torch.cat((truth, target[-1, :].squeeze(1).cpu()), 0)
    
    # test_result_ = scaler.inverse_transform(test_result[:700])
    test_result_ = scaler.inverse_transform(test_result)
    truth_ = scaler.inverse_transform(truth)
    print(test_result.shape, truth.shape)
    
    # WS_S1 WS_S4 FLOW_S25A GATE_S25A HWS_S25A TWS_S25A FLOW_S25B GATE_S25B HWS_S25B TWS_S25B FLOW_S26 GATE_S26 HWS_S26 TWS_S26 PUMP_S26 mean
    MAE = []
    for col in indices:#[0, 1, 5, 9, 13]:
    #for col in [0, 1, 5, 9, 13]:
        mae = mean_absolute_error(test_result_[:, col], truth_[:, col])
        MAE.append(mae)
    print("MAE:", MAE)
    
    
    station = stations#['WS_S1', 'WS_S4', 'TWS_S25A', 'TWS_S25B', 'TWS_S26']
    for i, m in enumerate([0, 1, 5, 9, 13]):
        # pyplot.rcParams["figure.figsize"] = (8, 6)
        test_result = test_result_[:700, m]
        truth = truth_[:700, m]
        fig = pyplot.figure(1, figsize=(8, 6))
        fig.patch.set_facecolor('xkcd:white')
        # pyplot.plot([k + 510 for k in range(190)], test_result[510:], color="red")
        pyplot.title('Predicted & Actual Value of {}'.format(station[i]), fontsize=18)
        pyplot.plot(test_result, label='prediction', linewidth=2)
        pyplot.plot(truth, label='truth', linewidth=2)
        pyplot.legend(fontsize=14)
        # ymin, ymax = pyplot.ylim()
        # pyplot.vlines(510, ymin, ymax, color="blue", linestyles="dashed", linewidth=2)
        # pyplot.ylim(ymin, ymax)
        pyplot.xlabel("Time", fontsize=16)
        pyplot.ylabel("Water Stage (ft)", fontsize=16)
        pyplot.xticks(fontsize=14)
        pyplot.yticks(fontsize=14)
        pyplot.show()
        pyplot.close()
    return total_loss / i

model = TransAm().to(device)
criterion = nn.MSELoss()

## import dataset
class TransformerMultiplePlugin:
    def input(self, inputfile):
        self.parameters = dict()
        infile = open(inputfile, 'r')
        for line in infile:
            contents = line.strip().split('\t')
            self.parameters[contents[0]] = contents[1]
            
        dataset = pd.read_csv(PyPluMA.prefix()+"/"+self.parameters["inputfile"], index_col=0)
        dataset.fillna(0, inplace=True)
        self.data_ = dataset[:int(self.parameters["divide"])]

    def run(self):
       stationfile = open(PyPluMA.prefix()+"/"+self.parameters['stations'], 'r')
       stations = []
       for line in stationfile:
          stations.append(line.strip())
       indexfile = open(PyPluMA.prefix()+"/"+self.parameters['indices'], 'r')
       indices = []
       for line in indexfile:
           indices.append(int(line.strip()))
       train_data, val_data, scaler = get_data(self.data_, self.parameters["target"])
       print(train_data.size())
       # print(train_data.size(), val_data.size())
       train, test = get_batch(train_data, 0, batch_size)
       print(train.shape, test.shape)
       train_data, val_data, scaler = get_data(self.data_, self.parameters["target"])

       lr = float(self.parameters["lr"])
       global optimizer
       optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
       global scheduler
       scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 1.0, gamma=0.98)

       self.all_train_loss, self.all_val_loss = [], []
       epochs = int(self.parameters["epochs"])

       for epoch in range(1, epochs + 1):
           epoch_start_time = time.time()
           train_model(train_data, epoch)
           if epoch % 120 == 0:
               val_loss = plot(model, val_data, epoch, scaler, stations, indices)
               # train_loss = plot(model, train_data, epoch, scaler)
               # predict_future(model, val_data,200,epoch,scaler)
           else:
               train_loss = evaluate(model, train_data)
               self.all_train_loss.append(train_loss)

               val_loss = evaluate(model, val_data)
               self.all_val_loss.append(val_loss)
    
           print('-' * 89)
           print('| end of epoch {:3d} | time: {:5.2f}s | train loss {:5.5f} | train ppl {:8.2f}'.format(epoch, (
                   time.time() - epoch_start_time), train_loss, math.exp(train_loss)))
           print('| end of epoch {:3d} | time: {:5.2f}s | valid loss {:5.5f} | valid ppl {:8.2f}'.format(epoch, (
                       time.time() - epoch_start_time), val_loss, math.exp(val_loss)))
           print('-' * 89)
    
           scheduler.step()

    def output(self, filename):
       pyplot.plot(self.all_train_loss, label='train')
       pyplot.plot(self.all_val_loss, label='test')
       pyplot.xlabel('Loss', fontsize=16)
       pyplot.ylabel('Epoch', fontsize=16)
       pyplot.xticks(fontsize=14)
       pyplot.yticks(fontsize=14)
       pyplot.title("Training loss vs Testing loss", fontsize=18)
       pyplot.legend()
       pyplot.savefig(filename)
       pyplot.show()
