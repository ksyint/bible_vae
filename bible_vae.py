import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
import torch.nn.functional as F
import pandas as pd
import random
import math

'''
Hyperparameters
'''

lr = 0.001
batch_size = 200
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
input_dim = 1536  # number of sentence imbeding vector dimension

df = pd.read_json(r"/root/shinyee_kang/data/bible_embeddings.jsonl", lines=True)

import numpy as np

embeding_vecs2 = df[['KJV_embedding', 'NET_embedding', 'ASV_embedding', 'ASVS_embedding'
    , 'Coverdale_embedding', 'Geneva_embedding', 'KJV_Strongs_embedding']]

embeding_tensors2 = torch.tensor(np.array(embeding_vecs2.values.tolist())).clone().detach()


class CustomDataset2(torch.utils.data.Dataset):
    def __init__(self, tensors):
        self.kjv_vec = torch.tensor(tensors[:, 0, :], dtype=torch.float32).clone().detach()
        self.net_vec = torch.tensor(tensors[:, 1, :], dtype=torch.float32).clone().detach()
        self.asv_vec = torch.tensor(tensors[:, 2, :], dtype=torch.float32).clone().detach()
        self.asvs_vec = torch.tensor(tensors[:, 3, :], dtype=torch.float32).clone().detach()
        self.cov_vec = torch.tensor(tensors[:, 4, :], dtype=torch.float32).clone().detach()
        self.gen_vec = torch.tensor(tensors[:, 5, :], dtype=torch.float32).clone().detach()
        self.kjvs_vec = torch.tensor(tensors[:, 6, :], dtype=torch.float32).clone().detach()

    def __len__(self):
        return len(self.kjv_vec)

    def __getitem__(self, idx):
        return (self.kjv_vec[idx], self.net_vec[idx], self.asv_vec[idx],
                self.asvs_vec[idx], self.cov_vec[idx], self.gen_vec[idx], self.kjvs_vec[idx])


dataset2 = CustomDataset2(embeding_tensors2)

data_size = len(embeding_tensors2)
train_size = int(data_size * 0.9)
test_size = data_size - train_size

train_loader, test_loader = torch.utils.data.random_split(dataset2, [train_size, test_size], generator=torch.Generator().manual_seed(0))

train_loader = torch.utils.data.DataLoader(dataset=train_loader, batch_size=batch_size, shuffle=True)
test_loader = torch.utils.data.DataLoader(dataset=test_loader, batch_size=batch_size, shuffle=False)

name_ls = ['NET', 'ASV', 'ASVS', 'Coverdale', 'Geneva', 'KJV_Strongs']
color_ls = ['b', 'g', 'r', 'c', 'm', 'y']

def train(feature_dim: int, fc_dim_ls: list, epochs: int = 300, epoch_mul: int = 1) -> bool:
    '''
    Modules : Encoder, Decoder

    from prototype2, use batch norm
    '''

    # Encoder
    class Encoder(nn.Module):
        def __init__(self):
            super(Encoder, self).__init__()
            self.fc_in = nn.Linear(input_dim, fc_dim_ls[0])
            self.fc_out = nn.Linear(fc_dim_ls[-1], feature_dim)

            if len(fc_dim_ls) >= 2:
                self.fc_list = nn.ModuleList(
                    [nn.Linear(fc_dim_ls[i], fc_dim_ls[i + 1]) for i in range(len(fc_dim_ls) - 1)])
            self.bn_list = nn.ModuleList([nn.BatchNorm1d(i) for i in fc_dim_ls])

        def forward(self, x):
            x = x.view(x.size(0), -1)
            x = F.leaky_relu(self.bn_list[0](self.fc_in(x)))
            if len(fc_dim_ls) >= 2:
                for fc_hid, bn in zip(self.fc_list, self.bn_list[1:]):
                    x = F.leaky_relu(bn(fc_hid(x)))
            x = self.fc_out(x)
            return x

    # Decoder
    class Decoder(nn.Module):
        def __init__(self):
            super(Decoder, self).__init__()
            self.fc_out = nn.Linear(fc_dim_ls[0], input_dim)
            self.fc_in = nn.Linear(feature_dim, fc_dim_ls[-1])

            if len(fc_dim_ls) >= 2:
                self.fc_list = nn.ModuleList(
                    [nn.Linear(fc_dim_ls[i], fc_dim_ls[i - 1]) for i in range(len(fc_dim_ls) - 1, 0, -1)])
            self.bn_list = nn.ModuleList([nn.BatchNorm1d(i) for i in fc_dim_ls[::-1]])

        def forward(self, x):
            x = x.view(x.size(0), -1)
            x = F.leaky_relu(self.bn_list[0](self.fc_in(x)))
            if len(fc_dim_ls) >= 2:
                for fc_hid, bn in zip(self.fc_list, self.bn_list[1:]):
                    x = F.leaky_relu(bn(fc_hid(x)))
            x = F.sigmoid(self.fc_out(x)) * 2 - 1
            return x

    '''
    Traing
    '''

    enc = Encoder().to(device)
    dec = Decoder().to(device)

    loss_function = nn.MSELoss()
    optimizer = optim.Adam(list(enc.parameters()) + list(dec.parameters()), lr=lr)

    train_loss_list = []
    test_loss_list = []

    import time
    start = time.time()
    enc.train()
    dec.train()

    for epoch in range(epochs):
        print("{}th epoch starting.".format(epoch))
        for batch, vec in enumerate(train_loader):
            kjv_vec = vec[0]
            asv_vec = vec[2]
            # print(kjv_vec.shape, asv_vec.shape)
            sentence_vec = kjv_vec - asv_vec
            # print(sentence_vec.shape)
            sentence_vec = sentence_vec.to(device)
            z = enc(sentence_vec)
            reconstructed_sentence_vec = dec(z)

            optimizer.zero_grad()
            train_loss = loss_function(sentence_vec, reconstructed_sentence_vec)
            train_loss.backward()
            train_loss_list.append(train_loss.item())

            optimizer.step()

            print(f"[Epoch {epoch:3d}] Processing batch #{batch:3d} reconstruction loss: {train_loss.item():.6f}",
                  end='\r')
        with torch.no_grad():
            test_err_ls = []
            for vec in test_loader:
                kjv_vec = vec[0]
                asv_vec = vec[2]

                sentence_vec = (kjv_vec - asv_vec).to(device)
                z = enc(sentence_vec)
                reconstructed_sentence_vec = dec(z)

                optimizer.zero_grad()
                test_err_ls.append(loss_function(sentence_vec, reconstructed_sentence_vec))
                optimizer.zero_grad()
            epoch_train_loss = sum(test_err_ls) / len(test_err_ls)
            test_loss_list.append(epoch_train_loss)
            print(f"{epoch}th Epoch Complete, Test set err(avg) :\t {epoch_train_loss}")
    end = time.time()
    print("Time ellapsed in training is: {}".format(end - start))

    '''
    Show & Save result (Plot)
    '''

    from matplotlib import pyplot as plt
    plt.plot(train_loss_list[1000:], label="train loss")
    plt.semilogy()
    plt.legend()
    plt.savefig(f'train (trunc.){fc_dim_ls}, {feature_dim}, {epochs * epoch_mul}, kjv-asv.png')
    plt.close()

    from matplotlib import pyplot as plt
    plt.plot(train_loss_list, label="train loss")
    plt.semilogy()
    plt.legend()
    plt.savefig(f'train {fc_dim_ls}, {feature_dim}, {epochs * epoch_mul}, kjv-asv.png')
    plt.close()

    temp_ls = []
    for i in test_loss_list:
        temp_ls.append(i.item())
    plt.plot(temp_ls, 'r-', label="test loss")
    plt.legend()
    plt.savefig(f'test {fc_dim_ls}, {feature_dim}, {epochs * epoch_mul}, kjv-asv.png')
    plt.close()

    '''
    Save Results (Params)
    '''

    import json
    with open(f'train_loss {fc_dim_ls}, {feature_dim}, epochs={epochs * epoch_mul}, kjv-asv.json', 'w') as f:
        json.dump(train_loss_list, f)

    temp_ls = []
    for i in test_loss_list:
        temp_ls.append(i.item())
    with open(f'test_loss {fc_dim_ls}, {feature_dim}, epochs={epochs * epoch_mul}, kjv-asv.json', 'w') as f:
        json.dump(temp_ls, f)

    torch.save(enc.state_dict(),
               r'/root/shinyee_kang/' + f'Enc params {fc_dim_ls}, {feature_dim}, epochs={epochs * epoch_mul}, kjv-asv')
    torch.save(dec.state_dict(),
               r'/root/shinyee_kang/' + f'Dec params {fc_dim_ls}, {feature_dim}, epochs={epochs * epoch_mul}, kjv-asv')

    anomaly_test(enc, dec, feature_dim, fc_dim_ls, epochs=epochs)

    print("Train completed")
    return True

def anomaly_test(enc, dec, feature_dim: int, fc_dim_ls: list, epochs: int = 300, ylog_scale=False)->bool:
    '''
    Anomaly Test, \w Test dataset
    '''

    anomaly_err_ls = [[] for _ in range(6)]

    with torch.no_grad():
        for vec in test_loader:
            kjv_vec = vec[0]
            left_vec = vec[1:]

            sentence_vec = []
            for j in left_vec:
                sentence_vec.append((kjv_vec - j).to(device))

            for idx, i in enumerate(sentence_vec):
                z = enc(i)
                reconstructed_sentence_vec = dec(z)
                res = reconstructed_sentence_vec - i
                loss = torch.norm(res, dim=1).tolist()
                anomaly_err_ls[idx] += loss

    mean_ls = []
    std_ls = []

    for idx, ls in enumerate(anomaly_err_ls):
        mean_ls.append(np.mean(ls))
        std_ls.append(math.sqrt(np.var(ls)))
        plt.hist(ls, bins=100, label=name_ls[idx], alpha=0.4,
                 color=color_ls[idx], stacked=True)
    #mean_lsls.append(mean_ls[:])
    #std_lsls.append(std_ls[:])
    print(mean_ls, '\n', std_ls)
    stat_dic[f"{fc_dim_ls}, {feature_dim}"] = {"mean":mean_ls[:], "std":std_ls[:]}

    if ylog_scale == True:
        plt.semilogy()
    plt.xlabel("L2 error")
    plt.ylabel("# of data")
    plt.legend()
    plt.savefig(f'L2 lost dist, model={fc_dim_ls}, {feature_dim}, {epochs}.png')
    plt.close()
    return True

'''
Main
'''
stat_dic = {}  # to save model(s) statistic values: mean, std. and will be saved by json file
feature_dim_ls = [16,32,64,128,256]

for feature_dim in feature_dim_ls:
    for fc_dim_ls_len in range(1, 4):

        fc_dim_ls = [round(1536 * (feature_dim / 1536) ** ((i + 1) / (fc_dim_ls_len + 1))) for i in
                     range(fc_dim_ls_len)]  # dim.s of layers
        print("\n\n==========================================================\n"
              "\tStart training\n"+\
              f"Featrue dim = {feature_dim}\n"+\
              f"Fc layer structure = {fc_dim_ls}\n"+\
              f"Fc layer depth = {len(fc_dim_ls)}\n"+\
              "==========================================================\n\n")


        if len(fc_dim_ls) >= 3:
            epochs = 1000
        elif len(fc_dim_ls) == 2:
            epochs = 700
        else:
            epochs = 500
        if feature_dim == 96:
            if len(fc_dim_ls) <= 3:
                epochs = 600
            else:
                epochs = 1000

        train(feature_dim, fc_dim_ls, epochs=epochs)

with open("statistic.json", 'w') as json_file:
    json.dump(stat_dic, json_file)

print("All Train(s) complete")
