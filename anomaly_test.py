import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
import torch.nn.functional as F
import pandas as pd

'''
Hyperparameters
'''

lr = 0.001
batch_size = 100
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
input_dim = 1536 # number of sentence imbeding vector dimension

df = pd.read_json(r"/root/shinyee_kang/data/bible_embeddings.jsonl",lines=True,encoding_errors='ignore')

import numpy as np

embeding_vecs2 = df[['KJV_embedding', 'NET_embedding', 'ASV_embedding', 'ASVS_embedding' \
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

train_loader = torch.utils.data.DataLoader(dataset=dataset2, batch_size=batch_size, shuffle=False)

import math

mean_lsls = []
std_lsls = []

name_ls = ['NET', 'ASV', 'ASVS', 'Coverdale', 'Geneva', 'KJV_Strongs']
color_ls = ['b', 'g', 'r', 'c', 'm', 'y']
feature_dim_ls = [32, 64, 96]

for feature_dim in feature_dim_ls:
    for fc_dim_ls_len in range(1, 7):
        fc_dim_ls = [round(1536 * (feature_dim / 1536) ** ((i + 1) / (fc_dim_ls_len + 1))) for i in
                     range(fc_dim_ls_len)]  # dim.s of layers

        if len(fc_dim_ls) > 4:
            epochs = 600
        else:
            epochs = 300
        if feature_dim == 96:
            if len(fc_dim_ls) == 1:
                epoch = 300
            elif len(fc_dim_ls) <= 3:
                epochs = 600
            else:
                epochs = 1000

        '''
        Modules : Encoder, Decoder

        from prototype2, use batch norm
        '''

        print("\n\n==========================================================\n" + "\tTest Start\n" + \
              f"\tFeatrue dim = {feature_dim}\n" + \
              f"\tFc layer structure = {fc_dim_ls}\n" + \
              f"\tFc layer depth = {len(fc_dim_ls)}\n" + \
              "==========================================================\n\n")


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


        enc = Encoder().to(device)
        dec = Decoder().to(device)
        loss_function = nn.MSELoss()

        dec.load_state_dict(
            torch.load(f'/root/shinyee_kang/ASV VAE/Dec params {fc_dim_ls}, {feature_dim}, epochs={epochs}, kjv-asv',
                       map_location="cuda:0"))
        enc.load_state_dict(
            torch.load(f'/root/shinyee_kang/ASV VAE/Enc params {fc_dim_ls}, {feature_dim}, epochs={epochs}, kjv-asv',
                       map_location="cuda:0"))

        enc.eval()
        dec.eval()

        import time

        start = time.time()

        anomaly_err_ls = [[] for _ in range(6)]

        with torch.no_grad():
            for vec in train_loader:
                kjv_vec = vec[0]
                left_vec = vec[1:]

                sentence_vec = []
                for j in left_vec:
                    sentence_vec.append((kjv_vec - j).to(device))
                # print(sentence_vec[0].shape)
                # print(len(sentence_vec))

                for idx, i in enumerate(sentence_vec):
                    z = enc(i)
                    reconstructed_sentence_vec = dec(z)
                    res = reconstructed_sentence_vec - i
                    loss = torch.norm(res, dim=1).tolist()
                    anomaly_err_ls[idx] += loss


        mean_ls = []
        std_ls = []
        print(len(anomaly_err_ls[0]))

        for idx, ls in enumerate(anomaly_err_ls):
            mean_ls.append(np.mean(ls))
            std_ls.append(math.sqrt(np.var(ls)))
            #print(len(ls))
            #print(len(ls[0]))
            plt.hist(ls, bins=100, label=name_ls[idx], alpha=0.5, density=True,
                     color=color_ls[idx], stacked=True)
        mean_lsls.append(mean_ls[:])
        std_lsls.append(std_ls[:])
        print(mean_ls, '\n', std_ls)
        plt.legend()
        # plt.semilogx()
        plt.savefig(f'L2 lost dist, model={fc_dim_ls}, {feature_dim}, {epochs}.png')
        plt.close()

        end = time.time()
        print("Time ellapsed in training is: {}".format(end - start))
