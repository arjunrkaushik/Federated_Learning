# -*- coding: utf-8 -*-
"""VAE_test.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/16_PN37E2D_6_v5wIjw3cUneFanf6WPx1
"""

!nvidia-smi

from google.colab import drive
drive.mount('/content/drive')

"""# Header Files"""

import pandas as pd
import numpy as np

import torch
import torchvision
from torch import nn
from torch.utils.data import DataLoader
from torch.utils.data import Dataset, random_split
from torchvision import transforms
from torch.distributions import Normal, Bernoulli, kl_divergence as kl
import torch.nn.functional as F


from matplotlib import pyplot as plt
from matplotlib.pyplot import figure

from tqdm.auto import tqdm
import time
from sklearn.metrics import accuracy_score

import scipy

!pip install umap-learn datashader bokeh holoviews

import umap
import umap.plot

"""###FEDERATED LEARNING ALGOS"""

def split_iid(dataset, n_centers):
    """ Split PyTorch dataset randomly into n_centers """
    n_obs_per_center = [len(dataset) // n_centers for _ in range(n_centers)]
    return random_split(dataset, n_obs_per_center)

def federated_averaging(models, n_obs_per_client):
    assert len(models) > 0, 'An empty list of models was passed.'
    assert len(n_obs_per_client) == len(models), 'List with number of observations must have ' \
                                                 'the same number of elements that list of models.'

    # Compute proportions
    n_obs = sum(n_obs_per_client)
    
    proportions = [n_k / n_obs for n_k in n_obs_per_client]

    # Empty model parameter dictionary
    avg_params = models[0].state_dict()
    for key, val in avg_params.items():
        avg_params[key] = torch.zeros_like(val)

    # Compute average
    for model, proportion in zip(models, proportions):
        for key in avg_params.keys():
            avg_params[key] += proportion * model.state_dict()[key]

    # Copy one of the models and load trained params
    avg_model = copy.deepcopy(models[0])
    avg_model.load_state_dict(avg_params)
  
    return avg_model

"""###PARAMETERS"""

N_CENTERS = 2
N_ROUNDS = 50   

BATCH_SIZE = 8
LR = 1e-6

"""# Dataloader for sparse matrix"""

class SparseDataset(Dataset):
    def __init__(self, mat_csc):
        self.dim = mat_csc.shape

        csr = mat_csc.tocsr(copy=True) # Converts matrix to Compressed Sparse Row format

        self.data = torch.tensor(csr.data, dtype=torch.float32)                 # array containing all the non zero elements of sparse matrix
        self.indices = torch.tensor(csr.indices, dtype=torch.int64)             # array mapping each element of data to columns of sparse matrix
        self.indptr = torch.tensor(csr.indptr, dtype=torch.int64)               # maps elements of indices and data to the rows of sparse matrix
        

    def __len__(self):
        return self.dim[0]

    def __getitem__(self, idx):
        obs = torch.zeros((self.dim[1],), dtype=torch.float32)
        ind1,ind2 = self.indptr[idx],self.indptr[idx+1]
        obs[self.indices[ind1:ind2]] = self.data[ind1:ind2]

        return obs

train_sparse = scipy.io.mmread("/content/drive/MyDrive/Federated Learning/Data/Fixed_depth_Fixed_rou/3000 depth/rou0.6/peakMat.fixLen.400_400_400_400_400.rou0.6.rdepth3000.mtx")
type(train_sparse)

train_dataset = SparseDataset(train_sparse)

train_dataset[0].shape

federated_dataset = split_iid(train_dataset, n_centers=N_CENTERS)

print('Number of centers:', len(federated_dataset))
#train_data_loader = DataLoader(dataset= federated_dataset, batch_size=BATCH_SIZE, shuffle=False)

"""###W"""

ones = train_sparse.count_nonzero()
ones

total_elements = train_sparse.shape[0] * train_sparse.shape[1]
zeros = total_elements - ones
ω = zeros/ones

print(f"Total elements: {total_elements} \t Ones: {ones} \t Zeros: {zeros} \n ω: {ω}")

"""#  Labels"""

labels = pd.read_csv("/content/drive/MyDrive/Federated Learning/Data/Fixed_depth_Fixed_rou/3000 depth/rou0.6/cellLabel.fixLen.400_400_400_400_400.rou0.6.rdepth3000.txt", sep="\t", header=None)

labels.head()

"""# VAE MODEL

## My model
"""

d = 10
class VAE(nn.Module):
    
    def __init__(self):
        super().__init__()

        

        self.encoder = nn.Sequential(
            nn.Linear(train_sparse.shape[1], d ** 3),
            nn.ReLU(), 
            nn.Linear(d ** 3, d ** 2),
            nn.ReLU(),
            nn.Linear(d ** 2, d * 2)          
        )

        self.decoder = nn.Sequential(
            nn.Linear(d, d ** 2),
            nn.ReLU(), 
            nn.Linear(d ** 2, d ** 3),
            nn.ReLU(),
            nn.Linear(d ** 3, train_sparse.shape[1]),
            nn.Sigmoid(),
          
            
        )

    def reparameterise(self, mu, logvar):
        if self.training:
            std = logvar.mul(0.5).exp_()
            eps = std.data.new(std.size()).normal_()
            return eps.mul(std).add_(mu)
        else:
            return mu

    def forward(self, x):
        x = x.view(-1, train_sparse.shape[1])

        mu_logvar = self.encoder(x).view(-1, 2, d)
        mu = mu_logvar[:, 0, :]
        logvar = mu_logvar[:, 1, :]
        z = self.reparameterise(mu, logvar)

        
        return self.decoder(z), mu, logvar, z

"""# Train"""

device = torch.device("cuda:0") #if torch.cuda.is_available() else "cpu")
#device = torch.device( "cpu")
device

train_sparse.shape[1]

model = VAE().to(device)

def get_data(subset, shuffle=False):
    """ Extracts data from a Subset torch dataset in the form of a tensor"""
    loader = DataLoader(subset, batch_size=len(subset), shuffle=shuffle)
    return iter(loader).next()

#x = next(iter(train_data_loader)).to(device)

#x.shape

model(get_data(federated_dataset[0]).to(device))

import copy
optimizer = torch.optim.Adam(model.parameters(), lr=LR)
models = [copy.deepcopy(model) for _ in range(N_CENTERS)]
n_obs_per_client = [len(client_data) for client_data in federated_dataset]

n_obs_per_client

for model in models:
  optimizer= torch.optim.Adam(model.parameters(),lr=LR)

params = sum(p.numel() for p in model.parameters() if p.requires_grad)
print("Trainable Parameters: "+str(params))

def plot_UMAP(combined, z):
    mapper = umap.UMAP(n_neighbors=20).fit(combined )
    umap.plot.points(mapper, labels=labels[1],color_key_cmap='Paired', background='black')
    plt.title("COMBINED")
    plt.show()

    mapper = umap.UMAP(n_neighbors=20).fit(z)
    umap.plot.points(mapper, labels=labels[1],color_key_cmap='Paired', background='black')
    plt.title("MEAN")
    plt.show()

    plt.clf()

"""## Train with my loss"""

# Reconstruction + KL divergence losses summed over all elements and batch

def loss_function(x_hat, x, mu, logvar,ω=1):
    eps = 1e-12
    #x = x.view(-1, train_sparse.shape[1])

    #BCE = nn.functional.binary_cross_entropy(x_hat, x.view(-1, 90635), reduction='sum')
    
    own_BCE = ω * x * x_hat.clamp(min=eps).log() + (1 - x) * (1 - x_hat).clamp(min=eps).log()
    own_BCE = torch.neg(torch.sum(own_BCE))

    KLD = 0.5 * torch.sum(logvar.exp() - logvar - 1 + mu.pow(2))


    return own_BCE, KLD

init_params = model.state_dict()
server_model=copy.deepcopy(model)
train_losses = []
KLD_losses = []
BCE_losses = []
for round_i in range(N_ROUNDS):
    train_loss = 0
    KLD_loss = 0
    BCE_loss = 0
    # latent_rep = []
    # latent_rep_mean = []
    print(f'============ Epoch: {round_i} ============')
    start_time = time.time()
    print(init_params['encoder.0.weight'])
    for client_dataset, client_model in zip(federated_dataset, models):
        optimizer = torch.optim.Adam(client_model.parameters(), lr=LR)
        X = get_data(client_dataset)
        client_model.load_state_dict(init_params)
        #client_model.data = [X.view(-1, 90635)]  # Set data attribute in client's model (list wraps the number of channels)
        for x in tqdm(X):
          x = x.to(device)
        
          # ===================forward=====================
          x_hat, mu, logvar, hidden = client_model(x.float())
        
          # latent_rep.extend(hidden.detach().clone().cpu().numpy())
          # latent_rep_mean.extend(mu.detach().clone().cpu().numpy())
        
          BCE, KLD = loss_function(x_hat, x, mu, logvar,ω)
          loss = KLD+BCE
          train_loss += loss.item()*x.size(0) 
          KLD_loss += KLD.item()*x.size(0) 
          BCE_loss += BCE.item()*x.size(0) 
        
          # ===================backward====================
          optimizer.zero_grad()
          loss.backward()
          optimizer.step()
    
        # Load client's model parameters and train
        #client_model.optimize(epochs=N_EPOCHS, data=client_model.data)
        avg_loss = train_loss / len(client_dataset.dataset)
        avg_KLD_loss = KLD_loss / len(client_dataset.dataset)
        avg_BCE_loss = BCE_loss / len(client_dataset.dataset)
        
    
        train_losses.append(avg_loss)
        KLD_losses.append(avg_KLD_loss)
        BCE_losses.append(avg_BCE_loss)

        trained_model = federated_averaging(models, n_obs_per_client)
        server_model = copy.deepcopy(trained_model)
        init_params = server_model.state_dict()


        print(f'''Average loss: {avg_loss :.4f}\t Avg KLD loss: {avg_KLD_loss :.4f}\t 
               Avg BCE loss: {avg_BCE_loss :.4f}''')


plt.plot(train_losses, label="Average Loss")
plt.plot(KLD_losses, label="KLD Loss", linestyle="-.")
plt.plot(BCE_losses, label="BCE Loss", linestyle=":")
plt.legend()
plt.show()
plt.clf()
    

# plot_UMAP(latent_rep, latent_rep_mean)
    
#torch.save(model.state_dict(), '/gdrive/MyDrive/UCI/VAE high quality data/weights_100/epoch_'+ str(epoch)+'.pt')
#torch.save(model.state_dict(), 'epoch_'+ str(epoch)+'.pt')
#torch.save(latent_rep, '/gdrive/MyDrive/UCI/VAE high quality data/hidden_rep_100/hidden_dim_epoch_'+ str(epoch) +'.pt')   

print(f"{((time.time() - start_time)) :.4f} secs")     
    # Aggregate models using federated averaging

#optimizer = torch.optim.Adam(server_model.parameters(), lr=LR)
train_losses = []
KLD_losses = []
BCE_losses = []
for round_i in range(0,1):
    train_loss = 0
    KLD_loss = 0
    BCE_loss = 0
    latent_rep = []
    latent_rep_mean = []
    print(f'============ Epoch: {round_i} ============')
    start_time = time.time()
    #print(init_params['decoder.0.weight'])
    for x in tqdm(DataLoader(dataset= train_dataset, batch_size=BATCH_SIZE, shuffle=False)):
        x = x.to(device)
        
          # ===================forward=====================
        x_hat, mu, logvar, hidden = server_model(x.float())
        
        latent_rep.extend(hidden.detach().clone().cpu().numpy())
        latent_rep_mean.extend(mu.detach().clone().cpu().numpy())
        
        # BCE, KLD = loss_function(x_hat, x, mu, logvar,ω)
        # loss = KLD+BCE
        # train_loss += loss.item()*x.size(0) 
        # KLD_loss += KLD.item()*x.size(0) 
        # BCE_loss += BCE.item()*x.size(0) 
        
        #   # ===================backward====================
        # optimizer.zero_grad()
        # loss.backward()
        # optimizer.step()
    
        # Load client's model parameters and train
        #client_model.optimize(epochs=N_EPOCHS, data=client_model.data)
    avg_loss = train_loss / 2000
    avg_KLD_loss = KLD_loss / 2000
    avg_BCE_loss = BCE_loss / 2000
        
    
    train_losses.append(avg_loss)
    KLD_losses.append(avg_KLD_loss)
    BCE_losses.append(avg_BCE_loss)

    print(f'''Average loss: {avg_loss :.4f}\t Avg KLD loss: {avg_KLD_loss :.4f}\t 
               Avg BCE loss: {avg_BCE_loss :.4f}''')

    

plot_UMAP(latent_rep, latent_rep_mean)
