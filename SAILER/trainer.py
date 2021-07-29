import numpy as np
import pandas as pd 
import os
import torch
from torch.utils.data import DataLoader, random_split
import torch.optim as optim
from tqdm import tqdm
import torch.nn as nn
from torch.distributions import Normal, Bernoulli, kl_divergence as kl
import torch.nn.functional as F
from torch.autograd import Variable
import copy

from dataset import MouseAtlas, PBMC, MergeSim, SimATAC_peak as SimATAC
from model import VAE2, VAEInv


WARM_UP = 10
CYCLE = 100
CUT_OFF = None

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
    # n_obs=len(models)
    proportions = [n_k / n_obs for n_k in n_obs_per_client]

    # Empty model parameter dictionary
    avg_params = models[0].module.state_dict()
    for key, val in avg_params.items():
        avg_params[key] = torch.zeros_like(val)

    # Compute average
    # for model, proportion in zip(models, proportions):
    #     for key in avg_params.keys():
    #         avg_params[key] += proportion * model.state_dict()[key]

    for model in models:
        for key in avg_params.keys():
            avg_params[key] += model.module.state_dict()[key]

    # for key in avg_params.keys():
    #     avg_params[key] /= float(len(models))

    # Copy one of the models and load trained params
    avg_model = copy.deepcopy(models[0])
    avg_model.module.load_state_dict(avg_params)

    return avg_model


def dice_loss(pred, target, with_logit=True):
    if with_logit:
        pred = torch.sigmoid(pred)
    smooth = 1e-13
    iflat = pred.contiguous().view(-1)
    tflat = target.contiguous().view(-1)
    intersection = (iflat * tflat).sum()
    A_sum = torch.sum(iflat * iflat)
    B_sum = torch.sum(tflat * tflat)
    return 1 - ((2. * intersection + smooth) / (A_sum + B_sum + smooth) )

def apprx_kl(mu, sigma):
    '''Adapted from https://github.com/dcmoyer/invariance-tutorial/
    Function to calculate approximation for KL(q(z|x)|q(z))
        Args:
            mu: Tensor, (B, z_dim)
            sigma: Tensor, (B, z_dim)
    '''
    var = sigma.pow(2)
    var_inv = var.reciprocal()

    first = torch.matmul(var, var_inv.T)

    r = torch.matmul(mu * mu, var_inv.T)
    r2 = (mu * mu * var_inv).sum(axis=1)

    second = 2 * torch.matmul(mu, (mu * var_inv).T)
    second = r - second + (r2 * torch.ones_like(r)).T

    r3 = var.log().sum(axis=1)
    third = (r3 * torch.ones_like(r)).T - r3

    return 0.5 * (first + second + third)

          
class Trainer(object):
    def __init__(self, args):
        self.name = args.name
        self.max_epoch = args.max_epoch
        self.lr = args.lr
        self.weight_decay = args.weight_decay
        self.log = args.log
        self.out_every = args.out_every
        self.pos_w = args.pos_w,
        self.LAMBDA = args.LAMBDA
        self.Clients = args.clients
        if args.cuda_dev:
            torch.cuda.set_device(args.cuda_dev[0])
            self.cuda_dev = f'cuda:{args.cuda_dev[0]}'
            self.device = 'cuda'
        else:
            self.cuda_dev = None
            self.device = 'cpu'
        print(f'Using {self.device}')
        self.z_dim = args.z_dim
        self.batch_size = args.batch_size
        self.start_save = args.start_save
        self.start_epoch = args.start_epoch
        self.file_name = args.file_name
        self.type_name = args.type_name
        self.ckpt_dir = os.path.join(args.ckpt_dir, self.name)
        if args.data_type == 'simATAC':
            self.dataset = SimATAC(self.file_name, self.type_name, args.setting, args.signal, args.frags, args.bin_size, conv=args.conv)
        elif args.data_type == 'atlas':
            self.dataset = MouseAtlas(cutoff=CUT_OFF)
        elif args.data_type == 'pbmc':
            self.dataset = PBMC()
        elif args.data_type == 'mergeSim':
            if args.num:
                self.dataset = MergeSim(args.num)
            else:
                self.dataset = MergeSim()
        else:
            raise Exception(f'Dataset {args.data_type} does not exist!')
        self.federated_dataset = split_iid(self.dataset,n_centers=self.Clients)
        self.dataloader =  DataLoader(self.dataset,
                                        batch_size=args.batch_size,
                                        shuffle=False,
                                        num_workers=3*len(args.cuda_dev),
                                        pin_memory=True,
                                        drop_last=False)
        input_dim = self.dataset.padto
        if args.model_type == 'inv':
            if args.sample_batch:
                self.de_batch = True
                self.vae = VAE2(input_dim, args.z_dim, batch=True)
            else:
                self.de_batch = False
                self.vae = VAE2(input_dim, args.z_dim)
            self.vaeI = VAEInv(self.vae)
            self.model = nn.DataParallel(self.vaeI, device_ids=args.cuda_dev)
        else:
            raise Exception(f'Model type {args.model_type} does not exist!')
        self.model_type = args.model_type
        if args.load_ckpt:
            if os.path.isfile(args.load_ckpt):
                print('Loading ' + args.load_ckpt)
                if self.cuda_dev:
                    self.model.module.load_state_dict(torch.load(args.load_ckpt, map_location=self.cuda_dev))
                else:
                    self.model.module.load_state_dict(torch.load(args.load_ckpt, map_location='cpu'))
                print('Finished Loading ckpt...')
            else:
                raise Exception(args.load_ckpt + "\nckpt does not exist!")
        self.model.to(self.device)
        self.optim = optim.Adam(self.model.parameters(), lr=self.lr, weight_decay=self.weight_decay)
        self.cycle = CYCLE * self.dataset.__len__() // self.batch_size // len(args.cuda_dev)
        self.models = [copy.deepcopy(self) for _ in range(self.Clients)]
        for x in self.models:
            print("Model initiate")
            print(x.model.module.state_dict()['vae.Encoder.Encoder.0.weight'])
            print(x.model_type)
            # if args.model_type == 'inv':
            #     if args.sample_batch:
            #         x.de_batch = True
            #         x.vae = VAE2(input_dim, args.z_dim, batch=True)
            #     else:
            #         x.de_batch = False
            #         x.vae = VAE2(input_dim, args.z_dim)
            #     x.vaeI = VAEInv(x.vae)
            #     x.model = nn.DataParallel(x.vaeI, device_ids=args.cuda_dev)
            # else:
            #     raise Exception(f'Model type {args.model_type} does not exist!')
            # x.model_type = args.model_type
            # x.model.to(self.device)
            # self.optim = optim.Adam(x.model.parameters(), lr=self.lr, weight_decay=self.weight_decay)
        self.server_model = copy.deepcopy(self)

    def warm_up(self):
        if not os.path.exists(self.ckpt_dir):
            print(f'Making dir {self.ckpt_dir}')
            os.makedirs(self.ckpt_dir)
        for y in self.models:
            y.model.train()
        self.pbar = tqdm(total=WARM_UP)
        total_iter = 0
        for step in range(WARM_UP):
            for (y,data) in zip(self.models,self.federated_dataset):
                data_loaded = DataLoader(data,
                                        batch_size=self.batch_size,
                                        shuffle=False,
                                        num_workers=3,
                                        pin_memory=True,
                                        drop_last=False)
                for x,s,l in data_loaded:
                    l = l.unsqueeze(1).float().to(self.device).log()
                    # print("L value")
                    # print(l)
                    total_iter += 1
                    x = x.float().to(self.device)
                    # print("x value")
                    # print(x)
                    if y.model_type == 'adv':
                        _, _, _, rec, _ = y.model(x, l)
                    elif y.model_type == 'inv':
                        if y.de_batch:
                            s = s.unsqueeze(1).float().to(self.device)
                            _, _, _, rec = y.model(x, l, b=s)
                        else:
                            a, b, c, rec = y.model(x, l)
                            # print(f'a = {a}')
                            # print(f'b = {b}')
                            # print(f'c = {c}')
                            # print(f'rec = {rec}')
                    else:
                        _, _, _, rec = y.model(x)
                    pos_weight = torch.Tensor([self.pos_w]).to(self.device)
                    bce = nn.BCEWithLogitsLoss(pos_weight = pos_weight)
                    # rec_loss = focal(rec.view(-1), x.view(-1).long())
                    rec_loss = bce(rec, x)
                    y.optim.zero_grad()
                    rec_loss.backward()
                    y.optim.step()
                    if total_iter%50 == 0:
                        self.pbar.write(f'[{total_iter}] vae_recon_loss:{rec_loss.item()}')
            self.pbar.update(1)
            n_obs_per_client = [len(data) for data in self.federated_dataset]
            models = [copy.deepcopy(y.model) for y in self.models]
            self.server_model.model = federated_averaging(models,n_obs_per_client)
            # print(models[0].module.state_dict()['vae.Encoder.Encoder.0.weight'])
            # print(self.server_model.model.module.state_dict()['vae.Encoder.Encoder.0.weight'])
            # for y in self.models:
            #     y.model.module.load_state_dict(self.server_model.model.module.state_dict())

            #models = [copy.deepcopy(self.server_model) for _ in range(self.Clients)]

        torch.save(self.server_model.model.module.state_dict(), os.path.join(self.ckpt_dir, 'warmup.pt'))
        self.pbar.write("[Warmup Finished]")
        self.pbar.close()

    def rec_all(self, batch_size=1, same_depth=False):
        dataloader =  DataLoader(self.dataset,
                                batch_size=batch_size,
                                shuffle=False,
                                num_workers=0,
                                pin_memory=True,
                                drop_last=False)
        labels = []
        self.model.eval()
        for i, dp in tqdm(enumerate(dataloader)):
            x, l, d = dp
            x = x.float().to(self.device)
            labels = labels + l
            if same_depth:
                d = d.unsqueeze(1).float().to(self.device).log()
                # same_depth = (same_depth - self.dataset.d_mean) / self.dataset.d_std
                d = (torch.ones_like(d) * same_depth).log()
            else:
                d = d.unsqueeze(1).float().to(self.device).log()
            with torch.no_grad():
                _, _, _, rec = self.model.forward(x, d)
                # rec = torch.sigmoid(rec).cpu()
                rec = rec.cpu()
                if i==0:
                    out = rec
                else:
                    out = torch.cat((out, rec))
        return out, labels

    def rec_batch(self, batch_size=1, same_depth=False):
        dataloader =  DataLoader(self.dataset,
                                batch_size=batch_size,
                                shuffle=False,
                                num_workers=0,
                                pin_memory=True,
                                drop_last=False)
        labels = []
        self.model.eval()
        for i, dp in tqdm(enumerate(dataloader)):
            x, l, d = dp
            x = x.float().to(self.device)
            labels = labels + l
            if same_depth:
                d = d.unsqueeze(1).float().to(self.device).log()
                # same_depth = (same_depth - self.dataset.d_mean) / self.dataset.d_std
                d = (torch.ones_like(d) * same_depth).log()
                
            else:
                d = d.unsqueeze(1).float().to(self.device).log()
            b = torch.zeros_like(d).float().to(self.device)
            with torch.no_grad():
                _, _, _, rec = self.model.forward(x, d, b)
                # rec = torch.sigmoid(rec).cpu()
                rec = rec.cpu()
                if i==0:
                    out = rec
                else:
                    out = torch.cat((out, rec))
        return out, labels

    def inv_train(self):
        if not os.path.exists(self.ckpt_dir):
            print(f'Making dir {self.ckpt_dir}')
            os.makedirs(self.ckpt_dir)
        for y in self.models:
            y.model.train()
        kl_list, rec_list, dice_list, mkl_list = [], [], [], []
        print('Inv Training started')
        self.pbar = tqdm(total=self.max_epoch - self.start_epoch)
        total_iter = (self.start_epoch-1) * self.dataset.__len__() // self.batch_size + 1
        for epoch in range(self.start_epoch, self.start_epoch + self.max_epoch):
            for (y,data) in zip(self.models,self.federated_dataset):
                data_loaded = DataLoader(data,
                                         batch_size=self.batch_size,
                                         shuffle=False,
                                         num_workers=3,
                                         pin_memory=True,
                                         drop_last=False)
                epoch_kl, epoch_rec, epoch_dice, epoch_mkl = [], [], [], []
                kl_w = np.min([2 * (total_iter -(total_iter//self.cycle) * self.cycle) / self.cycle, 1])
                for x1, s1, l1 in data_loaded:
                    x1 = x1.float().to(self.device)
                    l1 = l1.log()
                    l1 = (l1 - self.dataset.d_mean) / self.dataset.d_std
                    l1 = l1.unsqueeze(1).float().to(self.device)
                    z_mean, z_log_var, _, rec = y.model(x1, l1)
                    mean = torch.zeros_like(z_mean)
                    var = torch.ones_like(z_log_var)
                    kld_z = kl(Normal(z_mean, torch.exp(z_log_var).sqrt()), Normal(mean, var)).sum()
                    pos_weight = torch.Tensor([self.pos_w]).to(self.device)
                    bce = F.binary_cross_entropy_with_logits(rec, x1, weight=pos_weight, reduction='sum')
                    rec_loss = bce
                    m_kld = apprx_kl(z_mean, torch.exp(z_log_var).sqrt()).mean() - 0.5 * self.z_dim
                    loss = kld_z*kl_w + (1+self.LAMBDA)*rec_loss + m_kld*kl_w*self.LAMBDA
                    y.optim.zero_grad()
                    loss.backward()
                    y.optim.step()
                    total_iter += 1
            self.pbar.update(1)
            self.pbar.write(f'[{epoch}], iter {total_iter}')
            n_obs_per_client = [len(data) for data in self.federated_dataset]
            models = [copy.deepcopy(y.model) for y in self.models]
            self.server_model.model = federated_averaging(models, n_obs_per_client)
            # self.models = [copy.deepcopy(self.server_model) for _ in range(self.Clients)]
            if epoch % self.out_every == 0:
                if epoch > self.start_save:
                    torch.save(self.server_model.model.module.state_dict(), os.path.join(self.ckpt_dir, f'{epoch}.pt'))
        self.pbar.write("[Inv Training Finished]")
        self.pbar.close()

    def encode_adv(self, batch_size=100):
        dataloader = DataLoader(self.dataset,
                                batch_size=batch_size,
                                shuffle=False,
                                num_workers=0,
                                pin_memory=True,
                                drop_last=False)
        labels = []
        latent = torch.zeros(self.dataset.__len__(), self.z_dim)
        depth = torch.zeros(self.dataset.__len__())
        self.server_model.model.eval()
        for i, dp in tqdm(enumerate(dataloader)):
            x, l, d = dp
            x = x.float().to(self.device)
            labels = labels + l
            depth[i*batch_size: (i+1)*batch_size] = d
            d = d.log()
            d = (d - self.dataset.d_mean) / self.dataset.d_std
            d = d.unsqueeze(1).float().to(self.device)
            with torch.no_grad():
                z_mean, _ = self.server_model.model.forward(x, d, no_rec=True)
                # z_mean, _, _, _ = self.model(x)
                latent[i*batch_size: (i+1)*batch_size] = z_mean.cpu()
        return latent, labels, depth
