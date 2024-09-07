# -*- coding: utf-8 -*-
"""
Created on Fri Sep  6 23:38:12 2024

@author: suchs
"""


import numpy as np
import cv2
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
from torch.nn import functional as F

import math
import sys
import os
import algo_imaging

global device

def setup_seed(seed):
     torch.manual_seed(seed)
     torch.cuda.manual_seed_all(seed)
     np.random.seed(seed)
     # random.seed(seed)
     torch.backends.cudnn.deterministic = True
     
def legendre(n, sampling):
    ddx = np.linspace(-1, 1, sampling)
    mesh_x, mesh_y = np.meshgrid(ddx, ddx, indexing='xy')
    if n == 0:
        return np.ones((sampling,sampling), dtype=np.float32)
    elif n == 1:
        return mesh_x
    else:
        _n = n - 1 
        return (2*_n+1)/(_n+1)*mesh_x*legendre(_n, sampling) - _n/(_n+1)*legendre(_n-1, sampling)

def gen_legendre_array(N, sampling):
    res = np.zeros((N,sampling,sampling), dtype=np.float32)
    for i in range(N//2):
        tmp = legendre(i, sampling)
        res[i*2,:,:] = tmp
        res[i*2+1,:,:] = tmp.T
    return res
    
def build_apod(sampling, NA_obj, wl_um, pix_img_um, mag):
    R = NA_obj / wl_um
    M = 1 / 2 / (pix_img_um / mag)  
    confidence = (M/2)/R
    ddx = torch.linspace(-M, M, sampling)
    mesh_x, mesh_y = torch.meshgrid(ddx, ddx, indexing='xy')
    mesh_rho = torch.sqrt(mesh_x ** 2 + mesh_y ** 2)
    apod = torch.zeros_like(mesh_rho)
    apod[mesh_rho <= R] = 1
    apod[mesh_rho > R] = 0
    return apod
    
def FFT(data):
    return torch.fft.fftshift(torch.fft.fft2(torch.fft.ifftshift(data)))

def iFFT(data):
    return torch.fft.fftshift(torch.fft.ifft2(torch.fft.ifftshift(data)))

def sigmoid_np(x, a):
    return 1 / (1 + np.exp(-a*(x)))
    
def sigmoid(x, a):
    return 1 / (1 + torch.exp(-a*(x)))
    
class mask_layer(nn.Module):
    def __init__(self, info_list):
        super(mask_layer, self).__init__()
        sampling = info_list[0]
        term_num = info_list[1]
        NA_obj = info_list[2]
        wl_um = info_list[3]
        mag = info_list[4]
        pix_img_um = info_list[5]
        
        self.sampling = sampling
        self.term_num = term_num
        self.NA_obj = NA_obj
        self.wl_um = wl_um
        self.mag = mag
        self.pix_img_um = pix_img_um
        
        self.scale = 3
        self.scale_half = (self.scale - 1) / 2
        self.pad_n_1 = int(self.sampling * self.scale_half)
        self.pad_n_2 = int(self.sampling * self.scale - self.sampling - self.pad_n_1)
        self.apod = build_apod(int(sampling*self.scale), NA_obj, wl_um, pix_img_um, mag).to(device) 
        lgd_fname = f"legendre_{term_num}x{sampling}x{sampling}"
        if os.path.exists(lgd_fname + ".npy"):
            print("legendre will be loaded")
            lgd_array = np.load(lgd_fname + ".npy")
            self.LGD = torch.from_numpy(lgd_array).to(device)
        else:
            print(f"legendre will be gennerated and save as {lgd_fname}")
            lgd_array = gen_legendre_array(self.term_num, self.sampling)
            np.save(lgd_fname + ".npy", lgd_array)
            self.LGD = torch.from_numpy(lgd_array).to(device)
        self.mask_coef_list = nn.Parameter(torch.Tensor(self.term_num))
        nn.init.uniform_(self.mask_coef_list, -0.1, 0.1)
        
    def mask_calc(self):
        self.mask = torch.zeros((self.sampling,self.sampling)).to(device)
        for i in range(self.term_num//2):
            self.mask = self.mask + self.mask_coef_list[i*2] * self.LGD[i*2]
            self.mask = self.mask + self.mask_coef_list[i*2+1] * self.LGD[i*2+1]
        self.mask = sigmoid(self.mask, coef_sigmoid)
        # self.mask[self.mask>=0.5] = 1
        # self.mask[self.mask<0.5] = 0
        mask_pad = F.pad(self.mask,(self.pad_n_1,self.pad_n_2,self.pad_n_1,self.pad_n_2))
        return mask_pad
        
    def coherence_image(self):
        data_in = self.mask_calc().unsqueeze(0).unsqueeze(0).to(device)       
        input_fft = FFT(data_in)
        output_fft = input_fft * self.apod
        output_ifft = iFFT(output_fft)
        # output = torch.abs(output_ifft) ** 2
        output = torch.real((output_ifft) * torch.conj(output_ifft))
        output = output[:,:,self.pad_n_1:-self.pad_n_2,self.pad_n_1:-self.pad_n_2]
        output = (output - 0.5) * 2
        output = sigmoid(output, coef_sigmoid)
        return output
    
    def forward(self):
        output = self.coherence_image()
        return output
 
class intensity_layer(nn.Module):
    def __init__(self):
        super(intensity_layer, self).__init__()
        self.K = nn.Parameter(torch.Tensor(1))
        nn.init.uniform_(self.K, 0.8, 1.2)
        
    def forward(self, input):
        output = input * self.K
        return output 
    
class Find_LGD(nn.Module):
    def __init__(self, info_list):
        super(Find_LGD, self).__init__()
        self.mask_forward = mask_layer(info_list)
    def forward(self):
        A = self.mask_forward()
        return A

def norm(data):
    return (data-np.min(data)) / (np.max(data)-np.min(data))


       
if __name__ == '__main__':
    
    print(torch.cuda.is_available())
    print(torch.cuda.device_count())

    coef_sigmoid = 10
    if torch.cuda.is_available():
        device = torch.device("cuda:0")
        torch.cuda.set_device(device)
    else:
        device = torch.device("cpu")
        
    print("Using {} ".format(device))
    
    setup_seed(1100)
    
    file_dir = "C:\\code\\opc\\"
    fname = "001"
    fname_out = "fig_pred3_" + fname
    
    img_gt = cv2.imread(file_dir + fname + "_in.png", 0).astype(np.float32)
    
    img_gt = (img_gt / 255 - 0.5) * 2
    img_gt = sigmoid_np(img_gt, coef_sigmoid)
    img_gt_tensor = torch.from_numpy(img_gt).unsqueeze(0).unsqueeze(0).to(device)
    
    sampling = 301
    term_num = 30
    
    NA_obj = 0.3
    wl_um = 0.193
    mag = 0.25
    pix_img_um = 0.004
    
    info_list = list()
    info_list.append(sampling)
    info_list.append(term_num)
    info_list.append(NA_obj)
    info_list.append(wl_um)
    info_list.append(mag)
    info_list.append(pix_img_um)
        
    epoch = int(1e5)
    loss_min = 1e-6 
    loss_delta_min = loss_min 
    loss_last = 1e5
    model = Find_LGD(info_list).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr = 0.005)
    loss_MSE = torch.nn.MSELoss()
    
    for i in range(epoch):
        model.train()
        img_pred_tensor = model()
        loss_mse = loss_MSE(img_pred_tensor, img_gt_tensor)
        optimizer.zero_grad()
        loss_mse.backward()
        optimizer.step()
        loss = loss_mse.item()
        
        if math.isnan(loss):
            print("break by nan")
            break
        
        if i % 500 == 1:
            mask_coef_list = model.mask_forward.mask_coef_list.detach().cpu().numpy()
            # print("epoch: "+str(i)+"/"+str(epoch), " loss:"+str(loss), mask_coef_list)
            print("epoch: "+str(i)+"/"+str(epoch), " loss:"+str(loss))
            mask_pred = model.mask_forward.mask.detach().cpu().numpy()
            img_pred = img_pred_tensor.detach().cpu().numpy()[0,0,:,:]
        
            if loss < loss_min:
                print(f"break by loss_min, {loss_min}")
                break
            if np.abs(loss-loss_last) < loss_delta_min:
                print(f"break by loss_delta_min, {loss_delta_min}")
                break
        
            loss_last = loss
    
    
    print("***   End   **********************************************************************")
    # print("epoch: "+str(i)+"/"+str(epoch), " loss:"+str(loss), mask_coef_list)
    print("mask_coef_list: ",  mask_coef_list)
    
    mask_pred_real_tmp = np.zeros_like(mask_pred)
    mask_pred_real = np.zeros_like(mask_pred)
    img_pred_real_tmp = np.zeros_like(mask_pred)
    img_pred_real = np.zeros_like(mask_pred)
    
    thres_good = 10
    err_min = 1000
    for thres_new in range(50,100,3):
        mask_pred_real_tmp[mask_pred>=thres_new/100] = 1
        mask_pred_real_tmp[mask_pred<thres_new/100] = 0
        img_pred_2 = algo_imaging.imaging(mask_pred_real_tmp, NA_obj, wl_um, pix_img_um, mag, 1)
        img_pred_real_tmp[img_pred_2>=0.5] = 1
        img_pred_real_tmp[img_pred_2<0.5] = 0
        err = np.mean(np.abs(img_gt - img_pred_real_tmp))
        print(f"thres = {thres_new/100}, err = {err}")
        if err < err_min:
            err_min = err
            thres_good = thres_new/100
            mask_pred_real = mask_pred_real_tmp.copy()
            img_pred_real = img_pred_real_tmp.copy()
    
    print(f"thres_good = {thres_good},   err_min = {err_min}")
    
    img_error = np.abs(img_gt - img_pred_real)
    
    
    file_dir = r"C:\code\opc" + "\\"
    num_y = 2
    num_x = 3
    plt.figure(num=1,dpi=200,figsize=(12,9))
    plt.subplot(num_y,num_x,1)
    plt.title("img_gt")
    plt.imshow(img_gt, cmap = 'rainbow', vmin = 0, vmax = 1)
    plt.colorbar()
    plt.subplot(num_y,num_x,2)
    plt.title("img_pred_pytorch")
    plt.imshow(img_pred, cmap = 'rainbow', vmin = 0, vmax = 1)
    plt.colorbar()
    plt.subplot(num_y,num_x,3)
    plt.title("img_pred_real")
    plt.imshow(img_pred_real, cmap = 'rainbow', vmin = 0, vmax = 1)
    plt.colorbar()
    plt.subplot(num_y,num_x,4)
    plt.title("img_error")
    plt.imshow(img_error, cmap = 'rainbow')
    plt.colorbar()
    plt.subplot(num_y,num_x,5)
    plt.title("mask_pred_pytorch")
    plt.imshow(mask_pred, cmap = 'rainbow')
    plt.colorbar()
    plt.subplot(num_y,num_x,6)
    plt.title("mask_pred_real")
    plt.imshow(mask_pred_real, cmap = 'rainbow')
    plt.colorbar()
    plt.savefig(file_dir + fname_out + ".png")
    plt.close()
    
    
    
    
