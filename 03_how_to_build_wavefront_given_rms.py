# -*- coding: utf-8 -*-

"""
@Title: How to build the wavefront given specific rms value, even for the not-full-shape aperture

@author: sheuchenko
"""

import numpy as np
import math
import os
import cv2
import sys
import random
import matplotlib.pyplot as plt

def inner_product_norm(A, B, M):
    C = np.nansum(A*B*M)
    D = np.nansum(M)
    return C/D

def array_stack(A):
    n = len(A)
    m = A[0].shape[0] * A[0].shape[1]
    out = np.zeros((m,n))
    for i in range(n):
        out[:,i] = A[i].flatten()
    return out

def re_ortho_FZP(list_of_U_nonan, mask_nan):
    nsize = len(list_of_U_nonan)
    for i in range(nsize):
        coef = inner_product_norm(list_of_U_nonan[i], list_of_U_nonan[i], mask_nan)
        list_of_U_nonan[i] = list_of_U_nonan[i] * np.sqrt(1.0/coef)
    mask_nonan = np.ones_like(mask_nan)
    mask_nonan[np.isnan(mask_nan)] = 0
    sampling = mask_nan.shape[0]
    list_of_V_nan = list()
    list_of_V_nonan = list()
    list_of_V_nan.append(list_of_U_nonan[0] * mask_nan)
    list_of_V_nonan.append(list_of_U_nonan[0] * mask_nonan)
    for i in range(1,nsize):
        add_tmp = np.zeros((sampling,sampling))
        for t in range(i):
            val = inner_product_norm(list_of_U_nonan[i], list_of_V_nonan[t], mask_nan)
            add_tmp += (val * list_of_V_nonan[t])
        tmp = list_of_U_nonan[i] - add_tmp
        tmp *= mask_nonan
        coef = inner_product_norm(tmp, tmp, mask_nan)
        tmp = tmp * np.sqrt(1.0/coef)
        tmp_nan = tmp * mask_nan
        list_of_V_nan.append(tmp_nan)
        list_of_V_nonan.append(tmp)
    matrix_transfer = np.zeros((nsize,nsize))
    for m in range(nsize):
        _array_U = array_stack(list_of_U_nonan[0:m+1]).T
        _array_V = array_stack(list_of_V_nonan[0:m+1]).T
        _array_C = np.dot(_array_V, np.dot(_array_U.T, np.linalg.inv(np.dot(_array_U, _array_V.T))))
        matrix_transfer[m, :m+1] = _array_C[-1,:]
        matrix_transfer[np.less(np.abs(matrix_transfer), 1e-4)] = 0
    return list_of_V_nan, list_of_V_nonan, matrix_transfer
    
def save_figure_of_fzp(list_of_fzp, list_of_Nnm_fzp, file_dir_tmp, file_dir_out, fname = "fig", verbose = False):
    if not os.path.exists(file_dir_tmp):
        os.makedirs(file_dir_tmp)
    if not os.path.exists(file_dir_out):
        os.makedirs(file_dir_out)
    term_num = len(list_of_Nnm_fzp)    
    for i in range(term_num):
        if verbose:
            print( f"{term_num}_{i}")
        N = list_of_Nnm_fzp[i][0]
        n = list_of_Nnm_fzp[i][1]
        m = list_of_Nnm_fzp[i][2]
        t = list_of_Nnm_fzp[i][3]
        fig = list_of_fzp[i]
        plt.figure(1)
        plt.title(f"i={i}: N={N}, n={n}, m={m}")
        plt.imshow(fig, cmap = 'rainbow')
        plt.colorbar()
        plt.savefig(file_dir_tmp + f"{fname}_N{N_term}_{term_num}_{i}" + ".png")
        plt.close()        
        fig_tmp = cv2.imread(file_dir_tmp + f"{fname}_N{N_term}_{term_num}_{i}" + ".png")
        if i == 0:
            fig_wid = fig_tmp.shape[1]
            fig_hei = fig_tmp.shape[0]
            fig_total = np.ones((fig_hei*N_term, fig_wid*(N_term*2), 3)) * 255
            if i < term_num - 1:
                fig_total[N*fig_hei:(N+1)*fig_hei, t*fig_wid:(t+1)*fig_wid, :] = fig_tmp.copy()
            else:
                fig_total[(N-1)*fig_hei:(N)*fig_hei, (t+1)*fig_wid:(t+2)*fig_wid, :] = fig_tmp.copy()
        else:
            if i < term_num - 1:
                fig_total[N*fig_hei:(N+1)*fig_hei, t*fig_wid:(t+1)*fig_wid, :] = fig_tmp.copy()
            else:
                fig_total[(N-1)*fig_hei:(N)*fig_hei, (N_term*2-1)*fig_wid:(N_term*2)*fig_wid, :] = fig_tmp.copy()
    
    cv2.imwrite(file_dir_out + f"{fname}_N{N_term}_{term_num}.png", fig_total)
    
def calc_zernike_norm(zernike_array):
    list_of_norm = list()
    sum_0 = np.nansum(zernike_array[0])
    for i in range(zernike_array.shape[0]):
        val = np.nansum(zernike_array[i] * zernike_array[i]) / sum_0
        val = np.sqrt(1 / val)
        list_of_norm.append(val)
    return list_of_norm
        
def data_mounting(data, shape):
    wid = shape[0]
    hei = shape[1]
    scale_0 = data.shape[1]/data.shape[0]
    scale_1 = wid/hei 
    if scale_1 < scale_0:
        data = cv2.resize(data, (wid,int(wid/scale_0)))
        val_1 = (hei - int(wid/scale_0)) // 2
        val_2 = hei - int(wid/scale_0) - val_1
        if data.ndim == 1:
            res = np.pad(data, ((val_1,val_2),(0,0)), "constant", constant_values=(150,150))
        if data.ndim == 3:
            res = np.pad(data, ((val_1,val_2),(0,0),(0,0)), "constant", constant_values=(150,150))
    elif scale_1 > scale_0:
        data = cv2.resize(data, (int(hei*scale_0),hei))
        val_1 = (int(hei/scale_0) - wid) // 2
        val_2 = int(hei/scale_0) - wid - val_1
        if data.ndim == 1:
            res = np.pad(data, ((0,0),(val_1,val_2)), "constant", constant_values=(150,150))
        if data.ndim == 3:
            res = np.pad(data, ((0,0),(val_1,val_2),(0,0)), "constant", constant_values=(150,150))
    else:
        res = cv2.resize(data, shape)
    
    return res

def count_decimal_places(val):
    places = 0
    value = val
    while True:
        if value == int(value):
            break
        value *= 10
        places += 1
    return places
    
def calc_zernike_given_rms(rms_set, list_of_fzp, list_of_flag):
    if len(list_of_fzp) != len(list_of_flag):
        print("Error: len(list_of_fzp) < len(list_of_flag)")
        sys.exit(0)
    term_num = len(list_of_fzp)
    zer_list_by_rms = [0] * term_num
    term_num = len(list_of_fzp)
    decimal = 2
    rms_sqr = rms_set ** 2
    space = count_decimal_places(rms_sqr)
    scale_1 = 10 ** space
    scale_2 = 10 ** decimal
    sum_v = int(rms_sqr * scale_1 * scale_2)
    term_num_of_flag = len([i for i in list_of_flag if i==1])
    a = random.sample(range(1,sum_v), k = term_num_of_flag-1)
    a.append(0)
    a.append(sum_v)
    a = sorted(a)
    b = [(a[i]-a[i-1])/scale_1/scale_2 for i in range(1,len(a))]
    t = 0
    list_of_fzp_norm = calc_zernike_norm(np.asarray(list_of_fzp))
    for i in range(term_num):
        if list_of_flag[i] == 1:
            norm = list_of_fzp_norm[i]
            val = np.sqrt(b[t]*norm*norm)
            t += 1
            zer_list_by_rms[i] = val
    return zer_list_by_rms  
 
def calc_wavefront_given_zernike(list_of_fzp, zer_list, sampling):
    if len(list_of_fzp) != len(zer_list):
        print("Error: len(list_of_fzp) < len(zer_list)")
        sys.exit(0)
    term_num = len(list_of_fzp)
    res = np.zeros((sampling,sampling))
    for i in range(term_num):
        res += list_of_fzp[i] * zer_list[i]
    return res   

def calc_rms(data, nan_flag = True):
    if nan_flag == True:
        num = len(data[~np.isnan(data)])
        rms = np.sqrt(np.nansum(data**2)/num - (np.nansum(data)/num)**2)
    else:
        num = len(data[~np.equal(data, 0)])
        rms = np.sqrt(np.sum(data**2)/num - (np.sum(data)/num)**2)
    return rms
      
def calc_pv(data, nan_flag = True):
    if nan_flag == True:
        pv = np.nanmax(data) - np.nanmin(data)
    else:
        pv = np.max(data) - np.min(data)
    return pv
      
       
if __name__ == '__main__':
    M = 1
    R = M
    sampling = 201
    
    ddx = np.linspace(-M, M, sampling)
    mesh_x, mesh_y = np.meshgrid(ddx, ddx)
    mesh_rho = np.sqrt(mesh_x ** 2 + mesh_y ** 2)
    mesh_theta = np.arctan2(mesh_y, mesh_x)
    mask = np.zeros_like(mesh_rho)
    mask_nan = np.zeros_like(mesh_rho)
    mask[mesh_rho <= R] = 1
    mask[mesh_rho > R] = 0
    mask_nan[mesh_rho <= R] = 1
    mask_nan[mesh_rho > R] = np.nan
    
    R_left = -R * 2/3
    mask_new = np.zeros_like(mesh_rho)
    mask_new_nan = np.zeros_like(mesh_rho)
    mask_new[mesh_rho <= R] = 1
    mask_new[mesh_rho > R] = 0
    mask_new[mesh_x < R_left] = 0
    mask_new_nan[mesh_rho <= R] = 1
    mask_new_nan[mesh_rho > R] = np.nan
    mask_new_nan[mesh_x < R_left] = np.nan
    
    mesh_rho = mesh_rho * mask
    mesh_theta = mesh_theta * mask
    
    # 6 --> 37
    N_term = 3
    
    list_of_Nnm_fzp = list()
    list_of_Nnm_fzp.append([0,0,0,0])
    for N in range(1, N_term):
        idx = 0
        for n in range(N, N*2+1):
            if N*2 - n == 0:
                list_of_Nnm_fzp.append([N,n,0,idx])
                idx += 1
            else:
                list_of_Nnm_fzp.append([N,n,N*2-n,idx])
                idx += 1
                list_of_Nnm_fzp.append([N,n,n-N*2,idx])
                idx += 1
    list_of_Nnm_fzp.append([N_term,N_term*2,0,0])
                
    term_num = len(list_of_Nnm_fzp)
    print(f"N_term = {N_term}   term_num = {term_num}")
    list_of_fzp = list()
    list_of_fzp_nonan = list()
    for idx in range(term_num):
        n = list_of_Nnm_fzp[idx][1]
        m = list_of_Nnm_fzp[idx][2]
        Rnm = np.zeros_like(mesh_rho)
        for s in range((n-np.abs(m))//2 + 1):
            tmp = (-1)**s * math.factorial(n-s) / math.factorial(s) / \
                math.factorial((n+np.abs(m))//2-s) / math.factorial((n-np.abs(m))//2-s)
            Rnm += (tmp * mesh_rho ** (n-s*2))
        if m >= 0:
            Rnm *= np.cos(mesh_theta * np.abs(m))
        else:
            Rnm *= np.sin(mesh_theta * np.abs(m))
        list_of_fzp.append(Rnm * mask_nan)
        list_of_fzp_nonan.append(Rnm)
        
    file_dir = "C:\\code\\optics\\How_To\\wf_given_rms\\"
    file_dir_tmp = file_dir + "tmp\\"
    file_dir_out = file_dir
    
    if not os.path.exists(file_dir_tmp):
        os.makedirs(file_dir_tmp)
    if not os.path.exists(file_dir_out):
        os.makedirs(file_dir_out)
    
    # save_figure_of_fzp(list_of_fzp, list_of_Nnm_fzp, file_dir_tmp, file_dir_out, "fig_A") 
    
    '''
        Given rms, the proper zernike values will be estimated, and then the corresponding wavefront will be built
    '''
    rms_set = 0.2
    list_of_flag = [0] * term_num
    for i in range(4, term_num):
        list_of_flag[i] = 1
        
    zer_list_by_rms = calc_zernike_given_rms(rms_set, list_of_fzp, list_of_flag)
    
    wf_given_rms = calc_wavefront_given_zernike(list_of_fzp, zer_list_by_rms, sampling)
    
    rms_real = round(calc_rms(wf_given_rms, True), 5)
    print(f"rms_real = {rms_real}")
    print("zer_list_by_rms:")
    print(zer_list_by_rms)    
    plt.figure(1)
    plt.title(f"rms: set={rms_set} , real={rms_real}")
    plt.imshow(wf_given_rms, cmap = 'rainbow')
    plt.colorbar()
    plt.savefig(file_dir_out + "fig_A_wf_given_rms" + ".png")
    plt.close()   
    
    '''
        if the apod is not full-shape
    '''
    list_of_fzp_new, list_of_fzp_new_nonan, matrix_transfer = re_ortho_FZP(list_of_fzp_nonan, mask_new_nan)   
    # save_figure_of_fzp(list_of_fzp_new, list_of_Nnm_fzp, file_dir_tmp, file_dir_out, "fig_D")
    
    zer_list_by_rms_new = calc_zernike_given_rms(rms_set, list_of_fzp_new, list_of_flag)
    wf_given_rms_new = calc_wavefront_given_zernike(list_of_fzp_new, zer_list_by_rms_new, sampling)
        
    rms_real_new = round(calc_rms(wf_given_rms_new, True), 5)
    print(f"rms_real_new = {rms_real_new}")
    print("zer_list_by_rms_new:")
    print(zer_list_by_rms_new)    
    plt.figure(1)
    plt.title(f"rms: set={rms_set} , real={rms_real_new}")
    plt.imshow(wf_given_rms_new, cmap = 'rainbow')
    plt.colorbar()
    plt.savefig(file_dir_out + "fig_D_wf_given_rms" + ".png")
    plt.close()        
    
    sys.exit(0)
    
    
    
