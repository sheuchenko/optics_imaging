# -*- coding: utf-8 -*-

import numpy as np
import cv2
import sys
import scipy
import matplotlib.pyplot as plt
from scipy.sparse.linalg import svds

def calc_rms(data):
    num = len(data[np.isnan(data)])
    # print("num", num)
    if num != 0:
        num = len(data[~np.isnan(data)])
        rms = np.sqrt(np.nansum(data**2)/num - (np.nansum(data)/num)**2)
    else:
        rms = -1
    return rms
      
def calc_pv(data):
    num = len(data[np.isnan(data)])
    if num != 0:
        pv = np.nanmax(data) - np.nanmin(data)
    else:
        pv = np.max(data) - np.min(data)
    return pv
      
def FFT(data):
    return np.fft.fftshift(np.fft.fft2(np.fft.ifftshift(data)))

def iFFT(data):
    return np.fft.fftshift(np.fft.ifft2(np.fft.ifftshift(data)))

def image_pad(data, m = 1):
    n = 2 * m + 1
    h = data.shape[0]
    w = data.shape[1]
    data_pad = np.pad(data,((h*m,h*m), (w*m,w*m)),'constant', constant_values=(data[0,0],data[-1,-1]))
    for i in range(n):
        for j in range(n):
            if i == m and j < m:
                data_pad[h*j:h*(j+1), w*i:w*(i+1)] = data[0,:]
            if i == m and j > m:
                data_pad[h*j:h*(j+1), w*i:w*(i+1)] = data[-1,:]
            if i < m and j == m:
                data_pad[h*j:h*(j+1), w*i:w*(i+1)] = data[:,0]
            if i > m and j == m:
                data_pad[h*j:h*(j+1), w*i:w*(i+1)] = data[:,-1]
    return data_pad

def image_unpad(data_pad, m = 1):
    n = 2 * m + 1
    h = data_pad.shape[0] // n
    w = data_pad.shape[1] // n
    data = data_pad[h*m:h*(m+1),w*m:w*(m+1)]
    return data

def resize_cplx(data, size_new):
    out_real = cv2.resize(np.real(data), size_new, interpolation=cv2.INTER_CUBIC)
    out_imag = cv2.resize(np.imag(data), size_new, interpolation=cv2.INTER_CUBIC)
    return out_real + 1j * out_imag
    
def norm(data):
    return (data-np.min(data)) / (np.max(data)-np.min(data))

def build_phase_base_on_zernike(zer_list, mesh_rho, mesh_theta):
    sampling = mesh_rho.shape[0]
    zernike_array = np.zeros((37,sampling,sampling))
    
    zernike_array[0,:,:] = 1
    zernike_array[1,:,:] = mesh_rho * np.cos(mesh_theta)
    zernike_array[2,:,:] = mesh_rho * np.sin(mesh_theta)
    zernike_array[3,:,:] = 2 * mesh_rho ** 2 - 1
    zernike_array[4,:,:] = mesh_rho ** 2 * np.cos(2 * mesh_theta)
    zernike_array[5,:,:] = mesh_rho ** 2 * np.sin(2 * mesh_theta)
    zernike_array[6,:,:] = (3*mesh_rho**2 - 2) * mesh_rho * np.cos(mesh_theta)
    zernike_array[7,:,:] = (3*mesh_rho**2 - 2) * mesh_rho * np.sin(mesh_theta)
    zernike_array[8,:,:] = 6*mesh_rho**4-6*mesh_rho**2+1
    zernike_array[9,:,:] = mesh_rho**3 * np.cos(3*mesh_theta)
    zernike_array[10,:,:] = mesh_rho**3 * np.sin(3*mesh_theta)
    zernike_array[11,:,:] = (4*mesh_rho**2-3)*mesh_rho**2*np.cos(2*mesh_theta)
    zernike_array[12,:,:] = (4*mesh_rho**2-3)*mesh_rho**2*np.sin(2*mesh_theta)
    zernike_array[13,:,:] = (10*mesh_rho**4-12*mesh_rho**2+3)*mesh_rho*np.cos(mesh_theta)
    zernike_array[14,:,:] = (10*mesh_rho**4-12*mesh_rho**2+3)*mesh_rho*np.sin(mesh_theta)
    zernike_array[15,:,:] = 20*mesh_rho**6-30*mesh_rho**4+12*mesh_rho**2-1
    zernike_array[16,:,:] = mesh_rho**4*np.cos(4*mesh_theta)
    zernike_array[17,:,:] = mesh_rho**4*np.sin(4*mesh_theta)
    zernike_array[18,:,:] = (5*mesh_rho**2-4)*mesh_rho**3*np.cos(2*mesh_theta)
    zernike_array[19,:,:] = (5*mesh_rho**2-4)*mesh_rho**3*np.sin(2*mesh_theta)
    zernike_array[20,:,:] = (15*mesh_rho**4-20*mesh_rho**2+6)*mesh_rho**2*np.cos(2*mesh_theta)
    zernike_array[21,:,:] = (15*mesh_rho**4-20*mesh_rho**2+6)*mesh_rho**2*np.sin(2*mesh_theta)
    zernike_array[22,:,:] = (35*mesh_rho**6-60*mesh_rho**4+30*mesh_rho**2-4)*mesh_rho*np.cos(mesh_theta)
    zernike_array[23,:,:] = (35*mesh_rho**6-60*mesh_rho**4+30*mesh_rho**2-4)*mesh_rho*np.sin(mesh_theta)
    zernike_array[24,:,:] = 70*mesh_rho**8-140*mesh_rho**6+90*mesh_rho**4-20*mesh_rho**2-1
    zernike_array[25,:,:] = mesh_rho**5*np.cos(5*mesh_theta)
    zernike_array[26,:,:] = mesh_rho**5*np.sin(5*mesh_theta)
    zernike_array[27,:,:] = (6*mesh_rho**2-5)*mesh_rho**4*np.cos(4*mesh_theta)
    zernike_array[28,:,:] = (6*mesh_rho**2-5)*mesh_rho**4*np.sin(4*mesh_theta)
    zernike_array[29,:,:] = (21*mesh_rho**4-30*mesh_rho**2+10)*mesh_rho**3*np.cos(3*mesh_theta)
    zernike_array[30,:,:] = (21*mesh_rho**4-30*mesh_rho**2+10)*mesh_rho**3*np.sin(3*mesh_theta)
    zernike_array[31,:,:] = (56*mesh_rho**6-105*mesh_rho**4+60*mesh_rho**2-10)*mesh_rho**2*np.cos(2*mesh_theta)
    zernike_array[32,:,:] = (56*mesh_rho**6-105*mesh_rho**4+60*mesh_rho**2-10)*mesh_rho**2*np.sin(2*mesh_theta)
    zernike_array[33,:,:] = (126*mesh_rho**8-280*mesh_rho**6+210*mesh_rho**4-60*mesh_rho**2+5)*mesh_rho*np.cos(mesh_theta)
    zernike_array[34,:,:] = (126*mesh_rho**8-280*mesh_rho**6+210*mesh_rho**4-60*mesh_rho**2+5)*mesh_rho*np.sin(mesh_theta)
    zernike_array[35,:,:] = 252*mesh_rho**10-630*mesh_rho**8+560*mesh_rho**6-210*mesh_rho**4+30*mesh_rho**2-1
    zernike_array[36,:,:] = 924*mesh_rho**12-2772*mesh_rho**10+3150*mesh_rho**8-1680*mesh_rho**6+420*mesh_rho**4-40*mesh_rho**2+1

    phase = np.zeros((sampling, sampling), dtype=np.float32)
    for idx in range(len(zer_list)):
        phase += zernike_array[idx,:,:] * zer_list[idx]
    phase = np.flipud(phase)
    
    return phase

def build_wf_cplx_ideal(sampling, NA_obj, wl_um, pix_img_um, mag, zer_list = [0]):
    R = NA_obj / wl_um
    M = 1 / 2 / (pix_img_um / mag)  
    confidence = (M/2)/R
    print(f"wf confidence is {confidence}(greater than 1 is OK), phase sampling is {int(sampling * R / M)}")
    if confidence <= 1:
        pix_img_um_should_be = 0.25 / R 
        print(f"pix_img_um is too large, and it shpuld be {pix_img_um_should_be} at least")
        
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
    
    mesh_rho = mesh_rho * mask
    mesh_theta = mesh_theta * mask
    
    apod = mask
    phase = build_phase_base_on_zernike(zer_list, mesh_rho, mesh_theta) * mask
    wf_cplx = apod * np.exp(1j * 1 * np.pi * phase)
    
    return wf_cplx, apod, phase, mask_nan
    
def build_source(sampling, NA_obj, wl_um, pix_img_um, mag, sigma_in, sigma_out):
    R = NA_obj / wl_um
    M = 1 / 2 / (pix_img_um / mag)      
    ddx = np.linspace(-M, M, sampling)
    mesh_x, mesh_y = np.meshgrid(ddx, ddx)
    mesh_rho = np.sqrt(mesh_x ** 2 + mesh_y ** 2)
    source = np.zeros_like(mesh_rho)
    source[mesh_rho <= sigma_out * R] = 1
    source[mesh_rho < sigma_in * R] = 0    
    return source
    
def imaging(data_in, NA_obj, wl_um, pix_img_um, mag, coef_cohere, zer_list = [0]):
    pad_scale = 1
    msize_pad = data_in.shape[0] * (2*pad_scale +1)
    wf_cplx, apod, phase, mask_nan = build_wf_cplx_ideal(msize, NA_obj, wl_um, pix_img_um, mag, zer_list)
    out = np.zeros_like(data_in)
    if coef_cohere <= 1e-3:
        # data_in should be amplitude
        amplitude = image_pad(np.sqrt(data_in), pad_scale)
        wf_cplx_resize = resize_cplx(wf_cplx, (msize_pad, msize_pad))
        out_amp = iFFT(FFT(amplitude) * wf_cplx_resize)
        out_intens = np.real(out_amp * np.conj(out_amp))
        out = image_unpad(out_intens, pad_scale)
    elif coef_cohere >= 1 - 1e-3:
        # data_in should be intensity
        intensity = image_pad(data_in, pad_scale)
        wf_cplx_resize = resize_cplx(wf_cplx, (msize_pad, msize_pad))
        psf = np.abs(iFFT(wf_cplx_resize))**2
        psf = psf / np.sum(psf)
        out = np.real(iFFT(FFT(intensity) * FFT(psf)))
        out = image_unpad(out, pad_scale)
        
    else:
        NA_source = NA_obj
        source = build_source(msize, NA_source, wl_um, pix_img_um, mag, 0, coef_cohere)
        
        list_of_source_index = np.argwhere(source*apod == 1)
        N = len(list_of_source_index)
        
        apod_pro = np.zeros((N, msize*msize))
        for i in range(N):
            tx = list_of_source_index[i][1] - msize//2
            ty = list_of_source_index[i][0] - msize//2
            apod_shift = np.roll(apod, tx, axis=1)
            apod_shift = np.roll(apod_shift, ty, axis=0)
            apod_shift_reshape = apod_shift.reshape(1,-1).T
            apod_pro[i,:] = apod_shift_reshape[:,0]
        
        N_select = np.min([100,N-1])
        # svd_U,svd_sigma,svd_VT = np.linalg.svd(apod_pro)
        svd_U,svd_sigma,svd_VT = svds(apod_pro, N_select)
        svd_sigma = svd_sigma[::-1]
        svd_VT = svd_VT[::-1]
        
        tcc_coefs_origin = svd_sigma ** 2
        N_select_cut = N_select
        norm_of_tcc_coefs = np.sqrt(np.max(tcc_coefs_origin))
        tcc_coefs_origin = tcc_coefs_origin / (norm_of_tcc_coefs**2)
        toler_of_tcc_coefs = 1e-3
        for i in range(len(svd_sigma) - 1):
            if tcc_coefs_origin[i-1] > toler_of_tcc_coefs and tcc_coefs_origin[i] < toler_of_tcc_coefs:
                N_select_cut = i
                break
        tcc_coefs = tcc_coefs_origin[:N_select_cut]
        print(f"TCC is cut: {N} -> {N_select} -> {N_select_cut}")
        
        tcc_array = np.zeros((N_select_cut,msize,msize))
        data_out_partial = np.zeros_like(data_in)
        for i in range(N_select_cut):
            tmp = svd_VT[i,:].reshape(msize,msize)
            tmp[np.less(np.abs(tmp), 1e-8)] = 0
            tcc_array[i,:,:] = tmp * norm_of_tcc_coefs
            
            amplitude = image_pad(np.sqrt(data_in), pad_scale)
            tcc_pad = resize_cplx(tcc_array[i,:,:], (msize_pad,msize_pad))
            out_amp = iFFT(FFT(amplitude) * tcc_pad)
            out_intens = np.real(out_amp * np.conj(out_amp))
            data_out_partial += (tcc_coefs[i] * image_unpad(out_intens, pad_scale))
        out = data_out_partial / N
        
    return out

if __name__ == '__main__':
    
    term_nums = 37
    zernike_list = list()
    zernike_list = [0] * term_nums
    zernike_list[4] = 0.0
    
    NA_obj = 0.5
    wl_um = 0.5
    mag = 100
    pix_img_um = 5
    msize = 201
    
    data_in = np.ones((msize,msize), dtype = np.float32)
    data_in = data_in * 0.2
    data_in[:, msize//2:] = 1
    # data_in[msize//2,msize//2] = 1
    
    data_out_coherence = imaging(data_in, NA_obj, wl_um, pix_img_um, mag, 0, zernike_list)
    
    data_out_incoherence = imaging(data_in, NA_obj, wl_um, pix_img_um, mag, 1, zernike_list)
    
    coef_cohere = 0.5
    
    list_of_coef_cohere = [val/10 for val in range(0,11,2)]
    list_of_data_out_partial = list()
    for coef_cohere in list_of_coef_cohere:      
        print(f"coef_cohere = {coef_cohere}")
        
        data_out_partial = imaging(data_in, NA_obj, wl_um, pix_img_um, mag, coef_cohere, zernike_list)
        list_of_data_out_partial.append(data_out_partial)
        # print(f"{np.max(data_out_coherence)}, {np.max(data_out_incoherence)}, {np.max(data_out_partial)}")
        
    fname = f"NA_obj = {NA_obj}, wl_um = {wl_um}, mag = {mag}, pix_img_um = {pix_img_um}"
    x_coord = np.linspace(0, msize, msize)
    plt.figure(num=1, figsize=(20,15))
    plt.title(fname)
    plt.plot(x_coord, data_in[msize//2,:], color="black", label="in", linestyle='-')
    plt.plot(x_coord, data_out_coherence[msize//2,:], label="coherence")
    plt.plot(x_coord, data_out_incoherence[msize//2,:], label="incoherence")
    for i in range(len(list_of_coef_cohere)):
        data_out_partial = list_of_data_out_partial[i]
        plt.plot(x_coord, data_out_partial[msize//2,:], label=f"partial {list_of_coef_cohere[i]}")
    # plt.ylim(0.0,1.2)
    plt.grid()
    plt.legend()
    # plt.show()
    plt.savefig("C:\\code\\optics\\tcc\\" + "fig_cohere.png")
    plt.close()
    
