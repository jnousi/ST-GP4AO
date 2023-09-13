from hcipy import *
import h5py


import numpy as np
import matplotlib
matplotlib.use('nbagg')
import matplotlib.pyplot as plt

import scipy.ndimage as ndimage
import scipy
import time
import os

from scipy.special import gamma, kv


def s_gp(grid, r0, L0):
       
    cov_function = phase_covariance_von_karman(r0, L0)
    nd = grid.shape[0]*grid.shape[1]
    cov_mat_spatial = np.zeros((nd,nd))

    for x in range(nd):

        shifted_grid = grid.shifted(grid[x])
        cov_mat_spatial[:,x] = np.flip(cov_function(shifted_grid), axis=0)
        #print(cov_function(shifted_grid))
        
    return cov_mat_spatial

def st_gp_ff(timesteps, Cn2, winds, cov_mat_spatial, grid, r0, L0):
    
    cov_function = phase_covariance_von_karman(r0, L0)
    nlayer = Cn2.shape[0]
    nd = grid.shape[0]*grid.shape[1]
    cov_mat = np.zeros((timesteps*nd,timesteps*nd))

    for nl in range(nlayer):

        cov_mat_layer = np.zeros((timesteps*nd,timesteps*nd))
        cov_mat_layer = scipy.linalg.block_diag(*([cov_mat_spatial] * timesteps))

        for t in range(1,timesteps):
            cov_mat_shifted= np.zeros((nd,nd))
            for x in range(nd):
                shifted_grid = grid.shifted(grid[x] + winds[nl]*(t) + 1e-8)            
                cov_mat_shifted[:,x] = np.flip(cov_function(shifted_grid), axis = 0)
            # fill right blocks
            temp = scipy.linalg.block_diag(*([cov_mat_shifted] * (timesteps-t)))
            cov_mat_layer[t*nd:timesteps*nd,0:(timesteps-t)*nd] += temp 
            cov_mat_layer[0:(timesteps-t)*nd,t*nd:timesteps*nd] += temp.transpose()                            

        cov_mat += (Cn2[nl]) * cov_mat_layer

    return cov_mat

def st_gp_waff(timesteps, n_int, avg_wind, cov_mat_spatial, grid, r0, L0):
    
    cov_function = phase_covariance_von_karman(r0, L0)   
    nd = grid.shape[0]*grid.shape[1]
    cov_mat = np.zeros((timesteps*nd,timesteps*nd))
    
    winds = avg_wind*evenly_spaced_unit_circle(n_int) 

    for nl in range(n_int):

        cov_mat_layer = np.zeros((timesteps*nd,timesteps*nd))
        cov_mat_layer = scipy.linalg.block_diag(*([cov_mat_spatial] * timesteps))

        for t in range(1,timesteps):
            cov_mat_shifted= np.zeros((nd,nd))
            for x in range(nd):
                shifted_grid = grid.shifted(grid[x] + winds[nl]*(t) + 1e-8)            
                cov_mat_shifted[:,x] = np.flip(cov_function(shifted_grid), axis = 0)
            # fill right blocks
            temp = scipy.linalg.block_diag(*([cov_mat_shifted] * (timesteps-t)))
            cov_mat_layer[t*nd:timesteps*nd,0:(timesteps-t)*nd] += temp 
            cov_mat_layer[0:(timesteps-t)*nd,t*nd:timesteps*nd] += temp.transpose()                            

        cov_mat += (1/n_int) * cov_mat_layer

    return cov_mat


def vec_to_img(n_pixels, xvalid, yvalid):
    
    def func(phase_vec):
        phase_im = np.zeros((n_pixels,n_pixels))
        phase_im[xvalid, yvalid] = phase_vec
        
        return phase_im
    return func

def img_to_vec(xvalid, yvalid):       
    
    def func(phase_im):
        return phase_im[xvalid, yvalid]
    return func

def vec_to_img_st(n_pixels, timesteps, tvalid, xvalid, yvalid):
    
    def func(phase_vec):
        phase_im = np.zeros((timesteps, n_pixels, n_pixels))      
        phase_im[tvalid, xvalid, yvalid] = phase_vec
        
        return phase_im
    return func

def evenly_spaced_unit_circle(num_vectors):
    angles = np.linspace(0, 2 * np.pi, num_vectors, endpoint=False)
    vectors = np.array([(np.cos(angle), np.sin(angle)) for angle in angles])
    return vectors