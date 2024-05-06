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
    cov_mat_spatial = np.zeros((nd,nd), dtype=np.float32)

    for x in range(nd):

        shifted_grid = grid.shifted(grid[x])
        cov_mat_spatial[:,x] = np.flip(cov_function(shifted_grid), axis=0)
        #print(cov_function(shifted_grid))
        
    return cov_mat_spatial

def s_gp_validpix(grid, r0, L0, valid_pixels):
       
    cov_function = phase_covariance_von_karman_validpix(r0, L0)
    nd = valid_pixels.shape[0]
    cov_mat_spatial = np.zeros((nd,nd), dtype=np.float32)
    index = 0
    for x in valid_pixels:

        shifted_grid = grid.shifted(grid[x])
        cov_mat_spatial[:,index] = np.flip(cov_function(shifted_grid, valid_pixels), axis=0)
        index += 1
        
    return cov_mat_spatial

def st_gp_ff(timesteps, Cn2, winds, cov_mat_spatial, grid, r0, L0):
    
    cov_function = phase_covariance_von_karman(r0, L0)
    nlayer = Cn2.shape[0]
    nd = grid.shape[0]*grid.shape[1]
    cov_mat = np.zeros((timesteps*nd,timesteps*nd), dtype=np.float32)

    for nl in range(nlayer):

        cov_mat_layer = np.zeros((timesteps*nd,timesteps*nd), dtype=np.float32)
        cov_mat_layer = scipy.linalg.block_diag(*([cov_mat_spatial] * timesteps))

        for t in range(1,timesteps):
            cov_mat_shifted= np.zeros((nd,nd), dtype=np.float32)
            for x in range(nd):
                shifted_grid = grid.shifted(grid[x] + winds[nl]*(t) + 1e-8)            
                cov_mat_shifted[:,x] = np.flip(cov_function(shifted_grid), axis = 0)
            # fill right blocks
            temp = scipy.linalg.block_diag(*([cov_mat_shifted] * (timesteps-t)))
            cov_mat_layer[t*nd:timesteps*nd,0:(timesteps-t)*nd] += temp 
            cov_mat_layer[0:(timesteps-t)*nd,t*nd:timesteps*nd] += temp.transpose()                            

        cov_mat += (Cn2[nl]) * cov_mat_layer

    return cov_mat

def st_gp_ff_validpix(timesteps, Cn2, winds, cov_mat_spatial, grid, r0, L0, valid_pixels):
    
    cov_function = phase_covariance_von_karman_validpix(r0, L0)
    nlayer = Cn2.shape[0]
    nd = valid_pixels.shape[0]
    cov_mat = np.zeros((timesteps*nd,timesteps*nd), dtype=np.float32)

    for nl in range(nlayer):

        cov_mat_layer = np.zeros((timesteps*nd,timesteps*nd), dtype=np.float32)
        cov_mat_layer = scipy.linalg.block_diag(*([cov_mat_spatial] * timesteps))

        for t in range(1,timesteps):
            cov_mat_shifted= np.zeros((nd,nd), dtype=np.float32)
            index = 0
            for x in valid_pixels:
                shifted_grid = grid.shifted(grid[x] + winds[nl]*(t) + 1e-8)            
                cov_mat_shifted[:,index] = np.flip(cov_function(shifted_grid, valid_pixels), axis = 0)
                index += 1
            # fill right blocks
            temp = scipy.linalg.block_diag(*([cov_mat_shifted] * (timesteps-t)))
            cov_mat_layer[t*nd:timesteps*nd,0:(timesteps-t)*nd] += temp 
            cov_mat_layer[0:(timesteps-t)*nd,t*nd:timesteps*nd] += temp.transpose()                            

        cov_mat += (Cn2[nl]) * cov_mat_layer

    return cov_mat

def add_recon_grid_ff(cov_mat_st, cov_mat_spatial, Cn2, winds, r0, L0,  n_history, n_predict, data_grid, recon_grid, valid_pixels, valid_pixels_recon):
        
    cov_function = phase_covariance_von_karman(r0, L0)
    nlayer = Cn2.shape[0]
    nd_recon = valid_pixels_recon.shape[0]
    nd_data  = valid_pixels.shape[0]
    
    cov_function = phase_covariance_von_karman_validpix(r0, L0)
    cov_mat = np.block([[cov_mat_st, np.zeros((cov_mat_st.shape[0], cov_mat_spatial.shape[1]), dtype=np.float32)],
                          [np.zeros((cov_mat_spatial.shape[0], cov_mat_st.shape[1]), dtype=np.float32), cov_mat_spatial]])
    
    for t in range(0,n_history):
        temp = np.zeros((nd_recon,nd_data), dtype=np.float32)
        for nl in range(nlayer):
            index = 0
            
            cov_mat_shifted = np.zeros((nd_recon,nd_data), dtype=np.float32)
            for x in valid_pixels:
                shifted_grid = recon_grid.shifted(data_grid[valid_pixels[index]] + winds[nl]*(t + (n_predict) ) + 1e-8)    
                cov_mat_shifted[:,index] = np.flip(cov_function(shifted_grid, valid_pixels_recon), axis = 0)
                # fill right blocks
                index +=1

            temp += (Cn2[nl]) * cov_mat_shifted
            
        cov_mat[nd_data*n_history:,-nd_recon-(t+1)*nd_data:-nd_recon-(t)*nd_data ] =  temp
        cov_mat[-nd_recon-(t+1)*nd_data:-nd_recon-(t)*nd_data,nd_data*n_history:]  = temp.transpose()
        
    return cov_mat

#def st_gp_ff_custom_grids(Cn2, winds, r0, L0, n_history, n_predict, data_grid, recon_grid, valid_pixels, valid_pixels_recon):
    
def st_gp_waff(timesteps, n_int, avg_wind, cov_mat_spatial, grid, r0, L0):
    
    cov_function = phase_covariance_von_karman(r0, L0)   
    nd = grid.shape[0]*grid.shape[1]
    cov_mat = np.zeros((timesteps*nd,timesteps*nd), dtype=np.float32)
    
    winds = avg_wind*evenly_spaced_unit_circle(n_int) 

    for nl in range(n_int):

        cov_mat_layer = np.zeros((timesteps*nd,timesteps*nd), dtype=np.float32)
        cov_mat_layer = scipy.linalg.block_diag(*([cov_mat_spatial] * timesteps))

        for t in range(1,timesteps):
            cov_mat_shifted= np.zeros((nd,nd), dtype=np.float32)
            for x in range(nd):
                shifted_grid = grid.shifted(grid[x] + winds[nl]*(t) + 1e-8)            
                cov_mat_shifted[:,x] = np.flip(cov_function(shifted_grid), axis = 0)
            # fill right blocks
            temp = scipy.linalg.block_diag(*([cov_mat_shifted] * (timesteps-t)))
            cov_mat_layer[t*nd:timesteps*nd,0:(timesteps-t)*nd] += temp 
            cov_mat_layer[0:(timesteps-t)*nd,t*nd:timesteps*nd] += temp.transpose()                            

        cov_mat += (1/n_int) * cov_mat_layer

    return cov_mat


def st_gp_waff_validpix(timesteps, n_int, avg_wind, cov_mat_spatial, grid, r0, L0, valid_pixels):
    
    cov_function = phase_covariance_von_karman_validpix(r0, L0)
    nd = valid_pixels.shape[0]
    cov_mat = np.zeros((timesteps*nd,timesteps*nd), dtype=np.float32)
    
    winds = avg_wind*evenly_spaced_unit_circle(n_int) 

    for nl in range(n_int):

        cov_mat_layer = np.zeros((timesteps*nd,timesteps*nd), dtype=np.float32)
        cov_mat_layer = scipy.linalg.block_diag(*([cov_mat_spatial] * timesteps))

        for t in range(1,timesteps):
            cov_mat_shifted= np.zeros((nd,nd), dtype=np.float32)
            index = 0
            for x in valid_pixels:
                shifted_grid = grid.shifted(grid[x] + winds[nl]*(t) + 1e-8)            
                cov_mat_shifted[:,index] = np.flip(cov_function(shifted_grid, valid_pixels), axis = 0)
                index += 1
            # fill right blocks
            temp = scipy.linalg.block_diag(*([cov_mat_shifted] * (timesteps-t)))
            cov_mat_layer[t*nd:timesteps*nd,0:(timesteps-t)*nd] += temp 
            cov_mat_layer[0:(timesteps-t)*nd,t*nd:timesteps*nd] += temp.transpose()                            

        cov_mat += (1/n_int) * cov_mat_layer

    return cov_mat



def add_recon_grid_waff(cov_mat_st, cov_mat_spatial, n_int, avg_wind, r0, L0,  n_history, n_predict, data_grid, recon_grid, valid_pixels, valid_pixels_recon):
        
    cov_function = phase_covariance_von_karman(r0, L0)
    nd_recon = valid_pixels_recon.shape[0]
    nd_data  = valid_pixels.shape[0]
    
    winds = avg_wind*evenly_spaced_unit_circle(n_int) 
    
    cov_function = phase_covariance_von_karman_validpix(r0, L0)
    cov_mat = np.block([[cov_mat_st, np.zeros((cov_mat_st.shape[0], cov_mat_spatial.shape[1]),  dtype=np.float32)],
                          [np.zeros((cov_mat_spatial.shape[0], cov_mat_st.shape[1]), dtype=np.float32), cov_mat_spatial]])
    
    for t in range(0,n_history):
        temp = np.zeros((nd_recon,nd_data), dtype=np.float32)
        for nl in range(n_int):
            index = 0
            
            cov_mat_shifted = np.zeros((nd_recon,nd_data), dtype=np.float32)
            for x in valid_pixels:
                shifted_grid = recon_grid.shifted(data_grid[valid_pixels[index]] + winds[nl]*(t + (n_predict) ) + 1e-8)    
                cov_mat_shifted[:,index] = np.flip(cov_function(shifted_grid, valid_pixels_recon), axis = 0)
                # fill right blocks
                index +=1

            temp += (1/n_int) * cov_mat_shifted
            
        cov_mat[nd_data*n_history:,-nd_recon-(t+1)*nd_data:-nd_recon-(t)*nd_data ] =  temp
        cov_mat[-nd_recon-(t+1)*nd_data:-nd_recon-(t)*nd_data,nd_data*n_history:]  = temp.transpose()
        
    return cov_mat



def temporal_cross_correlation(timediff, Cn2, winds, cov_mat_spatial, grid, r0, L0, valid_pixels):
    
    cov_function = phase_covariance_von_karman_validpix(r0, L0)
    nlayer = Cn2.shape[0]
    nd = valid_pixels.shape[0]
    cov_mat = np.zeros((nd,nd), dtype=np.float32)

    for nl in range(nlayer):
        
        cov_mat_shifted= np.zeros((nd,nd), dtype=np.float32)
        index = 0
        for x in valid_pixels:
            shifted_grid = grid.shifted(grid[x] + winds[nl]*(timediff) + 1e-8)            
            cov_mat_shifted[:,index] = np.flip(cov_function(shifted_grid, valid_pixels), axis = 0)
            index += 1
                          
        cov_mat += (Cn2[nl]) * cov_mat_shifted

    return cov_mat

def phase_covariance_von_karman_validpix(r0, L0):
    
    def func(grid, valid_pixels):
        r = grid.as_('polar').r + 1e-10
        r = r[valid_pixels] 
        a = (L0 / r0)**(5 / 3)
        b = gamma(11 / 6) / (2**(5 / 6) * np.pi**(8 / 3))
        c = (24 / 5 * gamma(6 / 5))**(5 / 6)
        d = (2 * np.pi * r / L0)**(5 / 6)
        e = kv(5 / 6, 2 * np.pi * r / L0)

        return Field(a * b * c * d * e, grid[valid_pixels])
    return func

def get_wind_vector(speeds, angles, framerate):
    
    winds = np.zeros((speeds.size, 2))
    
    for i in range(speeds.size):
        x = np.cos(angles[i])
        y = np.sin(angles[i])
        winds[i,:] = framerate*(speeds[i])*np.array([x, y])
    
    return winds


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

def get_non_empty_column_indices(matrix):
    empty_column_indices = np.where(~np.all(matrix == 0, axis=0))[0]
    return empty_column_indices