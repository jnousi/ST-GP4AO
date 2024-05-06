from hcipy import *
from stgp_util_32 import *
import h5py

import numpy as np
import matplotlib
matplotlib.use('nbagg')
import matplotlib.pyplot as plt

import scipy.ndimage as ndimage
import scipy
import time
import os

filepath = 'matrix_for_GP_16x16.mat'
arrays = {}
f = h5py.File(filepath,'r')
for k, v in f.items():
    arrays[k] = np.array(v)

G = arrays["G_4"]

G = G.transpose().astype(np.float32)
# the full phase matrix
mask_phase = arrays["mask_phase4"]
    
filepath = 'dm4.mat'
arrays = {}
f = h5py.File(filepath,'r')
for k, v in f.items():
    arrays[k] = np.array(v)

M2P = arrays["M2P_dm"]
M2P = M2P.transpose().astype(np.float32)   

P2M = np.linalg.pinv(M2P)
F = M2P @ P2M 

# telescope
telescope_diameter = 3.2

# recontruction grid
n_lenslet          = 16
n_pixels           = 4*n_lenslet

# atmosphere
r0             = 0.168 # m
L0             = 20 # meters
Cn2            = np.array([0.672, 0.051, 0.028, 0.106, 0.08, 0.052, 0.01])
wind_speeds    = np.array([8.5,    6.55,   6.6,   6.7,   22,   9.5,  5.6]) #m/s
wind_angles    = np.array([1.4,   1.57,  1.67,  1.77,  3.1,   3.2,  3.3])  #radian
avg_wind       = np.average(wind_speeds, weights=Cn2)
framerate      = 0.002
winds = get_wind_vector(wind_speeds, wind_angles, framerate)

wavelength_wfs = 0.5e-6 

# model
n_history = 20 # data steps

#n_history = np.array([0,1,2,3])
n_predict = 2 # predective steps

#timesteps = n_history + n_predict
signal_to_noise = 4

recon_grid =  make_pupil_grid(n_pixels, telescope_diameter)
data_grid =  make_pupil_grid(4*n_lenslet, telescope_diameter)

vecvalid          = np.nonzero(mask_phase.flatten())[0]
SH_grid = get_non_empty_column_indices(G)

(xvalid, yvalid) = np.nonzero(mask_phase)

vec_to_im    = vec_to_img(n_pixels, xvalid, yvalid)
im_to_vec    = img_to_vec(xvalid, yvalid)

act_pix    = vecvalid.shape[0]
SH_grid = get_non_empty_column_indices(G)

n_int = 20

Cs_data       = s_gp_validpix(data_grid, r0, L0, vecvalid[SH_grid])
Cs_recon       = s_gp_validpix(recon_grid, r0, L0, vecvalid)

Cff = st_gp_ff_validpix(n_history, Cn2, winds, Cs_data, data_grid, r0, L0, vecvalid[SH_grid])
print('ff data ready')
Cwaff = st_gp_waff_validpix(n_history, n_int, avg_wind*framerate, Cs_data, data_grid, r0, L0, vecvalid[SH_grid])
print('waff data ready')

Cff = add_recon_grid_ff(Cff, Cs_recon, Cn2, winds, r0, L0, n_history, n_predict, data_grid, recon_grid, vecvalid[SH_grid], vecvalid)
print('ff recon ready')
Cwaff = add_recon_grid_waff(Cwaff, Cs_recon, n_int, avg_wind*framerate, r0, L0, n_history, n_predict, data_grid, recon_grid, vecvalid[SH_grid], vecvalid)

print('waff recon ready')

recon_grid =  make_pupil_grid(n_pixels, telescope_diameter)
data_grid =  make_pupil_grid(4*n_lenslet, telescope_diameter)

vecvalid          = np.nonzero(mask_phase.flatten())[0]
SH_grid = get_non_empty_column_indices(G)

(xvalid, yvalid) = np.nonzero(mask_phase)

vec_to_im    = vec_to_img(n_pixels, xvalid, yvalid)
im_to_vec    = img_to_vec(xvalid, yvalid)

# numeber of active pixels
act_pix    = vecvalid.shape[0]
SH_grid = get_non_empty_column_indices(G)

N2=act_pix

signal_variance = G @ Cs_recon @ G.transpose()
signal_strength = np.mean(np.diag(signal_variance))

noise_to_signal = 1/signal_to_noise
noise_var = (noise_to_signal**2) * signal_strength

SH_grid = get_non_empty_column_indices(G)

G_recon = G
G = G[:,SH_grid]

Cs_inv     = np.linalg.inv(Cs_recon)
Cs_post     =  np.linalg.inv(G_recon.transpose() @ ((1/noise_var)*np.eye(G_recon.shape[0])) @ G_recon  +  Cs_inv)  

u, s, vh = np.linalg.svd(Cs_post, full_matrices=True)
s[0] = 0

#remove piston
u_crop      = u.copy()
u_crop[:,0] = 0

ff_ao = []
waff_ao = []
for i in range(1,n_history):    
    
    Cff_temp = Cff[-act_pix-i*SH_grid.shape[0]:,-act_pix-i*SH_grid.shape[0]:]
    Cwaff_temp = Cwaff[-act_pix-i*SH_grid.shape[0]:,-act_pix-i*SH_grid.shape[0]:]
    
    A          = scipy.linalg.block_diag(*([G] * i))
    A          = np.concatenate((A, np.zeros((A.shape[0], 1 * G_recon.shape[1]))),1).astype(np.float32)

    Cff_inv   = np.linalg.inv(Cff_temp)
    Cwaff_inv   = np.linalg.inv(Cwaff_temp)

    Cff_post   =  np.linalg.inv(A.transpose() @ ((1/noise_var)*np.eye(A.shape[0])) @ A  +  Cff_inv)
    Cwaff_post   =  np.linalg.inv(A.transpose() @ ((1/noise_var)*np.eye(A.shape[0])) @ A  +  Cwaff_inv)

    Cff_np   = u_crop @ u.transpose() @ Cff_post[-act_pix:,-act_pix:]
    Cwaff_np   = u_crop @ u.transpose() @ Cwaff_post[-act_pix:,-act_pix:]

    modal_var_ff = F @ Cff_np @ F.transpose()
    modal_var_waff = F @ Cwaff_np @ F.transpose()


    print(np.sqrt(np.trace(modal_var_ff)))
    print(np.sqrt(np.trace(modal_var_waff)))
    
    ff_ao.append(np.sqrt(np.trace(modal_var_ff)))
    waff_ao.append(np.sqrt(np.trace(modal_var_waff)))


np.save('ff_ao',ff_ao)
np.save('waff_ao',waff_ao)

