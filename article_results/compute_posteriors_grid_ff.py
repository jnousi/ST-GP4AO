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


folder_name = '16x16_sn100'

filepath = 'matrix_for_GP_16x16.mat'
arrays = {}
f = h5py.File(filepath,'r')
for k, v in f.items():
    arrays[k] = np.array(v)

G1 = arrays["G_fried"] 
G2 = arrays["G_3"] 
G4 = arrays["G_4"]
G6 = arrays["G_6"]
G8 = arrays["G_8"]

G1 = G1.transpose().astype(np.float32)
G2 = G2.transpose().astype(np.float32)
G4 = G4.transpose().astype(np.float32)
G6 = G6.transpose().astype(np.float32)
G8 = G8.transpose().astype(np.float32)

# the full phase matrix
mask_fried  = arrays["mask_fried"]
mask_phase2 = arrays["mask_phase3"]
mask_phase4 = arrays["mask_phase4"]
mask_phase6 = arrays["mask_phase6"]
mask_phase8 = arrays["mask_phase8"]


M2P = arrays["M2P"]
M2P = M2P.transpose().astype(np.float32)

# telescope
telescope_diameter = 3.2

# recontruction grid
n_lenslet          = 16
n_pixels           = 8*n_lenslet

# predictive control
n_history = 5 # data steps
n_predict = 2 # predective steps

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

#timesteps = n_history + n_predict
signal_to_noise = 100

recon_grid =  make_pupil_grid(n_pixels, telescope_diameter)

data_grid1 =  make_pupil_grid(1*n_lenslet + 1, telescope_diameter + telescope_diameter/(n_lenslet))
data_grid2 =  make_pupil_grid(2*n_lenslet + 1, telescope_diameter + telescope_diameter/(2*n_lenslet))
data_grid4 =  make_pupil_grid(4*n_lenslet, telescope_diameter)
data_grid6 =  make_pupil_grid(6*n_lenslet, telescope_diameter)
data_grid8 =  make_pupil_grid(8*n_lenslet, telescope_diameter)

vecvalid1          = np.nonzero(mask_fried.flatten())[0]
vecvalid2          = np.nonzero(mask_phase2.flatten())[0]
vecvalid4          = np.nonzero(mask_phase4.flatten())[0]
vecvalid6          = np.nonzero(mask_phase6.flatten())[0]
vecvalid8          = np.nonzero(mask_phase8.flatten())[0]

SH_grid2 = get_non_empty_column_indices(G2)
SH_grid4 = get_non_empty_column_indices(G4)
SH_grid6 = get_non_empty_column_indices(G6)
SH_grid8 = get_non_empty_column_indices(G8)

(xvalid, yvalid) = np.nonzero(mask_phase8)

vec_to_im    = vec_to_img(n_pixels, xvalid, yvalid)
im_to_vec    = img_to_vec(xvalid, yvalid)

# numeber of active pixels
SH_grid8 = get_non_empty_column_indices(G8)
act_pix  = vecvalid8.shape[0]


Cs_data1        = s_gp_validpix(data_grid1, r0, L0, vecvalid1)
Cs_data2        = s_gp_validpix(data_grid2, r0, L0, vecvalid2[SH_grid2])
Cs_data4        = s_gp_validpix(data_grid4, r0, L0, vecvalid4[SH_grid4])
Cs_data6        = s_gp_validpix(data_grid6, r0, L0, vecvalid6[SH_grid6])
Cs_data8        = s_gp_validpix(data_grid8, r0, L0, vecvalid8[SH_grid8])

Cs_recon       = s_gp_validpix(recon_grid, r0, L0, vecvalid8)

print("data priors...")

Cff1 = st_gp_ff_validpix(n_history, Cn2, winds, Cs_data1, data_grid1, r0, L0, vecvalid1)
Cff2 = st_gp_ff_validpix(n_history, Cn2, winds, Cs_data2, data_grid2, r0, L0, vecvalid2[SH_grid2])
Cff4 = st_gp_ff_validpix(n_history, Cn2, winds, Cs_data4, data_grid4, r0, L0, vecvalid4[SH_grid4])
Cff6 = st_gp_ff_validpix(n_history, Cn2, winds, Cs_data6, data_grid6, r0, L0, vecvalid6[SH_grid6])
Cff8 = st_gp_ff_validpix(n_history, Cn2, winds, Cs_data8, data_grid8, r0, L0, vecvalid8[SH_grid8])

print("add recon grid.....")
Cff1 = add_recon_grid_ff(Cff1, Cs_recon, Cn2, winds, r0, L0, n_history, n_predict, data_grid1, recon_grid, vecvalid1, vecvalid8)
Cff2 = add_recon_grid_ff(Cff2, Cs_recon, Cn2, winds, r0, L0, n_history, n_predict, data_grid2, recon_grid, vecvalid2[SH_grid2], vecvalid8)
Cff4 = add_recon_grid_ff(Cff4, Cs_recon, Cn2, winds, r0, L0, n_history, n_predict, data_grid4, recon_grid, vecvalid4[SH_grid4], vecvalid8)
Cff6 = add_recon_grid_ff(Cff6, Cs_recon, Cn2, winds, r0, L0, n_history, n_predict, data_grid6, recon_grid, vecvalid6[SH_grid6], vecvalid8)
Cff8 = add_recon_grid_ff(Cff8, Cs_recon, Cn2, winds, r0, L0, n_history, n_predict, data_grid8, recon_grid, vecvalid8[SH_grid8], vecvalid8)

print("priors ready!")

signal_variance = G8 @ Cs_recon @ G8.transpose()
signal_strength = np.mean(np.diag(signal_variance))

noise_to_signal = 1/signal_to_noise
noise_var = (noise_to_signal**2) * signal_strength

#remove empty column corresponding to empty pixels
G_recon = G8
G1 = G1
G2 = G2[:,SH_grid2]
G4 = G4[:,SH_grid4]
G6 = G6[:,SH_grid6]
G8 = G8[:,SH_grid8]

A1          = scipy.linalg.block_diag(*([G1] * n_history))/2
A1          = np.concatenate((A1, np.zeros((A1.shape[0], 1 * G_recon.shape[1]))),1).astype(np.float32)

A2          = scipy.linalg.block_diag(*([G2] * n_history))
A2          = np.concatenate((A2, np.zeros((A2.shape[0], 1 * G_recon.shape[1]))),1).astype(np.float32)

A4          = scipy.linalg.block_diag(*([G4] * n_history))
A4          = np.concatenate((A4, np.zeros((A4.shape[0], 1 * G_recon.shape[1]))),1).astype(np.float32)

A6          = scipy.linalg.block_diag(*([G6] * n_history))
A6          = np.concatenate((A6, np.zeros((A6.shape[0], 1 * G_recon.shape[1]))),1).astype(np.float32)

A8          = scipy.linalg.block_diag(*([G8] * n_history))
A8          = np.concatenate((A8, np.zeros((A8.shape[0], 1 * G_recon.shape[1]))),1).astype(np.float32)

Cs_inv     = np.linalg.inv(Cs_recon)
Cff1_inv   = np.linalg.inv(Cff1)
Cff2_inv   = np.linalg.inv(Cff2)
Cff4_inv   = np.linalg.inv(Cff4)
Cff6_inv   = np.linalg.inv(Cff6)
Cff8_inv   = np.linalg.inv(Cff8)

Cff_post1   =  np.linalg.inv(A1.transpose() @ ((1/noise_var)*np.eye(A1.shape[0])) @ A1  +  Cff1_inv)
Cff_post2   =  np.linalg.inv(A2.transpose() @ ((1/noise_var)*np.eye(A2.shape[0])) @ A2  +  Cff2_inv)
Cff_post4   =  np.linalg.inv(A4.transpose() @ ((1/noise_var)*np.eye(A4.shape[0])) @ A4  +  Cff4_inv)
Cff_post6   =  np.linalg.inv(A6.transpose() @ ((1/noise_var)*np.eye(A6.shape[0])) @ A6  +  Cff6_inv)
Cff_post8   =  np.linalg.inv(A8.transpose() @ ((1/noise_var)*np.eye(A8.shape[0])) @ A8  +  Cff8_inv)

Cs_post     =  np.linalg.inv(G_recon.transpose() @ ((1/noise_var)*np.eye(G_recon.shape[0])) @ G_recon  +  Cs_inv)    

Cc                   =  temporal_cross_correlation(n_predict, Cn2, winds, Cs_recon, recon_grid, r0, L0, vecvalid8)
Cs_post_temporal     =  Cs_post + 2*Cs_recon - Cc - Cc.transpose()

u, s, vh = np.linalg.svd(Cs_post, full_matrices=True)
s[0] = 0

u_crop      = u.copy()
u_crop[:,0] = 0

Cs_np   = u_crop @ u.transpose() @ Cs_post_temporal

Cff_np1   = u_crop @ u.transpose() @ Cff_post1[-act_pix:,-act_pix:]
Cff_np2   = u_crop @ u.transpose() @ Cff_post2[-act_pix:,-act_pix:]
Cff_np4   = u_crop @ u.transpose() @ Cff_post4[-act_pix:,-act_pix:]
Cff_np6   = u_crop @ u.transpose() @ Cff_post6[-act_pix:,-act_pix:]
Cff_np8   = u_crop @ u.transpose() @ Cff_post8[-act_pix:,-act_pix:]

Rs    = Cs_post @ G_recon.transpose() @ ((1/noise_var)*np.eye(G_recon.shape[0]))

Rff1   = Cff_post1 @ A1.transpose() @ ((1/noise_var)*np.eye(A1.shape[0]))
Rff2   = Cff_post2 @ A2.transpose() @ ((1/noise_var)*np.eye(A2.shape[0]))
Rff4   = Cff_post4 @ A4.transpose() @ ((1/noise_var)*np.eye(A4.shape[0]))
Rff6   = Cff_post6 @ A6.transpose() @ ((1/noise_var)*np.eye(A6.shape[0]))
Rff8   = Cff_post8 @ A8.transpose() @ ((1/noise_var)*np.eye(A8.shape[0]))


Chol_ff = np.linalg.cholesky(Cff8 + 1e-8*np.diag(np.random.randn(Cff8.shape[0]))).astype(np.float32)

os.makedirs(folder_name, exist_ok=True)

# Save matrices to the specified folder
np.save(os.path.join(folder_name, 'Cff_np1.npy'), Cs_np)
np.save(os.path.join(folder_name, 'Cff_np1.npy'), Cff_np1)
np.save(os.path.join(folder_name, 'Cff_np2.npy'), Cff_np2)
np.save(os.path.join(folder_name, 'Cff_np4.npy'), Cff_np4)
np.save(os.path.join(folder_name, 'Cff_np6.npy'), Cff_np6)
np.save(os.path.join(folder_name, 'Cff_np8.npy'), Cff_np8)

np.save(os.path.join(folder_name, 'Cs_np.npy'), Cs_np)
np.save(os.path.join(folder_name, 'Chol_ff.npy'), Chol_ff)

np.save(os.path.join(folder_name, 'Cff8.npy'), Cff8)

np.save(os.path.join(folder_name, 'u.npy'), u)
np.save(os.path.join(folder_name, 'Rs.npy'), Rs)
np.save(os.path.join(folder_name, 'Rff1.npy'), Rff1)
np.save(os.path.join(folder_name, 'Rff2.npy'), Rff2)
np.save(os.path.join(folder_name, 'Rff4.npy'), Rff4)
np.save(os.path.join(folder_name, 'Rff6.npy'), Rff6)
np.save(os.path.join(folder_name, 'Rff8.npy'), Rff8)



np.save(os.path.join(folder_name, 'noise_var.npy'), noise_var)
