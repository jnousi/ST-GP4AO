'''
This file runs the OED test for investigating the effects of\n",
various wind strengths and noise levels on the usefulness of\n",
various lengths of time series\n",
'''

import sys, os
from hcipy import *
import h5py
import numpy as np

import scipy

# Location of imports for this project
curdir = os.getcwd(); srcdir = "src"
sys.path.append(os.path.join(curdir, srcdir))

from gp_util import *
from stgp_util import *
import helpers


# Locations of input and output files
indata = "input_data"
outdata = "output_data"

filepath = 'matrix_for_GP_16x16.mat'
arrays = {}
f = h5py.File(filepath,'r')
for k, v in f.items():
    arrays[k] = np.array(v)


G = arrays["G_4"]

G = G.transpose().astype(np.float32)


filepath = 'dm4.mat'
arrays = {}
f = h5py.File(filepath,'r')
for k, v in f.items():
    arrays[k] = np.array(v)

M2P = arrays["M2P_dm"]
M2P = M2P.transpose().astype(np.float32)   

P2M = np.linalg.pinv(M2P)
F = M2P @ P2M 

## telescope parameters, defined the same as in the file more_memory_gentle_posterior.ipynb

# telescope
telescope_diameter = 3.2

# recontruction grid
n_lenslet          = 16
n_pixels           = 4*n_lenslet

# data_grid

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

Cs_recon       = s_gp_validpix(recon_grid, r0, L0, vecvalid)

Cs_inv     = np.linalg.inv(Cs_recon)
Cs_post     =  np.linalg.inv(G_recon.transpose() @ ((1/noise_var)*np.eye(G_recon.shape[0])) @ G_recon  +  Cs_inv)  

u, s, vh = np.linalg.svd(Cs_post, full_matrices=True)
s[0] = 0

#remove piston
u_crop      = u.copy()
u_crop[:,0] = 0

# model
n_history = 16 # data steps
n_predict = 2 # predective steps
timesteps = n_history + n_predict
signal_to_noise = 100


data = np.load('output_data/cov_matrices_large.npz')
C = data['Cwaff']
Cff = data['Cff']
G1 = data['G1']
N = data['N']
N2 = data['N2']
noise_var = data['noise_var']



Cs_recon       = s_gp_validpix(recon_grid, r0, L0, vecvalid)


history_max = 16
timesteps = [14,13,12,11,10,9,8,7,6,5,4,3,2,1,0] # possible steps still remaining

def construct_subm(C,steps_in,N,h_max):
    # Helper function to construct the covariance matrix
    # including only the selected timesteps
    # args:
    # C : full covariance matrix
    # steps_in : steps to be included
    # N : size of reconstruction
    C_tmp = C
    step = h_max
    C_tmp = np.delete(C_tmp, slice(N2*step,N2*(step+1)), axis=0)
    C_tmp = np.delete(C_tmp, slice(N2*step,N2*(step+1)), axis=1)
    for step in reversed(range(h_max)):
        if step not in steps_in:
            C_tmp = np.delete(C_tmp, slice(N*step,N*(step+1)), axis=0)
            C_tmp = np.delete(C_tmp, slice(N*step,N*(step+1)), axis=1)
    return C_tmp

# Always include the last timestep and predictions
steps_in = [16]
opt_vals_dict = {}

for i in range(1,history_max-7):
    min_val = 1000
    opt_vals = []
    # Create the system matrix
    A = scipy.linalg.block_diag(*([G1] * (i+1)))
    A = np.concatenate((A, np.zeros((A.shape[0], 1 * N2))),1)
#     steps_tmp = timesteps
    for elem in timesteps:
        steps_in_tmp = steps_in.copy()
        steps_in_tmp.append(elem)
        print(elem, end="")
        C_tmp = construct_subm(C, steps_in_tmp, N,history_max)
        C_inv = np.linalg.inv(C_tmp)

        # posterior covariance C_post
        C_post_inv = A.transpose() @ ((1/noise_var)*np.eye(A.shape[0])) @ A  +  C_inv
        C_post =  np.linalg.inv(C_post_inv)

        C_post_pred = C_post[-N:,-N:]
                
        C_post_pred  = u_crop @ u.transpose() @ C_post_pred[-act_pix:,-act_pix:]
        modal_var = F @ C_post_pred  @ F.transpose()

        opt_val = np.sqrt(np.trace(modal_var))
        opt_vals.append(opt_val)
        if opt_val < min_val:
            opt_elem = elem
            min_val = opt_val
    print('\n')
    print(f'Chosen timestep: {opt_elem}')
    print(f'Optimal values:')
    opt_vals_dict[i] = opt_vals
    print(opt_vals)
    steps_in.append(opt_elem)
    timesteps.remove(opt_elem)
    print(steps_in)
    print('\n### ', end="")
    "


# For comparison, compute the utility including the latest 6 timesteps
steps_in_ref = [10,11,12,13,14,15,16]
A = scipy.linalg.block_diag(*([G1] * (6)))
A = np.concatenate((A, np.zeros((A.shape[0], 1 * N2))),1)
C_tmp = construct_subm(C, steps_in_ref, N,history_max)
C_inv = np.linalg.inv(C_tmp)

# posterior covariance C_post
C_post_inv = A.transpose() @ ((1/noise_var)*np.eye(A.shape[0])) @ A  +  C_inv
C_post =  np.linalg.inv(C_post_inv)

C_post_pred = C_post[-N:,-N:]
# filter uncertainty
u, s, vh = np.linalg.svd(C_post_pred, full_matrices=True)
s[0] = 0
C_post_pred2 = u @ np.diag(s) @ vh
opt_val = np.sqrt(np.trace(C_post_pred2))
print(f'Target value (unoptimized): {opt_val}')
print(f'Target value (optimized): {min(opt_vals_dict[5])}')

# for i in timesteps:
#     Opt_vals.append(A_opt_data(i, G_full, winds_iso, Cn2_iso, noise_var))

def output_pgfplot(values):
    # Helper function to output values in suitable format
    # for plotting in LaTeX
    output = "\\addplot coordinates {\n"
    time = -framerate*len(values)
#     timesteps = np.arange(0,-0.002*len(steps),-0.002)
    for val in values:
        str = f'({time:.3f},{val:.3f}) '
        output += str
        time += framerate
    output += "\n}"
    return output


def plot_helper(opt_vals_dict, steps_in):
    # Helper function to output values in suitable format
    # for plotting in LaTeX
    steps_in = steps_in[:]

    output = ""
    opt_vals_tmp = opt_vals_dict[1]
    opt_vals_tmp.reverse()
    output += output_pgfplot(opt_vals_tmp)

    opt_vals_tmp = opt_vals_dict[2]
    opt_vals_tmp.reverse()
#    opt_vals_tmp.insert(steps_in[0],0)\n",
    output += output_pgfplot(opt_vals_tmp)

    opt_vals_tmp = opt_vals_dict[3]
    opt_vals_tmp.reverse()
#    opt_vals_tmp.insert(steps_in[0],0)
#    opt_vals_tmp.insert(steps_in[1],0)
    output += output_pgfplot(opt_vals_tmp)

    return output

opt_vals_dict_tmp = opt_vals_dict.copy()
print(plot_helper(opt_vals_dict_tmp,steps_in))