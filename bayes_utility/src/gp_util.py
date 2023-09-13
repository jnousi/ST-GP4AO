from hcipy import *

import numpy as np
import scipy
import helpers

from scipy.special import gamma, kv


def spatial_covmat(grid, r0, L0):
    '''
    Generate spatial covariance matrix for given grid, using
    environment parameters r0 and L0
    '''

    cov_function = phase_covariance_von_karman(r0, L0)
    nd = grid.shape[0]*grid.shape[1]
    cov_mat_spatial = np.zeros((nd,nd))


    for x in range(nd):

        shifted_grid = grid.shifted(grid[x])
        cov_mat_spatial[:,x] = np.flip(cov_function(shifted_grid), axis=0)
        #print(cov_function(shifted_grid))

    return cov_mat_spatial

def spatio_temporal_covmat(timesteps, Cn2, winds, cov_mat_spatial, grid, r0, L0):
    '''
    Generate spatio-temporal covariance matrix from given spatial
    covariance matrix and using given wind speeds
    '''

    cov_function = phase_covariance_von_karman(r0, L0)
    nlayer = Cn2.shape[0]
#     timesteps = 8
    nd = grid.shape[0]*grid.shape[1]
    cov_mat = np.zeros((timesteps*nd,timesteps*nd))

    for nl in range(nlayer):

        cov_mat_layer = np.zeros((timesteps*nd,timesteps*nd))
        cov_mat_layer = scipy.linalg.block_diag(*([cov_mat_spatial] * timesteps))

        for t in range(1,timesteps):
            cov_mat_shifted= np.zeros((nd,nd))
            for x in range(nd):
                shifted_grid = grid.shifted(grid[x] + winds[nl]*(t))
                cov_mat_shifted[:,x] = np.flip(cov_function(shifted_grid), axis = 0)
            # fill right blocks
            temp = scipy.linalg.block_diag(*([cov_mat_shifted] * (timesteps-t)))
            cov_mat_layer[t*nd:timesteps*nd,0:(timesteps-t)*nd] += temp
            cov_mat_layer[0:(timesteps-t)*nd,t*nd:timesteps*nd] += temp.transpose()

        cov_mat += (Cn2[nl]) * cov_mat_layer

    return cov_mat

def spatio_temporal_covmat_isotropic(timesteps, Cn2, winds, cov_mat_spatial, grid, r0, L0):

    cov_function_iso = phase_covariance_von_karman_3d(r0, L0)
    nd = grid.shape[0]*grid.shape[1]
    cov_mat = np.zeros((timesteps*nd,timesteps*nd))
    average_wind = np.linalg.norm(winds[0])

    cov_mat = np.zeros((timesteps*nd,timesteps*nd))
    cov_mat = scipy.linalg.block_diag(*([cov_mat_spatial] * timesteps))

    for t in range(1,timesteps):
        cov_mat_shifted= np.zeros((nd,nd))
        for x in range(nd):
            shifted_grid = grid.shifted(grid[x])
            cov_mat_shifted[:,x] = np.flip(cov_function_iso(shifted_grid, (t)*average_wind + 1e-8), axis = 0)
        # fill right blocks
        temp = scipy.linalg.block_diag(*([cov_mat_shifted] * (timesteps-t)))
        cov_mat[t*nd:timesteps*nd,0:(timesteps-t)*nd] += temp
        cov_mat[0:(timesteps-t)*nd,t*nd:timesteps*nd] += temp.transpose()


    return cov_mat

def phase_covariance_von_karman_timeshift(r0, L0):

    def func(grid, timeshift):
        r = grid.as_('polar').r + 1e-10
        r = r + timeshift
        a = (L0 / r0)**(5 / 3)
        b = gamma(11 / 6) / (2**(5 / 6) * np.pi**(8 / 3))
        c = (24 / 5 * gamma(6 / 5))**(5 / 6)
        d = (2 * np.pi * r / L0)**(5 / 6)
        e = kv(5 / 6, 2 * np.pi * r / L0)

        return Field(a * b * c * d * e, grid)
    return func



def phase_covariance_von_karman_3d(r0, L0):

    def func(grid, timeshift):

        r = np.sqrt(np.sum(grid[:]**2 ,1) + timeshift**2)+ 1e-10
        #r = grid.as_('polar').r + 1e-10
        #r = r + timeshift
        a = (L0 / r0)**(5 / 3)
        b = gamma(11 / 6) / (2**(5 / 6) * np.pi**(8 / 3))
        c = (24 / 5 * gamma(6 / 5))**(5 / 6)
        d = (2 * np.pi * r / L0)**(5 / 6)
        e = kv(5 / 6, 2 * np.pi * r / L0)

        return Field(a * b * c * d * e, grid)
    return func

def create_ts_prior(n_history, winds, Cn2, nLenslet, alias=True, params=None):
    '''
    Create temporal-spatial prior

    Inputs:
    n_history - number of timesteps in data
    winds - (2,) np.array of wind speeds in x and y directions
    (1,0) is left to right and (0,1) up-down m/framerate (= 0.002)
    params - dictionary of telescope parameters (optional with default
    values supplied)
    '''
    if not params:
        if not alias:
            telescope_diameter = 8. # meter
            wavelength_wfs = 0.7e-6

#            nLenslet = 40
            ndm = nLenslet + 1
        else:
            telescope_diameter = 8.
            ndm = nLenslet + 1
            npx = 2
#            nLenslet = 16

        # atm parameters
        r0 = 0.168
        # r0 = 16.8
        L0 = 40 # meter
        # Cn2 = np.array([0.5, 0.3, 0.2])
    else:
        telescope_diameter = params['telescope_diameter']
        wavelength_wfs     = params['wavelengths_wfs']
        nLenslet           = params['nLenslet']
        ndm                = params['ndm']
        r0                 = params['r0']
        L0                 = params['L0']
        Cn2                = params['Cn2']

    timesteps = 2 + n_history

    if not alias:
        fried_pixel = ndm
        fried_grid = make_pupil_grid(fried_pixel, telescope_diameter)

        cov_mat_spatial = spatial_covmat(fried_grid, r0, L0)
        cov_mat         = spatio_temporal_covmat(timesteps, Cn2, winds, cov_mat_spatial, fried_grid, r0, L0)
    else:
        fried_extend = telescope_diameter + telescope_diameter/(ndm -1) # meter
        phase_extend = telescope_diameter + telescope_diameter/(ndm*npx-1) # meter
        phase_d = telescope_diameter
        fried_grid = make_pupil_grid(ndm, fried_extend)
        fried_full_grid = make_pupil_grid(ndm*npx, phase_extend)
        full_grid =  make_pupil_grid((ndm-1)*npx, phase_d)

        cov_mat_spatial = spatial_covmat(full_grid, r0, L0)
        cov_mat         = spatio_temporal_covmat(timesteps, Cn2, winds, cov_mat_spatial, full_grid, r0, L0)


    return cov_mat, cov_mat_spatial

def D_opt_data(n_history, G_full, winds, prior="ST", noise_var=0.00001):
    '''
    Compute D-optimality criterion prediction using n_history
    number of timesteps of data
    '''
    wavelength_wfs = 0.7e-6

    predict = 2 # predictive steps
    N = G_full.shape[1] # dimension of the reconstruction

    A = scipy.linalg.block_diag(*([G_full] * n_history))
    A = np.concatenate((A, np.zeros((A.shape[0], predict * N))),1)

        # construct permutation matrix for the prior
    C = create_ts_prior(n_history, winds, int(np.sqrt(N))-1, prior=prior, alias=False)
    # C = C*(wavelength_wfs/(2*np.pi))*1e6 # Scale units to micrometers

    C_inv = np.linalg.inv(C)

    # posterior covariance C_post
    C_post_inv = A.transpose() @ ((1/noise_var)*np.eye(A.shape[0])) @ A  +  C_inv
    C_post =  np.linalg.inv(C_post_inv)

    C_post_pred = C_post[-N:,-N:]

    return helpers.log_det(C_post_pred, inv=False)

def A_opt_data(n_history, G_full, winds, Cn2, signal_to_noise=0.00001):
    '''
    Compute A-optimality criterion prediction using n_history
    number of timesteps of data
    '''
    wavelength_wfs = 0.7e-6

    predict = 2 # predictive steps
    N = G_full.shape[1] # dimension of the reconstruction

    A = scipy.linalg.block_diag(*([G_full] * n_history))
    A = np.concatenate((A, np.zeros((A.shape[0], predict * N))),1)

    C, C_spatial = create_ts_prior(n_history, winds, Cn2, int(np.sqrt(N))-1, alias=False)
    # C = C*(wavelength_wfs/(2*np.pi))*1e6 # Scale units to micrometers

    signal_variance = G_full@C_spatial@G_full.transpose()
    signal_strength = np.mean(np.diag(signal_variance))
    noise_var = (signal_to_noise**2) * signal_strength

    C_inv = np.linalg.inv(C)

    # posterior covariance C_post
    C_post_inv = A.transpose() @ ((1/noise_var)*np.eye(A.shape[0])) @ A  +  C_inv
    C_post =  np.linalg.inv(C_post_inv)

    C_post_pred = C_post[-N:,-N:]
    # filter uncertainty ?????
    u, s, vh = np.linalg.svd(C_post_pred, full_matrices=True)
    s[0] = 0
    C_post_pred2 = u @ np.diag(s) @ vh

    return np.sqrt(np.trace(C_post_pred2))
