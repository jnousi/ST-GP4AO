import numpy as np
import scipy
import matplotlib.pyplot as plt
import helpers

def log_det(B, inv=True):
    '''
    Function to compute logarithm of the determinant of
    a symmetric matrix B using a Cholesky decomposition. Useful
    for computing D-optimality targets.
    '''
    C = np.linalg.cholesky(B);
    dC = np.diag(C);
    if inv:
        logdet = -np.sum(np.log(dC));
    else:
        logdet = np.sum(np.log(dC));
    return logdet

def perm_prior(C_hat, steps, dim):
    '''
    Construct permutation matrix for the prior.
    Inputs:
    C_hat - the covariance matrix for the temporal
    dependence of an individual pixel.
    steps  - the total number of timesteps
    of the model (data points + prediction steps)
    dim - dimension of the problem (number of pixels
    in the detector)
    '''
    T = np.zeros_like(C_hat)

    temp = 0
    for ii in range(dim):
        for tt in range(steps):
            T[tt + (ii*(steps)), ii + tt*(dim)] =  1 

    C = T.transpose() @ C_hat @ T
    return C

def plot_reconstruction(reconstruction, variances, ground_truth, Z_loc, indexes='', loc=[5,5]):
    '''
    Plot temporal evolution of estimate for an individual pixel, including
    confidence intervals.
    '''
    steps = reconstruction.shape[0]
    
    # If location not specified, default is used.
    xx = loc[0]
    yy = loc[1]

    f_maps1 = reconstruction[:,xx,yy]
    f_var   = variances[:,xx,yy]

    fig_prediction= plt.figure(figsize = (10,5))
    plt.rcParams.update({'font.size': 10})

    # Draw a mean function and 95% confidence interval
    if indexes:
        Z_loc = Z_loc[indexes]
        plt.plot(Z_loc, f_maps1, 'b-', label='Fried data')
        plt.plot(Z_loc, ground_truth[indexes, xx, yy], 'r-', label='True signal')   
    else:
        plt.plot(Z_loc, f_maps1, 'b-', label='Fried data')
        plt.plot(Z_loc, ground_truth[0:steps, xx, yy], 'r-', label='True signal')   
        
    upper_bound = f_maps1.reshape(steps,) + 1.96 * np.sqrt(f_var.squeeze())
    lower_bound = f_maps1.reshape(steps,) - 1.96 * np.sqrt(f_var.squeeze())

    plt.fill_between(Z_loc.ravel(), lower_bound, upper_bound, color = 'b', alpha = 0.1,
                     label='95% confidence interval')
    plt.xlabel('$x$')
    plt.ylabel('$f(x)$')
    plt.legend(loc='upper left')
    plt.title('A GP posterior')
    plt.show()
    
    
def average_pixels(N, axes='both'):
    '''
    Function to produce a matrix that averages together pixels from in the slope data.
    Inputs:
    N: pixels per edge in slope data.
    level: number of pixesl to average together; 1 would join combine 4 pixels into one, 2 for 16 etc.  
    '''
    Np = N//2
    if axes=='both':
        P = np.zeros((Np*Np,N*N))
        for row in range(Np):
            for column in range(Np):
                P[Np*row + column,N*row + 2*column] = 0.25
                P[Np*row + column,N*row + 2*column + 1] = 0.25
                P[Np*row + column,N*(2*row+1) + column] = 0.25
                P[Np*row + column,N*(2*row+1) + column + 1] = 0.25
        return np.vstack( (np.hstack((P,np.zeros((Np*Np,N*N)))), np.hstack((np.zeros((Np*Np,N*N)), P))))
    elif axes=='x':
        P = np.zeros((N*Np,N*N))
        for row in range(N):
            for column in range(Np):
                P[Np*row + column,Np*row + 2*column] = 0.5
                P[Np*row + column,Np*row + 2*column + 1] =  0.5
        return np.vstack( (np.hstack((P,np.zeros((N*Np,N*N)))), np.hstack((np.zeros((N*Np,N*N)), P))))
    elif axes=='y':
        for row in range(Np):
            for column in range(N):
                P[N*row + column,N*row + column] = 0.5
                P[N*row + column,N*(row+1) + column] =  0.5
        return np.vstack( (np.hstack((P,np.zeros((N*Np,N*N)))), np.hstack((np.zeros((N*Np,N*N)), P))))
    else:
        raise Exception("Invalid value for parameters 'axes', only 'both,'x', or 'y' accepted")

def DorA_opt(A, C_inv, noise_var, N, DorA='D'):
    '''
    Function to compute D-optimality criterion for a given system
    matrix A, inverse covariance C_inv, and noise variance noise_var.
    '''
    # recontruction matrix R and posterior covariance C_post
    C_post_inv = A.transpose() @ ((1/noise_var)*np.eye(A.shape[0])) @ A  +  C_inv
    C_post =  np.linalg.inv(C_post_inv)
    
    C_post_pred = C_post[-N*N:,-N*N:]

    # Computing the optimality targets
    # Note that smaller is better
    if DorA == 'D':
        return helpers.log_det(C_post_pred, inv=False)
    else:
        return helpers.trace(C_post_pred, inv=False)



def create_A_from_indexes(G_full, levels, n_history, predict):
    '''
    Function to create system matrix A for (n_history + predict) timesteps
    from initial matrix G_full, with averaging levels defined by input variable
    levels (list).
    
    Note: shape of the initial must be suitable so that the shape remains an integer
    after averaging (i.e. must be divisible by sufficiently many powers of 2)
    '''
    N = int(np.sqrt(G_full.shape[0]/2))
    P = [np.eye(2*N*N), average_pixels(N,1)]
    PP = average_pixels(N//2,1)
    PPP = average_pixels(N//4,1)
    P.append(PP @ P[-1])
    P.append(PPP @ P[-1])

    matrices = []
    for level in levels:
        matrices.append(P[level] @ G_full)
    A = scipy.linalg.block_diag(*matrices)
    A = np.concatenate((A, np.zeros((A.shape[0], predict * G_full.shape[1]))),1)
    return A

def pixels(level, N):
    '''
    Helper function to compute the increase in pixels when
    densening data
    '''
    if level == 0:
        Npost = 2*N*N
    else:
        Npost = N/(2*level); Npost = 2*Npost*Npost
    return Npost
