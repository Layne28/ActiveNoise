import numpy as np
import os
import sys
import argparse
#import GPUtil

try:
    import cupy as cp
    CUPY_IMPORTED = True
except ImportError:
    CUPY_IMPORTED = False

def main():

    parser = argparse.ArgumentParser(description='Generate active noise trajectory')
    parser.add_argument('--N', default=400, help='linear system size')
    parser.add_argument('--do_output', default=0, help='whether to print noise field to file')
    parser.add_argument('--print_freq', default=100, help='how often to print information (in timesteps)')
    parser.add_argument('--output_freq', default=100, help='how often to output noise configurations (in timesteps)')
    parser.add_argument('--nsteps', default=500, help='number of timesteps')
    parser.add_argument('--chunksize', default=50, help='how big to make chunks of random numbers (in timesteps)')
    parser.add_argument('--xpu', default='gpu', help='cpu or gpu')

    args = parser.parse_args()
    N = args.N
    do_output = args.do_output
    print_freq = args.print_freq
    output_freq = args.output_freq
    chunksize = args.chunksize
    nsteps = args.nsteps
    xpu = args.xpu

    params = {}

    params['N'] = int(N)
    params['dx'] = 1.0
    params['print_freq'] = int(print_freq)
    params['do_output'] = int(do_output)
    params['output_freq'] = int(output_freq)
    params['lambda'] = 1.0
    params['tau'] = 1.0
    params['dim'] = 2
    params['nsteps'] = int(nsteps)
    params['chunksize'] = int(chunksize)
    params['dt'] = 1e-2
    params['D'] = 1.0
    params['cov_type'] = 'exponential'
    params['xpu'] = xpu
    params['verbose'] = True

    gen_trajectory(**params)


def gen_trajectory(**kwargs):

    """Generate an active noise trajectory

    INPUT: Dictionary containing parameters
    OUTPUT: last noise field in trajectory (numpy array)
    """

    N = kwargs['N']
    dx = kwargs['dx']
    print_freq = kwargs['print_freq']
    do_output = kwargs['do_output']
    output_freq = kwargs['output_freq']
    Lambda = kwargs['lambda']
    tau = kwargs['tau']
    dim = kwargs['dim']
    nsteps = kwargs['nsteps']
    chunksize = kwargs['chunksize']
    dt = kwargs['dt']
    D = kwargs['D']
    cov_type = kwargs['cov_type']
    xpu = kwargs['xpu']
    verbose = kwargs['verbose']

    #Create output directory
    if not os.path.exists('./data') and do_output==1:
        os.makedirs('./data')

    if CUPY_IMPORTED:
        if xpu=='gpu':
            print('Using GPU')
            xp = cp
        else:
            print('Using CPU')
            xp = np
    else:
        print('Using CPU')
        xp = np

    '''
    if dim==1:
        init_arr = xp.zeros((dim,N))
    elif dim==2:
        init_arr = xp.zeros((dim,N,N))
    else:
        init_arr = xp.zeros((dim,N,N,N))
    ''' 
    init_arr = xp.array([])
    #Get trajectory
    traj_arr, fourier_noise = run(init_arr, **kwargs)

    return traj_arr

def run(init_fourier_arr, **kwargs):

    N = kwargs['N']
    dx = kwargs['dx']
    print_freq = kwargs['print_freq']
    do_output = kwargs['do_output']
    output_freq = kwargs['output_freq']
    Lambda = kwargs['lambda']
    tau = kwargs['tau']
    dim = kwargs['dim']
    nsteps = kwargs['nsteps']
    chunksize = kwargs['chunksize']
    dt = kwargs['dt']
    D = kwargs['D']
    cov_type = kwargs['cov_type']
    xpu = kwargs['xpu']
    verbose = kwargs['verbose']

    #derived variables
    L = N*dx
    nchunks = nsteps//chunksize

    if CUPY_IMPORTED:
        if xpu=='gpu':
            if verbose:
                print('Using GPU for active noise')
            xp = cp
        else:
            if verbose:
                print('Using CPU for active noise')
            xp = np
    else:
        if verbose:
            print('Using CPU for active noise')
        xp = np


    if dim==1:
        traj_arr = np.zeros((dim,N,nsteps+1))
    elif dim==2:
        traj_arr = np.zeros((dim,N,N,nsteps+1))
    else:
        traj_arr = np.zeros((dim,N,N,N,nsteps+1))

    ck = get_spatial_covariance(N, dx, dim, cov_type, Lambda, xpu)
    fourier_noise = init_fourier_arr

    #Generate a random field if init array is empty
    if fourier_noise.size==0:
        spat_corr_field = gen_field(N, dx, dim, 1, ck)
        fourier_noise = xp.sqrt(D*L**dim)*spat_corr_field[...,0]

    #Put in step 0 because hoomd requires it for multiple runs
    if xp==np:
        traj_arr[...,0] = get_real_field(fourier_noise[...,np.newaxis], N)[...,0]
    else: 
        traj_arr[...,0] = xp.asnumpy(get_real_field(fourier_noise[...,np.newaxis], N)[...,0])

    for c in range(nchunks):

        #print('chunk', c)

        if dim==1:
            fourier_arr = xp.zeros((1,N//2+1,chunksize), dtype=xp.complex64)
        elif dim==2:
            fourier_arr = xp.zeros((2,N,N//2+1,chunksize), dtype=xp.complex64)
        else:
            fourier_arr = xp.zeros((3,N,N,N//2+1,chunksize), dtype=xp.complex64)
            
        spat_corr_field = gen_field(N, dx, dim, chunksize, ck)

        #print(chunksize)
        for n in range(chunksize):

            if n%print_freq==0 and verbose:
                print('active noise step', n)

            noise_incr = xp.sqrt(2*D*L**dim*dt/tau)*spat_corr_field[...,n]
            fourier_noise = (1.0-dt/tau)*fourier_noise + noise_incr
            fourier_arr[...,n] = fourier_noise

            #if do_output==1 and n%output_freq==0:
            #    xp.savez('data/field_%04d.npz' % (n+c*chunksize), real_noise)

        #Take inverse fourier transform of noise trajectory    
        #print('computing FFT...')
        if xp==np:
            traj_arr[...,(c*chunksize+1):((c+1)*chunksize+1)] = get_real_field(fourier_arr, N)
        else:
            traj_arr[...,(c*chunksize+1):((c+1)*chunksize+1)] = xp.asnumpy(get_real_field(fourier_arr, N))
            
    #print(traj_arr.shape)
    #print('rms noise: %f' % np.sqrt(np.average(traj_arr**2)))
    return traj_arr, fourier_noise
    #return traj_arr

def get_spatial_covariance(N, dx, dim, cov_type, l, xpu):

    """Create a matrix containing the spatial covariance function
    at each allowed wavevector.

    INPUT: Linear size of noise field (int),
           grid spacing (float)
           dimensions (int),
           mathematical form of covariance function (string),
           correlation length (float)
    OUTPUT: Numpy array containing spatial correlation at each wavevector
    """

    if CUPY_IMPORTED:
        if xpu=='gpu':
            xp = cp
        else:
            xp = np
    else:
        xp = np
 
    #Define wavevectors
    myvec = xp.zeros(N)
    for i in range(N//2+1):
        myvec[i] = i
    for i in range(N//2+1,N):
        myvec[i] = i-N
    kvec = 2*xp.pi/(N*dx)*myvec
    if dim==1:
        kx = kvec
    elif dim==2:
        kx, ky = xp.meshgrid(kvec, kvec)
    else:
        kx, ky, kz = xp.meshgrid(kvec, kvec, kvec)

    #Get spatial correlation function
    if cov_type=='exponential':
        if dim==1:
            ck = 1.0/(1+l**2*kx**2)
        elif dim==2:
            ck = 1.0/pow(1+l**2*(kx**2+ky**2),3.0/2.0)
        else:
            ck = 1.0/pow(1+l**2*(kx**2+ky**2+kz**2),2.0)
    elif cov_type=='gaussian':
        if dim==1:
            ck = xp.exp(-l**2*kx**2/2.0)
        elif dim==2:
            ck = xp.exp(-l**2*(kx**2+ky**2)/2.0)
        else:
            ck = xp.exp(-l**2*(kx**2+ky**2+kz**2)/2.0)
    else:
        ck = 0.0*kx
    ck = ck/(xp.sum(ck)*(N*dx)**dim)
    #print(np.sum(ck)/(N*dx)**dim)

    return ck

def gen_field(N, dx, dim, nsteps, ck):

    """Create a spatially correlated noise field in Fourier space

    INPUT: Linear size of field (int),
           dimension of field (int),
           number of time steps (int),
           spatial correlation function (numpy array)

    OUTPUT: Noise field (numpy array of size (dim, N, ..., N/2+1, nsteps) with N repeated dim-1 times)
    """

    if CUPY_IMPORTED:
        xp = cp.get_array_module(ck)
    else:
        xp = np

    if dim==1:
        noise = xp.zeros((1,N//2+1,nsteps), dtype=xp.complex64)
    elif dim==2:
        noise = xp.zeros((2,N,N//2+1,nsteps), dtype=xp.complex64)
    else:
        noise = xp.zeros((3,N,N,N//2+1,nsteps), dtype=xp.complex64)

    #print([dim] + [N]*dim + [nsteps])
    #print('generating big white noise array...')
    white_noise = xp.sqrt(N**dim)*xp.random.normal(loc=0.0, scale=1.0, size=tuple([dim] + [N]*dim + [nsteps]))
    #print('done')

    for t in range(nsteps):
        for d in range(dim):
            if dim==1:
                fourier_white_noise = xp.fft.rfft(white_noise[d,...,t], axis=0)
                noise[d,...,t] = xp.multiply(xp.sqrt(ck[...,:(N//2+1)]), fourier_white_noise)
            elif dim==2:
                fourier_white_noise = xp.fft.rfft2(white_noise[d,...,t], axes=(0,1))
                noise[d,...,t] = xp.multiply(xp.sqrt(ck[...,:(N//2+1)]), fourier_white_noise)
            else:
                fourier_white_noise = xp.fft.rfftn(white_noise[d,...,t], axes=(0,1,2))
                noise[d,...,t] = xp.multiply(xp.sqrt(ck[...,:(N//2+1)]), fourier_white_noise)
            
    noise = xp.array(noise)
    #print('done coloring noise')

    return noise

def get_real_field(field, N):

    """Transform fourier space field to real space

    INPUT: Fourier-space field (numpy array),
           linear size of field (int)

    OUTPUT: Real-space field (numpy array of size (dim, N, ..., N, nsteps) with N repeated dim times)
    """

    if CUPY_IMPORTED:
        xp = cp.get_array_module(field)
    else:
        xp = np

    dim = field.shape[0]
    nsteps = field.shape[-1]

    if dim==1:
        real_field = xp.zeros((1,N,nsteps))
        real_field[0] = xp.fft.irfft(field, axis=0)
    elif dim==2:
        real_field = xp.zeros((2,N,N,nsteps))
        real_field[0] = xp.fft.irfft2(field[0,...], axes=(0,1))
        real_field[1] = xp.fft.irfft2(field[1,...], axes=(0,1))
    else:
        real_field = xp.zeros((3,N,N,N,nsteps))
        real_field[0] = xp.fft.irfftn(field[0,...], axes=(0,1,2))
        real_field[1] = xp.fft.irfftn(field[1,...], axes=(0,1,2))
        real_field[2] = xp.fft.irfftn(field[2,...], axes=(0,1,2))

    real_field = xp.array(real_field)
    return real_field

if __name__ == "__main__":
    main()
