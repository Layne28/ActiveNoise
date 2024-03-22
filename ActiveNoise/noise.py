import numpy as np
import os
import sys
import argparse
import GPUtil

if len(GPUtil.getAvailable())>0:
    import cupy as cp

def main():

    parser = argparse.ArgumentParser(description='Generate active noise trajectory')
    parser.add_argument('--N', default=400, help='linear system size')
    parser.add_argument('--do_output', default=0, help='whether to print noise field to file')
    parser.add_argument('--print_freq', default=100, help='how often to output noise configurations (in timesteps)')
    parser.add_argument('--nsteps', default=100, help='number of timesteps')
    parser.add_argument('--xpu', default='gpu', help='cpu or gpu')

    args = parser.parse_args()
    N = args.N
    do_output = args.do_output
    print_freq = args.print_freq
    nsteps = args.nsteps
    xpu = args.xpu

    params = {}

    params['N'] = int(N)
    params['dx'] = 1.0
    params['print_freq'] = 100
    params['do_output'] = int(do_output)
    params['output_freq'] = int(print_freq)
    params['lambda'] = 10.0
    params['tau'] = 1.0
    params['dim'] = 2
    params['nsteps'] = int(nsteps)
    params['dt'] = 1e-2
    params['D'] = 1.0
    params['cov_type'] = 'exponential'
    params['xpu'] = xpu

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
    dt = kwargs['dt']
    D = kwargs['D']
    cov_type = kwargs['cov_type']
    xpu = kwargs['xpu']

    #Create output directory
    if not os.path.exists('./data') and do_output==1:
        os.makedirs('./data')

    if len(GPUtil.getAvailable())>0:
        if xpu=='gpu':
            print('Using GPU')
            xp = cp
        else:
            print('Using CPU')
            xp = np
    else:
        print('Using CPU')
        xp = np

    if dim==1:
        init_arr = xp.zeros((dim,N))
    elif dim==2:
        init_arr = xp.zeros((dim,N,N))
    else:
        init_arr = xp.zeros((dim,N,N,N))
 
    #Get trajectory
    traj_arr, fourier_noise = run(init_arr, **kwargs)
    #traj_arr, fourier_noise = run(init_arr, N, dx, print_freq, output_freq, do_output, Lambda, tau, dim, nsteps, dt, D, cov_type, xpu)
    #traj_arr = run(init_arr, N, dx, print_freq, output_freq, do_output, Lambda, tau, dim, nsteps, dt, D, cov_type, xpu)

    return traj_arr

def run(init_fourier_arr, **kwargs):
#def run(init_fourier_arr, N, dx, print_freq, do_output, output_freq, Lambda, tau, dim, nsteps, dt, D, cov_type, xpu):

    N = kwargs['N']
    dx = kwargs['dx']
    print_freq = kwargs['print_freq']
    do_output = kwargs['do_output']
    output_freq = kwargs['output_freq']
    Lambda = kwargs['lambda']
    tau = kwargs['tau']
    dim = kwargs['dim']
    nsteps = kwargs['nsteps']
    dt = kwargs['dt']
    D = kwargs['D']
    cov_type = kwargs['cov_type']
    xpu = kwargs['xpu']
    L = N*dx

    if len(GPUtil.getAvailable())>0:
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
        traj_arr = xp.zeros((dim,N,nsteps))
    elif dim==2:
        traj_arr = xp.zeros((dim,N,N,nsteps))
    else:
        traj_arr = xp.zeros((dim,N,N,N,nsteps))
    '''
    
    if dim==1:
        traj_arr = xp.zeros((1,N//2+1,nsteps), dtype=xp.complex64)
    elif dim==2:
        traj_arr = xp.zeros((2,N,N//2+1,nsteps), dtype=xp.complex64)
    else:
        traj_arr = xp.zeros((3,N,N,N//2+1,nsteps), dtype=xp.complex64)
    
    ck = get_spatial_covariance(N, dx, dim, cov_type, Lambda, xpu)
    fourier_noise = init_fourier_arr

    spat_corr_field = gen_field(N, dx, dim, nsteps, ck)
    for n in range(nsteps):

        if n%print_freq==0:
            print('active noise step', n)

        if n==0: #and xp.count_nonzero(init_fourier_arr)==0:
            print('generating initial field')
            fourier_noise = xp.sqrt(D*L**dim)*spat_corr_field[...,n]#gen_field(N, dx, dim, ck)
            traj_arr[...,0] = fourier_noise
            #real_noise = get_real_field(fourier_noise, N)
            #traj_arr[...,0] = real_noise

        else:
            noise_incr = xp.sqrt(2*D*L**dim*dt/tau)*spat_corr_field[...,n]#gen_field(N, dx, dim, ck);
            fourier_noise = (1.0-dt/tau)*fourier_noise + noise_incr
            traj_arr[...,n] = fourier_noise
            #real_noise = get_real_field(fourier_noise, N)
            #traj_arr[...,n] = real_noise

        if do_output==1 and n%output_freq==0:
            xp.savez('data/field_%04d.npz' % n, real_noise)

    #Take inverse fourier transform of noise trajectory    
    print('computing FFT...')
    traj = get_real_field(traj_arr, N)

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

    if len(GPUtil.getAvailable())>0:
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

    return ck

def gen_field(N, dx, dim, nsteps, ck):

    """Create a spatially correlated noise field in Fourier space

    INPUT: Linear size of field (int),
           dimension of field (int),
           number of time steps (int),
           spatial correlation function (numpy array)

    OUTPUT: Noise field (numpy array of size (dim, N, ..., N/2+1, nsteps) with N repeated dim-1 times)
    """

    if len(GPUtil.getAvailable())>0:
        xp = cp.get_array_module(ck)
    else:
        xp = np

    if dim==1:
        noise = xp.zeros((1,N//2+1,nsteps), dtype=xp.complex64)
    elif dim==2:
        noise = xp.zeros((2,N,N//2+1,nsteps), dtype=xp.complex64)
    else:
        noise = xp.zeros((3,N,N,N//2+1,nsteps), dtype=xp.complex64)

    print([dim] + [N]*dim + [nsteps])
    print('generating big white noise array...')
    white_noise = xp.sqrt(N**dim)*xp.random.normal(loc=0.0, scale=1.0, size=tuple([dim] + [N]*dim + [nsteps]))
    print('done')

    for t in range(nsteps):
        for d in range(dim):
            fourier_white_noise = xp.fft.rfft(white_noise[d,...,t])
            noise[d,...,t] = xp.multiply(xp.sqrt(ck[...,:(N//2+1)]), fourier_white_noise)
    noise = xp.array(noise)
    print('done coloring noise')

    return noise

def get_real_field(field, N):

    """Transform fourier space field to real space

    INPUT: Fourier-space field (numpy array),
           linear size of field (int)

    OUTPUT: Real-space field (numpy array of size (dim, N, ..., N, nsteps) with N repeated dim times)
    """

    if len(GPUtil.getAvailable())>0:
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
