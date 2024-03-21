import numpy as np
import os

def main():

    params = {}

    params['N'] = 100
    params['print_freq'] = 100
    params['do_output'] = 1
    params['output_freq'] = 100
    params['lambda'] = 10.0
    params['tau'] = 1.0
    params['dim'] = 2
    params['nsteps'] = 1000
    params['dt'] = 1e-2
    params['D'] = 1.0
    params['cov_type'] = 'exponential'

    gen_trajectory(**params)


def gen_trajectory(**kwargs):

    """Generate an active noise trajectory

    INPUT: Dictionary containing parameters
    OUTPUT: noise field trajectory (numpy array)
    """

    N = kwargs['N']
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

    if dim==1:
        traj_arr = np.zeros((dim,N,nsteps))
    elif dim==2:
        traj_arr = np.zeros((dim,N,N,nsteps))
    else:
        traj_arr = np.zeros((dim,N,N,N,nsteps))

    #Create output directory
    if not os.path.exists('./data') and do_output==1:
        os.makedirs('./data')
 
    #Generate initial field
    ck = get_spatial_covariance(N, dim, cov_type, Lambda)
    fourier_noise = np.sqrt(D*N**dim)*gen_field(N, dim, ck)
    real_noise = get_real_field(fourier_noise, N)
    traj_arr[...,0] = real_noise
    if do_output==1:
        np.savez('data/field_0000.npz', real_noise)

    #Get trajectory
    for n in range(nsteps-1):
        if n%print_freq==0:
            print('active noise step', n)

        noise_incr = np.sqrt(2*D*N**dim*dt/tau)*gen_field(N, dim, ck);
        fourier_noise = (1.0-dt/tau)*fourier_noise + noise_incr
        real_noise = get_real_field(fourier_noise, N)

        traj_arr[...,n+1] = real_noise

        if do_output==1 and n%output_freq==0:
            np.savez('data/field_%04d.npz' % (n+1), real_noise)

    print('TEST:', traj_arr.shape)
    return traj_arr


def get_spatial_covariance(N, dim, cov_type, l):

    """Create a matrix containing the spatial covariance function
    at each allowed wavevector.

    INPUT: Linear size of noise field (int),
           dimensions (int),
           mathematical form of covariance function (string),
           correlation length (float)
    OUTPUT: Numpy array containing spatial correlation at each wavevector
    """
           
    #Define wavevectors
    myvec = np.zeros(N)
    for i in range(N//2+1):
        myvec[i] = i
    for i in range(N//2+1,N):
        myvec[i] = i-N
    kvec = 2*np.pi/N*myvec
    if dim==1:
        kx = kvec
    elif dim==2:
        kx, ky = np.meshgrid(kvec, kvec)
    else:
        kx, ky, kz = np.meshgrid(kvec, kvec, kvec)

    #Get spatial correlation function
    if cov_type=='exponential':
        if dim==1:
            ck = 1.0/(1+l**2*kx**2)
        elif dim==2:
            ck = 2*np.pi*l**2*N**2/pow(1+l**2*(kx**2+ky**2),3.0/2.0)
        else:
            ck = 1.0/pow(1+l**2*(kx**2+ky**2+kz**2),2.0)
    elif cov_type=='gaussian':
        if dim==1:
            ck = np.exp(-l**2*kx**2/2.0)
        elif dim==2:
            ck = np.exp(-l**2*(kx**2+ky**2)/2.0)
        else:
            ck = np.exp(-l**2*(kx**2+ky**2+kz**2)/2.0)
    else:
        ck = 0.0*kx
    ck = ck/(np.sum(ck)*N**dim)

    return ck

def gen_field(N, dim, ck):

    """Create a spatially correlated noise field in Fourier space

    INPUT: Linear size of field (int),
           dimension of field (int),
           spatial correlation function (numpy array)

    OUTPUT: Noise field (numpy array of size (dim, N, ..., N/2+1) with N repeated dim-1 times)
    """

    if dim==1:
        noise = np.zeros((1,N//2+1), dtype=np.complex128)
    elif dim==2:
        noise = np.zeros((2,N,N//2+1), dtype=np.complex128)
    else:
        noise = np.zeros((3,N,N,N//2+1), dtype=np.complex128)
    for d in range(dim):
        #white_noise_real[d] = np.random.normal(loc=0.0, scale=1.0, size=tuple([N]*dim))
        white_noise = np.sqrt(N**dim)*np.random.normal(loc=0.0, scale=1.0, size=tuple([N]*dim))
        if dim==1:
            fourier_white_noise = np.fft.rfft(white_noise)
            noise[d] = np.multiply(np.sqrt(ck[:(N//2+1)]), fourier_white_noise)
        elif dim==2:
            fourier_white_noise = np.fft.rfft2(white_noise)
            noise[d] = np.multiply(np.sqrt(ck[:,:(N//2+1)]), fourier_white_noise)
        else:
            fourier_white_noise = np.fft.rfftn(white_noise)
            noise[d] = np.multiply(np.sqrt(ck[:,:,:(N//2+1)]), fourier_white_noise)

    noise = np.array(noise)

    return noise

def get_real_field(field, N):

    """Transform fourier space field to real space

    INPUT: Fourier-space field (numpy array),
           linear size of field (int)

    OUTPUT: Real-space field (numpy array of size (dim, N, ..., N) with N repeated dim times)
    """

    dim = field.shape[0]

    if dim==1:
        real_field = np.zeros((1,N))
        real_field[0] = np.fft.irfft(field)
    elif dim==2:
        real_field = np.zeros((2,N,N))
        real_field[0] = np.fft.irfft2(field[0,:,:])
        real_field[1] = np.fft.irfft2(field[1,:,:])
    else:
        real_field = np.zeros((3,N,N,N))
        real_field[0] = np.fft.irfftn(field[0,:,:,:])
        real_field[1] = np.fft.irfftn(field[1,:,:,:])
        real_field[2] = np.fft.irfftn(field[2,:,:,:])

    real_field = np.array(real_field)
    return real_field

if __name__ == '__main__':
    main()
