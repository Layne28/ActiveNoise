import numpy as np
import os

def main():

    N = 100
    l = 10.0
    tau = 1.0
    dim = 2
    nsteps = 1000
    dt = 1e-2
    D = 1.0

    cov_type = 'exponential'

    #Define noise array
    noise = [0]*dim
    pos_dims = tuple([N]*dim)
    print(pos_dims)
    for d in range(dim):
        noise[d] = np.zeros(pos_dims)
    noise = np.array(noise)
    print(noise.shape)

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
    
    #Generate initial field
    fourier_noise = np.sqrt(D*N**dim)*gen_field(N, dim, ck)
    real_noise = get_real_field(fourier_noise, N)
    np.savez('data/field_0000.npz', real_noise)

    #Get trajectory
    for n in range(nsteps):
        print('step', n)
        noise_incr = np.sqrt(2*D*N**dim*dt/tau)*gen_field(N, dim, ck);
        print(1.0-dt/tau)
        fourier_noise = (1.0-dt/tau)*fourier_noise + noise_incr
        print(np.sum(fourier_noise))
        real_noise = get_real_field(fourier_noise, N)
        if not os.path.exists('./data'):
            os.makedirs('./data')
        np.savez('data/field_%04d.npz' % (n+1), real_noise)


def gen_field(N, dim, ck):

    if dim==1:
        noise = np.zeros((1,N//2+1), dtype=np.complex128)
    elif dim==2:
        noise = np.zeros((2,N,N//2+1), dtype=np.complex128)
    else:
        noise = np.zeros((3,N,N,N//2+1), dtype=np.complex128)
    print('noise', noise.shape)
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

    dim = field.shape[0]
    if dim==1:
        real_field = np.zeros((1,N))
    elif dim==2:
        real_field = np.zeros((2,N,N))
    else:
        real_field = np.zeros((3,N,N,N))
    if dim==1:
        real_field[0] = np.fft.irfft(field)
    elif dim==2:
        real_field[0] = np.fft.irfft2(field[0,:,:])
        real_field[1] = np.fft.irfft2(field[1,:,:])
    else:
        real_field[0] = np.fft.irfftn(field[0,:,:,:])
        real_field[1] = np.fft.irfftn(field[1,:,:,:])
        real_field[2] = np.fft.irfftn(field[2,:,:,:])

    real_field = np.array(real_field)
    return real_field

if __name__ == '__main__':
    main()
