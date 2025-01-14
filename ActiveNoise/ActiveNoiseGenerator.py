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

class ActiveNoiseGenerator():

    def __init__(self, **kwargs):
        self.dim = kwargs['dim']
        if 'N' in kwargs:
            self.Nx = kwargs['N']
            self.Ny = kwargs['N']
            self.Nz = kwargs['N']
        else:
            self.Nx = kwargs['Nx']
            if self.dim>1:
                self.Ny = kwargs['Ny']
            if self.dim>2:
                self.Nz = kwargs['Nz']
        self.dx = kwargs['dx']
        if 'dy' in kwargs:
            self.dy = kwargs['dy']
        else:
            self.dy = self.dx
        if 'dz' in kwargs:
            self.dz = kwargs['dz']
        else:
            self.dz = self.dx
        self.print_freq = kwargs['print_freq']
        self.do_output = kwargs['do_output']
        self.output_freq = kwargs['output_freq']
        self.Lambda = kwargs['lambda']
        self.tau = kwargs['tau'] 
        self.nsteps = kwargs['nsteps']
        self.chunksize = kwargs['chunksize']
        self.dt = kwargs['dt']
        self.D = kwargs['D']
        self.cov_type = kwargs['cov_type']
        self.compressibility = kwargs['compressibility']
        self.do_lattice_correction = kwargs['do_lattice_correction']
        self.xpu = kwargs['xpu']
        self.verbose = kwargs['verbose']
        self.seed = kwargs['seed']

        #derived variables
        self.Lx = self.Nx*self.dx
        if self.dim>1:
            self.Ly = self.Ny*self.dy
        if self.dim>2:
            self.Lz = self.Nz*self.dz
        self.nchunks = self.nsteps//self.chunksize

        if CUPY_IMPORTED:
            if self.xpu=='gpu':
                if self.verbose:
                    print('Using GPU for active noise')
                self.xp = cp
            else:
                if self.verbose:
                    print('Using CPU for active noise')
                self.xp = np
        else:
            if self.verbose:
                print('Using CPU for active noise')
            self.xp = np

        if self.dim==1:
            self.traj_arr = np.zeros((self.dim,self.Nx,self.nsteps+1))
        elif self.dim==2:
            self.traj_arr = np.zeros((self.dim,self.Nx,self.Ny,self.nsteps+1))
        else:
            self.traj_arr = np.zeros((self.dim,self.Nx,self.Ny,self.Nz,self.nsteps+1))

        if self.dim==1:
            self.Narr = np.array([self.Nx])
            self.dxarr = np.array([self.dx])
            self.vol = self.Lx
        elif self.dim==2:
            self.Narr = np.array([self.Nx,self.Ny])
            self.dxarr = np.array([self.dx,self.dy])
            self.vol = self.Lx*self.Ly
        else:
            self.Narr = np.array([self.Nx,self.Nyself.Nz])
            self.dxarr = np.array([self.dx,self.dy,self.dz])
            self.vol = self.Lx*self.Ly*self.Lz

        #Set random seed
        self.rng = self.xp.random.default_rng(self.seed)
    
    def run(self, init_fourier_arr):

        if self.dim==1:
            self.traj_arr = np.zeros((self.dim,self.Nx,self.nsteps+1))
        elif self.dim==2:
            self.traj_arr = np.zeros((self.dim,self.Nx,self.Ny,self.nsteps+1))
        else:
            self.traj_arr = np.zeros((self.dim,self.Nx,self.Ny,self.Nz,self.nsteps+1))

        ck = self.get_spatial_covariance()
        fourier_noise = init_fourier_arr

        #Generate a random field if init array is empty
        if fourier_noise.size==0:
            spat_corr_field = self.gen_field(1, ck)
            fourier_noise = self.xp.sqrt(self.D*self.vol)*spat_corr_field[...,0]

        #Put in step 0 because hoomd requires it for multiple runs
        if self.xp==np:
            self.traj_arr[...,0] = self.get_real_field(fourier_noise[...,np.newaxis], self.Narr)[...,0]
        else: 
            self.traj_arr[...,0] = self.xp.asnumpy(self.get_real_field(fourier_noise[...,np.newaxis], self.Narr)[...,0])

        for c in range(self.nchunks):
            if self.dim==1:
                fourier_arr = self.xp.zeros((1,self.Nx//2+1,self.chunksize), dtype=self.xp.complex64)
            elif self.dim==2:
                fourier_arr = self.xp.zeros((2,self.Nx,self.Ny//2+1,self.chunksize), dtype=self.xp.complex64)
            else:
                fourier_arr = self.xp.zeros((3,self.Nx,self.Ny,self.Nz//2+1,self.chunksize), dtype=self.xp.complex64)
            spat_corr_field = self.gen_field(self.chunksize, ck)

            for n in range(self.chunksize):
                if n%self.print_freq==0 and self.verbose:
                    print('active noise step', n)
                noise_incr = self.xp.sqrt(2*self.D*self.vol*self.dt/self.tau)*spat_corr_field[...,n]
                fourier_noise = (1.0-self.dt/self.tau)*fourier_noise + noise_incr
                fourier_arr[...,n] = fourier_noise

            #Take inverse fourier transform of noise trajectory    
            if self.xp==np:
                self.traj_arr[...,(c*self.chunksize+1):((c+1)*self.chunksize+1)] = self.get_real_field(fourier_arr, self.Narr)
            else:
                self.traj_arr[...,(c*self.chunksize+1):((c+1)*self.chunksize+1)] = self.xp.asnumpy(self.get_real_field(fourier_arr, self.Narr))

            if self.do_output==1:
                self.xp.savez('data/field_%04d.npz' % (c*self.chunksize), self.traj_arr[...,(c*self.chunksize+1):((c+1)*self.chunksize+1)])
                
        return self.traj_arr, fourier_noise

    def get_spatial_covariance(self):

        """Create a matrix containing the spatial covariance function
        at each allowed wavevector.

        INPUT: Size of noise field (int array),
            grid spacing (int array)
            dimensions (int),
            mathematical form of covariance function (string),
            correlation length (float)
        OUTPUT: Numpy array containing spatial correlation at each wavevector
        """
    
        #Define wavevectors
        myvecs = []
        for d in range(self.dim):
            myvec = self.xp.zeros(self.Narr[d])
            for i in range(self.Narr[d]//2+1):
                myvec[i] = i
            for i in range(self.Narr[d]//2+1,self.Narr[d]):
                myvec[i] = i-self.Narr[d]
            kvec = 2*self.xp.pi/(self.Narr[d]*self.dxarr[d])*myvec
            myvecs.append(kvec)
        if self.dim==1:
            kx = myvecs[0]
        elif self.dim==2:
            kx, ky = self.xp.meshgrid(myvecs[0], myvecs[1], indexing='ij')
        else:
            kx, ky, kz = self.xp.meshgrid(myvecs[0], myvecs[1], myvecs[2], indexing='ij')

        vol = 1.0
        for d in range(self.dim):
            vol *= self.Narr[d]*self.dxarr[d]

        #Get spatial correlation function
        if self.cov_type=='exponential':
            if self.dim==1:
                ck = 1.0/(1+self.Lambda**2*kx**2)
            elif self.dim==2:
                ck = 1.0/pow(1+self.Lambda**2*(kx**2+ky**2),3.0/2.0)
            else:
                ck = 1.0/pow(1+self.Lambda**2*(kx**2+ky**2+kz**2),2.0)
        elif self.cov_type=='gaussian':
            if self.dim==1:
                ck = self.xp.exp(-self.Lambda**2*kx**2/2.0)
            elif self.dim==2:
                ck = self.xp.exp(-self.Lambda**2*(kx**2+ky**2)/2.0)
            else:
                ck = self.xp.exp(-self.Lambda**2*(kx**2+ky**2+kz**2)/2.0)
        else:
            ck = 0.0*kx
        ck = ck/(self.xp.sum(ck)*vol)

        return ck

    def gen_field(self, nsteps, ck):

        """Create a spatially correlated noise field in Fourier space

        INPUT: Linear size of field (int),
            dimension of field (int),
            number of time steps (int),
            spatial correlation function (numpy array),
            compressibility (string)

        OUTPUT: Noise field (numpy array of size (dim, N, ..., N/2+1, nsteps) with N repeated dim-1 times)
        """

        if self.dim==1:
            noise = self.xp.zeros((1,self.Narr[0]//2+1,nsteps), dtype=self.xp.complex64)
        elif self.dim==2:
            noise = self.xp.zeros((2,self.Narr[0],self.Narr[1]//2+1,nsteps), dtype=self.xp.complex64)
        else:
            noise = self.xp.zeros((3,self.Narr[0],self.Narr[1],self.Narr[2]//2+1,self.nsteps), dtype=self.xp.complex64)

        #print([dim] + [N]*dim + [nsteps])
        #print('generating big white noise array...')
        if self.dim==1:
            white_noise = self.xp.sqrt(self.Narr[0])*self.rng.standard_normal(size=tuple([self.dim] + [self.Narr[0]] + [nsteps]))
        elif self.dim==2:
            white_noise = self.xp.sqrt(self.Narr[0]*self.Narr[1])*self.rng.standard_normal(size=tuple([self.dim] + [self.Narr[0]] + [self.Narr[1]] + [self.nsteps]))
        else:
            white_noise = self.xp.sqrt(self.Narr[0]*self.Narr[1]*self.Narr[2])*self.rng.standard_normal( size=tuple([self.dim] + [self.Narr[0]] + [self.Narr[1]] + [self.Narr[2]] + [self.nsteps]))
        #print('done')

        if self.compressibility=='incompressible':
            #Define wavevectors
            myvecs = []
            for d in range(self.dim):
                myvec = self.xp.zeros(self.Narr[d])
                for i in range(self.Narr[d]//2+1):
                    myvec[i] = i
                for i in range(self.Narr[d]//2+1,self.Narr[d]):
                    myvec[i] = i-self.Narr[d]
                kvec = 2*self.xp.pi/(self.Narr[d]*self.dxarr[d])*myvec
                if self.do_lattice_correction==True:
                    print('doing correction')
                    kvec = np.sin(kvec*self.dxarr[d])/self.dxarr[d]
                myvecs.append(kvec)
            if self.dim==1:
                kx = myvecs[0]
            elif self.dim==2:
                kx, ky = self.xp.meshgrid(myvecs[0], myvecs[1], indexing='ij')
                kmat = np.stack((kx, ky), axis=-1)
            else:
                kx, ky, kz = self.xp.meshgrid(myvecs[0], myvecs[1], myvecs[2], indexing='ij')
                kmat = np.stack((kx, ky, kz), axis=-1)

        for t in range(nsteps):
            for d in range(self.dim):
                if self.dim==1:
                    fourier_white_noise = self.xp.fft.rfft(white_noise[d,...,t], axis=0)
                    noise[d,...,t] = self.xp.multiply(self.xp.sqrt(ck[...,:(self.Narr[-1]//2+1)]), fourier_white_noise)
                elif self.dim==2:
                    fourier_white_noise = self.xp.fft.rfft2(white_noise[d,...,t], axes=(0,1))
                    noise[d,...,t] = self.xp.multiply(self.xp.sqrt(ck[...,:(self.Narr[-1]//2+1)]), fourier_white_noise)
                else:
                    fourier_white_noise = self.xp.fft.rfftn(white_noise[d,...,t], axes=(0,1,2))
                    noise[d,...,t] = self.xp.multiply(self.xp.sqrt(ck[...,:(self.Narr[-1]//2+1)]), fourier_white_noise)
            if self.compressibility=='incompressible':
                #projector = np.identity(dim)
                if self.dim==1:
                    print('WARNING: incompressibility invalid in 1d. Not doing projection.')
                elif self.dim==2:
                    k2 = self.xp.einsum('ijk,ijk->ij', kmat, kmat)
                    #print(k2.shape)
                    k2mat = self.xp.einsum('ijk,ijl->ijkl', kmat/np.sqrt(k2[:,:,None]), kmat/np.sqrt(k2[:,:,None]))
                    k2mat = self.xp.nan_to_num(k2mat) #corrects for dividing by k=0
                    id = self.xp.zeros(k2mat.shape)
                    for i in range(self.dim):
                        for j in range(self.dim):
                            if i==j:
                                id[:,:,i,j] = 1
                            else:
                                id[:,:,i,j] = 0
                    projector = id-k2mat
                    projector = projector[:,:(self.Narr[-1]//2+1),:,:]
                    k2mat = k2mat[:,:(self.Narr[-1]//2+1),:,:]
                    #print(noise[...,t].shape)
                    noise[...,t] = np.einsum('ijkl,lij->kij', projector, noise[...,t])
                    #print(np.max(np.abs(np.einsum('ijkl,lij->kij', k2mat, noise[...,t])))) #this should be very small
                else:
                    k2 = self.xp.einsum('ijkl,ijkl->ijk', kmat, kmat)
                    #print(k2.shape)
                    k2mat = self.xp.einsum('ijkl,ijkm->ijklm', kmat/np.sqrt(k2[:,:,:,None]), kmat/np.sqrt(k2[:,:,:,None]))
                    k2mat = self.xp.nan_to_num(k2mat) #corrects for dividing by k=0
                    id = self.xp.zeros(k2mat.shape)
                    for i in range(self.dim):
                        for j in range(self.dim):
                            if i==j:
                                id[:,:,:,i,j] = 1
                            else:
                                id[:,:,:,i,j] = 0
                    projector = id-k2mat
                    projector = projector[:,:,:(self.Narr[-1]//2+1),:,:]
                    k2mat = k2mat[:,:,:(self.Narr[-1]//2+1),:,:]
                    #print(noise[...,t].shape)
                    noise[...,t] = np.einsum('ijklm,mijk->lijk', projector, noise[...,t])
                    #print(np.max(np.abs(np.einsum('ijklm,mijk->lijk', k2mat, noise[...,t])))) #this should be very small
                
        noise = self.xp.array(noise)
        #print('done coloring noise')

        return noise

    def get_real_field(self, field, Narr):

        """Transform fourier space field to real space

        INPUT: Fourier-space field (numpy array),
            linear size of field (int)

        OUTPUT: Real-space field (numpy array of size (dim, N, ..., N, nsteps) with N repeated dim times)
        """

        nsteps = field.shape[-1]

        if self.dim==1:
            real_field = self.xp.zeros((1,self.Narr[0],nsteps))
            real_field[0] = self.xp.fft.irfft(field, axis=0)
        elif self.dim==2:
            real_field = self.xp.zeros((2,self.Narr[0],self.Narr[1],nsteps))
            real_field[0] = self.xp.fft.irfft2(field[0,...], axes=(0,1))
            real_field[1] = self.xp.fft.irfft2(field[1,...], axes=(0,1))
        else:
            real_field = self.xp.zeros((3,self.Narr[0],self.Narr[1],self.Narr[2],nsteps))
            real_field[0] = self.xp.fft.irfftn(field[0,...], axes=(0,1,2))
            real_field[1] = self.xp.fft.irfftn(field[1,...], axes=(0,1,2))
            real_field[2] = self.xp.fft.irfftn(field[2,...], axes=(0,1,2))

        real_field = self.xp.array(real_field)
        return real_field
