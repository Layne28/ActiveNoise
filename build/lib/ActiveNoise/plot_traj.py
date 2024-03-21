import numpy as np
import os
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
import AnalysisTools.field_analysis as field_analysis

def main():

    compressibility='compressible'
    thetype='exponential'

    N = 100
    #ls = [1.0, 2.0, 5.0, 10.0, 20.0, 50.0, 100.0]

    nframes=100
    ncells = np.array([N,N])
    spacing = np.array([1.0,1.0])

    for i in range(nframes):

        print(i)

        fig, axs = plt.subplots(1, 3, figsize=(13,5), sharex=True, sharey=True)
        field = np.load('data/field_%04d.npz' % i)['arr_0']
        field = np.transpose(field, (1,2,0))
        mag = field[:,:,0]**2 + field[:,:,1]**2
        div = field_analysis.get_divergence(2, ncells, spacing, field)
        curl = field_analysis.get_curl(2, ncells, spacing, field)

        if i==0:
            mymax_mag = np.max(mag)
            mymin_mag = 0
        im1 = axs[0].imshow(mag, cmap='viridis', vmin=mymin_mag, vmax=mymax_mag)

        if i==0:
            mymax_curl = np.max(np.abs(curl))
            mymin_curl = -mymax_curl
        im2 = axs[1].imshow(div, cmap='RdBu_r', vmin=mymin_curl, vmax=mymax_curl)
        im3 = axs[2].imshow(curl, cmap='PiYG', vmin=mymin_curl, vmax=mymax_curl)

        axs[0].set_title(r'magnitude')
        axs[1].set_title(r'div')
        axs[2].set_title(r'curl')

        divider = make_axes_locatable(axs[0])
        cax = divider.append_axes("right", size="5%", pad=0.05)
        fig.colorbar(im1, cax=cax)

        divider = make_axes_locatable(axs[1])
        cax = divider.append_axes("right", size="5%", pad=0.05)
        fig.colorbar(im2, cax=cax)

        divider = make_axes_locatable(axs[2])
        cax = divider.append_axes("right", size="5%", pad=0.05)
        fig.colorbar(im3, cax=cax)

        plt.suptitle('%s %s' % (compressibility, thetype))

        if not os.path.exists('./plots'):
            os.makedirs('./plots')
        plt.savefig('plots/field_%s_%s_%04d.png' % (compressibility, thetype, i))
        plt.close()


if __name__ == "__main__":
    main()
