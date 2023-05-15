import numpy as np

import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib.colors import LogNorm, SymLogNorm
mpl.use('agg')

from pathlib import Path

from tqdm import tqdm
from astropy.io import fits

import pippin.auxiliary_functions as af

# Setting the length of progress bars
pbar_format = '{l_bar}{bar:20}{r_bar}{bar:-20b}'

################################################################################
# Make figures
################################################################################

def plot_reduction(path_SCIENCE_files_selected, 
                   path_reduced_files_selected=None, 
                   path_skysub_files_selected=None, 
                   path_beams_files_selected=None, 
                   beam_centers=None,
                   size_to_crop=None, 
                   Wollaston_45=False
                   ):

    if (beam_centers is not None) and np.isnan(beam_centers[0][:,1]).all():
        width_ratios, nrows = [1,1,1,1], 1
    else:
        width_ratios, nrows = [1,1,1,0.5], 2

    cmap = mpl.colors.LinearSegmentedColormap.from_list('', ['k','C0','w'])
    cmap_skysub = mpl.colors.LinearSegmentedColormap.from_list('', ['k','C1','w'])

    for i in tqdm(range(len(path_SCIENCE_files_selected)), \
                  bar_format=pbar_format):

        fig = plt.figure(figsize=(12,4))
        gs = fig.add_gridspec(ncols=4, nrows=nrows, wspace=0.05, hspace=0.05,
                              width_ratios=width_ratios, left=0.02, right=0.98,
                              bottom=0.02, top=0.98)

        # Create the axes objects
        if nrows == 1:
            ax = [fig.add_subplot(gs[0]), fig.add_subplot(gs[1]),
                  fig.add_subplot(gs[2]), fig.add_subplot(gs[3])]
        else:
            ax = [fig.add_subplot(gs[:,0]), fig.add_subplot(gs[:,1]),
                  fig.add_subplot(gs[:,2]), fig.add_subplot(gs[0,3]),
                  fig.add_subplot(gs[1,3])]

        ax[0].set_title('Raw SCIENCE image')
        ax[1].set_title('Calibrated image')
        ax[2].set_title('Sky-subtracted image')
        ax[3].set_title('Ord./Ext. beams')

        if path_reduced_files_selected is not None:
            # Plot the raw SCIENCE image
            file_i = path_SCIENCE_files_selected[i]
            SCIENCE_i, _ = af.read_FITS_as_cube(file_i)
            SCIENCE_i = np.nanmedian(SCIENCE_i, axis=0) # Median-combine

            yp, xp = np.mgrid[0:SCIENCE_i.shape[0], 0:SCIENCE_i.shape[1]]

            vmin, vmax = np.nanmedian(SCIENCE_i), np.nanmax(SCIENCE_i)
            if (vmin >= vmax):
                vmin = 0.1*vmax
            #print(vmin, vmax)
            ax[0].imshow(SCIENCE_i, cmap=cmap, aspect='equal',
                         interpolation='none',
                         norm=SymLogNorm(linthresh=1e-16, vmin=vmin, vmax=vmax))

            # Plot the calibrated SCIENCE image
            file_i = path_reduced_files_selected[i]
            reduced_i = fits.getdata(file_i).astype(np.float32)
            reduced_i = np.nanmedian(reduced_i, axis=0) # Median-combine

            yp, xp = np.mgrid[0:reduced_i.shape[0], 0:reduced_i.shape[1]]


            vmin, vmax = np.nanmedian(reduced_i), np.nanmax(reduced_i)
            if (vmin >= vmax):
                vmin = 0.1*vmax
            #print(vmin, vmax)
            ax[1].imshow(reduced_i, cmap=cmap, aspect='equal',
                         interpolation='none',
                         extent=(xp.min(), xp.max(), yp.max(), yp.min()),
                         norm=SymLogNorm(linthresh=1e-16, vmin=vmin, vmax=vmax))

        if path_skysub_files_selected is not None:
            # Plot the sky-subtracted image
            file_i = path_skysub_files_selected[i]
            skysub_i = fits.getdata(file_i).astype(np.float32)
            skysub_i = np.nanmedian(skysub_i, axis=0) # Median-combine

            yp, xp = np.mgrid[0:skysub_i.shape[0], 0:skysub_i.shape[1]]

            skysub_i_pos = np.ma.masked_array(skysub_i, mask=~(skysub_i > 0))
            vmin, vmax = np.nanmedian(skysub_i_pos), np.nanmax(skysub_i_pos)
            if vmin >= vmax:
                vmin = 0.1*vmax
            #print(vmin, vmax)
            ax[2].imshow(skysub_i_pos, cmap=cmap, aspect='equal',
                         interpolation='none',
                         extent=(xp.min(), xp.max(), yp.max(), yp.min()),
                         norm=SymLogNorm(linthresh=1e-16, vmin=vmin, vmax=vmax))

            skysub_i_neg = np.ma.masked_array(skysub_i, mask=~(skysub_i < 0))
            vmin, vmax = np.nanmedian(-skysub_i_neg), np.nanmax(-skysub_i_neg)
            if vmin >= vmax:
                vmin = 0.1*vmax
            #print(vmin, vmax)
            ax[2].imshow(-skysub_i_neg, cmap=cmap_skysub, aspect='equal',
                         interpolation='none',
                         extent=(xp.min(), xp.max(), yp.max(), yp.min()),
                         norm=SymLogNorm(linthresh=1e-16, vmin=vmin, vmax=vmax))

            # Plot the ord./ext. beams
            file_i = path_beams_files_selected[i]
            beams_i = fits.getdata(file_i).astype(np.float32)

            vmin, vmax = np.nanmedian(beams_i), np.nanmax(beams_i)
            if vmin >= vmax:
                vmin = 0.1*vmax
            #print(vmin, vmax)
            #print()
            if np.isnan(vmin) or np.isnan(vmax):
                vmin, vmax = 0, 1
            ax[3].imshow(beams_i[0], cmap=cmap, aspect='equal',
                         interpolation='none',
                         norm=SymLogNorm(linthresh=1e-16, vmin=vmin, vmax=vmax))

            if beams_i.shape[0] != 1:
                ax[4].imshow(beams_i[1], cmap=cmap, aspect='equal',
                             interpolation='none',
                             norm=SymLogNorm(linthresh=1e-16,
                                             vmin=vmin, vmax=vmax))

        if beam_centers is not None:
            ord_beam_center_i = np.median(beam_centers[i][:,0,:], axis=0)
            ext_beam_center_i = np.median(beam_centers[i][:,1,:], axis=0)

            for ax_i in ax[1:3]:
                ax_i.scatter(ord_beam_center_i[0], ord_beam_center_i[1],
                             marker='+', color='C3')
                rect = mpl.patches.Rectangle(xy=(ord_beam_center_i[0]-
                                                 size_to_crop[1]/2,
                                                 ord_beam_center_i[1]-
                                                 size_to_crop[0]/2),
                                             width=size_to_crop[1],
                                             height=size_to_crop[0],
                                             edgecolor='C3',
                                             facecolor='none')
                ax_i.add_patch(rect)

                ax_i.scatter(ext_beam_center_i[0], ext_beam_center_i[1],
                             marker='x', color='C3')
                rect = mpl.patches.Rectangle(xy=(ext_beam_center_i[0]-
                                                 size_to_crop[1]/2,
                                                 ext_beam_center_i[1]-
                                                 size_to_crop[0]/2),
                                             width=size_to_crop[1],
                                             height=size_to_crop[0],
                                             edgecolor='C3',
                                             facecolor='none')
                ax_i.add_patch(rect)

        for ax_i in ax:
            ax_i.set(xticks=[], yticks=[])
            ax_i.set_facecolor('w')
            ax_i.invert_yaxis()

        # Path to figure file. Create directory if it does not exist yet.
        path_to_fig = Path(path_reduced_files_selected[i].parent, 'plots',
                           path_reduced_files_selected[i].name.replace('_reduced.fits', '.pdf'))
        if not path_to_fig.parent.is_dir():
            Path(path_to_fig.parent).mkdir()

        fig.savefig(path_to_fig)#, dpi=200)
        plt.close() # Remove from memory

def plot_open_AO_loop(path_reduced_files_selected, 
                      max_counts, 
                      bounds_ord_beam, 
                      bounds_ext_beam
                      ):
    '''
    Plot the maximum beam-intensities and the open AO-loop bounds.

    Input
    -----
    max_counts : 2D-array
        Maximum beam-intensities with shape
        (observations, ordinary/extra-ordinary beam).
    bounds_ord_beam : 1D-array
        Minimum/maximum open AO-loop bounds for ordinary beam.
    bounds_ext_beam : 1D-array
        Minimum/maximum open AO-loop bounds for extra-ordinary beam.
    '''

    fig, ax = plt.subplots(figsize=(6,4))
    ax.hlines(bounds_ord_beam, xmin=0, xmax=len(max_counts)+1,
              color='r', ls='--')
    ax.hlines(bounds_ext_beam, xmin=0, xmax=len(max_counts)+1,
              color='b', ls='--')

    ax.plot(np.arange(1,len(max_counts)+1,1), max_counts[:,0],
            c='r', label='Ordinary beam')
    if max_counts.shape[1] != 1:
        ax.plot(np.arange(1,len(max_counts)+1,1), max_counts[:,1],
                c='b', label='Extra-ordinary beam')

    ax.legend(loc='best')
    ax.set(ylabel='Maximum counts', xlabel='Observation',
           xlim=(0,len(max_counts)+1), xticks=np.arange(1,len(max_counts)+1,5))
    fig.tight_layout()
    #plt.show()

    # Path to figure file. Create directory if it does not exist.
    path_to_fig = Path(path_reduced_files_selected[0].parent,
                       'plots', 'max_counts.pdf')
    if not path_to_fig.parent.is_dir():
        Path(path_to_fig.parent).mkdir()

    plt.savefig(path_to_fig)
    plt.close()
