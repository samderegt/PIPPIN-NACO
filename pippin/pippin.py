import numpy as np

import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib.colors import LogNorm, SymLogNorm
mpl.use('agg')

from astropy.io import fits
from astropy.modeling import models, fitting
from astropy.stats import sigma_clip
from astropy.time import Time
from astropy import units as u
from astropy.coordinates import SkyCoord
from astropy.utils.exceptions import AstropyUserWarning

from astroquery.simbad import Simbad

from scipy import ndimage, signal
from scipy.stats import sigmaclip
from scipy.optimize import minimize

import urllib
from pathlib import Path

import configparser
from ast import literal_eval

from tqdm import tqdm

import textwrap
import time
import datetime
import warnings
import sys

################################################################################
# Auxiliary functions
################################################################################

# Setting the length of progress bars
pbar_format = '{l_bar}{bar:20}{r_bar}{bar:-20b}'

def r_phi(im, xc, yc):
    '''
    Get a radius- and angle-array around a center.

    Input
    -----
    im : 2D-array
        Array with same size as the output.
    xc : scalar
        x-coordinate of the center.
    yc : scalar
        y-coordinate of the center.

    Output
    ------
    r : 2D-array
        Radius around (xc, yc).
    phi: 2D-array
        Angle around (xc, yc).
    '''

    yp, xp = np.mgrid[0:im.shape[0], 0:im.shape[1]]

    r   = np.sqrt((yp-yc)**2 + (xp-xc)**2)
    phi = np.arctan2((yp-yc), (xc-xp))

    return r, phi

def Wollaston_beam_separation(camera, filter=''):
    '''
    Get a beam-separation belonging to the utilised camera and filter.

    Input
    -----
    camera : str
        Camera that was used.
    filter : str
        Filter that was used.

    Output
    ------
    beam_separation : int
        Number of pixels separating the beams.
    '''

    # From NACO user manual: Separation in pixels
    all_offsets = {'S13_H':260, 'S13_Ks':254, 'S13':257,
                   'S27_H':126, 'S27_Ks':122, 'S27':124,
                   'S54_H':62, 'S54_Ks':61, 'S54':61.5,
                   'L27':110, 'L54':55}

    key_camera = camera
    key_camera_filter = f'{camera}_{filter}'

    if key_camera_filter in all_offsets.keys():
        # Wavelength-specific separation
        return all_offsets[key_camera_filter]
    elif key_camera in all_offsets.keys():
        # Separation based on utilised camera
        return all_offsets[key_camera]
    else:
        raise KeyError('\nCamera \'{}\' not recognized. Use \'S13\', \'S27\', \'S54\', \'L27\' or \'L54\'.')

def assign_Stokes_parameters(files, HWP_used, Wollaston_used):
    '''
    Assign Stokes parameters based on HWP angles or position angles.

    Input
    -----
    files : 1D-array
        Filenames.
    HWP_used : bool
        If True, HWP was used, else position angle was changed.
    Wollaston_used : bool
        If True, Wollaston was used, else wiregrid was used.

    Output
    ------
    StokesPara : 1D-array
        Stokes parameter-strings ('Q+', 'U+', 'Q-', 'U-')
    '''
    pos_angles = np.array([fits.getheader(x)['ESO ADA POSANG']
                           for x in files])

    def closest_angle(angle_x):

        valid_angles = np.arange(0., 360+22.5, 22.5)
        corr_angle_x = valid_angles[np.argmin(np.abs(valid_angles - angle_x))]

        if corr_angle_x in [0., 90., 180., 270., 360.]:
            corr_angle_x = 0.0    # Replace 360 degree with 0 degree
        if corr_angle_x in [22.5, 112.5, 202.5, 292.5]:
            corr_angle_x = 22.5
        if corr_angle_x in [45., 135., 225., 315.]:
            corr_angle_x = 45.0
        if corr_angle_x in [67.5, 157.5, 247.5, 337.5]:
            corr_angle_x = 67.5

        return corr_angle_x

    if HWP_used:
        # HWP (+ Wollaston)

        HWP_angles = np.zeros(len(files))
        for i, x in enumerate(files):
            # Read the HWP encoder
            HWP_encoder_i = fits.getheader(x)['ESO INS ADC1 ENC']

            # According to the NACO manual
            HWP_angle_i = ((HWP_encoder_i + 205) / (4096/360)) % 360

            # Closest valid HWP angle
            HWP_angles[i] = closest_angle(HWP_angle_i)

        # Assign Stokes parameters based on HWP angles
        StokesPara = np.array(['Unassigned']*len(files))
        StokesPara[HWP_angles==0.00] = 'Q+'
        StokesPara[HWP_angles==22.5] = 'U+'
        StokesPara[HWP_angles==45.0] = 'Q-'
        StokesPara[HWP_angles==67.5] = 'U-'

    elif not HWP_used and Wollaston_used:
        # Rotator + Wollaston

        pos_angles = np.array([fits.getheader(x)['ESO ADA POSANG']
                               for x in files])

        # Subtract smallest position angle
        pos_angles -= pos_angles.min()

        # Closest valid HWP angle
        pos_angles = np.mod(pos_angles, 180)

        Qplus_mask = np.ma.mask_or((pos_angles==0.0), (pos_angles==180.0))
        Umin_mask  = np.ma.mask_or((pos_angles==45.0), (pos_angles==-135.0))
        Qmin_mask  = np.ma.mask_or((pos_angles==90.0), (pos_angles==-90.0))
        Uplus_mask = np.ma.mask_or((pos_angles==-45.0), (pos_angles==135.0))

        # Assign Stokes parameters based on position angles
        StokesPara = np.array(['Unassigned']*len(files))
        StokesPara[Qplus_mask] = 'Q+'
        StokesPara[Umin_mask]  = 'U-'
        StokesPara[Qmin_mask]  = 'Q-'
        StokesPara[Uplus_mask] = 'U+'

    elif not HWP_used and not Wollaston_used:
        # Wiregrid

        wiregrids = []
        for i, x in enumerate(files):
            wiregrid_i = fits.getheader(x)['ESO INS OPTI4 ID']
            wiregrids.append(wiregrid_i)
        wiregrids = np.array(wiregrids)

        Qplus_mask = (wiregrids == 'Pol_00')
        Umin_mask  = (wiregrids == 'Pol_45')
        Qmin_mask  = (wiregrids == 'Pol_90')
        Uplus_mask = (wiregrids == 'Pol_135')

        # Assign Stokes parameters based on position angles
        StokesPara = np.array(['Unassigned']*len(files))
        StokesPara[Qplus_mask] = 'Q+'
        StokesPara[Umin_mask]  = 'U-'
        StokesPara[Qmin_mask]  = 'Q-'
        StokesPara[Uplus_mask] = 'U+'

    return StokesPara

################################################################################
# Make figures
################################################################################

def plot_reduction(plot_reduced=False, plot_skysub=False, beam_centers=None,
                   size_to_crop=None, Wollaston_45=False):

    # path_SCIENCE_files_selected, path_reduced_files_selected,
    # path_skysub_files_selected, path_beams_files_selected

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

        if plot_reduced:
            # Plot the raw SCIENCE image
            file_i = path_SCIENCE_files_selected[i]
            SCIENCE_i, _ = read_FITS_as_cube(file_i)
            SCIENCE_i = np.nanmedian(SCIENCE_i, axis=0) # Median-combine

            yp, xp = np.mgrid[0:SCIENCE_i.shape[0], 0:SCIENCE_i.shape[1]]

            vmin, vmax = np.nanmedian(SCIENCE_i), np.nanmax(SCIENCE_i)
            if vmin >= vmax:
                vmin = 0.1*vmax
            #print(vmin, vmax)
            ax[0].imshow(SCIENCE_i, cmap=cmap, aspect='equal',
                         norm=LogNorm(vmin=vmin, vmax=vmax))

            # Plot the calibrated SCIENCE image
            file_i = path_reduced_files_selected[i]
            reduced_i = fits.getdata(file_i).astype(np.float32)
            reduced_i = np.nanmedian(reduced_i, axis=0) # Median-combine

            yp, xp = np.mgrid[0:reduced_i.shape[0], 0:reduced_i.shape[1]]


            vmin, vmax = np.nanmedian(reduced_i), np.nanmax(reduced_i)
            if vmin >= vmax:
                vmin = 0.1*vmax
            #print(vmin, vmax)
            ax[1].imshow(reduced_i, cmap=cmap, aspect='equal',
                         extent=(xp.min(), xp.max(), yp.max(), yp.min()),
                         norm=LogNorm(vmin=vmin, vmax=vmax))

        if plot_skysub:
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
                         extent=(xp.min(), xp.max(), yp.max(), yp.min()),
                         norm=LogNorm(vmin=vmin, vmax=vmax))

            skysub_i_neg = np.ma.masked_array(skysub_i, mask=~(skysub_i < 0))
            vmin, vmax = np.nanmedian(-skysub_i_neg), np.nanmax(-skysub_i_neg)
            if vmin >= vmax:
                vmin = 0.1*vmax
            #print(vmin, vmax)
            ax[2].imshow(-skysub_i_neg, cmap=cmap_skysub, aspect='equal',
                         extent=(xp.min(), xp.max(), yp.max(), yp.min()),
                         norm=LogNorm(vmin=vmin, vmax=vmax))

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
                         norm=LogNorm(vmin=vmin, vmax=vmax))

            if beams_i.shape[0] != 1:
                ax[4].imshow(beams_i[1], cmap=cmap, aspect='equal',
                             norm=LogNorm(vmin=vmin, vmax=vmax))

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

def plot_open_AO_loop(max_counts, bounds_ord_beam, bounds_ext_beam):
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

################################################################################
# Reading and writing files
################################################################################

def write_FITS_file(path_to_file, cube, header=None):
    '''
    Write a FITS file.

    Input
    -----
    path_to_file : str
        Filename.
    cube : 3D-array
    header : astropy header
    '''

    # Save the cube with a header
    fits.writeto(path_to_file, cube.astype(np.float32), header,
                 output_verify='silentfix', overwrite=True)

    return path_to_file

def read_FITS_as_cube(path_to_file):
    '''
    Read a FITS file as a cube, reshape if necessary.

    Input
    -----
    path_to_file : str
        Filename.

    Output
    ------
    data : 3D-array
    header : astropy header
    '''

    # Read the data from the file
    data, header = fits.getdata(path_to_file, header=True)

    data = data.astype(np.float32)

    if data.ndim == 2:
        data = np.expand_dims(data, axis=0)

    elif data.ndim==3:

        if len(data) > 1:
            # Remove the last, mean frame if cube consists multiple frames
            data = data[:-1]

        if data.shape[1]!=data.shape[2]:
            # Remove the top 2 rows of pixels
            data = data[:,:-2]

    return data, header

def read_from_FITS_header(path_to_file, key):
    '''
    Read a keyword from a FITS header.

    Input
    -----
    path_to_file : str
        Filename.
    key : str
        Keyword to read.

    Output
    ------
    val
        Keyword value.
    '''

    return fits.getval(path_to_file, key)

def write_config_file(path_config_file):

    config_file = open(path_config_file, 'w+')

    pre_processing_options = ['[Pre-processing options]',
                              'run_pre_processing        = True',
                              'remove_data_products      = True',
                              'split_observing_blocks    = True',
                              'y_pixel_range             = [0,1024]']
    config_file.write('{}\n\n{}\n{}\n{}\n{}\n\n\n'.format(
                      *pre_processing_options))

    sky_subtraction_options = ['[Sky-subtraction]',
                               'sky_subtraction_method     = dithering-offset',
                               'sky_subtraction_min_offset = 100',
                               'remove_horizontal_stripes  = False']
    config_file.write('{}\n\n{}\n{}\n{}\n\n\n'.format(*sky_subtraction_options))

    centering_options = ['[Centering]',
                         'centering_method = single-Moffat',
                         'tied_offset = False']
    config_file.write('{}\n\n{}\n{}\n\n\n'.format(*centering_options))

    PDI_options = ['[PDI options]',
                   'size_to_crop         = [121,121]',
                   'r_inner_IPS          = [0,3,6,9,12]',
                   'r_outer_IPS          = [3,6,9,12,15]',
                   'crosstalk_correction = False',
                   'minimise_U_phi       = False',
                   'r_crosstalk          = [7,17]']
    config_file.write('{}\n\n{}\n{}\n{}\n{}\n{}\n{}\n\n\n'.format(*PDI_options))

    # Use the file-path to guess the object's name
    object_name = path_config_file.parts[-2].replace('_', ' ')
    object_information = ['[Object information]',
                          f'object_name      = {object_name}',
                          'disk_pos_angle   = 0.0',
                          'disk_inclination = 0.0']
    config_file.write('{}\n\n{}\n{}\n{}'.format(*object_information))

def read_config_file(path_config_file):
    '''
    Read the settings from the configuration file.

    Input
    -----
    path_config_file : str
        Path to the .config file

    Output
    ------
    run_pre_processing : bool
    remove_data_products : bool
    split_observing_blocks : bool
    y_pixel_range : list
    sky_subtraction_method : str
    sky_subtraction_min_offset : float
    remove_horizontal_stripes : bool
    centering_method : str
    tied_offset : bool
    size_to_crop : list
    r_inner_IPS : list
    r_outer_IPS : list
    crosstalk_correction : bool
    minimise_U_phi : bool
    r_crosstalk : list
    object_name : str
    disk_pos_angle : float
    disk_inclination : float
    '''

    # Check if configuration file exists
    if not path_config_file.is_file():
        string = f'Configuration file {str(path_config_file.resolve())} does not exist. Do you want to create a default configuration file? (y/n)'
        user_input = input('\n'+textwrap.fill(string, width=80)+'\n')

        if user_input == 'y':
            # Create a default configuration file
            write_config_file(path_config_file)

            print_and_log(f'\nA default configuration file {str(path_config_file.resolve())} is created, please confirm that the input parameters are appropriate for your reduction.\n')

        else:
            print_and_log(f'\nConfiguration file {str(path_config_file.resolve())} does not exist and a default file is not created.\n')

        # Exit out of the reduction
        sys.exit()

    # Read the config file with a configparser object
    config      = configparser.ConfigParser()
    config_read = config.read(path_config_file)


    run_pre_processing     = literal_eval(config.get('Pre-processing options',
                                                     'run_pre_processing'))
    remove_data_products   = literal_eval(config.get('Pre-processing options',
                                                     'remove_data_products'))
    split_observing_blocks = literal_eval(config.get('Pre-processing options',
                                                     'split_observing_blocks'))
    y_pixel_range          = literal_eval(config.get('Pre-processing options',
                                                     'y_pixel_range'))


    sky_subtraction_method     = config.get('Sky-subtraction',
                                            'sky_subtraction_method')
    sky_subtraction_min_offset = float(config.get('Sky-subtraction',
                                            'sky_subtraction_min_offset'))
    remove_horizontal_stripes  = literal_eval(config.get('Sky-subtraction',
                                            'remove_horizontal_stripes'))
    # Confirm that the sky-subtraction method is valid
    if sky_subtraction_method not in ['dithering-offset', 'box-median']:
        raise ValueError('\nsky_subtraction_method should be \'dithering-offset\' or \'box-median\'')


    centering_method = config.get('Centering', 'centering_method')
    # Confirm that the centering method is valid
    if centering_method not in ['single-Moffat', 'double-Moffat', 'maximum']:
        raise ValueError('\ncentering_method should be \'single-Moffat\', \'double-Moffat\' or \'maximum\'')

    if centering_method=='single-Moffat' or centering_method=='double-Moffat':
        tied_offset = literal_eval(config.get('Centering', 'tied_offset'))
    else:
        tied_offset = False


    size_to_crop = literal_eval(config.get('PDI options', 'size_to_crop'))
    # Change the size to crop to odd lengths
    size_to_crop_is_even = (np.mod(size_to_crop, 2) == 0)
    old_size_to_crop     = np.copy(size_to_crop)
    size_to_crop = np.array(size_to_crop) + 1*size_to_crop_is_even
    size_to_crop = list(size_to_crop)

    if size_to_crop_is_even.any():
        print_and_log(f'\nsize_to_crop = {old_size_to_crop} had axes of even lengths, automatically changed to {size_to_crop}\n')

    r_inner_IPS          = literal_eval(config.get('PDI options',
                                                   'r_inner_IPS'))
    r_outer_IPS          = literal_eval(config.get('PDI options',
                                                   'r_outer_IPS'))
    crosstalk_correction = literal_eval(config.get('PDI options',
                                                   'crosstalk_correction'))
    minimise_U_phi       = literal_eval(config.get('PDI options',
                                                   'minimise_U_phi'))
    if crosstalk_correction or minimise_U_phi:
        r_crosstalk = literal_eval(config.get('PDI options', 'r_crosstalk'))
    else:
        r_crosstalk = [None,None]


    object_name      = config.get('Object information', 'object_name')
    disk_pos_angle   = float(config.get('Object information', 'disk_pos_angle'))
    disk_inclination = float(config.get('Object information',
                                        'disk_inclination'))

    # Query the SIMBAD archive
    query_result = Simbad.query_object(object_name)
    try:
        # Read RA and Dec
        RA  = query_result['RA']
        DEC = query_result['DEC']
    except TypeError:
        raise ValueError(f'\nobject_name \'{object_name}\' not found in the SIMBAD archive')

    print_and_log('')
    print_and_log('--- Configuration file parameters:')
    with open(path_config_file, 'r') as file:
        for line in file.readlines():
            print_and_log(line.replace('\n',''))

    return run_pre_processing, \
           remove_data_products, \
           split_observing_blocks, \
           y_pixel_range, \
           sky_subtraction_method, \
           sky_subtraction_min_offset, \
           remove_horizontal_stripes, \
           centering_method, \
           tied_offset, \
           size_to_crop, \
           r_inner_IPS, \
           r_outer_IPS, \
           crosstalk_correction, \
           minimise_U_phi, \
           r_crosstalk, \
           object_name, \
           disk_pos_angle, \
           disk_inclination

def print_and_log(string, new_file=False, pad=None, pad_character='-'):
    '''
    Print a string and record to a log-file.

    Input
    -----
    string : str
        String to print and log.
    new_file : bool
        If True, create a new file using the
        path_log_file global variable.
    '''

    # Create the log-file if it does not exist
    if new_file:
        open(path_log_file, 'w+')

    if pad is not None:
        # Pad the string with '-'
        string = string.ljust(pad, pad_character)

    # Wrap to a maximum of 80 characters
    string = textwrap.fill(string, width=80, replace_whitespace=False,
                           drop_whitespace=True, break_long_words=False)

    # Log to file and print in terminal
    print(string, file=open(path_log_file, 'a'))
    print(string)

################################################################################
# Re-centering functions
################################################################################

def fit_initial_guess(im, xp, yp, Wollaston_used, camera_used, filter_used):
    '''
    Retrieve an initial guess avoiding bad pixels with a minimum filter.

    Input
    -----
    im : 2D-array
        Image.
    Wollaston_used : bool
        If True, Wollaston was used, else wiregrid was used.
    camera_used: str
        Camera that was used ('S13','S27','L27','S54','L54').
    filter_used : str
        filter that was used.

    Output
    ------
    (x0_1, y0_1) : tuple
        Coordinates of one of the beams.
    (x0_2, y0_2) : tuple
        Coordinates of one of the beams.
    '''

    # Crop the image to avoid initial guesses near the edges
    im = im[:, 20:-20]
    yp = yp[:, 20:-20]
    xp = xp[:, 20:-20]

    if Wollaston_used:
        # Estimate two beam locations with the maxima of a filtered image
        # Set the separation based on the used camera
        Moffat_y_offset = Wollaston_beam_separation(camera_used)

        # Apply a long, horizontal filter to approximate polarimetric mask
        box = np.ones((1,20))
        mask_approx = ndimage.median_filter(im, footprint=box, mode='constant')

        # Apply the minimum filter
        box = np.zeros((Moffat_y_offset,3))
        box[:+3,:] = 1
        box[-3:,:] = 1

        filtered_im = ndimage.minimum_filter(im - mask_approx,
                                             footprint=box, mode='constant')

        # x and y values of each pixel
        y_idx, x_idx = np.unravel_index(np.argmax(
                                            filtered_im[Moffat_y_offset//2:
                                            -Moffat_y_offset//2]),
                                        filtered_im.shape)
        x0_center = xp[Moffat_y_offset//2:-Moffat_y_offset//2][y_idx, x_idx]
        y0_center = yp[Moffat_y_offset//2:-Moffat_y_offset//2][y_idx, x_idx]

        x0_1, x0_2 = x0_center, x0_center
        y0_1 = y0_center + Moffat_y_offset/2
        y0_2 = y0_center - Moffat_y_offset/2

    else:
        # Estimate one beam location with a filtered image
        # Apply the minimum filter
        box = np.ones((5,5))
        filtered_im = ndimage.median_filter(im, footprint=box, mode='constant')

        # x and y coordinates of one beam
        y_idx, x_idx = np.unravel_index(np.argmax(filtered_im),
                                        filtered_im.shape)
        x0_1, y0_1 = xp[y_idx, x_idx], yp[y_idx, x_idx]

        # Set second beam to NaN-values
        x0_2, y0_2 = np.nan, np.nan

    return (x0_1, y0_1), (x0_2, y0_2)

def fit_maximum(Wollaston_used, camera_used, filter_used):
    '''
    Use pixels with maximum counts as PSF centers.

    Input
    -----
    Wollaston_used : bool
        If True, Wollaston was used, else wiregrid was used.
    camera_used : str
        Camera that was used ('S13','S27','L27','S54','L54').
    filter_used : str
        Filter that was used.

    Output
    ------
    PSF : list of 3D-arrays
        Coordinates of the PSF centers for each cube. Each cube's
        3D-array has shape (cube-frames, ordinary/extra-ordinary beam, x/y).
    '''

    min_cube = 0
    #if filter_used in ['L_prime', 'NB_3.74']:
    if len(path_reduced_files_selected) > 1:
        for i, file in enumerate(path_reduced_files_selected):
            cube = fits.getdata(file).astype(np.float32)

            if i == 0:
                min_cube = np.mean(cube, axis=0, keepdims=True)
            else:
                min_cube = np.min(np.array([min_cube[0],
                                  np.mean(cube, axis=0)]),
                                  axis=0, keepdims=True)

    PSF = []
    for i, file in enumerate(tqdm(path_reduced_files_selected, \
                                  bar_format=pbar_format)):

        # Read the data
        cube = fits.getdata(file).astype(np.float32)

        # x and y values of each pixel
        yp, xp = np.mgrid[0:cube.shape[1], 0:cube.shape[2]]

        # Find one of the PSFs from the first frame in the cube
        (x0_1, y0_1), (x0_2, y0_2) = fit_initial_guess((cube-min_cube)[0],
                                                       xp, yp, Wollaston_used,
                                                       camera_used, filter_used)

        # Set the background to 0
        cube -= min_cube
        cube -= np.nanmedian(cube, axis=(1,2), keepdims=True)

        PSF.append(np.ones((len(cube),2,2))*np.nan)

        # Fit each frame in the cube
        for j, im in enumerate(cube):

            box = np.ones((3,3))
            filtered_im = ndimage.median_filter(im, footprint=box)

            if Wollaston_used:
                x0, y0 = [x0_1, x0_2], [y0_1, y0_2]
            else:
                x0, y0 = [x0_1], [y0_1]

            # Fit the (extra)-ordinary beams
            for k, x0_k, y0_k in zip(range(len(x0)), x0, y0):

                # Cut around the beam
                x_min = max([xp.min(), x0_k - 30])
                x_max = min([xp.max(), x0_k + 30])
                y_min = max([yp.min(), y0_k - 30])
                y_max = min([yp.max(), y0_k + 30])

                # Find the indices to retain the 2D array shape
                x_min_idx = np.argwhere(xp[0,:]>=x_min)[0,0]
                x_max_idx = np.argwhere(xp[0,:]<=x_max)[-1,0]
                y_min_idx = np.argwhere(yp[:,0]>=y_min)[0,0]
                y_max_idx = np.argwhere(yp[:,0]<=y_max)[-1,0]

                filtered_im_k = filtered_im[y_min_idx:y_max_idx,
                                            x_min_idx:x_max_idx]
                yp_k = yp[y_min_idx:y_max_idx, x_min_idx:x_max_idx]
                xp_k = xp[y_min_idx:y_max_idx, x_min_idx:x_max_idx]

                # Indices where maximum exists
                y_idx, x_idx = np.unravel_index(np.argmax(filtered_im_k),
                                                filtered_im_k.shape)

                # Record the maximum
                PSF[-1][j,k,:] = xp_k[y_idx,x_idx], yp_k[y_idx,x_idx]

            # Sort the PSF locations so that ordinary beam comes first
            idx_argsort = np.argsort(PSF[-1][j,:,1])
            PSF[-1][j,:,:] = PSF[-1][j,idx_argsort,:]

            """
            plt.imshow(filtered_im, extent=(xp.min(), xp.max(), yp.max(), yp.min()), norm=LogNorm())
            plt.scatter(PSF[-1][j,:,0], PSF[-1][j,:,1], c='r')
            plt.show()
            """

    return PSF

def fit_double_Moffat(im, xp, yp, x0_ext, y0_ext, camera_used,
                      filter_used, tied_offset):
    '''
    Fit two Moffat functions with the same center to retrieve the PSF center.
    The flat, saturated top of the PSF is simulated by subtracting a Moffat
    function from another.

    The ordinary and extra-ordinary beams are fitted simultaneously.

    Input
    -----
    im : 2D-array
        Image.
    xp, yp : 2D-arrays
        x-, y-coordinates of pixels.
    x0_ext, y0_ext : floats
        Initial guess of extra-ordinary (bottom) beam.
    camera_used : str
        Camera that was used ('S13','S27','L27','S54','L54').
    filter_used : str
        Filter that was used.
    tied_offset : bool
        Use a fixed beam-separation.

    Output
    ------
    [[x_ext,y_ext], [x_ord,y_ord]] : array
        x-, y-coordinate of fitted beam-centers.
    '''

    # Set the separation based on the used camera
    Moffat_y_offset = Wollaston_beam_separation(camera_used, filter_used)

    # Functions to constrain the model
    def tie_to_x_0_0(model):
        return model.x_0_0
    def tie_to_y_0_0(model):
        return model.y_0_0

    def tie_to_y_0_0_offset(model):
        return model.y_0_0 + Moffat_y_offset

    def tie_to_amp_0(model):
        return model.amplitude_0
    def tie_to_gamma_0(model):
        return model.gamma_0
    def tie_to_amp_1(model):
        return model.amplitude_1
    def tie_to_gamma_1(model):
        return model.gamma_1

    def tie_to_x_0_2(model):
        return model.x_0_2
    def tie_to_y_0_2(model):
        return model.y_0_2

    # Crop the image to fit -------------------------------
    x_min = max([xp.min(), x0_ext-40])
    x_max = min([xp.max(), x0_ext+40])
    y_min_0 = max([yp.min(), y0_ext-40])
    y_max_0 = min([yp.max(), y0_ext+40])
    y_min_2 = max([yp.min(), y0_ext+Moffat_y_offset-40])
    y_max_2 = min([yp.max(), y0_ext+Moffat_y_offset+40])

    # Mask the arrays
    mask_xp = (xp >= x_min) & (xp <= x_max)
    mask_yp = np.ma.mask_or(((yp >= y_min_0) & (yp <= y_max_0)),
                            ((yp >= y_min_2) & (yp <= y_max_2))
                            )
    mask_im = mask_xp & mask_yp

    im = im[mask_im]
    xp = xp[mask_im]
    yp = yp[mask_im]

    # Bounds of the Moffat center -------------------------
    x_min = max([xp.min(), x0_ext-10])
    x_max = min([xp.max(), x0_ext+10])
    y_min_0 = max([yp.min(), y0_ext-10])
    y_max_0 = min([yp.max(), y0_ext+10])
    y_min_2 = max([yp.min(), y0_ext+Moffat_y_offset-10])
    y_max_2 = min([yp.max(), y0_ext+Moffat_y_offset+10])

    # Extra-ordinary beam
    Moffat_0 = models.Moffat2D(x_0=x0_ext, y_0=y0_ext, amplitude=20000, gamma=5,
                               bounds={'x_0':(x_min, x_max),
                                       'y_0':(y_min_0, y_max_0)},
                               fixed={'alpha':True, 'gamma':True}
                              )
    Moffat_1 = models.Moffat2D(x_0=x0_ext, y_0=y0_ext, amplitude=1000, gamma=5,
                               bounds={'x_0':(x_min, x_max),
                                       'y_0':(y_min_0, y_max_0)},
                               tied={'x_0':tie_to_x_0_0, 'y_0':tie_to_y_0_0},
                               fixed={'alpha':True, 'gamma':True}
                              )
    # Double Moffat for the extra-ordinary beam
    double_Moffat_ext = Moffat_0 - Moffat_1


    # Ordinary beam
    if tied_offset:
        # Tie the x and y coordinates, amplitudes, and gamma
        tied_2 = {'x_0':tie_to_x_0_0, 'y_0':tie_to_y_0_0_offset,
                  'amplitude':tie_to_amp_0, 'gamma':tie_to_gamma_0}
        tied_3 = {'x_0':tie_to_x_0_0, 'y_0':tie_to_y_0_0_offset,
                  'amplitude':tie_to_amp_1, 'gamma':tie_to_gamma_1}

        Moffat_2 = models.Moffat2D(x_0=x0_ext, y_0=y0_ext+Moffat_y_offset,
                                   amplitude=20000, gamma=5,
                                   bounds={'x_0':(x_min, x_max),
                                           'y_0':(y_min_2, y_max_2)},
                                   tied=tied_2,
                                   fixed={'alpha':True, 'gamma':True}
                                  )
    else:
        Moffat_2 = models.Moffat2D(x_0=x0_ext, y_0=y0_ext+Moffat_y_offset,
                                   amplitude=20000, gamma=5,
                                   bounds={'x_0':(x_min, x_max),
                                           'y_0':(y_min_2, y_max_2)},
                                   fixed={'alpha':True, 'gamma':True}
                                  )
        # Tie the (x,y)-coordinates
        tied_3 = {'x_0':tie_to_x_0_2, 'y_0':tie_to_y_0_2}

    Moffat_3 = models.Moffat2D(x_0=x0_ext, y_0=y0_ext+Moffat_y_offset,
                               amplitude=1000, gamma=5,
                               bounds={'x_0':(x_min, x_max),
                                       'y_0':(y_min_2, y_max_2)},
                               tied=tied_3, fixed={'alpha':True, 'gamma':True}
                              )
    # Double Moffat for the ordinary beam
    double_Moffat_ord = Moffat_2 - Moffat_3


    # Combine the two beams into a single model
    complete_model = double_Moffat_ext + double_Moffat_ord

    # Fit the model to the image
    LevMar_fitter = fitting.LevMarLSQFitter()
    fitted = LevMar_fitter(complete_model, xp, yp, im, maxiter=10000, acc=1e-12)

    # x- and y-coordinates of the beam centers
    x_ext, y_ext = fitted[0].parameters[1:3]
    x_ord, y_ord = fitted[2].parameters[1:3]

    return np.array([[x_ext, y_ext],
                     [x_ord, y_ord]])

def fit_single_Moffat(im, xp, yp, x0_ord, y0_ord, x0_ext, y0_ext,
                      camera_used, filter_used, tied_offset):
    '''
    Fit a single Moffat function to retrieve a PSF center.

    Input
    -----
    im : 2D-array
        Image.
    xp, yp : 2D-arrays
        x-, y-coordinates of pixels.
    x0_ord, y0_ord : floats
        Initial guess of ordinary (top) beam.
    x0_ext, y0_ext : floats
        Initial guess of extra-ordinary (bottom) beam.
    camera_used : str
        Camera that was used ('S13','S27','L27','S54','L54').
    filter_used : str
        Filter that was used.
    tied_offset : bool
        Use a fixed beam-separation.

    Output
    ------
    [[x_ext,y_ext], [x_ord,y_ord]] : array
        x-, y-coordinate of fitted beam-centers.
    '''

    plt.imshow(im, extent=(xp.min(), xp.max(), yp.max(), yp.min()),
               norm=LogNorm())

    # Set the separation based on the used camera
    Moffat_y_offset = Wollaston_beam_separation(camera_used, filter_used)

    # Functions to constrain the model
    def tie_to_x_0_0(model):
        return model.x_0_0
    def tie_to_y_0_0_offset(model):
        return model.y_0_0 + Moffat_y_offset

    def tie_to_amp_0(model):
        return model.amplitude_0
    def tie_to_gamma_0(model):
        return model.gamma_0

    # Crop the image to fit -------------------------------
    x_min = max([xp.min(), x0_ord-40])
    x_max = min([xp.max(), x0_ord+40])
    y_min_0 = max([yp.min(), y0_ord-Moffat_y_offset-40])
    y_max_0 = min([yp.max(), y0_ord-Moffat_y_offset+40])
    y_min_1 = max([yp.min(), y0_ord-40])
    y_max_1 = min([yp.max(), y0_ord+40])

    # Mask the arrays
    mask_xp = (xp >= x_min) & (xp <= x_max)
    mask_yp = np.ma.mask_or(((yp >= y_min_0) & (yp <= y_max_0)),
                            ((yp >= y_min_1) & (yp <= y_max_1))
                            )
    mask_im = mask_xp & mask_yp

    im = im[mask_im]
    xp = xp[mask_im]
    yp = yp[mask_im]

    # Bounds of the Moffat center -------------------------
    x_min = max([xp.min(), x0_ord-10])
    x_max = min([xp.max(), x0_ord+10])
    y_min_0 = max([yp.min(), y0_ord-Moffat_y_offset-10])
    y_max_0 = min([yp.max(), y0_ord-Moffat_y_offset+10])
    y_min_1 = max([yp.min(), y0_ord-10])
    y_max_1 = min([yp.max(), y0_ord+10])

    # Extra-ordinary beam
    single_Moffat_ext = models.Moffat2D(x_0=x0_ext, y_0=y0_ext,
                                        amplitude=15000, gamma=3, alpha=1,
                                        bounds={'x_0':(x_min, x_max),
                                                'y_0':(y_min_0, y_max_0),
                                                'amplitude':(1,40000),
                                                'gamma':(0.1,30),
                                                'alpha':(0,10)}
                                       )
    """
    single_Moffat_ext = models.Moffat2D(x_0=x0_ext, y_0=y0_ext,
                                        amplitude=20000, gamma=5,
                                        bounds={'x_0':(x_min, x_max),
                                                'y_0':(y_min_0, y_max_0)},
                                        fixed={'alpha':True, 'gamma':True}
                                       )
    """
    # Ordinary beam
    if tied_offset:
        # Tie the x and y coordinates, amplitudes, and gamma
        tied_1 = {'x_0':tie_to_x_0_0, 'y_0':tie_to_y_0_0_offset,
                  'amplitude':tie_to_amp_0, 'gamma':tie_to_gamma_0}
        """
        single_Moffat_ord = models.Moffat2D(x_0=x0_ext,
                                            y_0=y0_ext+Moffat_y_offset,
                                            amplitude=20000, gamma=5,
                                            bounds={'x_0':(x_min, x_max),
                                                    'y_0':(y_min_1, y_max_1)},
                                            tied=tied_1,
                                            fixed={'alpha':True,
                                                   'gamma':True}
                                           )
        """
        single_Moffat_ord = models.Moffat2D(x_0=x0_ext,
                                            y_0=y0_ext+Moffat_y_offset,
                                            amplitude=15000, gamma=3, alpha=1,
                                            bounds={'x_0':(x_min, x_max),
                                                    'y_0':(y_min_1, y_max_1),
                                                    'amplitude':(1,40000),
                                                    'gamma':(0.1,30),
                                                    'alpha':(0,10)},
                                            tied=tied_1
                                           )
    else:
        """
        single_Moffat_ord = models.Moffat2D(x_0=x0_ext,
                                            y_0=y0_ext+Moffat_y_offset,
                                            amplitude=20000, gamma=5,
                                            bounds={'x_0':(x_min, x_max),
                                                    'y_0':(y_min_1, y_max_1)},
                                            fixed={'alpha':True, 'gamma':True}
                                           )
        """
        single_Moffat_ord = models.Moffat2D(x_0=x0_ext,
                                            y_0=y0_ext+Moffat_y_offset,
                                            amplitude=15000, gamma=3, alpha=1,
                                            bounds={'x_0':(x_min, x_max),
                                                    'y_0':(y_min_1, y_max_1),
                                                    'amplitude':(1,40000),
                                                    'gamma':(0.1,30),
                                                    'alpha':(0,10)},
                                           )

    # Combine the two beams into a single model
    complete_model = single_Moffat_ord
    if not np.isnan(x0_ext) and not np.isnan(y0_ext):
        complete_model += single_Moffat_ext

    # Fit the model to the image
    LevMar_fitter = fitting.LevMarLSQFitter()
    fitted = LevMar_fitter(complete_model, xp, yp, im,
                           maxiter=10000, acc=1e-12)

    # x- and y-coordinates of the beam centers
    if np.isnan(x0_ext) and np.isnan(y0_ext):
        x_ord, y_ord = fitted.parameters[1:3]
        x_ext, y_ext = x0_ext, y0_ext
    else:
        x_ord, y_ord = fitted[0].parameters[1:3]
        x_ext, y_ext = fitted[1].parameters[1:3]

    plt.scatter(x_ord, y_ord, c='r')
    plt.scatter(x_ext, y_ext, c='r')
    #plt.show()
    plt.close()

    return np.array([[x_ext, y_ext],
                     [x_ord, y_ord]])

def fit_beam_centers_Moffat(method, Wollaston_used, Wollaston_45,
                            camera_used, filter_used, tied_offset):
    '''
    Fit the beam-centers using 1 or 2 Moffat functions.

    Input
    -----
    method : str
        Method to use ('single-Moffat', 'double-Moffat').
    Wollaston_used : bool
        If True, Wollaston was used, else wiregrid was used.
    Wollaston_45 : bool
        If True, Wollaston_45 was used, else Wollaston_00 was used.
    camera_used : str
        Camera that was used ('S13','S27','L27','S54','L54').
    filter_used : str
        Filter that was used.
    tied_offset : bool
        Use a fixed beam-separation.

    Output
    ------
    Moffat_PSF : list of 3D-arrays
        Coordinates of the Moffat PSF centers for each cube. Each cube's
        3D-array has shape (cube-frames, ordinary/extra-ordinary beam, x/y).
    '''

    min_cube = 0
    #if filter_used in ['L_prime', 'NB_3.74']:
    for i, file in enumerate(path_reduced_files_selected):
        cube = fits.getdata(file).astype(np.float32)
        cube[np.isnan(cube)] = 0

        if i == 0:
            min_cube = np.mean(cube, axis=0, keepdims=True)
        else:
            min_cube = np.min(np.array([min_cube[0], np.mean(cube, axis=0)]),
                              axis=0, keepdims=True)

    Moffat_PSF = []
    for i, file in enumerate(tqdm(path_reduced_files_selected, \
                                  bar_format=pbar_format)):

        # Read the data
        cube = fits.getdata(file).astype(np.float32)
        cube[np.isnan(cube)] = 0

        # x and y values of each pixel
        yp, xp = np.mgrid[0:cube.shape[1], 0:cube.shape[2]]

        if not Wollaston_45:
            mask = (yp > -1.5*xp+2430)[None,:]
            mask = np.repeat(mask, cube.shape[0], axis=0)
            cube[mask] = np.nanmean(cube[~mask])

        # Find the PSFs from the first frame in the cube
        (x0_ord, y0_ord), \
        (x0_ext, y0_ext) \
        = fit_initial_guess((cube-min_cube)[0], xp, yp, Wollaston_used,
                            camera_used, filter_used)

        # Set the background to 0
        cube -= np.nanmedian(cube, axis=(1,2), keepdims=True)

        # Fit each frame in the cube
        Moffat_PSF.append(np.ones((len(cube),2,2))*np.nan)
        for j, im in enumerate(cube):

            # Fit the (extra)-ordinary beams
            if method=='double-Moffat':
                # Fit a double-Moffat function to find the location of the beam
                Moffat_PSF[-1][j,:,:] = fit_double_Moffat(im, xp, yp,
                                                          x0_ext, y0_ext,
                                                          camera_used,
                                                          filter_used,
                                                          tied_offset)
            elif method=='single-Moffat':
                # Fit a single-Moffat
                Moffat_PSF[-1][j,:,:] = fit_single_Moffat(im, xp, yp,
                                                          x0_ord, y0_ord,
                                                          x0_ext, y0_ext,
                                                          camera_used,
                                                          filter_used,
                                                          tied_offset)

            # Sort the PSF locations so that ordinary beam comes first
            idx_argsort = np.argsort(Moffat_PSF[-1][j,:,1])
            Moffat_PSF[-1][j,:,:] = Moffat_PSF[-1][j,idx_argsort,:]

    return Moffat_PSF

def fit_beam_centers(method, Wollaston_used, Wollaston_45,
                     camera_used, filter_used, tied_offset):
    '''
    Fit the beam-centers using a specified method.

    Input
    -----
    method : str
        Method to fit the beam-centers.
    Wollaston_used : bool
        If True, Wollaston was used, else wiregrid was used.
    Wollaston_45 : bool
        If True, Wollaston_45 was used, else Wollaston_00 was used.
    camera_used : str
        Camera that was used ('S13','S27','L27','S54','L54').
    filter_used : str
        Filter that was used.
    tied_offset : bool
        Use a fixed beam-separation.

    Output
    ------
    beam_centers : list of 3D-arrays
        Coordinates of the beam-centers for each cube. Each cube's 3D-array
        has shape (cube-frames, ordinary/extra-ordinary beam, x/y).
    '''

    print_and_log(f'--- Fitting the beam centers using method \'{method}\'')

    if method=='single-Moffat' or method=='double-Moffat':
        # Fit a 2D Moffat function
        beam_centers = fit_beam_centers_Moffat(method, Wollaston_used,
                                               Wollaston_45, camera_used,
                                               filter_used, tied_offset)

    elif method=='maximum':
        # Find 2 maxima in the images
        beam_centers = fit_maximum(Wollaston_used, camera_used, filter_used)

    return beam_centers

def center_beams(beam_centers, size_to_crop, Wollaston_used, Wollaston_45):
    '''
    Re-center the beams and crop the images.

    Input
    -----
    beam_centers : list of 3D-arrays
        Coordinates of the beam-centers for each cube. Each cube's 3D-array
        has shape (cube-frames, ordinary/extra-ordinary beam, x/y).
    size_to_crop : list
        [height, width] to crop.
    Wollaston_used : bool
        If True, Wollaston was used, else wiregrid was used.
    Wollaston_45 : bool
        If True, Wollaston_45 was used, else Wollaston_00 was used.
    '''

    print_and_log('--- Centering the beams')

    global path_beams_files_selected

    path_beams_files_selected = []
    beams = []
    for i, file in enumerate(tqdm(path_skysub_files_selected, \
                                  bar_format=pbar_format)):

        # Read the data
        cube, header = fits.getdata(file, header=True)
        cube = cube.astype(np.float32)
        cube[np.isnan(cube)] = 0

        # x and y values of each pixel
        yp, xp = np.mgrid[0:cube.shape[1], 0:cube.shape[2]]

        ord_beam_i, ext_beam_i = [], []
        for j, im in enumerate(cube):

            """
            fig, ax = plt.subplots(figsize=(25,10), ncols=3)
            ax[0].imshow(im)
            ax[0].scatter(beam_centers[i][j,0,0], beam_centers[i][j,0,1])

            ax[0].set(xlim=(beam_centers[i][j,0,0] - size_to_crop[1]/2,
                            beam_centers[i][j,0,0] + size_to_crop[1]/2),
                      ylim=(beam_centers[i][j,0,1] - size_to_crop[0]/2,
                            beam_centers[i][j,0,1] + size_to_crop[0]/2)
                      )
            """

            # Padding the image for large cropping sizes
            pad_width = ((0, 0), (im.shape[1], im.shape[1]))
            im = np.pad(im, pad_width, constant_values=0.0)

            # Mask of values outside the image
            im_mask = (im == 0)

            # Shift the ordinary beam to the center of the image
            y_shift = im.shape[0]/2 - (beam_centers[i][j,0,1] - yp.min())
            x_shift = im.shape[1]/2 - (beam_centers[i][j,0,0] + pad_width[1][0]
                                       - xp.min())

            ord_beam_ij = ndimage.shift(im, [y_shift, x_shift], order=3)
            # Replace values outside of image with 0
            ord_beam_ij_mask = ndimage.shift(im_mask, [y_shift, x_shift],
                                             order=0, cval=0.0)
            ord_beam_ij[ord_beam_ij_mask] = 0

            # Shift the extra-ordinary beam to the center of the image
            y_shift = im.shape[0]/2 - (beam_centers[i][j,1,1] - yp.min())
            x_shift = im.shape[1]/2 - (beam_centers[i][j,1,0] + pad_width[1][0]
                                       - xp.min())
            ext_beam_ij = ndimage.shift(im, [y_shift, x_shift], order=3)
            # Replace values outside of image with 0
            ext_beam_ij_mask = ndimage.shift(im_mask, [y_shift, x_shift],
                                             order=0, cval=0.0)
            ext_beam_ij[ext_beam_ij_mask] = 0

            """
            ax[1].imshow(ord_beam_ij)
            ax[1].scatter(ord_beam_ij.shape[1]/2, ord_beam_ij.shape[0]/2)
            ax[1].set(xlim=(ord_beam_ij.shape[1]/2 - size_to_crop[1]/2,
                            ord_beam_ij.shape[1]/2 + size_to_crop[1]/2),
                      ylim=(ord_beam_ij.shape[0]/2 - size_to_crop[0]/2,
                            ord_beam_ij.shape[0]/2 + size_to_crop[0]/2))
            """

            # Indices to crop between
            y_idx_low  = int(ord_beam_ij.shape[0]/2 - size_to_crop[0]/2 + 1/2)
            y_idx_high = int(ord_beam_ij.shape[0]/2 + size_to_crop[0]/2 + 1/2)
            x_idx_low  = int(ord_beam_ij.shape[1]/2 - size_to_crop[1]/2 + 1/2)
            x_idx_high = int(ord_beam_ij.shape[1]/2 + size_to_crop[1]/2 + 1/2)

            # Crop the images
            ord_beam_ij = ord_beam_ij[y_idx_low:y_idx_high,
                                      x_idx_low:x_idx_high]
            ext_beam_ij = ext_beam_ij[y_idx_low:y_idx_high,
                                      x_idx_low:x_idx_high]

            """
            ax[2].imshow(ord_beam_ij[::-1,:])
            plt.show()
            """

            ord_beam_i.append(ord_beam_ij)
            ext_beam_i.append(ext_beam_ij)

        # Median-combine the (extra)-ordinary beam for one file
        ord_beam_i = np.nanmedian(np.array(ord_beam_i), axis=0)
        ext_beam_i = np.nanmedian(np.array(ext_beam_i), axis=0)

        if Wollaston_45:

            # Masks of values outside the image
            ord_beam_i_mask = (ord_beam_i==0)
            ext_beam_i_mask = (ext_beam_i==0)

            # Rotate the images back to their initial orientation
            ord_beam_i = ndimage.rotate(ord_beam_i, angle=45,
                                        reshape=True, cval=np.nan)
            ext_beam_i = ndimage.rotate(ext_beam_i, angle=45,
                                        reshape=True, cval=np.nan)

            # Rotate masks to replace values outside the image with nan
            ord_beam_i_mask = ndimage.rotate(ord_beam_i_mask, angle=45,
                                             reshape=True, order=0, cval=1)
            ext_beam_i_mask = ndimage.rotate(ext_beam_i_mask, angle=45,
                                             reshape=True, order=0, cval=1)

            ord_beam_i[ord_beam_i_mask] = np.nan
            ext_beam_i[ext_beam_i_mask] = np.nan

        if Wollaston_used:
            # Concatenate the ordinary and extra-ordinary beam and save as cube
            beams_i = np.concatenate((ord_beam_i[None,:,:],
                                      ext_beam_i[None,:,:]),
                                     axis=0)
            beams.append(beams_i)
        else:
            # Save the only beam as a cube
            beams_i = ord_beam_i[None,:,:]
            beams.append(beams_i)

        # Save the calibrated data as a FITS file
        file_beams = Path(str(file).replace('_skysub.fits', '_beams.fits'))
        write_FITS_file(file_beams, beams_i, header=header)
        path_beams_files_selected.append(file_beams)

    path_beams_files_selected = np.sort(path_beams_files_selected)

    # Perform sigma-clipping on all beams
    open_AO_loop(np.array(beams), sigma_max=3)

################################################################################
# Sky-subtraction
################################################################################

def sky_background_fit(im, offset, next_offset, min_offset, y_ord_ext,
                       remove_horizontal_stripes):
    '''
    Fit each row of pixels to approximate a background gradient.

    Input
    -----
    im : 2D-array
        Image.
    offset : float
        Offset of the current dithering-position.
    next_offset : float
        Offset of the next dithering-position.
    min_offset : float
        Minimum offset between the dithering-positions.
    y_ord_ext : 1D-array
        y-coordinates of the ordinary and extra-ordinary beam.
    remove_horizontal_stripes : bool
        If True, remove the horizontal stripes found in some observations.

    Output
    ------
    background_model : 2D-array
        Model of the background gradient.
    '''
    # Retrieve pixel coordinates
    yp, xp = np.mgrid[0:im.shape[0], 0:im.shape[1]]

    # Ignore certain rows in the fit
    y_min = (y_ord_ext.min() - np.diff(y_ord_ext)*3/5)[0]
    y_max = (y_ord_ext.max() + np.diff(y_ord_ext)*3/5)[0]

    # Masks to not fit to the (offset) beams
    mask_x_1 = np.ma.mask_or((xp < offset - 1.25*min_offset),
                             (xp > offset + 1.25*min_offset))
    mask_x_2 = np.ma.mask_or((xp < next_offset - 1.25*min_offset),
                             (xp > next_offset + 1.25*min_offset))
    mask_x = mask_x_1 & mask_x_2

    if mask_x.sum() == 0:
        # Mask covers entire frame
        mask_x_1 = np.ma.mask_or((xp < offset - 0.5*min_offset),
                                 (xp > offset + 0.5*min_offset))
        mask_x_2 = np.ma.mask_or((xp < next_offset - 0.5*min_offset),
                                 (xp > next_offset + 0.5*min_offset))
        mask_x = mask_x_1 & mask_x_2

    # Masks to not fit additional sources
    _, low, high = sigma_clip(np.ma.masked_array(im, mask=~mask_x),
                              sigma=2.5, maxiters=5, cenfunc='median',
                              return_bounds=True, axis=1)
    mask_sources = (im > low[:,None]) & (im < high[:,None])

    mask_total = mask_x & mask_sources

    # Fit the data using astropy.modeling
    p_init = models.Linear1D()
    fit_p = fitting.LevMarLSQFitter()

    if remove_horizontal_stripes:
        # Linear fit to each row
        n_rows_combined = 1
    else:
        # Linear fit to every 5th row
        n_rows_combined = 5

    background_model = np.zeros(im.shape)
    for i in range((n_rows_combined-1), len(im), n_rows_combined):

        i_min = i - (n_rows_combined-1)
        i_max = i

        if (i_max < y_max) and (i_min > y_min):

            # Fit a linear function to each row
            if remove_horizontal_stripes:
                xp_masked = xp[i_max][mask_total[i_max]]
                im_masked_median = im[i_max][mask_total[i_max]]

            else:
                # Mask of the current rows
                mask_rows = mask_total & (yp >= i_min) & (yp <= i_max)

                # x-coordinates of row
                mask_rows_flatten = (mask_rows.sum(axis=0) != 0)
                xp_masked = np.ma.masked_array(xp[i_max],
                                               mask=~mask_rows_flatten)
                xp_masked = np.ma.compressed(xp_masked)

                # Median-combine along vertical axis
                im_masked = np.ma.masked_array(im, mask=~mask_rows)
                im_masked_median = np.nanmedian(im_masked[i_min:i_max+1],
                                                axis=0)
                im_masked_median = im_masked_median[np.isfinite(im_masked_median)]

            if np.isfinite(im_masked_median).any():
                p = fit_p(p_init, xp_masked, im_masked_median)

                # Store the horizontal representation
                for j in range(i_min, i_max+1):
                    background_model[j] = p(xp[j])

    if not remove_horizontal_stripes:
        # Smoothen the horizontal polynomial models
        background_model = ndimage.gaussian_filter(background_model, sigma=5)

    return background_model

def sky_subtraction_box_median(files, beam_centers, min_offset,
                               remove_horizontal_stripes):
    '''
    Sky-subtraction using two rectangles at an offset.

    Input
    -----
    files : list
        Filenames to sky-subtract
    beam_centers : list of 3D-arrays
        Coordinates of the beam-centers for each cube. Each cube's 3D-array
        has shape (cube-frames, ordinary/extra-ordinary beam, x/y).
    min_offset : float
        Minimum offset from the beam-centers.
    remove_horizontal_stripes : bool
        If True, remove the horizontal stripes found in some observations.
    '''

    if len(files) == 1:
        pbar_disable = True
    else:
        pbar_disable = False

    with tqdm(total=len(files), bar_format=pbar_format, \
              disable=pbar_disable) as pbar:

        for i, file in enumerate(files):

            if len(files) != 1:
                pbar.update(1)

            # Read the data
            cube, header = fits.getdata(file, header=True)
            cube = cube.astype(np.float32)

            # Pixel coordinates
            yp, xp = np.mgrid[0:cube.shape[1], 0:cube.shape[2]]

            # Retrieve the location of the beams
            x = np.nanmedian(beam_centers[i][:,:,0])

            # Mask for pixel sufficiently offset
            mask_x = np.ma.mask_or((xp[0] > int(x+min_offset)),
                                   (xp[0] < int(x-min_offset)))

            # Take the median along the x-axis
            sky = np.nanmedian(cube[:,:,mask_x], axis=2, keepdims=True)

            # Subtract the sky
            cube -= sky

            # Remove any leftover background signal with a linear fit
            for j in range(len(cube)):
                x_j = np.nanmean(beam_centers[i][j,:,0])
                y_j = beam_centers[i][j,:,1]
                cube[j] -= sky_background_fit(cube[j], x_j, x_j, min_offset,
                                              y_j, remove_horizontal_stripes)

            # Add filename
            file_skysub = Path(str(file).replace('_reduced.fits',
                                                 '_skysub.fits'))
            path_skysub_files_selected.append(file_skysub)

            # Save the sky-subtracted image to a FITS file
            write_FITS_file(file_skysub, cube, header=header)

def sky_subtraction_dithering(beam_centers, min_offset, HWP_used,
                              Wollaston_used, remove_horizontal_stripes):
    '''
    Sky-subtraction using the next dithering-offset
    with the same Stokes parameter.

    Input
    -----
    beam_centers : list of 3D-arrays
        Coordinates of the beam-centers for each cube. Each cube's 3D-array
        has shape (cube-frames, ordinary/extra-ordinary beam, x/y).
    min_offset : float
        Minimum offset from the beam-centers.
    HWP_used : bool
        If True, HWP was used, else position angle was changed.
    Wollaston_used : bool
        If True, Wollaston was used, else wiregrid was used.
    remove_horizontal_stripes : bool
        If True, remove the horizontal stripes found in some observations.
    '''

    StokesPara = assign_Stokes_parameters(path_reduced_files_selected,
                                          HWP_used, Wollaston_used)
    for i in range(len(StokesPara)):
        StokesPara[i] = StokesPara[i].replace('-', '')
        StokesPara[i] = StokesPara[i].replace('+', '')

    offsets = []
    for i in range(len(path_reduced_files_selected)):
        offsets.append(np.nanmean(beam_centers[i][:,:,0], axis=(0,1)))
    offsets = np.array(offsets)

    idx_offsets = np.arange(len(offsets))

    for i, file in enumerate(tqdm(path_reduced_files_selected, \
                                  bar_format=pbar_format)):

        # Read the data
        cube, header = fits.getdata(file, header=True)
        cube = cube.astype(np.float32)

        # Mask for the cubes with the same Stokes parameter
        mask_StokesPara = (StokesPara == StokesPara[i])

        # New minimal offset which can be decreased
        new_min_offset = min_offset

        # Mask of cubes with sufficient offsets
        mask_sufficient_offset = (np.abs(offsets-offsets[i]) >= new_min_offset)

        while not np.any(mask_sufficient_offset):
            # Decrease the offset and try again
            new_min_offset -= 5
            mask_sufficient_offset = (np.abs(offsets-offsets[i]) >= new_min_offset)

            if (new_min_offset < 60):
                sky_subtraction_box_median([file], np.array([beam_centers[i]]),
                                         min_offset, remove_horizontal_stripes)
                break

        if (new_min_offset < 60):
            continue

        mask_next_offsets = (idx_offsets > i)
        mask_prev_offsets = (idx_offsets < i)

        # Next dithering position with same Stokes parameter
        mask_next_same = mask_sufficient_offset * mask_next_offsets * \
                         mask_StokesPara
        # Previous dithering position with same Stokes parameter
        mask_prev_same = mask_sufficient_offset * mask_prev_offsets * \
                         mask_StokesPara

        # Next dithering position with different Stokes parameter
        mask_next_diff = mask_sufficient_offset * mask_next_offsets
        # Previous dithering position with different Stokes parameter
        mask_prev_diff = mask_sufficient_offset * mask_prev_offsets

        if np.any(mask_next_same):
            # First offset with same Stokes parameter
            idx_next_offset = idx_offsets[mask_next_same][0]

        elif np.any(mask_prev_same):
            # Previous offset with same Stokes parameter
            idx_next_offset = idx_offsets[mask_prev_same][-1]

        elif np.any(mask_next_diff):
            # First offset with different Stokes parameter
            idx_next_offset = idx_offsets[mask_next_diff][0]

        elif np.any(mask_prev_diff):
            # Previous offset with different Stokes parameter
            idx_next_offset = idx_offsets[mask_prev_diff][-1]


        # Read the data of the next dithering offset
        file_next_offset = path_reduced_files_selected[idx_next_offset]
        cube_next_offset = fits.getdata(file_next_offset).astype(np.float32)

        # Subtract the next dithering position from the original
        cube -= np.nanmedian(cube_next_offset, axis=0, keepdims=True)

        # Remove any leftover background signal with linear fits
        for j in range(len(cube)):
            y_j = beam_centers[i][j,:,1]
            cube[j] -= sky_background_fit(cube[j], offsets[i],
                                          offsets[idx_next_offset],
                                          min_offset, y_j,
                                          remove_horizontal_stripes)

        # Add filename
        file_skysub = Path(str(file).replace('_reduced.fits', '_skysub.fits'))
        path_skysub_files_selected.append(file_skysub)

        # Save the sky-subtracted image to a FITS file
        write_FITS_file(file_skysub, cube, header=header)

    if (new_min_offset < 35):
        print_and_log('    Sky-subtraction not possible with method \'dithering-offset\', used method \'box-median\'')

    if (new_min_offset!=min_offset):
        print_and_log(f'    sky_subtraction_min_offset too high, reduced to {new_min_offset} pixels')

def sky_subtraction(method, min_offset, beam_centers, HWP_used,
                    Wollaston_used, remove_horizontal_stripes):
    '''
    Sky-subtraction using a specified method.

    Input
    -----
    method : str
        Method for sky-subtraction.
    min_offset : float
        Minimum offset from the beam-centers.
    beam_centers : list of 3D-arrays
        Coordinates of the beam-centers for each cube. Each cube's 3D-array
        has shape (cube-frames, ordinary/extra-ordinary beam, x/y).
    HWP_used : bool
        If True, HWP was used, else position angle was changed.
    Wollaston_used : bool
        If True, Wollaston was used, else wiregrid was used.
    remove_horizontal_stripes : bool
        If True, remove the horizontal stripes found in some observations.
    '''

    print_and_log(f'--- Sky-subtraction using method: \'{method}\'')

    global path_skysub_files_selected
    path_skysub_files_selected = []

    if method=='dithering-offset':
        sky_subtraction_dithering(beam_centers, min_offset, HWP_used,
                                  Wollaston_used, remove_horizontal_stripes)

    elif method=='box-median':
        sky_subtraction_box_median(path_reduced_files_selected, beam_centers,
                                   min_offset, remove_horizontal_stripes)

    path_skysub_files_selected = np.sort(path_skysub_files_selected)

################################################################################
# Pre-processing functions
################################################################################

def remove_bad_pixels(cube, combined_bpm):
    '''
    Remove bad pixels with a median filter.

    Input
    -----
    cube : 3D-array
        Cube to remove bad pixels from.
    combined_bpm : 2D-array
        Bad-pixel mask.
    '''

    # Take median from 5x5 box of pixels, excluding the central pixel
    box = np.ones((1, 5, 5))
    box[:,2,2] = 0.

    # Apply the median filter
    filtered_cube = ndimage.median_filter(cube, footprint=box)

    # Apply the bad-pixel mask so that flagged pixels are
    # replaced with the median value
    nan_mask = np.ones(cube.shape)
    nan_mask[np.isnan(cube)] = 0.

    replace_mask = (nan_mask*combined_bpm)
    return filtered_cube + replace_mask*(cube - filtered_cube)

def open_AO_loop(beams, sigma_max=5):
    '''
    Determine if any open AO-loop images exist and save
    their filenames in a text-file.

    Input
    -----
    beams : 4D-array
        Array with shape (observations, beams, y, x)
    sigma_max : int
        Clipping limit for sigma-clip.
    '''

    # Find the maximum in the (extra)-ordinary beam
    max_counts = np.nanmax(beams[:,:,beams.shape[2]//2-20:beams.shape[2]//2+20,
                                 beams.shape[3]//2-20:beams.shape[3]//2+20],
                           axis=(2,3))

    # Perform an iterative sigma-clipping on the maximum counts
    filtered_max_counts_ord_beam, \
    low_ord_beam, \
    high_ord_beam \
    = sigma_clip(max_counts[:,0], sigma_max, maxiters=1,
                 masked=True, return_bounds=True)
    # Lower and upper bounds of the sigma-clip
    bounds_ord_beam    = (low_ord_beam, high_ord_beam)
    bounds_ext_beam    = (-np.inf, np.inf)
    mask_clip_ord_beam = np.ma.getmask(filtered_max_counts_ord_beam)
    mask_clip_beams    = mask_clip_ord_beam

    # Mask for lower limit
    #mask_low_ord_beam = (filtered_max_counts_ord_beam < \
    #                     0.1*np.nanmedian(filtered_max_counts_ord_beam))
    mask_low_ord_beam = (filtered_max_counts_ord_beam < 0)
    mask_low_beams    = mask_low_ord_beam

    if max_counts.shape[1] != 1:
        # Multiple beams, Wollaston was used
        filtered_max_counts_ext_beam, \
        low_ext_beam, \
        high_ext_beam \
        = sigma_clip(max_counts[:,1], sigma_max, maxiters=2,
                     masked=True, return_bounds=True)

        bounds_ext_beam    = (low_ext_beam, high_ext_beam)
        mask_clip_ext_beam = np.ma.getmask(filtered_max_counts_ext_beam)
        mask_clip_beams    = (mask_clip_ord_beam + mask_clip_ext_beam != 0)

        #mask_low_ext_beam = (filtered_max_counts_ext_beam < \
        #                     0.1*np.nanmedian(filtered_max_counts_ext_beam))
        mask_low_ext_beam = (filtered_max_counts_ext_beam < 0)
        mask_low_beams    = np.ma.mask_or(mask_low_ord_beam, mask_low_ext_beam)

    # Combine the lower limit mask and the sigma-clipping mask
    mask_open_loop_beams = np.ma.mask_or(mask_clip_beams, mask_low_beams)

    # Store the open AO loop files in a file
    path_open_loop_files = Path(path_reduced_files_selected[0].parent,
                                'open_loop_files.txt')
    open_loop_files = open(path_open_loop_files, 'w+')
    open_loop_files.close()

    if np.any(mask_open_loop_beams):
        print_and_log('    Possible open AO-loop image(s) found:')
        for file in path_beams_files_selected[mask_open_loop_beams]:

            print_and_log(f'    {file.name}')

            # Save the absolute paths of open AO-loop images
            open_loop_files = open(path_open_loop_files, 'a')
            open_loop_files.write(str(file.resolve())+'\n')
            open_loop_files.close()

    plot_open_AO_loop(max_counts, bounds_ord_beam, bounds_ext_beam)

#####################################
# Calibration FLAT, BPM and DARK
#####################################

def prepare_calib_files(path_SCIENCE_dir, path_FLAT_dir, path_master_BPM_dir,
                        path_DARK_dir):
    '''
    Prepare the master FLATs, DARKs and BPMs from the supplied raw
    calibration files.

    Input
    -----
    path_SCIENCE_dir : str
        Path to raw SCIENCE files.
    path_FLAT_dir : str
        Path to raw FLAT files.
    path_master_BPM_dir : str
        Path to store master BPMs. If None, directory is created in the
        same parent directory of path_FLAT_dir.
    path_DARK_dir : str
        Path to raw DARK files.

    Output
    ------
    path_master_FLAT_dir : str
        Path where master FLATs are stored.
    path_master_BPM_dir : str
        Path where master BPMs are stored.
    path_master_DARK_dir : str
        Path where master DARKs are stored.
    '''

    # Create the path of the output directory
    global path_output_dir
    path_output_dir = Path(path_SCIENCE_dir, 'pipeline_output')

    if not path_output_dir.is_dir():
        path_output_dir.mkdir()

    # Create the log file
    global path_log_file
    path_log_file = Path(path_output_dir, 'log.txt')

    print_and_log('')
    print_and_log('=== Welcome to PIPPIN (PdI PiPelIne for Naco data) ===',
                  new_file=True, pad=80, pad_character='=')
    print_and_log('')
    print_and_log('')
    print_and_log(f'Created output directory {str(path_output_dir.resolve())}')
    print_and_log('')
    print_and_log(f'Created log file {str(path_log_file)}')
    print_and_log('')

    print_and_log('')
    print_and_log('=== Creating the master calibration files ===',
                  pad=80, pad_character='=')
    print_and_log('')

    # Check if the directories exist and are not empty -------------------------
    if not path_FLAT_dir.is_dir():
        raise IOError(f'\nThe FLAT directory {str(path_FLAT_dir.resolve())} does not exist.')

    path_FLAT_files = sorted(Path(path_FLAT_dir).glob('*.fits'))
    # Ensure that FLATs and DARKs have the same image shapes
    path_FLAT_files = [file_i for file_i in path_FLAT_files
                       if (read_from_FITS_header(file_i, 'NAXIS')==2)
                       and (read_from_FITS_header(file_i, 'ESO DET WIN NX') == 1024)
                       and (read_from_FITS_header(file_i, 'ESO DET WIN NY') == 1024)
                      ]
    path_FLAT_files = np.array(path_FLAT_files)

    if len(path_FLAT_files) == 0:
        raise IOError(f'\nThe FLAT directory {str(path_FLAT_dir.resolve())} does not contain FITS-files. Please ensure that any FITS-files are uncompressed.')


    if not path_DARK_dir.is_dir():
        raise IOError(f'\nThe DARK directory {str(path_DARK_dir.resolve())} does not exist.')

    path_DARK_files = sorted(Path(path_DARK_dir).glob('*.fits'))

    # Ensure that FLATs and DARKs have the same image shapes
    path_DARK_files = [file_i for file_i in path_DARK_files
                       if (read_from_FITS_header(file_i, 'ESO DET WIN NX') == 1024)
                       and (read_from_FITS_header(file_i, 'ESO DET WIN NY') == 1024)
                      ]
    path_DARK_files = np.array(path_DARK_files)

    if len(path_DARK_files) == 0:
        raise IOError(f'\nThe DARK directory {str(path_DARK_dir.resolve())} does not contain FITS-files. Please ensure that any FITS-files are uncompressed.')

    # Create directories for master calib files --------------------------------
    if path_master_BPM_dir is None:
        path_master_BPM_dir = Path(str(path_FLAT_dir).replace('FLAT',
                                                              'master_BPM'))

    if not path_master_BPM_dir.is_dir():
        path_master_BPM_dir.mkdir()

    path_master_FLAT_dir = Path(path_FLAT_dir, 'master_FLAT')
    if not path_master_FLAT_dir.is_dir():
        path_master_FLAT_dir.mkdir()

    path_master_DARK_dir = Path(path_DARK_dir, 'master_DARK')
    if not path_master_DARK_dir.is_dir():
        path_master_DARK_dir.mkdir()

    # FLATs --------------------------------------------------------------------
    FLAT_cameras, FLAT_filters, FLAT_expTimes, FLAT_lampStatus, FLAT_OPTI1_ID \
    = [], [], [], [], []
    for i, path_FLAT_file_i in enumerate(path_FLAT_files):

        # Read the detector keyword
        camera_i = read_from_FITS_header(path_FLAT_file_i, 'ESO INS OPTI7 ID')

        # Read the filter keyword(s)
        filter_i = read_from_FITS_header(path_FLAT_file_i, 'ESO INS OPTI6 NAME')
        if filter_i == 'empty':
            filter_i = read_from_FITS_header(path_FLAT_file_i,
                                             'ESO INS OPTI5 NAME')

        # Read the exposure time
        expTime_i = read_from_FITS_header(path_FLAT_file_i, 'EXPTIME')

        # Read the lamp status (on/off). True if on, False if off.
        lampStatus_i = (read_from_FITS_header(path_FLAT_file_i,
                                              'ESO INS LAMP2 SET') != 0)

        # Read whether the Wollaston prism was inserted
        OPTI1_ID_i = read_from_FITS_header(path_FLAT_file_i, 'ESO INS OPTI1 ID')

        FLAT_cameras.append(camera_i)
        FLAT_filters.append(filter_i)
        FLAT_expTimes.append(expTime_i)
        FLAT_lampStatus.append(lampStatus_i)
        FLAT_OPTI1_ID.append(OPTI1_ID_i)

    # Create arrays for easier masking
    FLAT_cameras    = np.array(FLAT_cameras)
    FLAT_filters    = np.array(FLAT_filters)
    FLAT_expTimes   = np.array(FLAT_expTimes)
    FLAT_lampStatus = np.array(FLAT_lampStatus)
    FLAT_OPTI1_ID   = np.array(FLAT_OPTI1_ID)

    # Determine the unique configurations
    FLAT_configs = np.vstack((FLAT_cameras, FLAT_filters, FLAT_expTimes,
                              FLAT_lampStatus, FLAT_OPTI1_ID)).T
    FLAT_configs_unique = np.unique(FLAT_configs, axis=0)

    print_and_log('--- Unique FLAT types:')
    print_and_log('Camera'.ljust(10) + 'Filter'.ljust(10) + \
                  'Exp. Time (s)'.ljust(15) + 'Lamp status'.ljust(15) + \
                  'OPTI1 ID'.ljust(15))
    for (camera_i, filter_i, expTime_i, lampStatus_i, OPTI1_ID_i) in \
        FLAT_configs_unique:
        lampStatus_i = 'On' if lampStatus_i=='True' else 'Off'

        print_and_log(camera_i.ljust(10) + filter_i.ljust(10) + \
                      expTime_i.ljust(15) + lampStatus_i.ljust(15) + \
                      OPTI1_ID_i.ljust(15))

    # DARKs --------------------------------------------------------------------
    DARK_cameras, DARK_expTimes = [], []
    for i, path_DARK_file_i in enumerate(path_DARK_files):

        # Read the detector keyword
        camera_i = read_from_FITS_header(path_DARK_file_i, 'ESO INS OPTI7 ID')

        # Read the exposure time
        expTime_i = read_from_FITS_header(path_DARK_file_i, 'EXPTIME')

        DARK_cameras.append(camera_i)
        DARK_expTimes.append(expTime_i)

    # Create arrays for easier masking
    DARK_cameras  = np.array(DARK_cameras)
    DARK_expTimes = np.array(DARK_expTimes)

    # Determine the unique configurations
    DARK_configs = np.vstack((DARK_cameras, DARK_expTimes)).T
    DARK_configs_unique = np.unique(DARK_configs, axis=0)

    print_and_log('')
    print_and_log('--- Unique DARK types:')
    print_and_log('Camera'.ljust(10) + 'Exp. Time (s)'.ljust(15))
    for (camera_i, expTime_i) in DARK_configs_unique:
        print_and_log(camera_i.ljust(10) + expTime_i.ljust(15))

    # Loop over the unique DARK configurations
    master_DARKs, master_DARKs_header = [], []
    for DARK_config_i in DARK_configs_unique:

        # Read the files with the same configuration
        mask_i = np.prod((DARK_configs == DARK_config_i),
                         axis=1, dtype=np.bool)
        DARKs_i = []
        for path_DARK_file_j in path_DARK_files[mask_i]:

            # Read the file
            DARK_j, DARK_header_j = fits.getdata(path_DARK_file_j, header=True)
            if DARK_j.ndim > 2:
                DARK_j = np.nanmedian(DARK_j.astype(np.float32), axis=0)

            DARKs_i.append(DARK_j)

        # Median combine over the DARKs
        DARKs_i = np.array(DARKs_i)
        master_DARK_i = np.nanmedian(DARKs_i, axis=0)

        master_DARKs.append(master_DARK_i)
        master_DARKs_header.append(DARK_header_j)

    master_DARKs = np.array(master_DARKs)

    # Combine the FLATs and DARKs ----------------------------------------------
    master_FLATs_lamp_on, master_FLATs_lamp_off = [], []
    master_FLATs_lamp_on_header = []
    for FLAT_config_i in FLAT_configs_unique:

        camera_i, filter_i, expTime_i, lampStatus_i, OPTI1_ID_i = FLAT_config_i

        # Master DARK should use the same camera
        master_DARK_mask_i = (DARK_configs_unique[:,0]==camera_i)

        if master_DARK_mask_i.sum() == 0:
            raise IOError(f'\nMaster FLAT has no corresponding master DARK with detector \'{camera_i}\'.')

        expTime_ratio = np.float32(expTime_i) / \
                        np.float32(DARK_configs_unique[:,1][master_DARK_mask_i])
        master_DARK_i = master_DARKs[master_DARK_mask_i] * \
                        expTime_ratio[:,None,None]
        master_DARK_i = np.nanmedian(master_DARK_i, axis=0)

        # Read the files with the same configuration
        FLAT_mask_i = np.prod((FLAT_configs == FLAT_config_i),
                              axis=1, dtype=np.bool)
        FLATs_i = []
        for path_FLAT_file_j in path_FLAT_files[FLAT_mask_i]:

            # Read the file
            FLAT_j, FLAT_header_j = fits.getdata(path_FLAT_file_j, header=True)
            FLAT_j = FLAT_j.astype(np.float32)

            # DARK-subtract the FLAT
            FLAT_j -= master_DARK_i

            FLATs_i.append(FLAT_j)

        # Median combine over the FLATs
        FLATs_i = np.array(FLATs_i)
        master_FLAT_i = np.nanmedian(FLATs_i, axis=0)

        if lampStatus_i == 'True':
            master_FLATs_lamp_on.append(master_FLAT_i)
            master_FLATs_lamp_on_header.append(FLAT_header_j)
        elif lampStatus_i == 'False':
            master_FLATs_lamp_off.append(master_FLAT_i)

    master_FLATs_lamp_on  = np.array(master_FLATs_lamp_on)
    master_FLATs_lamp_off = np.array(master_FLATs_lamp_off)

    # In case there are fewer/more lamp-on than lamp-off observations
    len_lamp_on  = len(master_FLATs_lamp_on)
    len_lamp_off = len(master_FLATs_lamp_off)
    master_FLATs_lamp_on  = master_FLATs_lamp_on[:min([len_lamp_on,
                                                       len_lamp_off])]
    master_FLATs_lamp_off = master_FLATs_lamp_off[:min([len_lamp_on,
                                                        len_lamp_off])]

    # Bad-pixel masks from non-linear pixel responses --------------------------
    print_and_log('')
    print_and_log('--- Creating bad-pixel masks from (non)-linear pixel response between lamp-off and lamp-on FLATs')
    master_BPMs = np.ones(master_FLATs_lamp_on.shape)

    for i in range(len(master_BPMs)):

        # Factor by which the pixels should have increased
        linear_factor_i = np.nanmedian(master_FLATs_lamp_on[i]) / \
                          np.nanmedian(master_FLATs_lamp_off[i])

        # Factor by which pixels actually increased
        mask_i = (master_FLATs_lamp_off[i] != 0.)
        actual_factor_i         = np.ones(master_FLATs_lamp_on[i].shape)
        actual_factor_i[mask_i] = master_FLATs_lamp_on[i][mask_i] / \
                                  master_FLATs_lamp_off[i][mask_i]

        # Sigma-clip the actual increases and take the standard deviation
        clipped_actual_factor_i = sigmaclip(actual_factor_i, low=5, high=5)[0]
        std_i = np.nanstd(clipped_actual_factor_i)

        # Flag pixels that deviate by more than 1.5 sigma from a linear response
        master_BPMs[i, np.abs(actual_factor_i-linear_factor_i)>1.5*std_i] = 0

    # Normalise the FLATs
    master_FLATs_lamp_on /= np.nanmedian(master_FLATs_lamp_on,
                                         axis=(1,2), keepdims=True)

    # Save the FLATs, BPMs and DARKs -------------------------------------------

    print_and_log('')
    print_and_log(f'Saving master FLATs in directory {path_master_FLAT_dir}')
    print_and_log(f'Saving master BPMs in directory {path_master_BPM_dir}')
    print_and_log(f'Saving master DARKs in directory {path_master_DARK_dir}')
    print_and_log('')
    print_and_log('')


    for i in range(len(master_FLATs_lamp_on)):

        camera_i, filter_i, _, _, OPTI1_ID_i \
        = FLAT_configs_unique[FLAT_configs_unique[:,3]=='True'][i]

        if OPTI1_ID_i in ['FLM_13', 'FLM_27', 'FLM_54']:
            OPTI1_ID_i = '_FLM'
        elif OPTI1_ID_i == 'Wollaston_45':
            OPTI1_ID_i = '_Wollaston_45'
        else:
            OPTI1_ID_i = ''

        # Save the FLAT
        path_FLAT_file_i = f'master_FLAT_{camera_i}_{filter_i}{OPTI1_ID_i}_NACO.{master_FLATs_lamp_on_header[i]["DATE-OBS"]}.fits'
        path_FLAT_file_i = Path(path_master_FLAT_dir, path_FLAT_file_i)
        fits.writeto(path_FLAT_file_i,
                     master_FLATs_lamp_on[i].astype(np.float32),
                     output_verify='silentfix', overwrite=True)

        ############
        path_FLAT_file_i = Path('/home/sam/Documents/Master-2/MRP/PIPPIN-NACO/pippin/data/master_FLAT', f'master_FLAT_{camera_i}_{filter_i}{OPTI1_ID_i}_NACO.{master_FLATs_lamp_on_header[i]["DATE-OBS"]}.fits')
        fits.writeto(path_FLAT_file_i,
                     master_FLATs_lamp_on[i].astype(np.float32),
                     output_verify='silentfix', overwrite=True)
        ############

        # Save the BPM
        path_BPM_file_i = f'master_BPM_{camera_i}_{filter_i}{OPTI1_ID_i}_NACO.{master_FLATs_lamp_on_header[i]["DATE-OBS"]}.fits'
        path_BPM_file_i = Path(path_master_BPM_dir, path_BPM_file_i)
        fits.writeto(path_BPM_file_i, master_BPMs[i].astype(np.float32),
                     output_verify='silentfix', overwrite=True)

        ############
        path_BPM_file_i = Path('/home/sam/Documents/Master-2/MRP/PIPPIN-NACO/pippin/data/master_BPM', f'master_BPM_{camera_i}_{filter_i}{OPTI1_ID_i}_NACO.{master_FLATs_lamp_on_header[i]["DATE-OBS"]}.fits')
        fits.writeto(path_BPM_file_i, master_BPMs[i].astype(np.float32),
                     output_verify='silentfix', overwrite=True)
        ############

    for i in range(len(master_DARKs)):

        camera_i, _ = DARK_configs_unique[i]

        # Save the DARK
        path_DARK_file_i = f'master_DARK_{camera_i}_NACO.{master_DARKs_header[i]["DATE-OBS"]}.fits'
        path_DARK_file_i = Path(path_master_DARK_dir, path_DARK_file_i)
        fits.writeto(path_DARK_file_i, master_DARKs[i].astype(np.float32),
                     header=master_DARKs_header[i], output_verify='silentfix',
                     overwrite=True)

        ############
        path_DARK_file_i = Path('/home/sam/Documents/Master-2/MRP/PIPPIN-NACO/pippin/data/master_DARK', f'master_DARK_{camera_i}_NACO.{master_DARKs_header[i]["DATE-OBS"]}.fits')
        fits.writeto(path_DARK_file_i, master_DARKs[i].astype(np.float32),
                     header=master_DARKs_header[i], output_verify='silentfix', overwrite=True)
        ############

    return path_master_FLAT_dir, path_master_BPM_dir, path_master_DARK_dir

def read_master_CALIB(SCIENCE_file, filter_used, path_FLAT_files,
                      path_BPM_files, path_DARK_files, FLAT_pol_mask,
                      Wollaston_45):
    '''
    Read master FLAT, bad-pixel mask and DARK closest to the observing date.

    Input
    -----
    file : str
        Filename of the SCIENCE observation.
    filter : str
        Filter that was used.
    path_FLAT_files : str
        Filenames of FLAT files.
    path_BPM_files : str
        Filenames of BPM files.
    path_DARK_files : str
        Filenames of DARK files.
    FLAT_pol_mask : bool
        If True, read a FLAT with polarimetric mask.
    Wollaston_45 : bool
        If True, Wollaston_45 was used, else Wollaston_00 was used.

    Output
    ------
    master_FLAT : 3D-array
        FLAT closest to the observing date.
    master_BPM : 3D-array
        BPM closest to the observing date.
    master_DARK : 3D-array
        DARK closest to the observing date.
    DARK_expTime : float
        Exposure time of the DARK.
    SCIENCE_expTime : float
        Exposure time of the SCIENCE.
    '''

    # Check for correct filter and if polarimetric mask was used
    new_path_FLAT_files, new_path_BPM_files = [], []
    FLAT_DATE_OBS, BPM_DATE_OBS = [], []
    for i in range(len(path_FLAT_files)):

        if FLAT_pol_mask:
            # Polarimetric mask was used
            replacing_str = filter_used
        elif not FLAT_pol_mask and not Wollaston_45:
            # Mask was not used, add '_FLM_' to FLAT/BPM filenames
            replacing_str = f'{filter_used}_FLM'
        elif not FLAT_pol_mask and Wollaston_45:
            # Rotated Wollaston was used, add '_Wollaston_45_'
            replacing_str = f'{filter_used}_FLM_Wollaston_45'

        if replacing_str in path_FLAT_files[i].name:
            # Select only FLATs with the correct filter
            new_path_FLAT_files.append(path_FLAT_files[i])
            new_path_BPM_files.append(path_BPM_files[i])

            # Store the observing dates of the FLAT/BPM
            FLAT_DATE_OBS_i = str(path_FLAT_files[i]).split('NACO.')[-1].replace('.fits', '')
            FLAT_DATE_OBS.append(Time(FLAT_DATE_OBS_i,
                                      format='isot', scale='utc')
                                )

            BPM_DATE_OBS_i = str(path_BPM_files[i]).split('NACO.')[-1].replace('.fits', '')
            BPM_DATE_OBS.append(Time(BPM_DATE_OBS_i,
                                      format='isot', scale='utc')
                               )

    path_FLAT_files = np.array(new_path_FLAT_files)
    path_BPM_files  = np.array(new_path_BPM_files)

    FLAT_DATE_OBS = np.array(FLAT_DATE_OBS)
    BPM_DATE_OBS  = np.array(BPM_DATE_OBS)

    if path_FLAT_files.size == 0:
        raise IOError('\nNo FLATs found for the observation configuration.')

    # Read the observing dates of the DARK
    DARK_DATE_OBS = []
    for i in range(len(path_DARK_files)):
        DARK_DATE_OBS_i = str(path_DARK_files[i]).split('NACO.')[-1].replace('.fits', '')
        DARK_DATE_OBS.append(Time(DARK_DATE_OBS_i, format='isot', scale='utc'))
    DARK_DATE_OBS = np.array(DARK_DATE_OBS)

    # Read SCIENCE observing date from header
    DATE_OBS = read_from_FITS_header(SCIENCE_file, 'DATE-OBS')
    DATE_OBS = Time(DATE_OBS, format='isot', scale='utc')

    # Difference observing date between SCIENCE and FLAT/BPM
    FLAT_DATE_delta = DATE_OBS - FLAT_DATE_OBS + 0.5
    BPM_DATE_delta  = DATE_OBS - BPM_DATE_OBS + 0.5
    DARK_DATE_delta = DATE_OBS - DARK_DATE_OBS + 0.5

    # Choose the current FLAT and BPM
    if np.any(FLAT_DATE_delta > 0):
        path_FLAT_file = path_FLAT_files[(FLAT_DATE_delta > 0)][-1]
        path_BPM_file  = path_BPM_files[(BPM_DATE_delta > 0)][-1]
    else:
        path_FLAT_file = path_FLAT_files[np.argmin(np.abs(FLAT_DATE_delta))]
        path_BPM_file  = path_BPM_files[np.argmin(np.abs(BPM_DATE_delta))]

    # Choose the current DARK
    path_DARK_file = path_DARK_files[np.argmin(np.abs(DARK_DATE_delta))]

    # Load the master FLAT, BPM and DARK
    master_FLAT = read_FITS_as_cube(path_FLAT_file)[0]
    master_BPM  = read_FITS_as_cube(path_BPM_file)[0]
    master_DARK = read_FITS_as_cube(path_DARK_file)[0]

    # DARK exposure time
    DARK_expTime = read_from_FITS_header(path_DARK_file, 'EXPTIME')
    # SCIENCE exposure time
    SCIENCE_expTime = read_from_FITS_header(SCIENCE_file, 'EXPTIME')

    return master_FLAT, master_BPM, master_DARK, DARK_expTime, SCIENCE_expTime

def reshape_master_CALIB(data, window_shape, window_start):
    '''
    Reshape bad-pixel mask or FLAT to a specified window shape.

    Input
    -----
    data : 3D-array
        BPM or FLAT to reshape.
    window_shape : list
        [height, width] of the window.
    window_start : list
        [y, x] origin pixels of the window.

    Output
    ------
    data : 3D-array
        Reshaped BPM or FLAT.
    '''

    if window_shape != [1024,1024]:
        x_low  = window_start[1]-1
        x_high = window_start[1]-1 + window_shape[1]

        y_low  = window_start[0]-1
        y_high = window_start[0]-1 + window_shape[0]

        # Crop the data to the requested window size
        data = data[:,y_low:y_high, x_low:x_high]

    return data

#####################################
# SCIENCE calibration
#####################################

def read_unique_obsTypes(path_SCIENCE_files, split_observing_blocks, HWP_used,
                         Wollaston_used, Wollaston_45, camera_used):
    '''
    Read the unique observation types. Observations are separated
    by OBS IDs, exposure times, filters.

    Input
    -----
    path_SCIENCE_files : list
        Paths to the SCIENCE FITS-files.
    split_observing_blocks : bool
        If True, split the observing blocks by OBS IDs,
        else combine all observing blocks.
    HWP_used : bool
        If True, HWP was used, else position angle was changed.
    Wollaston_used : bool
        If True, Wollaston was used, else wiregrid was used.
    Wollaston_45 : bool
        If True, Wollaston_45 was used, else Wollaston_00 was used.
    camera_used : str
        Camera that was used ('S13','S27','L27','S54','L54').
    '''

    global obsTypes
    global unique_obsTypes
    global path_output_dirs

    # Read information from the FITS headers
    expTimes = np.array([read_from_FITS_header(x, 'EXPTIME')
                         for x in path_SCIENCE_files])
    OBS_IDs, filters = [], []
    for x in path_SCIENCE_files:
        try:
            OBS_IDs.append(read_from_FITS_header(x, 'ESO OBS ID'))
        except KeyError:
            OBS_IDs.append(0)

        filter_i = read_from_FITS_header(x, 'ESO INS OPTI6 NAME')
        if filter_i == 'empty':
            filter_i = read_from_FITS_header(x, 'ESO INS OPTI5 NAME')
        filters.append(filter_i)

    filters = np.array(filters)
    OBS_IDs = np.array(OBS_IDs)

    # If keyword OBS ID does not exist, assume a single observing block
    if np.any(OBS_IDs==0) or not split_observing_blocks:
        OBS_IDs = np.array([0]*len(OBS_IDs))

    # Unique observation-types (OBS ID, expTime, filter)
    obsTypes = np.vstack((OBS_IDs, expTimes, filters)).T
    unique_obsTypes = np.unique(obsTypes, axis=0)

    print_and_log('')
    print_and_log('')
    print_and_log('--- Unique observation types:')
    print_and_log('HWP'.ljust(10) + 'Wollaston'.ljust(15) + \
                  'Camera'.ljust(10) + 'OBS ID'.ljust(15) + \
                  'Exp. Time (s)'.ljust(15) + 'Filter'.ljust(10))

    if not Wollaston_used:
        Wollaston = 'False'
    elif Wollaston_used and Wollaston_45:
        Wollaston = 'Wollaston_45'
    elif Wollaston_used and not Wollaston_45:
        Wollaston = 'Wollaston_00'

    for unique_obsType_i in unique_obsTypes:
        print_and_log(str(HWP_used).ljust(10) + \
                      Wollaston.ljust(15) + \
                      camera_used.ljust(10) + \
                      unique_obsType_i[0].ljust(15) + \
                      unique_obsType_i[1].ljust(15) + \
                      unique_obsType_i[2].ljust(10)
                      )

    # Create output directories
    path_output_dirs = np.array([Path(path_output_dir, '{}_{}_{}'.format(*x))
                                 for x in unique_obsTypes])
    for x in path_output_dirs:
        if not x.is_dir():
            x.mkdir()

def calibrate_SCIENCE(path_SCIENCE_files, path_FLAT_files, path_BPM_files,
                      path_DARK_files, window_shape, window_start,
                      y_pixel_range, filter_used, FLAT_pol_mask, Wollaston_45):
    '''
    Calibrate the SCIENCE observations by FLAT-normalizing
    and bad-pixel masking.

    Input
    -----
    path_SCIENCE_files : 1D-array
        Paths to the SCIENCE FITS-files.
    path_FLAT_files : 1D-array
        Paths to the FLAT FITS-files.
    path_BPM_files : 1D-array
        Paths to the BPM FITS-files.
    path_DARK_files : 1D-array
        Paths to the DARK FITS-files.
    window_shape : list
        [height, width] of the window.
    window_start : list
        [y, x] origin pixels of the window.
    y_pixel_range : list
        [y_low, y_high] pixel range to cut between.
    filter_used : str
        Filter that was used
    FLAT_pol_mask : bool
        If True, read a FLAT with polarimetric mask.
    Wollaston_45 : bool
        If True, Wollaston_45 was used, else Wollaston_00 was used.
    '''

    global path_reduced_files_selected

    print_and_log('')
    print_and_log('--- Calibrating SCIENCE data')

    path_reduced_files_selected = []

    for i, file in enumerate(tqdm(path_SCIENCE_files, bar_format=pbar_format)):

        # Reduced file names
        reduced_file = Path(path_output_dir_selected,
                            file.name.replace('.fits', '_reduced.fits'))
        path_reduced_files_selected.append(reduced_file)

        # Load the un-calibrated data
        cube, header = read_FITS_as_cube(file)

        # Read the corresponding master FLAT, BPM and DARK
        master_FLAT, \
        master_BPM, \
        master_DARK, \
        DARK_expTime, \
        SCIENCE_expTime \
        = read_master_CALIB(file, filter_used, path_FLAT_files,
                            path_BPM_files, path_DARK_files,
                            FLAT_pol_mask, Wollaston_45)

        # Reshape the master FLAT, BPM and DARK
        if (window_shape == [1024,1024]) or (window_shape == [1026,1024]):
            cube = cube[:,y_pixel_range[0]:y_pixel_range[1]]

            master_FLAT = master_FLAT[:,y_pixel_range[0]:y_pixel_range[1]]
            master_BPM  = master_BPM[:,y_pixel_range[0]:y_pixel_range[1]]
            master_DARK = master_DARK[:,y_pixel_range[0]:y_pixel_range[1]]

        else:
            master_FLAT = reshape_master_CALIB(master_FLAT, cube.shape[1:],
                                               window_start)
            master_BPM  = reshape_master_CALIB(master_BPM, cube.shape[1:],
                                               window_start)
            master_DARK = reshape_master_CALIB(master_DARK, cube.shape[1:],
                                               window_start)

        # DARK-subtract the SCIENCE image
        cube -= master_DARK * SCIENCE_expTime/DARK_expTime

        # Normalize by the DARK-subtracted, normalized FLAT
        master_FLAT[master_FLAT==0] = 1
        cube /= master_FLAT

        # Replace the bad pixels with the median value
        cube = remove_bad_pixels(cube, master_BPM)

        if Wollaston_45:
            # Rotate the cube, because Wollaston_45 was used
            cube = ndimage.rotate(cube, angle=-45, axes=(1,2),
                                  reshape=True, cval=np.nan)

        # Save the calibrated data
        write_FITS_file(reduced_file, cube, header=header)

    path_reduced_files_selected = np.sort(path_reduced_files_selected)

def pre_processing(window_shape, window_start, remove_data_products,
                   y_pixel_range, sky_subtraction_method,
                   sky_subtraction_min_offset, remove_horizontal_stripes,
                   centering_method, tied_offset, size_to_crop, HWP_used,
                   Wollaston_used, Wollaston_45, camera_used, filter_used,
                   path_SCIENCE_files, path_FLAT_files, path_BPM_files,
                   path_DARK_files, FLAT_pol_mask
                   ):
    '''
    Apply the pre-processing functions.

    Input
    -----
    window_shape : list
        [height, width] of the window.
    window_start : list
        [y, x] origin pixels of the window.
    remove_data_products : bool
        If True, remove the '_reduced.fits' and '_skysub.fits'
        files at the end.
    y_pixel_range : list
        [y_low, y_high] pixel range to cut between.
    sky_subtraction_method : str
        Method for sky-subtraction.
    sky_subtraction_min_offset : float
        Minimum sky-subtraction offset from the beam-centers.
    remove_horizontal_stripes : bool
        If True, remove the horizontal stripes found in some observations.
    centering_method : str
        Beam-center fitting method to use.
    tied_offset : bool
        Use a fixed beam-separation.
    size_to_crop : list
        [height, width] to crop.
    HWP_used : bool
        If True, HWP was used, else position angle was changed.
    Wollaston_used : bool
        If True, Wollaston was used, else wiregrid was used.
    Wollaston_45 : bool
        If True, Wollaston_45 was used, else Wollaston_00 was used.
    camera_used : str
        Camera that was used ('S13','S27','L27','S54','L54').
    filter_used : str
        Filter that was used.
    path_SCIENCE_files : 1D-array
        Paths to the SCIENCE FITS-files.
    path_FLAT_files : 1D-array
        Paths to the FLAT FITS-files.
    path_BPM_files : 1D-array
        Paths to the BPM FITS-files.
    path_DARK_files : 1D-array
        Paths to the DARK FITS-files.
    FLAT_pol_mask : bool
        If True, read a FLAT with polarimetric mask.
    '''

    # Calibrate the SCIENCE images
    calibrate_SCIENCE(path_SCIENCE_files, path_FLAT_files,
                      path_BPM_files, path_DARK_files,
                      window_shape, window_start, y_pixel_range,
                      filter_used, FLAT_pol_mask, Wollaston_45)
    print_and_log('--- Plotting the raw and reduced images')
    plot_reduction(plot_reduced=True, plot_skysub=False)

    # Find the beam centers
    beam_centers = fit_beam_centers(centering_method, Wollaston_used,
                                    Wollaston_45, camera_used,
                                    filter_used, tied_offset)

    # Subtract the sky from the images
    sky_subtraction(sky_subtraction_method, sky_subtraction_min_offset,
                    beam_centers, HWP_used, Wollaston_used,
                    remove_horizontal_stripes)

    # Center the beams and save
    center_beams(beam_centers, size_to_crop, Wollaston_used, Wollaston_45)
    print_and_log('--- Plotting the sky-subtracted and cropped images')
    plot_reduction(plot_reduced=True, plot_skysub=True,
                   beam_centers=beam_centers, size_to_crop=size_to_crop,
                   Wollaston_45=Wollaston_45)

    if remove_data_products:
        # Remove the data products
        print_and_log('--- Removing temporary data products')

        for file in path_reduced_files_selected:
            # Remove the reduced files
            file.unlink()

        for file in path_skysub_files_selected:
            # Remove the sky-subtracted files
            file.unlink()

################################################################################
# Polarimetric differential imaging functions
################################################################################

def deprojected_radius(xp, yp, xc, yc, disk_pos_angle, disk_inclination):
    '''
    Compute the de-projected radius from the disk's inclination and
    position angle.

    Input
    -----
    xp, yp : 2D-array
        Coordinates of pixels.
    xc, yc : int
        Central coordinates of the array.
    disk_pos_angle : float
        Disk position angle.
    disk_inclination : float
        Disk inclination.

    Output
    ------
    r_corr : 2D-array
        Array of the de-projected radius from the
        central coordinates.
    '''

    # Convert to radians
    disk_pos_angle   = np.deg2rad(disk_pos_angle)
    disk_inclination = np.deg2rad(disk_inclination)

    # Rotate to correct for the disk's position angle
    xp_corr = (xp-xc)*np.sin(disk_pos_angle) + (yp-yc)*np.cos(disk_pos_angle)
    yp_corr = (yp-yc)*np.sin(disk_pos_angle) - (xp-xc)*np.cos(disk_pos_angle)

    # Correct for the inclined disk
    r_corr = np.sqrt(xp_corr**2 + (yp_corr/np.cos(disk_inclination))**2)

    return r_corr

def rotate_cube(cube, pos_angle, pad=False, rotate_axes=(1,2)):
    '''
    Rotate the cube using a position angle.

    Input
    -----
    cube : 3D-array
        Cube of images.
    pos_angle : float
        Angle to rotate.
    pad : bool
        If True, pad the cube to be square.

    Output
    ------
    rotated_cube : 3D-array
        Rotated cube with pos_angle.
    '''

    if pad:
        if cube.ndim == 2:
            pad_width = (
                         ((cube.shape[rotate_axes[1]] -
                           cube.shape[rotate_axes[0]]) // 2,
                          (cube.shape[rotate_axes[1]] -
                           cube.shape[rotate_axes[0]]) // 2),
                         (0, 0)
                         )
        elif cube.ndim == 3:
            pad_width = ((0, 0),
                         ((cube.shape[rotate_axes[1]] -
                           cube.shape[rotate_axes[0]]) // 2,
                          (cube.shape[rotate_axes[1]] -
                           cube.shape[rotate_axes[0]]) // 2),
                         (0, 0)
                        )
        elif cube.ndim == 4:
            pad_width = ((0, 0),
                         (0, 0),
                         ((cube.shape[rotate_axes[1]] -
                           cube.shape[rotate_axes[0]]) // 2,
                          (cube.shape[rotate_axes[1]] -
                           cube.shape[rotate_axes[0]]) // 2),
                         (0, 0)
                        )
        cube = np.pad(cube, pad_width, constant_values=0.0)

        mask = np.ma.mask_or(np.isnan(cube), (cube==0.0), shrink=False)
        cube[mask] = 0

        # Rotate a cube
        rotated_cube = ndimage.rotate(cube, pos_angle, reshape=False,
                                      axes=rotate_axes, cval=0.0)
        rotated_mask = ndimage.rotate(mask, pos_angle, reshape=False,
                                      axes=rotate_axes, cval=0.0, order=0)
        rotated_cube[rotated_mask] = np.nan

        # Set pixels outside the polarimetric mask to NaN
        pos_angle_rad = np.deg2rad(pos_angle)

        # Line through the image centre
        xc = (cube.shape[rotate_axes[1]]-1)/2
        yc = (cube.shape[rotate_axes[0]]-1)/2
        x = np.linspace(-1024, 1024, 2000) * np.cos(pos_angle_rad) + xc
        y = -np.linspace(-1024, 1024, 2000) * np.sin(pos_angle_rad) + yc

        # Bounding lines
        x1 = -(cube.shape[rotate_axes[0]]//2 - pad_width[1][0]) * \
             np.cos(pos_angle_rad+np.pi/2) + x
        y1 = +(cube.shape[rotate_axes[0]]//2 - pad_width[1][0]) * \
             np.sin(pos_angle_rad+np.pi/2) + y
        x2 = +(cube.shape[rotate_axes[0]]//2 - pad_width[1][0]) * \
             np.cos(pos_angle_rad+np.pi/2) + x
        y2 = -(cube.shape[rotate_axes[0]]//2 - pad_width[1][0]) * \
             np.sin(pos_angle_rad+np.pi/2) + y

        yp, xp = np.mgrid[0:rotated_cube.shape[rotate_axes[0]],
                          0:rotated_cube.shape[rotate_axes[1]]]

        # Interpolate onto the image grid
        xv = xp[0]
        y1_new = np.interp(xv, np.sort(x1), y1[np.argsort(x1)])
        y2_new = np.interp(xv, np.sort(x2), y2[np.argsort(x2)])
        y_min = np.min(np.array([y1_new,y2_new]), axis=0)
        y_max = np.max(np.array([y1_new,y2_new]), axis=0)

        if cube.ndim == 2:
            # Set pixels outside the polarimetric mask to NaN
            rotated_cube[~((yp >= y_min) & (yp <= y_max))] = np.nan
        elif cube.ndim == 3:
            rotated_cube[:,~((yp >= y_min) & (yp <= y_max))] = np.nan

    else:

        mask = np.isnan(cube)
        cube[mask] = 0

        # Rotate a cube
        rotated_cube = ndimage.rotate(cube, pos_angle, reshape=True,
                                      axes=rotate_axes, cval=0.0)
        rotated_mask = ndimage.rotate(mask, pos_angle, reshape=True,
                                      axes=rotate_axes, cval=0.0, order=0)
        rotated_cube[rotated_mask] = np.nan

    rotated_cube[rotated_cube==0] = np.nan
    return rotated_cube

def collapse_beams(beams):
    '''
    Collapse the beams array to remove the NaNs.

    Input
    -----
    beams : 4D-array
        Array of beam-images.

    Output
    ------
    masked_beams : 3D-array
        Collapsed array of beam-images
    mask : 2D-array
        Mask of the non-NaN values.
    '''
    # Locate all the NaNs in the beams and mask them
    mask = ~ np.all(np.isnan(beams), axis=(0,1))
    masked_beams = beams[:,:,mask]

    return masked_beams, mask

def remove_incomplete_HWP_cycles(path_beams_files, StokesPara):
    '''
    Remove any incomplete half-wave plate cycles.

    Input
    -----
    path_beams_files : list
        Filenames of 'beams.fits' files.
    StokesPara : 1D-array
        Stokes parameters ('Q+', 'U+', 'Q-', 'U-').

    Output
    ------
    path_beams_files : list
        Filenames of 'beams.fits' files with incomplete cycles removed.
    HWP_cycle_number : 1D-array
        Number of the corresponding HWP cycle.
    StokesPara : 1D-array
        Stokes parameters with incomplete HWP cycles removed.
    '''

    print_and_log('')
    print_and_log('--- Removing incomplete HWP cycles')

    # Save the HWP cycle number for each observation
    HWP_cycle_number = np.ones(len(StokesPara)) * np.nan

    if len(np.unique(StokesPara))==4:

        # Number of the current HWP cycle
        idx_HWP_cycle = 0

        # Stokes parameters not yet assigned to a HWP cycle
        unassigned_StokesPara     = StokesPara
        idx_unassigned_StokesPara = np.arange(0,len(StokesPara))

        # Continue iterating while there are still complete cycles
        while len(np.unique(unassigned_StokesPara))==4:

            # Only search different Stokes parameters
            mask = (unassigned_StokesPara != unassigned_StokesPara[0])

            # Next Stokes parameters
            next_StokesPara     = unassigned_StokesPara[mask]
            idx_next_StokesPara = idx_unassigned_StokesPara[mask]

            # Unique next Stokes parameters
            unique = np.unique(next_StokesPara)

            if (len(unique)==3):
                # Found a complete HWP cycle
                # Index of each Stokes parameter in the cycle
                idx_0 = idx_unassigned_StokesPara[0]
                idx_1 = idx_next_StokesPara[next_StokesPara == unique[0]][0]
                idx_2 = idx_next_StokesPara[next_StokesPara == unique[1]][0]
                idx_3 = idx_next_StokesPara[next_StokesPara == unique[2]][0]

                # Assign a number to this cycle
                HWP_cycle_number[[idx_0,idx_1,idx_2,idx_3]] = idx_HWP_cycle

            # Update the Stokes parameters yet to be assigned
            unassigned_StokesPara     = StokesPara[np.isnan(HWP_cycle_number)]
            idx_unassigned_StokesPara = np.arange(0,len(StokesPara))[np.isnan(HWP_cycle_number)]

            # Search for the next HWP cycle
            idx_HWP_cycle += 1

    elif ((StokesPara=='Q+').any() or (StokesPara=='Q-').any()) and \
         ((StokesPara=='U+').any() or (StokesPara=='U-').any()):

        # No full HWP cycle, but single measurements of Q and U
        mask_Q = np.ma.mask_or((StokesPara=='Q+'), (StokesPara=='Q-'))
        mask_U = np.ma.mask_or((StokesPara=='U+'), (StokesPara=='U-'))

        HWP_cycle_number[mask_Q] = np.arange(mask_Q.sum())
        HWP_cycle_number[mask_U] = np.arange(mask_U.sum())

        # Unequal number of Q and U measurements, remove observations
        if mask_Q.sum() != mask_U.sum():
            max_cycle = min([mask_Q.sum(), mask_U.sum()])
            HWP_cycle_number[HWP_cycle_number >= max_cycle] = np.nan


    # Remove the incomplete HWP cycles
    mask_to_remove = np.isnan(HWP_cycle_number)
    if np.any(mask_to_remove):
        print_and_log(f'    Removed {mask_to_remove.sum()} files:')
        for file_i, StokesPara_i in zip(path_beams_files[mask_to_remove],
                                        StokesPara[mask_to_remove]):
            print_and_log(f'    {StokesPara_i} {file_i.name}')

    path_beams_files = path_beams_files[~mask_to_remove]
    HWP_cycle_number = HWP_cycle_number[~mask_to_remove]
    StokesPara       = StokesPara[~mask_to_remove]

    # Return the cycle number to keep track of the HWP cycles
    return path_beams_files, HWP_cycle_number, StokesPara

def remove_open_AO_loop(path_beams_files, HWP_cycle_number, StokesPara):
    '''
    Remove any open AO-loop half-wave plate cycles.

    Input
    -----
    path_beams_files : list
        Filenames of 'beams.fits' files.
    HWP_cycle_number : 1D-array
        Number of the corresponding HWP cycle.
    StokesPara : 1D-array
        Stokes parameters ('Q+', 'U+', 'Q-', 'U-').

    Output
    ------
    path_beams_files : list
        Filenames of 'beams.fits' files with open AO-loops removed.
    HWP_cycle_number : 1D-array
        Number of the corresponding HWP cycle.
    StokesPara : 1D-array
        Stokes parameters.
    '''

    print_and_log('--- Removing open AO-loop observations')
    path_open_loop_files = Path(path_beams_files[0].parent,
                                'open_loop_files.txt')

    open_loop_files = []
    with open(path_open_loop_files, 'r') as f:
        for file_i in f.readlines():
            open_loop_files.append(Path(file_i.replace('\n','')))

    open_loop_files = np.array(open_loop_files)

    if len(open_loop_files) != 0:

        mask_to_remove = (path_beams_files[None,:]==open_loop_files[:,None])
        mask_to_remove = mask_to_remove.sum(axis=0, dtype=np.bool)

        cycles_to_remove = HWP_cycle_number[mask_to_remove]
        mask_to_remove   = (HWP_cycle_number[None,:]==cycles_to_remove[:,None])
        mask_to_remove   = mask_to_remove.sum(axis=0, dtype=np.bool)

        if np.any(mask_to_remove):
            print_and_log(f'    Removed {mask_to_remove.sum()} files:')
            for file_i, StokesPara_i in zip(path_beams_files[mask_to_remove],
                                            StokesPara[mask_to_remove]):
                print_and_log(f'    {StokesPara_i} {file_i.name}')

        path_beams_files = path_beams_files[~mask_to_remove]
        HWP_cycle_number = HWP_cycle_number[~mask_to_remove]
        StokesPara       = StokesPara[~mask_to_remove]

    return path_beams_files, HWP_cycle_number, StokesPara

def saturated_pixel_mask(beams, saturated_counts):
    '''
    Create a saturated pixel mask.

    Input
    -----
    beams : 4D-array
        Array of beam-images.
    saturated_counts : float
        Upper limit of pixel's linear response regime.

    Output
    ------
    spm : 2D-array
        Saturated pixel mask.
    '''

    # Mask saturated pixels
    spm = np.ones(beams.shape[2:])
    spm[beams.max(axis=(0,1)) > saturated_counts] = 0

    return spm


def equalise_ord_ext_flux(r, spm, beams, r_inner_IPS, r_outer_IPS):
    '''
    Re-scale the flux in the ordinary and extra-ordinary beams with annuli.

    Input
    -----
    r : 1D-array
        Radius-array.
    spm : 1D-array
        Saturated pixel mask.
    beams : 3D-array
        Array of beam-images.
    r_inner_IPS : list
        Inner radii of the annuli used in IP-subtraction and ord./ext.
        re-scaling.
    r_outer_IPS : list
        Outer radii of the annuli used in IP-subtraction and ord./ext.
        re-scaling.

    Output
    ------
    new_beams : 4D-array
        Array of beam-images, re-scaled ordinary and extra-ordinary beams.
    '''

    new_beams = []
    for r_inner, r_outer in zip(r_inner_IPS, r_outer_IPS):
        # Multiple annuli
        mask_annulus = (r >= r_inner) & (r <= r_outer)

        # Sum over pixels within annulus
        f_ord = np.nansum((beams[:,0]*spm[None,:])[:,mask_annulus], axis=1)
        f_ext = np.nansum((beams[:,1]*spm[None,:])[:,mask_annulus], axis=1)

        X_ord_ext_i = f_ord/f_ext
        X_ord_ext_i = X_ord_ext_i[:,None]

        new_ord_beam_i = beams[:,0] / np.sqrt(X_ord_ext_i)
        new_ext_beam_i = beams[:,1] * np.sqrt(X_ord_ext_i)
        new_beams_i    = np.concatenate((new_ord_beam_i[:,None,:],
                                         new_ext_beam_i[:,None,:]),
                                        axis=1, dtype=np.float32)

        new_beams.append(new_beams_i)

    new_beams = np.moveaxis(np.array(new_beams), 0, -1)

    return new_beams

def fit_U_efficiency(Q, U, I_Q, I_U, r, r_crosstalk):
    '''
    Assess the crosstalk-efficiency of the Stokes U component
    by counting pixels in Q and U.

    Input
    -----
    Q : 1D-array
        Stokes Q observation.
    U : 1D-array
        Stokes U observation.
    I_Q : 1D-array
        Stokes Q intensity.
    I_U : 1D-array
        Stokes U intensity.
    r : 1D-array
        Radius-array.
    r_crosstalk : list
        Inner and outer radius of the annulus used to correct for crosstalk.

    Output
    ------
    e_U : float
        Efficiency of the U observation.
    '''

    # Assess efficiency in an annulus with clear signal
    r_inner, r_outer = r_crosstalk
    mask_annulus = (r >= r_inner) & (r <= r_outer)

    # Evaluate efficiencies from 0 to 1 in steps of 0.01
    e_U_all = np.arange(0, 1+1e-5, 0.01)

    # Apply efficiencies to U image
    abs_Q = np.abs(Q[mask_annulus][:,None])
    abs_U = np.abs(U[mask_annulus][:,None]/e_U_all[None,:])

    # Count number of pixels where |Q|>|U/e_U| and |Q|<|U/e_U|
    num_pixels_over  = np.nansum(abs_Q > abs_U, axis=0)
    num_pixels_under = np.nansum(abs_Q < abs_U, axis=0)

    # Best efficiency is found when |Q|~|U/e_U|
    e_U = e_U_all[np.argmin(np.abs(num_pixels_over - num_pixels_under))]
    return np.round(e_U, 2)

def fit_offset_angle(r, phi, median_Q, median_U, r_crosstalk):
    '''
    Assess the offset angle to minimise the U_phi signal.

    Input
    -----
    r : 1D-array
        Radius-array.
    phi : 1D-array
        Azimuthal angle.
    PDI_frames : dict
        Dictionary of images resulting from PDI.
    r_crosstalk : list
        Inner and outer radius of the annulus used to correct for crosstalk.

    Output
    ------
    theta : float
        Offset angle in degrees.
    '''

    # Assess offset angles in an annulus with clear signal
    r_inner, r_outer = r_crosstalk
    mask_annulus = (r >= r_inner) & (r <= r_outer)

    # Evaluate offset angles from 0 to 90 in steps of 0.1 degrees
    theta_all = np.arange(0,90+1e-5,0.1)

    U_phi_sum = []
    for i, theta_i in enumerate(theta_all):

        # Create new phi with an offset angle
        phi_i = phi[:,None] + np.deg2rad(theta_i)

        # Calculate U_phi with the offset angle
        U_phi_i = + median_Q*np.sin(2*phi_i) - median_U*np.cos(2*phi_i)

        # Sum over pixels within the annulus
        U_phi_sum_i = np.abs(np.nansum(U_phi_i[mask_annulus], axis=0))
        U_phi_sum.append(U_phi_sum_i)

    U_phi_sum = np.array(U_phi_sum)

    # Best offset angle is found when the sum of U_phi is smallest
    theta = theta_all[np.argmin(U_phi_sum, axis=0)]
    return np.round(theta,2)

"""
def fit_offset_angle(r, phi, PDI_frames, r_crosstalk):
    '''
    Assess the offset angle to minimise the U_phi signal.

    Input
    -----
    r : 1D-array
        Radius-array.
    phi : 1D-array
        Azimuthal angle.
    PDI_frames : dict
        Dictionary of images resulting from PDI.
    r_crosstalk : list
        Inner and outer radius of the annulus used to correct for crosstalk.

    Output
    ------
    theta : float
        Offset angle in degrees.
    '''

    # Assess offset angles in an annulus with clear signal
    r_inner, r_outer = r_crosstalk
    mask_annulus = (r >= r_inner) & (r <= r_outer)

    # Evaluate offset angles from 0 to 90 in steps of 0.1 degrees
    theta_all = np.arange(0,90+1e-5,0.1)

    U_phi_sum = np.ones((len(theta_all),len(PDI_frames['P_I']))) * np.nan
    for i, theta_i in enumerate(theta_all):

        # Create new phi with an offset angle
        phi_i = phi[None,:] + np.deg2rad(theta_i)

        # Calculate U_phi with the offset angle
        U_phi_i = + PDI_frames['median_Q_IPS'] * np.sin(2*phi_i) \
                  - PDI_frames['median_U_IPS'] * np.cos(2*phi_i)

        # Sum over pixels within the annulus
        U_phi_sum[i] = np.abs(np.nansum(U_phi_i[:,mask_annulus], axis=1))

    # Best offset angle is found when the sum of U_phi is smallest
    theta = theta_all[np.argmin(U_phi_sum, axis=0)]
    return np.round(theta,2)
"""

def individual_Stokes_frames(beams):
    '''
    Add / subtract ordinary and extra-ordinary beam to
    retrieve I and Q/U images.

    Input
    -----
    beams : 4D-array
        Array of beam-images.

    Output
    ------
    ind_I : 3D-array
        Intensity image for each observation.
    ind_QU : 3D-array
        Stokes Q/U image for each observation.
    '''

    print_and_log('--- Combining the ord./ext. beams into Stokes Q and U images')

    if beams.shape[1] != 1:
        # Single difference
        ind_I  = beams[:,0] + beams[:,1]
        ind_QU = beams[:,0] - beams[:,1]
    else:
        ind_I  = beams[:,0]
        ind_QU = beams[:,0]

    return ind_I, ind_QU

def double_difference(ind_I, ind_QU, mask_beams, StokesPara,
                      crosstalk_correction, r, r_crosstalk,
                      Wollaston_used, path_PDI):
    '''
    Apply the double-difference method to remove instrumental polarisation.

    Input
    -----
    ind_I : 3D-array
        Intensity image for each observation.
    ind_QU : 3D-array
        Stokes Q/U image for each observation.
    mask_beams : 2D-array
        Mask of the non-NaN values.
    StokesPara : 1D-array
        Stokes parameters ('Q+', 'U+', 'Q-', 'U-').
    crosstalk_correction : bool
        Correct for crosstalk_correction if True.
    r : 1D-array
        Radius-array.
    r_crosstalk : list
        Inner and outer radius of the annulus used to correct for crosstalk.
    Wollaston_used : bool
        If True, Wollaston was used, else wiregrid was used.
    path_PDI : str
        Path to PDI output directory.

    Output
    ------
    PDI_frames : dict
        Dictionary of images resulting from PDI.
    '''

    print_and_log('--- Double-difference method to remove instrumental polarisation (IP)')

    # Keep track of the data products in dictionaries
    Q_frames = {'cube_Q': None, 'median_Q': None,
                'cube_Q+': None, 'median_Q+': None,
                'cube_Q-': None, 'median_Q-': None,
                }
    I_Q_frames = {'cube_I_Q': None, 'median_I_Q': None,
                  'cube_I_Q+': None, 'median_I_Q+': None,
                  'cube_I_Q-': None, 'median_I_Q-': None,
                  }

    U_frames = {'cube_U': None, 'median_U': None,
                'cube_U+': None, 'median_U+': None,
                'cube_U-': None, 'median_U-': None,
                }
    I_U_frames = {'cube_I_U': None, 'median_I_U': None,
                  'cube_I_U+': None, 'median_I_U+': None,
                  'cube_I_U-': None, 'median_I_U-': None,
                  }

    I_frames = {'cube_I': None, 'median_I': None}

    # Masks of the data
    mask_Qplus = (StokesPara=='Q+')
    mask_Qmin  = (StokesPara=='Q-')
    mask_Q = np.ma.mask_or(mask_Qplus, mask_Qmin)

    mask_Uplus = (StokesPara=='U+')
    mask_Umin  = (StokesPara=='U-')
    mask_U = np.ma.mask_or(mask_Uplus, mask_Umin)

    def double_difference_QU(QU_frames, I_QU_frames, key,
                             mask_QU_min, mask_QU_plus):

        # Retrieve the double-difference Stokes parameters if possible
        if mask_QU_plus.any() and mask_QU_min.any():
            QU_frames[f'cube_{key}']     = 1/2*(ind_QU[mask_QU_plus] - \
                                               ind_QU[mask_Qmin])
            I_QU_frames[f'cube_I_{key}'] = 1/2*(ind_I[mask_QU_plus] + \
                                               ind_I[mask_QU_min])
        elif mask_QU_plus.any() and not mask_QU_min.any():
            QU_frames[f'cube_{key}']     = ind_QU[mask_QU_plus]
            I_QU_frames[f'cube_I_{key}'] = ind_I[mask_QU_plus]
        elif not mask_QU_plus.any() and mask_QU_min.any():
            QU_frames[f'cube_{key}']     = - ind_QU[mask_QU_min]
            I_QU_frames[f'cube_I_{key}'] = ind_I[mask_QU_min]

        # Calculate the individual parameters
        if mask_QU_plus.any():
            QU_frames[f'cube_{key}+']     = ind_QU[mask_QU_plus]
            I_QU_frames[f'cube_I_{key}+'] = ind_I[mask_QU_plus]
        if mask_QU_min.any():
            QU_frames[f'cube_{key}-']     = ind_QU[mask_QU_min]
            I_QU_frames[f'cube_I_{key}-'] = ind_I[mask_QU_min]

        # Calculate the median parameters
        if mask_QU_plus.any() or mask_QU_min.any():
            QU_frames[f'median_{key}'] = np.nanmedian(QU_frames[f'cube_{key}'],
                                                      axis=0)
            I_QU_frames[f'median_I_{key}'] \
            = np.nanmedian(I_QU_frames[f'cube_I_{key}'], axis=0)

        # Calculate the individual parameters
        if mask_QU_plus.any():
            QU_frames[f'median_{key}+'] \
            = np.nanmedian(QU_frames[f'cube_{key}+'], axis=0)
            I_QU_frames[f'median_I_{key}+'] \
            = np.nanmedian(I_QU_frames[f'cube_I_{key}+'], axis=0)
        if mask_QU_min.any():
            QU_frames[f'median_{key}-'] \
            = np.nanmedian(QU_frames[f'cube_{key}-'], axis=0)
            I_QU_frames[f'median_I_{key}-'] \
            = np.nanmedian(I_QU_frames[f'cube_I_{key}-'], axis=0)

        return QU_frames, I_QU_frames

    # Stokes Q parameters
    Q_frames, I_Q_frames = double_difference_QU(Q_frames, I_Q_frames, 'Q',
                                                mask_Qmin, mask_Qplus)
    # Stokes U parameters
    U_frames, I_U_frames = double_difference_QU(U_frames, I_U_frames, 'U',
                                                mask_Umin, mask_Uplus)

    # Total intensity
    if (Q_frames['cube_Q'] is not None) and \
        (U_frames['cube_U'] is not None):

        I_frames['cube_I']   = 1/2*(I_Q_frames['cube_I_Q'] + \
                                    I_U_frames['cube_I_U'])
        I_frames['median_I'] = np.nanmedian(I_frames['cube_I'], axis=0,
                                            keepdims=True)

    return Q_frames, I_Q_frames, U_frames, I_U_frames, I_frames

def CTC(Q_frames, I_Q_frames, U_frames, I_U_frames, I_frames, r, r_crosstalk):
    '''
    Crosstalk correction.
    '''
    # Determine and correct for the U crosstalk-efficiency

    print_and_log('--- Correcting for the crosstalk-efficiency of U')

    # Loop over the ord./ext. flux-scaling annuli
    e_U = []
    for j in range(Q_frames['median_Q'].shape[-1]):
        e_U_j = fit_U_efficiency(Q_frames['median_Q'][:,j],
                                 U_frames['median_U'][:,j],
                                 I_Q_frames['median_I_Q'][:,j],
                                 I_U_frames['median_I_U'][:,j],
                                 r, r_crosstalk)
        e_U.append(e_U_j)

    print_and_log(f'    Efficiency per IPS annulus: e_U = {e_U}')

    # Correct for the reduced efficiency of the U parameter
    e_U = np.array(e_U)[None,None,:]
    for key in ['cube_U', 'cube_U+', 'cube_U-',
                'median_U', 'median_U+', 'median_U-']:
        if U_frames[key] is not None:
            U_frames[f'{key}_CTC'] = U_frames[key] / e_U

    for key in ['cube_I_U', 'cube_I_U+', 'cube_I_U-',
                'median_I_U', 'median_I_U+', 'median_I_U-']:
        if I_U_frames[key] is not None:
            I_U_frames[f'{key}_CTC'] = I_U_frames[key] / e_U

    I_frames['cube_I_CTC']   = 1/2*(I_Q_frames['cube_I_Q'] + \
                                    I_U_frames['cube_I_U_CTC'])
    I_frames['median_I_CTC'] = np.nanmedian(I_frames['cube_I_CTC'], axis=0,
                                            keepdims=True)

    return Q_frames, I_Q_frames, U_frames, I_U_frames, I_frames


    """
    # Save the Q, U and I images to a dictionary
    PDI_frames = {'cube_Q': None, 'cube_U': None,
                  'cube_I_Q': None, 'cube_I_U': None}

    mask_Q = np.ma.mask_or((StokesPara=='Q+'), (StokesPara=='Q-'))
    mask_U = np.ma.mask_or((StokesPara=='U+'), (StokesPara=='U-'))

    if len(np.unique(StokesPara)) == 4:
        # Stokes Q, U images
        PDI_frames['cube_Q'] = 1/2 * (ind_QU[StokesPara=='Q+'] -
                                      ind_QU[StokesPara=='Q-'])
        PDI_frames['cube_U'] = 1/2 * (ind_QU[StokesPara=='U+'] -
                                      ind_QU[StokesPara=='U-'])

        # Stokes Q, U intensity images
        PDI_frames['cube_I_Q'] = 1/2 * (ind_I[StokesPara=='Q+'] +
                                        ind_I[StokesPara=='Q-'])
        PDI_frames['cube_I_U'] = 1/2 * (ind_I[StokesPara=='U+'] +
                                        ind_I[StokesPara=='U-'])

    else:

        if mask_Q.any():
            PDI_frames['cube_Q'] = ind_QU[mask_Q]
            PDI_frames['cube_Q'][StokesPara[mask_Q]=='Q-'] *= -1 # Flip the sign

            PDI_frames['cube_I_Q'] = ind_I[mask_Q]

        if mask_U.any():
            PDI_frames['cube_U'] = ind_QU[mask_U]
            PDI_frames['cube_U'][StokesPara[mask_U]=='U-'] *= -1 # Flip the sign

            PDI_frames['cube_I_U'] = ind_I[mask_U]

    # Save PDI frames to disk and replace with path to save memory
    for key in PDI_frames.keys():
        if PDI_frames[key] is not None:
            PDI_frames[key] = write_FITS_file(Path(path_PDI, f'{key}.fits'),
                                              PDI_frames[key])

    # Determine the U crosstalk-efficiency and correct
    if crosstalk_correction and \
        (PDI_frames['cube_Q'] is not None) and \
        (PDI_frames['cube_U'] is not None):

        print_and_log('--- Correcting for the crosstalk-efficiency of U')

        # Applied to the median Q/U images
        median_Q = np.nanmedian(load_PDI_frames(PDI_frames, 'cube_Q'), axis=0)
        median_U = np.nanmedian(load_PDI_frames(PDI_frames, 'cube_U'), axis=0)
        median_I_Q = np.nanmedian(load_PDI_frames(PDI_frames, 'cube_I_Q'),
                                  axis=0)
        median_I_U = np.nanmedian(load_PDI_frames(PDI_frames, 'cube_I_U'),
                                  axis=0)

        # Loop over the ord./ext. flux-scaling annuli
        e_U_all = []
        for j in range(median_Q.shape[-1]):
            e_U = fit_U_efficiency(median_Q[:,j], median_U[:,j],
                                   median_I_Q[:,j], median_I_U[:,j],
                                   r, r_crosstalk)
            e_U_all.append(e_U)

        e_U_all = np.array(e_U_all)
        print_and_log(f'    Efficiency per IPS annulus: e_U = {list(e_U_all)}')

        # Correct for the reduced efficiency
        PDI_frames['cube_U']   = load_PDI_frames(PDI_frames, 'cube_U')
        PDI_frames['cube_I_U'] = load_PDI_frames(PDI_frames, 'cube_I_U')
        for key in ['cube_U', 'cube_I_U']:
            PDI_frames[key] /= e_U_all[None,None,:]

    if (PDI_frames['cube_Q'] is not None) and \
        (PDI_frames['cube_U'] is not None):
        # Total intensity images
        PDI_frames['cube_I'] = 1/2 * (load_PDI_frames(PDI_frames, 'cube_I_Q') +
                                      load_PDI_frames(PDI_frames, 'cube_I_U'))
        PDI_frames['median_I'] = np.nanmedian(PDI_frames['cube_I'], axis=0,
                                              keepdims=True)

    if (PDI_frames['cube_Q'] is not None):
        PDI_frames['median_Q']   = np.nanmedian(load_PDI_frames(PDI_frames,
                                                                'cube_Q'),
                                                axis=0, keepdims=True)
        PDI_frames['median_I_Q'] = np.nanmedian(load_PDI_frames(PDI_frames,
                                                                'cube_I_Q'),
                                                axis=0, keepdims=True)

    if (PDI_frames['cube_U'] is not None):
        PDI_frames['median_U']   = np.nanmedian(load_PDI_frames(PDI_frames,
                                                                'cube_U'),
                                                axis=0, keepdims=True)
        PDI_frames['median_I_U'] = np.nanmedian(load_PDI_frames(PDI_frames,
                                                                'cube_I_U'),
                                                axis=0, keepdims=True)

    for QU_sel, I_deg in zip(['Q+', 'Q-', 'U+', 'U-'], [0, 90, 45, 135]):

        mask_QU_sel = (StokesPara == QU_sel)
        if QU_sel[1]=='+':
            mask_QU_sel_alt = (StokesPara == QU_sel.replace('+','-'))
        elif QU_sel[1]=='-':
            mask_QU_sel_alt = (StokesPara == QU_sel.replace('-','+'))

        if np.any(mask_QU_sel) and np.any(mask_QU_sel_alt):
            if Wollaston_used:
                # Save the individual Q+- and U+- measurements
                PDI_frames[f'cube_I_{QU_sel}'] = ind_I[mask_QU_sel]
                PDI_frames[f'median_I_{QU_sel}'] \
                = np.nanmedian(ind_I[mask_QU_sel],
                               axis=0, keepdims=True)

                PDI_frames[f'cube_{QU_sel}'] = ind_QU[mask_QU_sel]
                PDI_frames[f'median_{QU_sel}'] \
                = np.nanmedian(ind_QU[mask_QU_sel],
                               axis=0, keepdims=True)

            else:
                # Save the measurements
                PDI_frames[f'cube_I_{I_deg}_deg'] = ind_I[mask_QU_sel]
                PDI_frames[f'median_I_{I_deg}_deg'] \
                = np.nanmedian(ind_I[mask_QU_sel],
                               axis=0, keepdims=True)

    for key in PDI_frames.keys():
        if isinstance(PDI_frames[key], np.ndarray):
            PDI_frames[key] = write_FITS_file(Path(path_PDI, f'{key}.fits'),
                                              PDI_frames[key])
    return PDI_frames
    """

def IPS(r, spm, r_inner_IPS, r_outer_IPS, Q, U, I_Q, I_U, I):
    '''
    Apply instrumental polarisation subtraction by using annuli.

    Input
    -----
    r : 1D-array
        Radius-array.
    spm : 1D-array
        Saturated pixel mask.
    r_inner_IPS : list
        Inner radii of the annuli used in IP-subtraction and ord./ext.
        re-scaling.
    r_outer_IPS : list
        Outer radii of the annuli used in IP-subtraction and ord./ext.
        re-scaling.
    Q : 3D-array
        Stokes Q observation.
    U : 3D-array
        Stokes U observation.
    I_Q : 3D-array
        Stokes Q intensity.
    I_U : 3D-array
        Stokes U intensity.
    I : 3D-array
        Total intensity.

    Output
    ------
    median_Q_IPS : 2D-array
        Median IP-subtracted Stokes Q observation.
    median_U_IPS : 2D-array
        Median IP-subtracted Stokes U observation.
    '''

    # Perform IP subtraction for each HWP cycle to avoid temporal differences
    Q_IPS, U_IPS = [], []
    for i in range(len(Q)):

        Q_IPS_i, U_IPS_i = [], []
        for j, r_inner, r_outer in zip(range(len(r_inner_IPS)), \
                                       r_inner_IPS, r_outer_IPS):

            if Q.ndim == 2:
                # Apply saturated-pixels mask before IPS
                Q_j = Q[i,:] * spm
                U_j = U[i,:] * spm
                I_Q_j = I_Q[i,:] * spm
                I_U_j = I_U[i,:] * spm
                I_j = I[i,:] * spm
            else:
                # Apply saturated-pixels mask before IPS
                Q_j = Q[i,:,j] * spm
                U_j = U[i,:,j] * spm
                I_Q_j = I_Q[i,:,j] * spm
                I_U_j = I_U[i,:,j] * spm
                I_j = I[i,:,j] * spm

            # Multiple annuli for instrumental polarisation correction
            mask_annulus = (r >= r_inner) & (r <= r_outer)

            # Median Q/I within the annulus
            c_Q_j = np.nanmedian(Q_j[mask_annulus]/I_j[mask_annulus])
            Q_IPS_j = Q_j - I_Q_j*c_Q_j
            Q_IPS_i.append(Q_IPS_j)

            # Median U/I within the annulus
            c_U_j = np.nanmedian(U_j[mask_annulus]/I_j[mask_annulus])
            U_IPS_j = U_j - I_U_j*c_U_j
            U_IPS_i.append(U_IPS_j)

        # Append to list with all HWP cycles
        Q_IPS.append(Q_IPS_i)
        U_IPS.append(U_IPS_i)

    Q_IPS = np.swapaxes(np.array(Q_IPS, dtype=np.float32), 1, 2)
    U_IPS = np.swapaxes(np.array(U_IPS, dtype=np.float32), 1, 2)

    # Median over all HWP cycles
    median_Q_IPS = np.nanmedian(Q_IPS, axis=0)
    median_U_IPS = np.nanmedian(U_IPS, axis=0)
    return median_Q_IPS, median_U_IPS

def final_Stokes_frames(r, phi, median_Q, median_U, theta=None):
    '''
    Compute the final Stokes images.

    Input
    -----
    r : 1D-array
        Radius-array.
    phi : 1D-array
        Azimuthal angle.
    PDI_frames : dict
        Dictionary of images resulting from PDI.
    minimise_U_phi : bool
        Minimise the signal in U_phi if True.
    r_crosstalk : list
        Inner and outer radius of the annulus used to correct for crosstalk.

    Output
    ------
    PDI_frames : dict
        Dictionary of images resulting from PDI.
    '''

    print_and_log('--- Creating final data products (PI, Q_phi, U_phi)')

    # Polarised intensity
    PI = np.sqrt(median_Q**2 + median_U**2)

    phi = phi[:,None]
    if theta is not None:
        # Add offset angles to the phi array
        phi = phi + np.deg2rad(theta[None,:])

    # Azimuthal Stokes parameters
    Q_phi = - median_Q*np.cos(2*phi) - median_U*np.sin(2*phi)
    U_phi = + median_Q*np.sin(2*phi) - median_U*np.cos(2*phi)

    return PI, Q_phi, U_phi

def UpC(r, phi, median_Q, median_U, r_crosstalk):
    '''
    U_phi correction.
    '''

    print_and_log('--- Minimising the U_phi signal')

    # Minimise the sum of U_phi in an annulus
    theta = fit_offset_angle(r, phi, median_Q, median_U, r_crosstalk)

    print_and_log(f'    Offset angle per IPS annulus: theta (deg) = {list(theta)}')

    _, Q_phi_UpC, U_phi_UpC = final_Stokes_frames(r, phi, median_Q, median_U,
                                                  theta=theta)
    return Q_phi_UpC, U_phi_UpC

"""
def final_Stokes_frames(r, phi, PDI_frames, minimise_U_phi, r_crosstalk):
    '''
    Compute the final Stokes images.

    Input
    -----
    r : 1D-array
        Radius-array.
    phi : 1D-array
        Azimuthal angle.
    PDI_frames : dict
        Dictionary of images resulting from PDI.
    minimise_U_phi : bool
        Minimise the signal in U_phi if True.
    r_crosstalk : list
        Inner and outer radius of the annulus used to correct for crosstalk.

    Output
    ------
    PDI_frames : dict
        Dictionary of images resulting from PDI.
    '''

    print_and_log('--- Creating final data products (PI, Q_phi, U_phi)')

    # De-projected radius
    PDI_frames['r'] = r[None,:]

    # polarised intensity image
    PDI_frames['P_I'] = np.sqrt(PDI_frames['median_Q_IPS']**2 + \
                                PDI_frames['median_U_IPS']**2)

    if minimise_U_phi:
        print_and_log('--- Minimising the U_phi signal')

        # Minimise the sum of U_phi in an annulus
        theta = fit_offset_angle(r, phi, PDI_frames, r_crosstalk)

        print_and_log(f'    Offset angle per IPS annulus: theta (deg) = {list(theta)}')

        # Add the best offset angles to the phi array
        phi = phi[None,:] + np.deg2rad(theta[:,None])

    # Azimuthal Stokes parameters
    PDI_frames['Q_phi'] = - PDI_frames['median_Q_IPS']*np.cos(2*phi) \
                          - PDI_frames['median_U_IPS']*np.sin(2*phi)
    PDI_frames['U_phi'] = + PDI_frames['median_Q_IPS']*np.sin(2*phi) \
                          - PDI_frames['median_U_IPS']*np.cos(2*phi)

    return PDI_frames, phi
"""

def extended_Stokes_frames(r, spm, r_inner_IPS, r_outer_IPS, PDI_frames):
    '''
    Produce extended total and polarised intensity images by
    exploiting greater sky coverage of non-HWP observations.

    Input
    -----
    r : 1D-array
        Radius-array.
    spm : 1D-array
        Saturated pixel mask.
    r_inner_IPS : list
        Inner radii of the annuli used in IP-subtraction and ord./ext.
        re-scaling.
    r_outer_IPS : list
        Outer radii of the annuli used in IP-subtraction and ord./ext.
        re-scaling.
    PDI_frames : dict
        Dictionary of images resulting from PDI.

    Output
    ------
    PDI_frames : dict
        Dictionary of images resulting from PDI.
    '''

    print_and_log('--- Producing extended data products')

    # Arrays to store the masks and extended frames in
    Q_extended = np.moveaxis(np.zeros(PDI_frames['cube_Q'].shape[1:]), -1, 0)
    U_extended = np.moveaxis(np.zeros(PDI_frames['cube_U'].shape[1:]), -1, 0)
    I_extended = np.moveaxis(np.zeros(PDI_frames['cube_I'].shape[1:]), -1, 0)

    mask_Q_extended = np.zeros(Q_extended.shape, dtype=np.int8)
    mask_U_extended = np.zeros(U_extended.shape, dtype=np.int8)
    mask_extended   = np.zeros(I_extended.shape, dtype=np.int8)

    # Loop over each combination of Q+, Q-, U+, and U-
    for pm_Q_i, pm_U_i in zip(['+','+','-','-'], ['+','-','+','-']):

        # Obtain the Q and U intensities
        Q_i = PDI_frames[f'cube_Q{pm_Q_i}']
        U_i = PDI_frames[f'cube_U{pm_U_i}']
        I_Q_i  = PDI_frames[f'cube_I_Q{pm_Q_i}']
        I_U_i  = PDI_frames[f'cube_I_U{pm_U_i}']
        I_QU_i = np.sqrt(I_Q_i**2 + I_U_i**2)

        # Perform IP-subtraction
        Q_i, U_i = IPS(r, spm, r_inner_IPS, r_outer_IPS,
                       Q_i, U_i, I_Q_i, I_U_i, I_QU_i)

        # Median-combine over cycles
        I_Q_i  = np.moveaxis(np.nanmedian(I_Q_i, axis=0), -1, 0)
        I_U_i  = np.moveaxis(np.nanmedian(I_U_i, axis=0), -1, 0)
        I_QU_i = np.sqrt(I_Q_i**2 + I_U_i**2)

        mask_Q_i = ~np.isnan(Q_i)
        mask_U_i = ~np.isnan(U_i)

        # Flip the signal of Q- and U- observations
        Q_extended[mask_Q_i] += float(pm_Q_i+'1') * Q_i[mask_Q_i]
        U_extended[mask_U_i] += float(pm_U_i+'1') * U_i[mask_U_i]

        I_extended[mask_Q_i*mask_U_i] += I_QU_i[mask_Q_i*mask_U_i]**2

        # Add to complete mask
        mask_Q_extended += mask_Q_i
        mask_U_extended += mask_U_i
        mask_extended   += mask_Q_i * mask_U_i

    # Normalise pixels by number of covering observations
    Q_extended[mask_Q_extended!=0] /= mask_Q_extended[mask_Q_extended!=0]
    U_extended[mask_U_extended!=0] /= mask_U_extended[mask_U_extended!=0]

    Q_extended[mask_Q_extended==0] = np.nan
    U_extended[mask_U_extended==0] = np.nan

    PDI_frames['median_Q_IPS_extended'] = Q_extended
    PDI_frames['median_U_IPS_extended'] = U_extended

    # Total I of overlapping Q and U signal
    I_extended[mask_extended!=0] /= mask_extended[mask_extended!=0]
    I_extended[mask_extended==0] = np.nan
    PDI_frames['median_I_extended'] = np.sqrt(I_extended)

    # PI of overlapping Q and U signal
    PDI_frames['P_I_extended'] = (PDI_frames['median_Q_IPS_extended']**2 + \
                                  PDI_frames['median_U_IPS_extended']**2)**(1/2)

    return PDI_frames

def write_header_coordinates(file, header, object_name, mask_beams):
    '''
    Create header keywords to add a world-coordinate system.

    Input
    -----
    file : str
        Filename of SCIENCE observations.
    header : astropy header
    object_name : str
        Object's name.
    mask_beams : 2D-array
        Mask of the non-NaN values.

    Output
    ------
    header : astropy header
    '''

    # Coordinate transformation matrix
    CD = np.array([[fits.getval(file, 'CD1_1'), fits.getval(file, 'CD1_2')],
                   [fits.getval(file, 'CD2_1'), fits.getval(file, 'CD2_2')]])

    pos_angle = np.deg2rad(-(fits.getval(file, 'ESO ADA POSANG')))

    # Rotation matrix
    R = np.array([[np.cos(pos_angle), -np.sin(pos_angle)],
                  [np.sin(pos_angle),  np.cos(pos_angle)]])

    # Rotate the transformation matrix
    new_CD = np.matmul(CD, R)

    # Fill in to the header
    header['CD1_1'] = new_CD[0,0]
    header['CD1_2'] = new_CD[0,1]
    header['CD2_1'] = new_CD[1,0]
    header['CD2_2'] = new_CD[1,1]

    # Query the SIMBAD archive to retrieve object coordinates
    query_result = Simbad.query_object(object_name, wildcard=True)
    # Convert the icrs coordinates to fk5
    coord_icrs = SkyCoord(ra=query_result['RA'], dec=query_result['DEC'],
                          frame='icrs', unit=(u.hourangle, u.deg))
    coord_fk5 = coord_icrs.transform_to('fk5')

    # Reference value
    header['CRVAL1'] = coord_fk5.ra.degree[0]
    header['CRVAL2'] = coord_fk5.dec.degree[0]

    # Reference pixel, first pixel has index 1
    im_shape = mask_beams.shape
    header['CRPIX1'] = im_shape[1]/2 + 1/2
    header['CRPIX2'] = im_shape[0]/2 + 1/2

    # Fill in RA, DEC
    header['RA']  = coord_fk5.ra.degree[0]
    header['DEC'] = coord_fk5.dec.degree[0]
    header.comments['RA']  = query_result['RA'][0].replace(' ', ':') + \
                             ' RA (J2000) pointing (deg)'
    header.comments['DEC'] = query_result['DEC'][0].replace(' ', ':') + \
                             ' DEC (J2000) pointing (deg)'

    return header

def write_header(object_name, mask_beams):
    '''
    Create header keywords.

    Input
    -----
    object_name : str
        Object's name.
    mask_beams : 2D-array
        Mask of the non-NaN values.

    Output
    ------
    hdu : astropy HDUList object
    '''

    # Read a header
    hdr = fits.getheader(path_beams_files_selected[-1])

    # Create a header
    hdu = fits.PrimaryHDU()
    #hdu = fits.BinTableHDU()


    hdu.header['SIMPLE'] = hdr['SIMPLE']
    hdu.header.comments['SIMPLE'] = 'Standard FITS'

    hdu.header['BITPIX'] = hdr['BITPIX']
    hdu.header.comments['BITPIX'] = '# of bits per pix value'

    hdu.header['NAXIS'] = hdr['NAXIS']
    hdu.header.comments['NAXIS'] = '# of axes in data array'

    hdu.header['NAXIS1'] = hdr['NAXIS1']
    hdu.header.comments['NAXIS1'] = '# of pixels in <axis direction>'

    hdu.header['NAXIS2'] = hdr['NAXIS2']
    hdu.header.comments['NAXIS2'] = '# of pixels in <axis direction>'

    hdu.header['NAXIS3'] = hdr['NAXIS3']
    hdu.header.comments['NAXIS3'] = '# of pixels in <axis direction>'

    hdu.header['EXTEND'] = True
    hdu.header.comments['EXTEND'] = 'FITS Extension may be present'


    keys_to_copy = ['ORIGIN', 'TELESCOP', 'INSTRUME', 'OBJECT',
                    'RA', 'DEC', 'EQUINOX', 'RADECSYS', 'EXPTIME',
                    'CTYPE1', 'CTYPE2', 'CRVAL1', 'CRVAL2', 'CRPIX1', 'CRPIX2',
                    'CD1_1', 'CD1_2', 'CD2_1', 'CD2_2',
                    'ESO INS GRP ID', 'ESO INS PIXSCALE',
                    'ESO INS OPTI1 ID', 'ESO INS OPTI7 ID', 'ESO OBS PROG ID',
                    'ESO OBS ID', 'ESO INS OPTI5 NAME', 'ESO INS OPTI6 NAME',
                    'ESO INS CWLEN', 'ESO TEL GEOELEV',
                    'ESO TEL GEOLAT', 'ESO TEL GEOLON'
                   ]

    all_DATE_OBS = []
    all_pos_ang  = []

    for i, file in enumerate(path_beams_files_selected):

        # Read a header
        hdr = fits.getheader(file)

        # Fill in some header values
        for key in keys_to_copy:
            try:
                hdu.header[key]          = hdr[key]
                hdu.header.comments[key] = hdr.comments[key]

                if key == 'ESO INS PIXSCALE':
                    hdu.header.comments[key] = hdr.comments[key] + ' (arcsec)'
                if key == 'EXPTIME':
                    hdu.header.comments[key] = hdr.comments[key] + ' (s)'
                if key == 'ESO INS CWLEN':
                    hdu.header.comments[key] = hdr.comments[key] + ' (micron)'

            except KeyError:
                # Read a different header
                pass

        # Save the observing date of each file
        all_DATE_OBS.append(hdr['DATE-OBS'])
        all_pos_ang.append(hdr['ESO ADA POSANG'])

    if np.all(np.array(all_pos_ang) == all_pos_ang[-1]):
        hdu.header['ESO ADA POSANG'] = all_pos_ang[-1]
    else:
        hdu.header['ESO ADA POSANG'] = 0.0
    hdu.header.comments['ESO ADA POSANG'] = 'Position angle before de-rotation (deg)'

    # Save observing dates at end of the header
    for i in range(len(path_beams_files_selected)):
        hdu.header['DATE-OBS'+str(i+1)]          = all_DATE_OBS[i]
        hdu.header.comments['DATE-OBS'+str(i+1)] = 'Observing date ' + str(i+1)

    hdu.header['DATE REDUCED'] = datetime.datetime.today().strftime('%Y-%m-%d-%H:%M:%S')

    hdu.header = write_header_coordinates(path_beams_files_selected[0],
                                          hdu.header, object_name, mask_beams)

    return hdu

def save_PDI_frames(path_PDI, frames, type, mask_beams, hdu,
                    HWP_used, pos_angle):

    if type == 'Q':
        keys_to_read = ['cube_Q', 'median_Q', 'median_Q_IPS',
                        'median_Q_CTC_IPS']
    elif type == 'I_Q':
        keys_to_read = ['cube_I_Q', 'median_I_Q']
    elif type == 'Q+':
        keys_to_read = ['cube_Q+', 'median_Q+']
    elif type == 'I_Q+':
        keys_to_read = ['cube_I_Q+', 'median_I_Q+']
    elif type == 'Q-':
        keys_to_read = ['cube_Q-', 'median_Q-']
    elif type == 'I_Q-':
        keys_to_read = ['cube_I_Q-', 'median_I_Q-']

    elif type == 'U':
        keys_to_read = ['cube_U', 'median_U', 'median_U_IPS',
                        'cube_U_CTC', 'median_U_CTC',
                        'median_U_CTC_IPS']
    elif type == 'I_U':
        keys_to_read = ['cube_I_U', 'median_I_U', 'cube_I_U_CTC',
                        'median_I_U_CTC']
    elif type == 'U+':
        keys_to_read = ['cube_U+', 'median_U+', 'cube_U+_CTC', 'median_U+_CTC']
    elif type == 'I_U+':
        keys_to_read = ['cube_I_U+', 'median_I_U+', 'cube_I_U+_CTC',
                        'median_I_U+_CTC']
    elif type == 'U-':
        keys_to_read = ['cube_U-', 'median_U-', 'cube_U-_CTC', 'median_U-_CTC']
    elif type == 'I_U-':
        keys_to_read = ['cube_I_U-', 'median_I_U-', 'cube_I_U-_CTC',
                        'median_I_U-_CTC']

    elif type == 'I':
        keys_to_read = ['cube_I', 'median_I', 'cube_I_CTC', 'median_I_CTC']

    elif type == 'PI':
        keys_to_read = ['PI', 'PI_IPS', 'PI_CTC_IPS']
    elif type == 'Q_phi':
        keys_to_read = ['Q_phi', 'Q_phi_IPS', 'Q_phi_CTC_IPS',
                        'Q_phi_UpC_CTC_IPS']
    elif type == 'U_phi':
        keys_to_read = ['U_phi', 'U_phi_IPS', 'U_phi_CTC_IPS',
                        'U_phi_UpC_CTC_IPS']

    print('')
    hdu_list = fits.HDUList(hdu)

    for key in frames.keys():
        if key in keys_to_read:

            # Move the pixel-axis to the first axis
            im_to_save = frames[key]
            if im_to_save.ndim == 3:
                im_to_save = np.moveaxis(im_to_save, 0, -1)

            # Reshape the array to form an image
            new_shape      = (*mask_beams.shape, *im_to_save.shape[1:])
            new_im_to_save = np.ones(new_shape) * np.nan

            # Replace the pixels within the mask with the image
            new_im_to_save[mask_beams] = im_to_save
            del im_to_save

            # Remove axes of length 1
            new_im_to_save = np.squeeze(new_im_to_save)

            if HWP_used:
                # Rotate the image
                new_im_to_save = rotate_cube(new_im_to_save,
                                             pos_angle, pad=False,
                                             rotate_axes=(0,1))

            # Swap axes for saving to a FITS file
            if new_im_to_save.ndim == 3:
                new_im_to_save = np.moveaxis(new_im_to_save, -1, 0)
            elif new_im_to_save.ndim == 4:
                new_im_to_save = np.moveaxis(new_im_to_save, -1, 0)
                new_im_to_save = np.moveaxis(new_im_to_save, -1, 0)
            print(key, new_im_to_save.shape)

            # Append to the HDU list
            hdu_list.append(fits.ImageHDU(new_im_to_save, name=key))

    hdu_list.writeto(Path(path_PDI, f'{type}.fits'),
                     output_verify='silentfix',
                     overwrite=True)

"""
def save_PDI_frames(path_PDI, PDI_frames, object_name, mask_beams,
                    HWP_used, pos_angle, keys='all'):
    '''
    Save the resulting images from PDI.

    Input
    -----
    path_PDI : str
        Path to PDI output directory.
    PDI_frames : dict
        Dictionary of images resulting from PDI.
    object_name : str
        Object's name.
    mask_beams : 2D-array
        Mask of the non-NaN values.
    HWP_used : bool
        If True, HWP was used, else position angle was changed.
    pos_angle : float
        Position angle of the observation.
    '''

    # Convert strings to a list of strings
    if isinstance(keys, list):
        keys_to_save = keys
    else:
        if keys=='all':
            keys_to_save = list(PDI_frames.keys())
        else:
            keys_to_save = list(keys)

    array_size = []
    for key in keys_to_save:
        if isinstance(PDI_frames[key], np.ndarray):
            array_size.append(PDI_frames[key].size)
        else:
            array_size.append(np.inf)

    keys_to_save = np.array(keys_to_save)[np.argsort(array_size)]

    # Create a header
    mask_beams_rotated = rotate_cube(mask_beams, pos_angle,
                                     pad=False, rotate_axes=(0,1))
    hdu = write_header(object_name, mask_beams_rotated)
    del mask_beams_rotated

    # Save all images in the PDI_frames dictionary
    for key in keys_to_save:

        im_to_save = load_PDI_frames(PDI_frames, key)
        # Remove the entry from the dictionary
        del PDI_frames[key]

        if im_to_save is not None:

            # Move the pixel-axis to the first axis
            im_to_save = np.moveaxis(im_to_save, 1, -1)

            # Reshape the array to form an image
            new_shape      = (*im_to_save.shape[:-1], *mask_beams.shape)
            new_im_to_save = np.ones(new_shape) * np.nan

            if len(new_shape)==2:
                new_im_to_save[mask_beams] = im_to_save
            elif len(new_shape)==3:
                new_im_to_save[:,mask_beams] = im_to_save
            elif len(new_shape)==4:
                new_im_to_save[:,:,mask_beams] = im_to_save
                new_im_to_save = np.swapaxes(new_im_to_save, 0, 1)

            im_to_save = new_im_to_save
            del new_im_to_save

            # Remove axes of length 1
            im_to_save = np.squeeze(im_to_save)

            if len(im_to_save.shape) == 2:
                im_to_save = im_to_save[None,:]

            if HWP_used:
                # Rotate the image
                im_to_save = list(im_to_save)
                for i in range(len(im_to_save)):
                    im_to_save[i] = rotate_cube(im_to_save[i][None,:],
                                                pos_angle, pad=False,
                                                rotate_axes=(-2,-1))[0]
                im_to_save = np.array(im_to_save)
            write_FITS_file(Path(path_PDI, f'{key}.fits'),
                            im_to_save, header=hdu.header)

        del im_to_save
"""

def load_PDI_frames(PDI_frames, key):

    if isinstance(PDI_frames[key], Path):
        # File was saved, key-value was replaced with path
        # Load and return data
        return fits.getdata(PDI_frames[key]).astype(np.float32)

    else:
        # If array or None, return key-value
        return PDI_frames[key]

def PDI(r_inner_IPS, r_outer_IPS, crosstalk_correction, minimise_U_phi,
        r_crosstalk, HWP_used, Wollaston_used, object_name, disk_pos_angle,
        disk_inclination, saturated_counts=10000):
    '''
    Apply the pre-processing functions.

    Input
    -----
    r_inner_IPS : list
        Inner radii of the annuli used in IP-subtraction and ord./ext.
        re-scaling.
    r_outer_IPS : list
        Outer radii of the annuli used in IP-subtraction and ord./ext.
        re-scaling.
    crosstalk_correction : bool
        Correct for crosstalk_correction if True.
    minimise_U_phi : bool
        Minimise the signal in U_phi if True.
    r_crosstalk : list
        Inner and outer radius of the annulus used to correct for crosstalk.
    HWP_used : bool
        If True, HWP was used, else position angle was changed.
    Wollaston_used : bool
        If True, Wollaston was used, else wiregrid was used.
    object_name : str
        Object's name.
    disk_pos_angle : float
        Disk position angle.
    disk_inclination : float
        Disk inclination.
    saturated_counts : float
        Upper limit of pixel's linear response regime.
    '''

    # Make the directory for the PDI images
    path_PDI = Path(path_output_dir_selected, 'PDI')
    if not path_PDI.is_dir():
        path_PDI.mkdir()

    global path_beams_files_selected
    path_beams_files_selected = sorted(
                        Path(path_output_dir_selected).glob('*_beams.fits')
                        )
    path_beams_files_selected = np.array(path_beams_files_selected)

    # Assign Stokes parameters to each observation
    StokesPara = assign_Stokes_parameters(path_beams_files_selected,
                                          HWP_used, Wollaston_used)

    mask_Q = np.ma.mask_or((StokesPara=='Q+'), (StokesPara=='Q-'))
    mask_U = np.ma.mask_or((StokesPara=='U+'), (StokesPara=='U-'))

    if mask_Q.any() and mask_U.any():

        # Remove any incomplete HWP cycles
        path_beams_files_selected, \
        HWP_cycle_number, \
        StokesPara \
        = remove_incomplete_HWP_cycles(path_beams_files_selected, StokesPara)

        if len(StokesPara)==0:
            # There are no complete cycles, continue to next output directory
            print_and_log('No complete HWP cycles')
            return
    else:
        HWP_cycle_number = np.arange(len(path_beams_files_selected))
        print_and_log('No complete HWP cycles, creating images of the available Stokes components')

    # Remove HWP cycles where open AO-loops were found
    path_beams_files_selected, \
    HWP_cycle_number, \
    StokesPara \
    = remove_open_AO_loop(path_beams_files_selected,
                          HWP_cycle_number, StokesPara)

    if len(StokesPara)==0:
        # There are no complete cycles, continue to next output directory
        print_and_log('No complete HWP cycles')
        return


    # Load the data
    beams = [fits.getdata(x).astype(np.float32) \
             for x in path_beams_files_selected]

    pos_angles = np.array([-(fits.getval(x, 'ESO ADA POSANG'))
                           for x in path_beams_files_selected])

    if not HWP_used:

        # Rotate the frames if HWP was not used
        for i, pos_angle_i in enumerate(pos_angles):
            beams[i] = rotate_cube(beams[i], pos_angle_i, pad=True)

    beams = np.array(beams)

    xc, yc = (beams.shape[3]-1)/2, (beams.shape[2]-1)/2
    r, phi = r_phi(beams[0,0], xc, yc)
    r, phi = r.astype(np.float32), phi.astype(np.float32)

    # Saturated-pixel mask
    spm = saturated_pixel_mask(beams, saturated_counts)

    # Collapse the beams to save memory
    beams, mask_beams = collapse_beams(beams)

    # Flatten the other arrays
    r, phi = r[mask_beams].flatten(), phi[mask_beams].flatten()
    spm = spm[mask_beams].flatten()

    # Retrieve the de-projected radius
    yp, xp = np.mgrid[0:mask_beams.shape[0], 0:mask_beams.shape[1]]
    r_deprojected = deprojected_radius(xp, yp,
                                       (mask_beams.shape[0]-1)/2,
                                       (mask_beams.shape[1]-1)/2,
                                       disk_pos_angle,
                                       disk_inclination)
    r_deprojected = r_deprojected.astype(np.float32)
    r_deprojected = r_deprojected[mask_beams].flatten()

    # Create a header
    mask_beams_rotated = rotate_cube(mask_beams, pos_angles[0],
                                     pad=False, rotate_axes=(0,1))
    hdu = write_header(object_name, mask_beams_rotated)
    del mask_beams_rotated

    # Re-scaling the ordinary and extra-ordinary beam fluxes
    if Wollaston_used:
        beams = equalise_ord_ext_flux(r, spm, beams, r_inner_IPS, r_outer_IPS)

    # Individual Stokes frames
    ind_I, ind_QU = individual_Stokes_frames(beams)
    del beams

    # Double-difference
    Q_frames, I_Q_frames, \
    U_frames, I_U_frames, \
    I_frames \
    = double_difference(ind_I, ind_QU, mask_beams, StokesPara,
                        crosstalk_correction, r, r_crosstalk,
                        Wollaston_used, path_PDI)
    del ind_I, ind_QU

    if (Q_frames['cube_Q'] is not None) and \
        (U_frames['cube_U'] is not None):

        # Instrumental polarisation correction
        print_and_log('--- IP-subtraction (IPS) using annuli with unpolarised stellar signal')
        Q_frames['median_Q_IPS'], \
        U_frames['median_U_IPS'] \
        = IPS(r, spm, r_inner_IPS, r_outer_IPS,
              Q=Q_frames['cube_Q'], U=U_frames['cube_U'],
              I_Q=I_Q_frames['cube_I_Q'], I_U=I_U_frames['cube_I_U'],
              I=I_frames['cube_I'])

        PI_frames, Q_phi_frames, U_phi_frames = {}, {}, {}
        # Final Stokes frames
        PI_frames['PI'], Q_phi_frames['Q_phi'], U_phi_frames['U_phi'] \
        = final_Stokes_frames(r, phi, Q_frames['median_Q'],
                              U_frames['median_U'])

        PI_frames['PI_IPS'], \
        Q_phi_frames['Q_phi_IPS'], \
        U_phi_frames['U_phi_IPS'] \
        = final_Stokes_frames(r, phi, Q_frames['median_Q_IPS'],
                              U_frames['median_U_IPS'])

        if crosstalk_correction:

            Q_frames, I_Q_frames, \
            U_frames, I_U_frames, \
            I_frames \
            = CTC(Q_frames, I_Q_frames, U_frames, I_U_frames,
                  I_frames, r, r_crosstalk)

            # Instrumental polarisation correction
            Q_frames['median_Q_CTC_IPS'], \
            U_frames['median_U_CTC_IPS'] \
            = IPS(r, spm, r_inner_IPS, r_outer_IPS,
                  Q=Q_frames['cube_Q'], U=U_frames['cube_U_CTC'],
                  I_Q=I_Q_frames['cube_I_Q'], I_U=I_U_frames['cube_I_U_CTC'],
                  I=I_frames['cube_I_CTC'])

            """
            # Final Stokes frames
            PI_frames['PI_CTC'], \
            Q_phi_frames['Q_phi_CTC'], \
            U_phi_frames['U_phi_CTC'] \
            = final_Stokes_frames(r, phi, Q_frames['median_Q_CTC'],
                                  U_frames['median_U_CTC'])
            """

            PI_frames['PI_CTC_IPS'], \
            Q_phi_frames['Q_phi_CTC_IPS'], \
            U_phi_frames['U_phi_CTC_IPS'] \
            = final_Stokes_frames(r, phi, Q_frames['median_Q_CTC_IPS'],
                                  U_frames['median_U_CTC_IPS'])

        if minimise_U_phi:

            Q_phi_frames['Q_phi_UpC_CTC_IPS'], \
            U_phi_frames['U_phi_UpC_CTC_IPS'] \
            = UpC(r, phi, Q_frames['median_Q_CTC_IPS'],
                  U_frames['median_U_CTC_IPS'], r_crosstalk)

        save_PDI_frames(path_PDI, PI_frames, 'PI', mask_beams,
                        hdu, HWP_used, pos_angles[0])
        save_PDI_frames(path_PDI, Q_phi_frames, 'Q_phi', mask_beams,
                        hdu, HWP_used, pos_angles[0])
        save_PDI_frames(path_PDI, Q_phi_frames, 'U_phi', mask_beams,
                        hdu, HWP_used, pos_angles[0])

    save_PDI_frames(path_PDI, Q_frames, 'Q', mask_beams,
                    hdu, HWP_used, pos_angles[0])
    save_PDI_frames(path_PDI, I_Q_frames, 'I_Q', mask_beams,
                    hdu, HWP_used, pos_angles[0])

    save_PDI_frames(path_PDI, U_frames, 'U', mask_beams,
                    hdu, HWP_used, pos_angles[0])
    save_PDI_frames(path_PDI, I_U_frames, 'I_U', mask_beams,
                    hdu, HWP_used, pos_angles[0])

    save_PDI_frames(path_PDI, I_frames, 'I', mask_beams,
                    hdu, HWP_used, pos_angles[0])

    """
    PDI_frames = double_difference(ind_I, ind_QU, mask_beams, StokesPara,
                                   crosstalk_correction, r, r_crosstalk,
                                   Wollaston_used, path_PDI)
    del ind_I, ind_QU

    if (PDI_frames['cube_Q'] is not None) and \
       (PDI_frames['cube_U'] is not None):

        # Instrumental polarisation correction
        print_and_log('--- IP-subtraction (IPS) using annuli with unpolarised stellar signal')
        PDI_frames['median_Q_IPS'], \
        PDI_frames['median_U_IPS'] \
        = IPS(r, spm, r_inner_IPS, r_outer_IPS,
              Q=load_PDI_frames(PDI_frames, 'cube_Q'),
              U=load_PDI_frames(PDI_frames, 'cube_U'),
              I_Q=load_PDI_frames(PDI_frames, 'cube_I_Q'),
              I_U=load_PDI_frames(PDI_frames, 'cube_I_U'),
              I=load_PDI_frames(PDI_frames, 'cube_I'))

        # Retrieve the de-projected radius
        yp, xp = np.mgrid[0:mask_beams.shape[0], 0:mask_beams.shape[1]]
        r_deprojected = deprojected_radius(xp, yp,
                                           (mask_beams.shape[0]-1)/2,
                                           (mask_beams.shape[1]-1)/2,
                                           disk_pos_angle,
                                           disk_inclination)
        r_deprojected = r_deprojected.astype(np.float32)
        r_deprojected = r_deprojected[mask_beams].flatten()

        # Final Stokes frames
        PDI_frames, phi = final_Stokes_frames(r_deprojected, phi,
                                              PDI_frames, minimise_U_phi,
                                              r_crosstalk)

        # r^2-scaled Stokes parameters
        PDI_frames['P_I_r2']   = PDI_frames['P_I'] * r_deprojected**2
        PDI_frames['Q_phi_r2'] = PDI_frames['Q_phi'] * r_deprojected**2
        PDI_frames['U_phi_r2'] = PDI_frames['U_phi'] * r_deprojected**2

        if not HWP_used and Wollaston_used:
            # r^2-scaled extended PI image
            PDI_frames = extended_Stokes_frames(r, spm, r_inner_IPS,
                                                r_outer_IPS, PDI_frames)
            PDI_frames['P_I_extended_r2'] = PDI_frames['P_I_extended'] * \
                                            r_deprojected**2

        del r_deprojected, xp, yp

    del r, phi, spm

    # Save the data products
    save_PDI_frames(path_PDI, PDI_frames, object_name,
                    mask_beams, HWP_used, pos_angles[0])

    """

def run_pipeline(path_SCIENCE_dir,
                 path_master_FLAT_dir='../data/master_FLAT/',
                 path_master_BPM_dir='../data/master_BPM/',
                 path_master_DARK_dir='../data/master_DARK/',
                 new_log_file=True,
                 ):
    '''
    Run the complete pipeline.

    Input
    -----
    path_SCIENCE_dir : str
        Path to SCIENCE directory.
    path_master_FLAT_dir : str
        Path to master-FLAT directory.
    path_master_BPM_dir : str
        Path to master-BPM directory.
    path_master_DARK_dir : str
        Path to master-DARK directory.
    new_log_file : bool
        If True, create a new log file, otherwise append to the existing file.
    '''

    global path_log_file
    global path_output_dir
    global path_config_file

    # Record the elapsed time
    start_time = time.time()

    # Check if the directories exist -------------------------------------------
    if not path_SCIENCE_dir.is_dir():
        raise IOError(f'\nThe SCIENCE directory {str(path_SCIENCE_dir.resolve())} does not exist.')

    if not path_master_DARK_dir.is_dir():
        raise IOError(f'\nThe master DARK directory {str(path_master_DARK_dir.resolve())} does not exist.')

    if not path_master_FLAT_dir.is_dir():
        raise IOError(f'\nThe master FLAT directory {str(path_master_FLAT_dir.resolve())} does not exist.')

    if not path_master_BPM_dir.is_dir():
        raise IOError(f'\nThe master BPM directory {str(path_master_BPM_dir.resolve())} does not exist.')

    # Create the path of the output directory
    path_output_dir = Path(path_SCIENCE_dir, 'pipeline_output')

    if not path_output_dir.is_dir():
        path_output_dir.mkdir()

    # FITS-files in SCIENCE directory, sorted by observing date ----------------
    path_SCIENCE_files = sorted(Path(path_SCIENCE_dir).glob('*.fits'))
    path_SCIENCE_files = np.array(path_SCIENCE_files)

    # Check if SCIENCE directory is empty
    if len(path_SCIENCE_files) == 0:
        raise IOError(f'\nThe SCIENCE directory {str(path_SCIENCE_dir.resolve())} does not contain FITS-files. Please ensure that any FITS-files are uncompressed.')


    # Create the log file ------------------------------------------------------
    path_log_file = Path(path_output_dir, 'log.txt')

    if new_log_file:
        print_and_log('')
        print_and_log('=== Welcome to PIPPIN (PdI PiPelIne for Naco data) ===',
                      new_file=new_log_file, pad=80, pad_character='=')
        print_and_log('')
        print_and_log('')
        print_and_log(f'Created output directory {str(path_output_dir.resolve())}')
        print_and_log('')
        print_and_log(f'Created log file {str(path_log_file.resolve())}')
        print_and_log('')


    # Read the configuration file ----------------------------------------------
    path_config_file = Path(path_SCIENCE_dir, 'config.conf')
    print_and_log(f'Reading configuration file {str(path_config_file.resolve())}')

    run_pre_processing, \
    remove_data_products, \
    split_observing_blocks, \
    y_pixel_range, \
    sky_subtraction_method, \
    sky_subtraction_min_offset, \
    remove_horizontal_stripes, \
    centering_method, \
    tied_offset, \
    size_to_crop, \
    r_inner_IPS, \
    r_outer_IPS, \
    crosstalk_correction, \
    minimise_U_phi, \
    r_crosstalk, \
    object_name, \
    disk_pos_angle, \
    disk_inclination \
    = read_config_file(path_config_file)

    # Read information from the SCIENCE headers --------------------------------
    dict_headers = {'ESO INS GRP ID':[],                      # HWP
                    'ESO INS OPTI1 ID':[],                    # Wollaston prism
                    #'ESO INS OPTI4 ID':[],                    # Wiregrid
                    'ESO INS OPTI7 ID':[],                    # Detector
                    'ESO DET WIN NX':[], 'ESO DET WIN NY':[], # Window shape
                    'ESO DET WIN STARTX':[], 'ESO DET WIN STARTY':[]
                    }
    for x in path_SCIENCE_files:
        for key in dict_headers.keys():
            dict_headers[key].append(read_from_FITS_header(x, key))

    # Unique combination of HWP/camera/window shape
    OBS_configs = np.array(list(dict_headers.values())).T

    for OBS_config_i in np.unique(OBS_configs, axis=0):

        # Usage of HWP, Wollaston prism, wiregrid and camera
        HWP_used       = (OBS_config_i[:2]=='Half_Wave_Plate').any()
        Wollaston_used = (np.ma.mask_or(OBS_config_i[:2]=='Wollaston_00',
                                        OBS_config_i[:2]=='Wollaston_45')
                         ).any()
        camera_used    = OBS_config_i[2]

        Wollaston_45 = False
        if Wollaston_used:
            Wollaston_45 = (OBS_config_i[:2]=='Wollaston_45').any()

        if OBS_config_i[1] in ['FLM_13', 'FLM_27', 'FLM_54']:
            # Polarimetric mask was not used
            FLAT_pol_mask = False
        else:
            FLAT_pol_mask = True

        # Set the window shape and window starting pixel
        window_shape = [int(float(OBS_config_i[-4])),
                        int(float(OBS_config_i[-3]))]
        window_start = [int(float(OBS_config_i[-2])),
                        int(float(OBS_config_i[-1]))]

        # Read the FLATs, BPMs and DARKs
        path_FLAT_files = np.array(sorted(Path(path_master_FLAT_dir
                                    ).glob(f'master_FLAT_{camera_used}*.fits')))
        # Read the BPMs
        path_BPM_files  = np.array(sorted(Path(path_master_BPM_dir
                                    ).glob(f'master_BPM_{camera_used}*.fits')))
        # Read the DARKs
        path_DARK_files = np.array(sorted(Path(path_master_DARK_dir
                                    ).glob(f'master_DARK_{camera_used}*.fits')))

        # Mask all the observations corresponding to this configuration
        SCIENCE_mask = np.prod((OBS_configs==OBS_config_i), axis=1, dtype=bool)
        path_SCIENCE_files_i = path_SCIENCE_files[SCIENCE_mask]

        # Split the data into unique observation types and create directories
        read_unique_obsTypes(path_SCIENCE_files_i, split_observing_blocks,
                             HWP_used, Wollaston_used, Wollaston_45,
                             camera_used)
        global path_SCIENCE_files_selected
        global path_output_dir_selected
        global obsType_selected

        # Run for each unique observation type
        for path_output_dir_selected, obsType_selected in zip(path_output_dirs,\
                                                              unique_obsTypes):

            # Mask of the same observation type
            mask_selected = ((obsTypes==obsType_selected).sum(axis=1)==3)
            filter_used = obsType_selected[2]

            # Paths of the reduced files for this observation type
            path_SCIENCE_files_selected = path_SCIENCE_files_i[mask_selected]

            if run_pre_processing:
                # Run the pre-processing functions
                print_and_log('')
                print_and_log('')
                print_and_log('=== Running the pre-processing functions ===',
                              pad=80, pad_character='=')
                print_and_log('')
                print_and_log('--- Reducing {} observations of type: ({}, {}, {}) ---'.format(
                              mask_selected.sum(), *obsType_selected),
                              pad=80, pad_character='-')

                pre_processing(window_shape=window_shape, \
                               window_start=window_start, \
                               remove_data_products=remove_data_products, \
                               y_pixel_range=y_pixel_range, \
                               sky_subtraction_method=sky_subtraction_method, \
                               sky_subtraction_min_offset=sky_subtraction_min_offset, \
                               remove_horizontal_stripes=remove_horizontal_stripes, \
                               centering_method=centering_method, \
                               tied_offset=tied_offset, \
                               size_to_crop=size_to_crop, \
                               HWP_used=HWP_used, \
                               Wollaston_used=Wollaston_used, \
                               Wollaston_45=Wollaston_45, \
                               camera_used=camera_used, \
                               filter_used=filter_used, \
                               path_SCIENCE_files=path_SCIENCE_files_selected, \
                               path_FLAT_files=path_FLAT_files, \
                               path_BPM_files=path_BPM_files, \
                               path_DARK_files=path_DARK_files, \
                               FLAT_pol_mask=FLAT_pol_mask
                               )

            # Run the polarimetric differential imaging functions
            print_and_log('')
            print_and_log('')
            print_and_log('=== Running the PDI functions ===',
                          pad=80, pad_character='=')
            PDI(r_inner_IPS=r_inner_IPS, \
                r_outer_IPS=r_outer_IPS, \
                crosstalk_correction=crosstalk_correction, \
                minimise_U_phi=minimise_U_phi, \
                r_crosstalk=r_crosstalk, \
                HWP_used=HWP_used, \
                Wollaston_used=Wollaston_used, \
                object_name=object_name, \
                disk_pos_angle=disk_pos_angle, \
                disk_inclination=disk_inclination)

    print_and_log('')
    print_and_log('')
    print_and_log('Elapsed time: {}'.format(str(datetime.timedelta(
                                                seconds=time.time()-start_time)
                                                )))
    print_and_log('')
    print_and_log('=== Finished running the pipeline ===',
                  pad=80, pad_character='=')
    print_and_log('')

def run_example(path_cwd):

    path_SCIENCE_dir    = Path(path_cwd, 'example_HD_135344B')
    path_FLAT_dir       = Path(path_SCIENCE_dir, 'FLATs')
    path_master_BPM_dir = Path(path_SCIENCE_dir, 'master_BPMs')
    path_DARK_dir       = Path(path_SCIENCE_dir, 'DARKs')

    # Define names of example data
    files_to_download = ['config.conf',
                         'NACO.2012-07-25T00:59:39.294.fits',
                         'NACO.2012-07-25T01:02:21.189.fits',
                         'NACO.2012-07-25T01:05:02.698.fits',
                         'NACO.2012-07-25T01:07:44.231.fits',
                         'NACO.2012-07-25T02:09:28.406.fits',
                         'NACO.2012-07-25T02:12:09.554.fits',
                         'NACO.2012-07-25T02:14:51.024.fits',
                         'NACO.2012-07-25T02:17:32.490.fits',
                         'DARKs/NACO.2012-07-25T10:37:01.343.fits',
                         'FLATs/NACO.2012-07-25T12:35:17.506.fits',
                         'FLATs/NACO.2012-07-25T12:35:46.300.fits',
                         ]

    # Check if data already exists
    files_exist = np.array([Path(path_SCIENCE_dir, file_i).is_file()
                             for file_i in files_to_download]).all()

    if not files_exist:

        # Data must be downloaded
        user_input = input('\nData is not found in the current directory. Proceed to download 48.4 MB? (y/n)\n')

        if user_input == 'y':
            print('\nDownloading data.')

            if not path_SCIENCE_dir.is_dir():
                path_SCIENCE_dir.mkdir()
            if not path_FLAT_dir.is_dir():
                path_FLAT_dir.mkdir()
            if not path_DARK_dir.is_dir():
                path_DARK_dir.mkdir()

            download_url = 'https://github.com/samderegt/PIPPIN-NACO/raw/master/pippin/example_HD_135344B/'

            for file_i in tqdm(files_to_download, bar_format=pbar_format):
                # Download the data from the git
                urllib.request.urlretrieve(download_url + file_i,
                                           str(Path(path_SCIENCE_dir, file_i))
                                           )

            files_exist = True

        elif user_input == 'n':
            print('\nNot downloading data.')
        else:
            print('\nInvalid input.')

    if files_exist:
        # Files exist, run the pipeline

        # Create master FLATs, BPMs and DARKs from the provided paths
        path_master_FLAT_dir, \
        path_master_BPM_dir, \
        path_master_DARK_dir \
        = prepare_calib_files(path_SCIENCE_dir=path_SCIENCE_dir,
                              path_FLAT_dir=path_FLAT_dir,
                              path_master_BPM_dir=path_master_BPM_dir,
                              path_DARK_dir=path_DARK_dir
                              )

        # Run the pipeline
        run_pipeline(path_SCIENCE_dir=path_SCIENCE_dir,
                     path_master_FLAT_dir=path_master_FLAT_dir,
                     path_master_BPM_dir=path_master_BPM_dir,
                     path_master_DARK_dir=path_master_DARK_dir,
                     new_log_file=False
                     )
