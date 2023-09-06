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

import pippin.auxiliary_functions as af
import pippin.figures as figs
import pippin.beam_fitting as beam
import pippin.sky_subtraction as sky

# Setting the length of progress bars
pbar_format = '{l_bar}{bar:20}{r_bar}{bar:-20b}'

################################################################################
# Reading and writing files
################################################################################

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

    figs.plot_open_AO_loop(
        path_reduced_files_selected, max_counts, bounds_ord_beam, bounds_ext_beam
        )

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
                       if (af.read_from_FITS_header(file_i, 'NAXIS')==2)
                       and (af.read_from_FITS_header(file_i, 'ESO DET WIN NX') == 1024)
                       and (af.read_from_FITS_header(file_i, 'ESO DET WIN NY') == 1024)
                      ]
    path_FLAT_files = np.array(path_FLAT_files)

    if len(path_FLAT_files) == 0:
        raise IOError(f'\nThe FLAT directory {str(path_FLAT_dir.resolve())} does not contain FITS-files. Please ensure that any FITS-files are uncompressed.')


    if not path_DARK_dir.is_dir():
        raise IOError(f'\nThe DARK directory {str(path_DARK_dir.resolve())} does not exist.')

    path_DARK_files = sorted(Path(path_DARK_dir).glob('*.fits'))

    # Ensure that FLATs and DARKs have the same image shapes
    path_DARK_files = [file_i for file_i in path_DARK_files
                       if (af.read_from_FITS_header(file_i, 'ESO DET WIN NX') == 1024)
                       and (af.read_from_FITS_header(file_i, 'ESO DET WIN NY') == 1024)
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
        camera_i = af.read_from_FITS_header(path_FLAT_file_i, 'ESO INS OPTI7 ID')

        # Read the filter keyword(s)
        filter_i = af.read_from_FITS_header(path_FLAT_file_i, 'ESO INS OPTI6 NAME')
        if filter_i == 'empty':
            filter_i = af.read_from_FITS_header(path_FLAT_file_i,
                                             'ESO INS OPTI5 NAME')

        # Read the exposure time
        expTime_i = af.read_from_FITS_header(path_FLAT_file_i, 'EXPTIME')

        # Read the lamp status (on/off). True if on, False if off.
        lampStatus_i = (af.read_from_FITS_header(path_FLAT_file_i,
                                              'ESO INS LAMP2 SET') != 0)

        # Read whether the Wollaston prism was inserted
        OPTI1_ID_i = af.read_from_FITS_header(path_FLAT_file_i, 'ESO INS OPTI1 ID')

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

    if (FLAT_cameras=='L27').any():
        if (FLAT_lampStatus[FLAT_cameras=='L27']==FLAT_lampStatus[FLAT_cameras=='L27'][0]).all():
            # Turn half the FLATs into lamp-on/off
            new_FLAT_lampStatus = FLAT_lampStatus[FLAT_cameras=='L27'].copy()
            new_FLAT_lampStatus[len(new_FLAT_lampStatus)//2:] = \
                'False' if FLAT_lampStatus[0]=='True' else 'True'

            FLAT_lampStatus[FLAT_cameras=='L27'] = new_FLAT_lampStatus

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
        camera_i = af.read_from_FITS_header(path_DARK_file_i, 'ESO INS OPTI7 ID')

        # Read the exposure time
        expTime_i = af.read_from_FITS_header(path_DARK_file_i, 'EXPTIME')

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
                         axis=1, dtype=bool)
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
                              axis=1, dtype=bool)
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

        '''
        ############
        path_FLAT_file_i = Path('/home/sam/Documents/Master-2/MRP/PIPPIN-NACO/pippin/data/master_FLAT', f'master_FLAT_{camera_i}_{filter_i}{OPTI1_ID_i}_NACO.{master_FLATs_lamp_on_header[i]["DATE-OBS"]}.fits')
        fits.writeto(path_FLAT_file_i,
                     master_FLATs_lamp_on[i].astype(np.float32),
                     output_verify='silentfix', overwrite=True)
        ############
        '''

        # Save the BPM
        path_BPM_file_i = f'master_BPM_{camera_i}_{filter_i}{OPTI1_ID_i}_NACO.{master_FLATs_lamp_on_header[i]["DATE-OBS"]}.fits'
        path_BPM_file_i = Path(path_master_BPM_dir, path_BPM_file_i)
        fits.writeto(path_BPM_file_i, master_BPMs[i].astype(np.float32),
                     output_verify='silentfix', overwrite=True)

        '''
        ############
        path_BPM_file_i = Path('/home/sam/Documents/Master-2/MRP/PIPPIN-NACO/pippin/data/master_BPM', f'master_BPM_{camera_i}_{filter_i}{OPTI1_ID_i}_NACO.{master_FLATs_lamp_on_header[i]["DATE-OBS"]}.fits')
        fits.writeto(path_BPM_file_i, master_BPMs[i].astype(np.float32),
                     output_verify='silentfix', overwrite=True)
        ############
        '''

    for i in range(len(master_DARKs)):

        camera_i, _ = DARK_configs_unique[i]

        # Save the DARK
        path_DARK_file_i = f'master_DARK_{camera_i}_NACO.{master_DARKs_header[i]["DATE-OBS"]}.fits'
        path_DARK_file_i = Path(path_master_DARK_dir, path_DARK_file_i)
        fits.writeto(path_DARK_file_i, master_DARKs[i].astype(np.float32),
                     header=master_DARKs_header[i], output_verify='silentfix',
                     overwrite=True)

        '''
        ############
        path_DARK_file_i = Path('/home/sam/Documents/Master-2/MRP/PIPPIN-NACO/pippin/data/master_DARK', f'master_DARK_{camera_i}_NACO.{master_DARKs_header[i]["DATE-OBS"]}.fits')
        fits.writeto(path_DARK_file_i, master_DARKs[i].astype(np.float32),
                     header=master_DARKs_header[i], output_verify='silentfix', overwrite=True)
        ############
        '''

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
    DATE_OBS = af.read_from_FITS_header(SCIENCE_file, 'DATE-OBS')
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
    master_FLAT = af.read_FITS_as_cube(path_FLAT_file)[0]
    master_BPM  = af.read_FITS_as_cube(path_BPM_file)[0]
    master_DARK = af.read_FITS_as_cube(path_DARK_file)[0]

    # DARK exposure time
    DARK_expTime = af.read_from_FITS_header(path_DARK_file, 'EXPTIME')
    # SCIENCE exposure time
    SCIENCE_expTime = af.read_from_FITS_header(SCIENCE_file, 'EXPTIME')

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
    expTimes = np.array([af.read_from_FITS_header(x, 'EXPTIME')
                         for x in path_SCIENCE_files])
    OBS_IDs, filters = [], []
    for x in path_SCIENCE_files:
        try:
            OBS_IDs.append(af.read_from_FITS_header(x, 'ESO OBS ID'))
        except KeyError:
            OBS_IDs.append(0)

        filter_i = af.read_from_FITS_header(x, 'ESO INS OPTI6 NAME')
        if filter_i == 'empty':
            filter_i = af.read_from_FITS_header(x, 'ESO INS OPTI5 NAME')
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
        cube, header = af.read_FITS_as_cube(file)

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
        af.write_FITS_file(reduced_file, cube, header=header)

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
    figs.plot_reduction(
        path_SCIENCE_files_selected=path_SCIENCE_files_selected, 
        path_reduced_files_selected=path_reduced_files_selected, 
        )

    # Find the beam centers
    print_and_log(f'--- Fitting the beam centers using method \'{centering_method}\'')
    beam_centers = beam.fit_beam_centers(
        centering_method, Wollaston_used, Wollaston_45, 
        camera_used, filter_used, tied_offset, 
        path_reduced_files_selected
        )

    # Subtract the sky from the images
    print_and_log(f'--- Sky-subtraction using method: \'{sky_subtraction_method}\'')
    global path_skysub_files_selected
    path_skysub_files_selected = sky.sky_subtraction(
        sky_subtraction_method, sky_subtraction_min_offset, beam_centers, 
        HWP_used, Wollaston_used, remove_horizontal_stripes, 
        path_reduced_files_selected
        )

    # Center the beams and save
    print_and_log('--- Centering the beams')
    global path_beams_files_selected
    beams, path_beams_files_selected = beam.center_beams(
        beam_centers, size_to_crop, Wollaston_used, 
        Wollaston_45, path_skysub_files_selected
        )
    # Perform sigma-clipping on all beams
    open_AO_loop(np.array(beams), sigma_max=3)

    print_and_log('--- Plotting the sky-subtracted and cropped images')
    figs.plot_reduction(
        path_SCIENCE_files_selected=path_SCIENCE_files_selected, 
        path_reduced_files_selected=path_reduced_files_selected, 
        path_skysub_files_selected=path_skysub_files_selected, 
        path_beams_files_selected=path_beams_files_selected, 
        beam_centers=beam_centers, size_to_crop=size_to_crop,
        Wollaston_45=Wollaston_45
        )

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
        mask_to_remove = mask_to_remove.sum(axis=0, dtype=bool)

        cycles_to_remove = HWP_cycle_number[mask_to_remove]
        mask_to_remove   = (HWP_cycle_number[None,:]==cycles_to_remove[:,None])
        mask_to_remove   = mask_to_remove.sum(axis=0, dtype=bool)

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
    median_Q : 2D-array
        Median Stokes Q parameter.
    median_U : 2D-array
        Median Stokes U parameter.
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

def double_difference(ind_I, ind_QU, mask_beams, StokesPara):
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

    Output
    ------
    Q_frames : dict
        Dictionary of Stokes Q images.
    I_Q_frames : dict
        Dictionary of Stokes Q intensity images.
    U_frames : dict
        Dictionary of Stokes U images.
    I_U_frames : dict
        Dictionary of Stokes U intensity images.
    I_frames : dict
        Dictionary of total intensity images.
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

        mask_QU = np.ma.mask_or(mask_QU_min, mask_QU_plus)

        # Retrieve the double-difference Stokes parameters if possible
        if (mask_QU_plus.sum() == mask_QU_min.sum()) and mask_QU_plus.any():
            QU_frames[f'cube_{key}']     = 1/2*(ind_QU[mask_QU_plus] - \
                                                ind_QU[mask_QU_min])
            I_QU_frames[f'cube_I_{key}'] = 1/2*(ind_I[mask_QU_plus] + \
                                                ind_I[mask_QU_min])

            # Calculate the individual parameters
            QU_frames[f'cube_{key}+']     = ind_QU[mask_QU_plus]
            I_QU_frames[f'cube_I_{key}+'] = ind_I[mask_QU_plus]

            QU_frames[f'cube_{key}-']     = ind_QU[mask_QU_min]
            I_QU_frames[f'cube_I_{key}-'] = ind_I[mask_QU_min]

        elif mask_QU_plus.any() or mask_QU_min.any():
            if ind_QU[mask_QU].ndim == 3:    
                QU_frames[f'cube_{key}'] = ind_QU[mask_QU] * \
                (-1*mask_QU_min[mask_QU][:,None,None] + 1*mask_QU_plus[mask_QU][:,None,None])
            elif ind_QU[mask_QU].ndim == 2:
                QU_frames[f'cube_{key}'] = ind_QU[mask_QU] * \
                (-1*mask_QU_min[mask_QU][:,None] + 1*mask_QU_plus[mask_QU][:,None])

            I_QU_frames[f'cube_I_{key}'] = ind_I[mask_QU]

        # Calculate the individual parameters
        if mask_QU_plus.any() and mask_QU_min.any():
            QU_frames[f'cube_{key}+']     = ind_QU[mask_QU_plus]
            I_QU_frames[f'cube_I_{key}+'] = ind_I[mask_QU_plus]

            QU_frames[f'cube_{key}-']     = ind_QU[mask_QU_min]
            I_QU_frames[f'cube_I_{key}-'] = ind_I[mask_QU_min]

        # Calculate the median parameters
        if mask_QU_plus.any() or mask_QU_min.any():
            QU_frames[f'median_{key}'] = np.nanmedian(QU_frames[f'cube_{key}'],
                                                      axis=0)
            I_QU_frames[f'median_I_{key}'] \
            = np.nanmedian(I_QU_frames[f'cube_I_{key}'], axis=0)

        # Calculate the individual parameters
        if mask_QU_plus.any() and mask_QU_min.any():
            QU_frames[f'median_{key}+'] \
            = np.nanmedian(QU_frames[f'cube_{key}+'], axis=0)
            I_QU_frames[f'median_I_{key}+'] \
            = np.nanmedian(I_QU_frames[f'cube_I_{key}+'], axis=0)

            QU_frames[f'median_{key}-'] \
            = np.nanmedian(QU_frames[f'cube_{key}-'], axis=0)
            I_QU_frames[f'median_I_{key}-'] \
            = np.nanmedian(I_QU_frames[f'cube_I_{key}-'], axis=0)

        """
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
        """

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
                                            keepdims=False)

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

def final_Stokes_frames(median_Q, median_U, r_deprojected, phi, theta=None):
    '''
    Compute the final Stokes images.

    Input
    -----
    median_Q : 2D-array
        Median Stokes Q parameter.
    median_U : 2D-array
        Median Stokes U parameter.
    r_deprojected : 1D-array
        De-projected radius array.
    phi : 1D-array
        Azimuthal angle.
    theta : float or None
        Offset-angle (in degrees) to minimise the U_phi-signal

    Output
    ------
    PI : 2D array
        Polarised intensity.
    PI_r2 : 2D array
        Polarised intensity scaled by the squared radius.
    Q_phi : 2D-array
        Azimuthal Stokes Q_phi parameter.
    Q_phi_r2 : 2D-array
        Azimuthal Stokes Q_phi parameter scaled by the squared radius.
    U_phi : 2D-array
        Azimuthal Stokes U_phi parameter.
    U_phi_r2 : 2D-array
        Azimuthal Stokes U_phi parameter scaled by the squared radius.
    '''

    if (median_Q.ndim == 2) and (median_U.ndim == 2):
        r_deprojected = r_deprojected[:,None]
        phi = phi[:,None]

    # Polarised intensity
    PI    = np.sqrt(median_Q**2 + median_U**2)
    PI_r2 = PI * r_deprojected**2

    if theta is not None:
        # Add offset angles to the phi array
        phi = phi + np.deg2rad(theta[None,:])
    
    # Azimuthal Stokes parameters
    Q_phi = - median_Q*np.cos(2*phi) - median_U*np.sin(2*phi)
    U_phi = + median_Q*np.sin(2*phi) - median_U*np.cos(2*phi)

    Q_phi_r2 = Q_phi * r_deprojected**2
    U_phi_r2 = U_phi * r_deprojected**2

    return PI, PI_r2, Q_phi, Q_phi_r2, U_phi, U_phi_r2

def UpC(median_Q, median_U, r, r_crosstalk, r_deprojected, phi):
    '''
    U_phi correction.
    '''

    print_and_log('--- Minimising the U_phi signal')

    # Minimise the sum of U_phi in an annulus
    theta = fit_offset_angle(r, phi, median_Q, median_U, r_crosstalk)

    print_and_log(f'    Offset angle per IPS annulus: theta (deg) = {list(theta)}')

    _, _, \
    Q_phi_UpC, Q_phi_UpC_r2, \
    U_phi_UpC, U_phi_UpC_r2 \
    = final_Stokes_frames(median_Q, median_U, r_deprojected, phi, theta=theta)

    return Q_phi_UpC, Q_phi_UpC_r2, U_phi_UpC, U_phi_UpC_r2

def extended_Stokes_frames(Q_frames, I_Q_frames, U_frames, I_U_frames,
                           I_frames, PI_frames,
                           spm, r_inner_IPS, r_outer_IPS,
                           r, r_deprojected, phi
                           ):
    '''
    Produce extended total and polarised intensity images by
    exploiting greater sky coverage of non-HWP observations.

    Input
    -----
    Q_frames : dict
    I_Q_frames : dict
    U_frames : dict
    I_U_frames : dict
    I_frames : dict
    PI_frames : dict

    spm : 1D-array
        Saturated pixel mask.
    r_inner_IPS : list
        Inner radii of the annuli used in IP-subtraction and ord./ext.
        re-scaling.
    r_outer_IPS : list
        Outer radii of the annuli used in IP-subtraction and ord./ext.
        re-scaling.
    r : 1D-array
        Radius-array.
    r_deprojected : 1D-array
        De-projected radius array.
    phi : 1D-array
        Azimuthal angle.

    Output
    ------
    Q_frames : dict
    I_Q_frames : dict
    U_frames : dict
    I_U_frames : dict
    I_frames : dict
    PI_frames : dict

    '''

    print_and_log('--- Generating extended data products')

    # Extended frames, averaged over the redundant (+,-) observations
    Q_frames['cube_Q_extended'] = np.nanmean(np.array([Q_frames['cube_Q+'],
                                                       -Q_frames['cube_Q-']]),
                                             axis=0)
    U_frames['cube_U_extended'] = np.nanmean(np.array([U_frames['cube_U+'],
                                                       -U_frames['cube_U-']]),
                                             axis=0)

    I_Q_frames['cube_I_Q_extended'] = np.nanmean(
                                          np.array([I_Q_frames['cube_I_Q+'],
                                                    I_Q_frames['cube_I_Q-']]),
                                          axis=0)
    I_U_frames['cube_I_U_extended'] = np.nanmean(
                                          np.array([I_U_frames['cube_I_U+'],
                                                    I_U_frames['cube_I_U-']]),
                                          axis=0)

    I_frames['cube_I_extended'] = np.nanmean(
                                    np.array([I_Q_frames['cube_I_Q_extended'],
                                              I_Q_frames['cube_I_Q_extended']]),
                                    axis=0)

    # Median extended frames
    Q_frames['median_Q_extended'] = np.nanmedian(Q_frames['cube_Q_extended'],
                                                 axis=0)
    U_frames['median_U_extended'] = np.nanmedian(U_frames['cube_U_extended'],
                                                 axis=0)
    I_Q_frames['median_I_Q_extended'] = np.nanmedian(
                                            I_Q_frames['cube_I_Q_extended'],
                                            axis=0)
    I_U_frames['median_I_U_extended'] = np.nanmedian(
                                            I_U_frames['cube_I_U_extended'],
                                            axis=0)
    I_frames['median_I_extended'] = np.nanmedian(I_frames['cube_I_extended'],
                                                 axis=0)

    # Final data products without IP-corrections
    PI_frames['PI_extended'], PI_frames['PI_r2_extended'], _, _, _, _ \
    = final_Stokes_frames(Q_frames['median_Q_extended'],
                          U_frames['median_U_extended'],
                          r_deprojected, phi)

    # Perform IP-subtraction
    Q_frames['median_Q_IPS_extended'], \
    U_frames['median_U_IPS_extended'] \
    = IPS(r, spm, r_inner_IPS, r_outer_IPS,
          Q_frames['cube_Q_extended'],
          U_frames['cube_U_extended'],
          I_Q_frames['cube_I_Q_extended'],
          I_U_frames['cube_I_U_extended'],
          I_frames['cube_I_extended']
          )

    # Final IP-subtracted data products
    PI_frames['PI_IPS_extended'], PI_frames['PI_IPS_r2_extended'], _, _, _, _ \
    = final_Stokes_frames(Q_frames['median_Q_IPS_extended'],
                          U_frames['median_U_IPS_extended'],
                          r_deprojected, phi)

    return Q_frames, I_Q_frames, U_frames, I_U_frames, I_frames, PI_frames

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
    query_result = Simbad.query_object(object_name)
    # Convert the icrs coordinates to fk5
    coord_icrs = SkyCoord(ra=query_result['RA'], dec=query_result['DEC'],
                          frame='icrs', unit=(u.hourangle, u.deg))
    coord_fk5 = coord_icrs.transform_to('fk5')

    # Reference value
    header['CRVAL1'] = coord_fk5.ra.degree[0]
    header['CRVAL2'] = coord_fk5.dec.degree[0]

    # Reference pixel, first pixel has index 1
    header['CRPIX1'] = mask_beams.shape[1]/2 - 1/2*(mask_beams.shape[1]%2) + 1
    header['CRPIX2'] = mask_beams.shape[0]/2 - 1/2*(mask_beams.shape[0]%2) + 1

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

def save_PDI_frames(type, frames, mask_beams, HWP_used,
                    pos_angle, hdu, path_PDI):

    '''
    Save the resulting images from PDI.

    Input
    -----
    type : str
        Type of frames to save.
    frames : dict
        Dictionary of frames to save.
    mask_beams : 2D-array
        Mask of the non_NaN values.
    HWP_used : bool
        If True, HWP was used, else position angle was changed.
    pos_angle : float
        Position angle of the observation.
    hdu : astropy HDUList object

    path_PDI : str
        Path to PDI output directory.
    '''

    if type == 'Q':
        keys_to_read = ['cube_Q', 'median_Q', 'median_Q_IPS',
                        'median_Q_CTC_IPS',
                        'cube_Q_extended', 'median_Q_extended',
                        'median_Q_IPS_extended']
    elif type == 'I_Q':
        keys_to_read = ['cube_I_Q', 'median_I_Q',
                        'cube_I_Q_extended', 'median_I_Q_extended']
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
                        'cube_U_CTC', 'median_U_CTC', 'median_U_CTC_IPS',
                        'cube_U_extended', 'median_U_extended',
                        'median_U_IPS_extended']
    elif type == 'I_U':
        keys_to_read = ['cube_I_U', 'median_I_U', 'cube_I_U_CTC',
                        'median_I_U_CTC',
                        'cube_I_U_extended', 'median_I_U_extended']
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
        keys_to_read = ['cube_I', 'median_I', 'cube_I_CTC', 'median_I_CTC',
                        'cube_I_extended', 'median_I_extended']

    elif type == 'PI':
        keys_to_read = ['PI', 'PI_r2', 'PI_IPS', 'PI_IPS_r2',
                        'PI_CTC_IPS', 'PI_CTC_IPS_r2',
                        'PI_extended', 'PI_r2_extended',
                        'PI_IPS_extended', 'PI_IPS_r2_extended']
    elif type == 'Q_phi':
        keys_to_read = ['Q_phi', 'Q_phi_r2',
                        'Q_phi_IPS', 'Q_phi_IPS_r2',
                        'Q_phi_CTC_IPS', 'Q_phi_CTC_IPS_r2',
                        'Q_phi_UpC_CTC_IPS', 'Q_phi_UpC_CTC_IPS_r2']
    elif type == 'U_phi':
        keys_to_read = ['U_phi', 'U_phi_r2',
                        'U_phi_IPS', 'U_phi_IPS_r2',
                        'U_phi_CTC_IPS', 'U_phi_CTC_IPS_r2',
                        'U_phi_UpC_CTC_IPS', 'U_phi_UpC_CTC_IPS_r2']

    hdu_list = fits.HDUList(hdu)

    for i, key in enumerate(frames.keys()):
        if (key in keys_to_read) and (frames[key] is not None):

            # Move the pixel-axis to the first axis
            im_to_save = frames[key]
            if (im_to_save.ndim == 3) or key.startswith('cube_'):
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

            # Append to the HDU list
            hdu_list.append(fits.ImageHDU(new_im_to_save, name=key,
                                          header=hdu.header)
                            )

    if len(hdu_list) > 1:
        # Save the HDU list if it is not empty
        hdu_list.writeto(Path(path_PDI, f'{type}.fits'),
                         output_verify='silentfix',
                         overwrite=True)

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
    StokesPara = af.assign_Stokes_parameters(path_beams_files_selected,
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

    if (not HWP_used):
        # De-rotate the frames if HWP was not used
        for i, pos_angle_i in enumerate(pos_angles):
            beams[i] = rotate_cube(beams[i], pos_angle_i, pad=True)

    beams = np.array(beams)

    xc, yc = (beams.shape[-1]-1)/2, (beams.shape[-2]-1)/2
    r, phi = af.r_phi(beams[0,0], xc, yc)
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
                                       (mask_beams.shape[-1]-1)/2,
                                       (mask_beams.shape[-2]-1)/2,
                                       disk_pos_angle,
                                       disk_inclination)
    r_deprojected = r_deprojected.astype(np.float32)
    r_deprojected = r_deprojected[mask_beams].flatten()

    # Create a header
    hdu = write_header(object_name, mask_beams)
    """
    # Create a header
    mask_beams_rotated = rotate_cube(mask_beams, pos_angles[0],
                                     pad=False, rotate_axes=(0,1))
    hdu = write_header(object_name, mask_beams_rotated)
    del mask_beams_rotated
    """

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
    = double_difference(ind_I, ind_QU, mask_beams, StokesPara)
    del ind_I, ind_QU

    if (Q_frames['cube_Q'] is not None) and (U_frames['cube_U'] is not None):

        # Dictionaries to store the final data products in
        PI_frames, Q_phi_frames, U_phi_frames = {}, {}, {}

        # Final data products without IP-corrections
        PI_frames['PI'], PI_frames['PI_r2'], \
        Q_phi_frames['Q_phi'], Q_phi_frames['Q_phi_r2'], \
        U_phi_frames['U_phi'], U_phi_frames['U_phi_r2'] \
        = final_Stokes_frames(Q_frames['median_Q'], U_frames['median_U'],
                              r_deprojected, phi)

        # Instrumental polarisation subtraction
        print_and_log('--- IP-subtraction (IPS) using annuli with unpolarised stellar signal')
        Q_frames['median_Q_IPS'], \
        U_frames['median_U_IPS'] \
        = IPS(r, spm, r_inner_IPS, r_outer_IPS,
              Q=Q_frames['cube_Q'],
              U=U_frames['cube_U'],
              I_Q=I_Q_frames['cube_I_Q'],
              I_U=I_U_frames['cube_I_U'],
              I=I_frames['cube_I'])

        # IP-subtracted final data products
        PI_frames['PI_IPS'], PI_frames['PI_IPS_r2'], \
        Q_phi_frames['Q_phi_IPS'], Q_phi_frames['Q_phi_IPS_r2'], \
        U_phi_frames['U_phi_IPS'], U_phi_frames['U_phi_IPS_r2'] \
        = final_Stokes_frames(Q_frames['median_Q_IPS'],
                              U_frames['median_U_IPS'],
                              r_deprojected, phi)

        if crosstalk_correction:

            # Fit for the reduced U-efficiency
            Q_frames, I_Q_frames, \
            U_frames, I_U_frames, \
            I_frames \
            = CTC(Q_frames, I_Q_frames, U_frames, I_U_frames,
                  I_frames, r, r_crosstalk)

            # Crosstalk-corrected + IP-subtracted Stokes Q and U parameters
            Q_frames['median_Q_CTC_IPS'], \
            U_frames['median_U_CTC_IPS'] \
            = IPS(r, spm, r_inner_IPS, r_outer_IPS,
                  Q=Q_frames['cube_Q'],
                  U=U_frames['cube_U_CTC'],
                  I_Q=I_Q_frames['cube_I_Q'],
                  I_U=I_U_frames['cube_I_U_CTC'],
                  I=I_frames['cube_I_CTC'])

            # Crosstalk-corrected final data products
            PI_frames['PI_CTC_IPS'], PI_frames['PI_CTC_IPS_r2'], \
            Q_phi_frames['Q_phi_CTC_IPS'], Q_phi_frames['Q_phi_CTC_IPS_r2'], \
            U_phi_frames['U_phi_CTC_IPS'], U_phi_frames['U_phi_CTC_IPS_r2'] \
            = final_Stokes_frames(Q_frames['median_Q_CTC_IPS'],
                                  U_frames['median_U_CTC_IPS'],
                                  r_deprojected, phi)

        if minimise_U_phi:

            # Minimise the signal in U_phi
            Q_phi_frames['Q_phi_UpC_CTC_IPS'], \
            Q_phi_frames['Q_phi_UpC_CTC_IPS_r2'], \
            U_phi_frames['U_phi_UpC_CTC_IPS'], \
            U_phi_frames['U_phi_UpC_CTC_IPS_r2'] \
            = UpC(Q_frames['median_Q_CTC_IPS'],
                  U_frames['median_U_CTC_IPS'],
                  r, r_crosstalk, r_deprojected, phi)

        if not HWP_used and Wollaston_used and \
            (Q_frames['cube_Q+'] is not None) and \
            (Q_frames['cube_Q-'] is not None) and \
            (U_frames['cube_U+'] is not None) and \
            (U_frames['cube_U-'] is not None):

            # Create extended images if the position angle was rotated
            Q_frames, I_Q_frames, U_frames, I_U_frames, I_frames, PI_frames \
            = extended_Stokes_frames(Q_frames, I_Q_frames,
                                     U_frames, I_U_frames,
                                     I_frames, PI_frames,
                                     spm, r_inner_IPS, r_outer_IPS, r,
                                     r_deprojected, phi)

        # Save the data frames
        save_PDI_frames('PI', PI_frames, mask_beams, HWP_used,
                        pos_angles[0], hdu, path_PDI)
        save_PDI_frames('Q_phi', Q_phi_frames, mask_beams, HWP_used,
                        pos_angles[0], hdu, path_PDI)
        save_PDI_frames('U_phi', U_phi_frames, mask_beams, HWP_used,
                        pos_angles[0], hdu, path_PDI)

    print_and_log('--- Saving the final data products')

    # Save the data frames
    for sign in ['', '+', '-']:
        if (Q_frames[f'cube_Q{sign}'] is not None):
            save_PDI_frames(f'Q{sign}', Q_frames, mask_beams, HWP_used,
                            pos_angles[0], hdu, path_PDI)
            save_PDI_frames(f'I_Q{sign}', I_Q_frames, mask_beams, HWP_used,
                            pos_angles[0], hdu, path_PDI)

        if (U_frames[f'cube_U{sign}'] is not None):
            save_PDI_frames(f'U{sign}', U_frames, mask_beams, HWP_used,
                            pos_angles[0], hdu, path_PDI)
            save_PDI_frames(f'I_U{sign}', I_U_frames, mask_beams, HWP_used,
                            pos_angles[0], hdu, path_PDI)

    if (Q_frames['cube_Q'] is not None) and (U_frames['cube_U'] is not None):
        save_PDI_frames('I', I_frames, mask_beams, HWP_used,
                        pos_angles[0], hdu, path_PDI)

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
                    'ESO INS OPTI4 ID':[],                    # Wiregrid
                    'ESO INS OPTI7 ID':[],                    # Detector
                    'ESO DET WIN NX':[], 'ESO DET WIN NY':[], # Window shape
                    'ESO DET WIN STARTX':[], 'ESO DET WIN STARTY':[]
                    }
    for x in path_SCIENCE_files:
        for key in dict_headers.keys():
            dict_headers[key].append(af.read_from_FITS_header(x, key))

    # Unique combination of HWP/camera/window shape
    OBS_configs = np.array(list(dict_headers.values())).T

    for OBS_config_i in np.unique(OBS_configs, axis=0):

        # Usage of HWP, Wollaston prism, wiregrid and camera
        HWP_used       = (OBS_config_i[:2]=='Half_Wave_Plate').any()
        Wollaston_used = (np.ma.mask_or(OBS_config_i[:3]=='Wollaston_00',
                                        OBS_config_i[:3]=='Wollaston_45')
                         ).any()
        camera_used    = OBS_config_i[3]

        Wollaston_45 = False
        if Wollaston_used:
            Wollaston_45 = (OBS_config_i[:3]=='Wollaston_45').any()

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

def download_data(path_cwd):

    import urllib, os

    path_request_file = list(path_cwd.glob('*.txt'))[0]
    request_file = open(path_request_file)
    
    content = request_file.readlines()

    FLATs, DARKs, SCIENCEs = [], [], []
    for file_name_i, file_i in zip(content[21::2], content[22::2]):
        
        file_name_i = file_name_i.replace(' \n', '')

        if ('CAL_FLAT_LAMP' in file_i) or ('CAL_FLAT_POL' in file_i):
            FLATs.append(file_name_i)
        elif 'CAL_DARK' in file_i:
            DARKs.append(file_name_i)
        elif 'Raw data for which no processed data is available' in file_i:
            SCIENCEs.append(file_name_i)
    
    all_files = np.concatenate((FLATs, DARKs, SCIENCEs))

    # Create FLAT and DARK directories
    path_FLAT_dir = Path(path_cwd, 'FLATs')
    path_DARK_dir = Path(path_cwd, 'DARKs')

    if not path_FLAT_dir.is_dir():
        path_FLAT_dir.mkdir()
    if not path_DARK_dir.is_dir():
        path_DARK_dir.mkdir()

    # Download the data
    url = 'https://dataportal.eso.org/dataportal_new/file/{}'
    for file_name_i in tqdm(all_files, bar_format=pbar_format):

        target_dir = path_cwd
        if file_name_i in FLATs:
            target_dir = path_FLAT_dir
        elif file_name_i in DARKs:
            target_dir = path_DARK_dir

        target_file_name_i = Path(target_dir, f'{file_name_i}.fits.Z')

        # Ignore any files that were already downloaded
        if target_file_name_i.is_file():
            continue
        if Path(target_dir, f'{file_name_i}.fits').is_file():
            continue

        # Download the requested file
        urllib.request.urlretrieve(url.format(file_name_i), target_file_name_i)
    
    # Extract the files
    os.system('gunzip *fits.Z')
    os.system('gunzip */*fits.Z')