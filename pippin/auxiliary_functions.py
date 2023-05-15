import numpy as np
from astropy.io import fits

################################################################################
# Auxiliary functions
################################################################################

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
    #phi = np.arctan((yp-yc)/(xc-xp))

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
    pos_angles = np.array([read_from_FITS_header(x, 'ESO ADA POSANG') 
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

        wiregrids = np.array([read_from_FITS_header(x, 'ESO INS OPTI4 ID') 
                              for x in files])

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
