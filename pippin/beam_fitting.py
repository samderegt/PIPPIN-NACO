import numpy as np

from scipy import ndimage

from astropy.io import fits
from astropy.modeling import models, fitting

from tqdm import tqdm
from pathlib import Path

import pippin.auxiliary_functions as af

# Setting the length of progress bars
pbar_format = '{l_bar}{bar:20}{r_bar}{bar:-20b}'

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
        Moffat_y_offset = af.Wollaston_beam_separation(camera_used)

        # Apply a long, horizontal filter to approximate polarimetric mask
        box = np.ones((1,50))
        mask_approx = ndimage.median_filter(im, footprint=box, mode='constant')

        # Apply the minimum filter
        box = np.zeros((Moffat_y_offset,3))
        box[:+3,:] = 1
        box[-3:,:] = 1

        filtered_im = ndimage.minimum_filter(im - mask_approx,
                                             footprint=box, mode='constant')

        # x and y values of each pixel
        y_idx, x_idx = np.unravel_index(
            np.argmax(filtered_im[Moffat_y_offset//2:-Moffat_y_offset//2]),
            filtered_im.shape
            )
        x0_center = xp[Moffat_y_offset//2:-Moffat_y_offset//2][y_idx, x_idx]
        y0_center = yp[Moffat_y_offset//2:-Moffat_y_offset//2][y_idx, x_idx]

        x0_1, x0_2 = x0_center, x0_center
        y0_1 = y0_center + Moffat_y_offset/2
        y0_2 = y0_center - Moffat_y_offset/2

        extent = (xp.min()-0.5, xp.max()-0.5, yp.max()-0.5, yp.min()-0.5)
        '''
        fig, ax = plt.subplots(figsize=(15,5), ncols=3, sharex=True,
                               sharey=True)
        ax[0].imshow(im, aspect='auto', interpolation='none', extent=extent)
        ax[1].imshow(im-mask_approx, aspect='auto', interpolation='none',
                     extent=extent)
        ax[2].imshow(filtered_im, aspect='auto', interpolation='none',
                     extent=extent)

        for ax_i in ax:
            ax_i.scatter([x0_1, x0_2], [y0_1, y0_2], marker='o',
                         facecolor='none', edgecolor='r', s=50)
        plt.show()
        '''

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

def fit_maximum(Wollaston_used, 
                camera_used, 
                filter_used, 
                path_reduced_files_selected
                ):
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
        (x0_1, y0_1), (x0_2, y0_2) = fit_initial_guess(
            (cube-min_cube)[0], xp, yp, Wollaston_used, camera_used, filter_used
            )

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
    Moffat_y_offset = af.Wollaston_beam_separation(camera_used, filter_used)

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

    # Set the separation based on the used camera
    Moffat_y_offset = af.Wollaston_beam_separation(camera_used, filter_used)

    # Functions to constrain the model
    def tie_to_x_0_1(model):
        return model.x_0_1
    def tie_to_y_0_1_offset(model):
        return model.y_0_1 + Moffat_y_offset

    def tie_to_amp_1(model):
        return model.amplitude_1
    def tie_to_gamma_1(model):
        return model.gamma_1

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
                                                'alpha':(0,10)},
                                        fixed={'alpha':True, 'gamma':True},
                                       )

    # Ordinary beam
    if tied_offset:
        # Tie the x and y coordinates, amplitudes, and gamma
        tied_1 = {'x_0':tie_to_x_0_1, 'y_0':tie_to_y_0_1_offset,
                  }#'amplitude':tie_to_amp_0, 'gamma':tie_to_gamma_0}

        single_Moffat_ord = models.Moffat2D(x_0=x0_ext,
                                            y_0=y0_ext+Moffat_y_offset,
                                            amplitude=15000, gamma=3, alpha=1,
                                            bounds={'x_0':(x_min, x_max),
                                                    'y_0':(y_min_1, y_max_1),
                                                    'amplitude':(1,40000),
                                                    'gamma':(0.1,30),
                                                    'alpha':(0,10)},
                                            fixed={'alpha':True, 'gamma':True},
                                            tied=tied_1
                                           )
    else:
        single_Moffat_ord = models.Moffat2D(x_0=x0_ext,
                                            y_0=y0_ext+Moffat_y_offset,
                                            amplitude=15000, gamma=3, alpha=1,
                                            bounds={'x_0':(x_min, x_max),
                                                    'y_0':(y_min_1, y_max_1),
                                                    'amplitude':(1,40000),
                                                    'gamma':(0.1,30),
                                                    'alpha':(0,10)},
                                            fixed={'alpha':True, 'gamma':True},
                                           )

    # Combine the two beams into a single model
    complete_model = single_Moffat_ord
    if not np.isnan(x0_ext) and not np.isnan(y0_ext):
        complete_model += single_Moffat_ext

    # Fit the model to the image
    LevMar_fitter = fitting.LevMarLSQFitter()
    fitted = LevMar_fitter(complete_model, xp, yp, im, maxiter=10000, acc=1e-12)

    # x- and y-coordinates of the beam centers
    if np.isnan(x0_ext) and np.isnan(y0_ext):
        x_ord, y_ord = fitted.parameters[1:3]
        x_ext, y_ext = x0_ext, y0_ext
    else:
        x_ord, y_ord = fitted[0].parameters[1:3]
        x_ext, y_ext = fitted[1].parameters[1:3]

    return np.array([[x_ext, y_ext],
                     [x_ord, y_ord]])

def fit_beam_centers_Moffat(method, 
                            Wollaston_used, 
                            Wollaston_45,
                            camera_used, 
                            filter_used, 
                            tied_offset, 
                            path_reduced_files_selected, 
                            ):
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
        (x0_ord, y0_ord), (x0_ext, y0_ext) = fit_initial_guess(
            (cube-min_cube)[0], xp, yp, Wollaston_used, camera_used, filter_used
            )

        # Set the background to 0
        cube -= np.nanmedian(cube, axis=(1,2), keepdims=True)

        # Fit each frame in the cube
        Moffat_PSF.append(np.ones((len(cube),2,2))*np.nan)
        for j, im in enumerate(cube):

            # Apply a median filter
            im = ndimage.median_filter(im, size=(3,3))

            # Fit the (extra)-ordinary beams
            if method=='double-Moffat':
                # Fit a double-Moffat function to find the location of the beam
                Moffat_PSF[-1][j,:,:] = fit_double_Moffat(
                    im, xp, yp, x0_ext, y0_ext, camera_used, filter_used, tied_offset
                    )
            elif method=='single-Moffat':
                # Fit a single-Moffat
                Moffat_PSF[-1][j,:,:] = fit_single_Moffat(
                    im, xp, yp, x0_ord, y0_ord, x0_ext, y0_ext, 
                    camera_used, filter_used, tied_offset
                    )

            # Sort the PSF locations so that ordinary beam comes first
            idx_argsort = np.argsort(Moffat_PSF[-1][j,:,1])
            Moffat_PSF[-1][j,:,:] = Moffat_PSF[-1][j,idx_argsort,:]

    return Moffat_PSF

def fit_beam_centers(method, 
                     Wollaston_used, 
                     Wollaston_45,
                     camera_used, 
                     filter_used, 
                     tied_offset, 
                     path_reduced_files_selected, 
                     ):
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

    if method=='single-Moffat' or method=='double-Moffat':
        # Fit a 2D Moffat function
        beam_centers = fit_beam_centers_Moffat(
            method, Wollaston_used, Wollaston_45, camera_used,
            filter_used, tied_offset, path_reduced_files_selected
            )

    elif method=='maximum':
        # Find 2 maxima in the images
        beam_centers = fit_maximum(
            Wollaston_used, camera_used, filter_used, 
            path_reduced_files_selected
            )

    return beam_centers


def center_beams(beam_centers, 
                 size_to_crop, 
                 Wollaston_used, 
                 Wollaston_45, 
                 path_skysub_files_selected
                 ):
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
        af.write_FITS_file(file_beams, beams_i, header=header)
        path_beams_files_selected.append(file_beams)

    path_beams_files_selected = np.sort(path_beams_files_selected)

    return beams, path_beams_files_selected