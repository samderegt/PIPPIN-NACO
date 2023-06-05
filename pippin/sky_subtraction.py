import numpy as np

from scipy import ndimage

from astropy.io import fits
from astropy.stats import sigma_clip
from astropy.modeling import models, fitting

from tqdm import tqdm

from pathlib import Path

import pippin.auxiliary_functions as af

# Setting the length of progress bars
pbar_format = '{l_bar}{bar:20}{r_bar}{bar:-20b}'

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
    if isinstance(low, np.ndarray) and isinstance(high, np.ndarray):
        mask_sources = (im > low[:,None]) & (im < high[:,None])
    elif isinstance(low, np.ndarray):
        mask_sources = (im > low[:,None]) & (im < high)
    elif isinstance(high, np.ndarray):
        mask_sources = (im > low) & (im < high[:,None])
    else:
        mask_sources = (im > low) & (im < high)

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

def sky_subtraction_box_median(files, 
                               beam_centers, 
                               min_offset,
                               remove_horizontal_stripes, 
                               path_skysub_files_selected
                               ):
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
            af.write_FITS_file(file_skysub, cube, header=header)

    return path_skysub_files_selected

def sky_subtraction_dithering(beam_centers, 
                              min_offset, 
                              HWP_used,
                              Wollaston_used, 
                              remove_horizontal_stripes, 
                              path_reduced_files_selected, 
                              path_skysub_files_selected, 
                              ):
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

    StokesPara = af.assign_Stokes_parameters(path_reduced_files_selected,
                                          HWP_used, Wollaston_used)
    for i in range(len(StokesPara)):
        StokesPara[i] = StokesPara[i].replace('-', '')
        StokesPara[i] = StokesPara[i].replace('+', '')

    offsets = []
    for i in range(len(path_reduced_files_selected)):
        offsets.append(np.nanmean(beam_centers[i][:,:,:], axis=(0,1)))
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
        mask_sufficient_offset = (np.sqrt(np.sum((offsets-offsets[i])**2,
                                                 axis=1)) >= new_min_offset)

        while not np.any(mask_sufficient_offset):
            # Decrease the offset and try again
            new_min_offset -= 5
            mask_sufficient_offset = (np.sqrt(np.sum((offsets-offsets[i])**2,
                                                     axis=1)) >= new_min_offset)

            if (new_min_offset < 60):
                path_skysub_files_selected = sky_subtraction_box_median(
                    [file], np.array([beam_centers[i]]), min_offset, 
                    remove_horizontal_stripes, path_skysub_files_selected
                    )
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

        #"""
        # Remove any leftover background signal with linear fits
        for j in range(len(cube)):
            y_j = beam_centers[i][j,:,1]
            cube[j] -= sky_background_fit(cube[j], offsets[i,0],
                                          offsets[idx_next_offset,0],
                                          min_offset, y_j,
                                          remove_horizontal_stripes)
        #"""
        # Add filename
        file_skysub = Path(str(file).replace('_reduced.fits', '_skysub.fits'))
        path_skysub_files_selected.append(file_skysub)

        # Save the sky-subtracted image to a FITS file
        af.write_FITS_file(file_skysub, cube, header=header)

    '''
    if (new_min_offset < 35):
        print_and_log('    Sky-subtraction not possible with method \'dithering-offset\', used method \'box-median\'')

    if (new_min_offset!=min_offset):
        print_and_log(f'    sky_subtraction_min_offset too high, reduced to {new_min_offset} pixels')
    '''

    return path_skysub_files_selected

def sky_subtraction(method, min_offset, beam_centers, HWP_used,
                    Wollaston_used, remove_horizontal_stripes, 
                    path_reduced_files_selected
                    ):
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

    path_skysub_files_selected = []

    if method=='dithering-offset':
        path_skysub_files_selected = sky_subtraction_dithering(
            beam_centers, min_offset, HWP_used, Wollaston_used, 
            remove_horizontal_stripes, path_reduced_files_selected, 
            path_skysub_files_selected
            )

    elif method=='box-median':
        path_skysub_files_selected = sky_subtraction_box_median(
            path_reduced_files_selected, beam_centers, min_offset, 
            remove_horizontal_stripes, path_skysub_files_selected
            )

    path_skysub_files_selected = np.sort(path_skysub_files_selected)

    return path_skysub_files_selected