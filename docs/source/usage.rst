
Usage instructions
==================

- :ref:`Running PIPPIN`
- :ref:`Pre-processing`
- :ref:`Polarimetric Differential Imaging`
- :ref:`Final data products`

Running PIPPIN
--------------
PIPPIN requires the raw SCIENCE FITS-files to be uncompressed and stored in a directory (e.g. :file:`example_HD_135344B/`). In the same directory, PIPPIN will search for a configuration file named :file:`config.conf`. A :ref:`configuration file <Configuration file>` with default parameters can be created if PIPPIN cannot find it.

To calibrate the SCIENCE images, PIPPIN makes use of master FLATs, master BPMs (Bad-Pixel Masks), and master DARKs. These master calibration files can be created by executing: 
::
   pippin_naco --prepare_calib_files --path_FLAT_dir /path/to/FLATs/ --path_DARK_dir /path/to/DARKs/ --path_BPM_dir /path/to/master_BPM/

PIPPIN will median-combine the FLATs and DARKs of each observation type and save the master calibration files in a sub-directory (e.g. :file:`/path/to/FLATs/master_FLAT/`). Master bad-pixel masks (BPMs) are generated using the (non)-linear pixel response between FLAT observations with the FLAT-lamp on or off. The BPMs are stored in the :file:`/path/to/master_BPM/` directory which will be created if it does not exist yet.

To run the pipeline from the terminal, navigate to the directory where the raw SCIENCE images are stored (e.g. :file:`cd /example_HD_135344B/`) and run:
::
   pippin_naco --run_pipeline --path_FLAT_dir /path/to/master_FLAT/ --path_DARK_dir /path/to/master_DARK/ --path_BPM_dir /path/to/master_BPM/

Alternatively, to create the master calibration files and run the reduction, one can execute:
::
   pippin_naco --run_pipeline --prepare_calib_files --path_FLAT_dir /path/to/FLATs/ --path_DARK_dir /path/to/DARKs/ --path_BPM_dir /path/to/master_BPM/

The data reduction is separated into a pre-processing and PDI part.

Pre-processing
--------------

Separating observation-types
^^^^^^^^^^^^^^^^^^^^^^^^^^^^
First, PIPPIN separates the raw SCIENCE data into different observation types using information from the headers. Observations are categorised by:

- HWP/rotator-usage
- Wollaston/wiregrid-usage (`Wollaston_00`, `Wollaston_45`, `wiregrid`)
- Detector (e.g. `S13`, `S27`, `L27`)
- Window-shape
- Exposure time
- Filter
- Observing block ID (if ``split_observing_blocks = True``; :ref:`config-file <Configuration file>`).

The results of the data reduction are stored in sub-directories of the generated :file:`pipeline_output/` directory. A log-file ``log.txt`` is created, storing information on the used reduction methods.

The pipeline continues by DARK-subtracting and FLAT-normalising the observations. The bad pixels are replaced by the median of a surrounding box of 5x5 pixels, excluding any other bad pixels.

Beam-centre fitting
^^^^^^^^^^^^^^^^^^^
Next, PIPPIN locates the centres of the ordinary and extra-ordinary beams. PIPPIN provides several methods for fitting the beam-centres (``centering_method``; :ref:`config-file <Configuration file>`):

- ``maximum``: The maximum pixel in an image median-filtered with a 3x3 kernel.
- ``single-Moffat``: A single 2D Moffat function (for each beam).
- ``double-Moffat``: Two 2D Moffat functions (for each beam) subtracted from each other to replicate the flat top of a saturated beam.

The two Moffat fitting methods allow the beam-offset to be tied, based on the expected pixel-separation with the utilised detector (``tied_offset = True``; :ref:`config-file <Configuration file>`). The tied offset can be useful in cases where the stellar light does not form a point source (e.g. for embedded stars).

.. note::
   Only one beam is identified if the data consists of wiregrid-observations instead of Wollaston-observations.

Sky-subtraction
^^^^^^^^^^^^^^^
The sky-subtraction can be performed with one of the following methods (``sky_subtraction_method``; :ref:`config-file <Configuration file>`):

- ``box-median``: The sky-signal is estimated from the median signal of pixels which are at least ``sky_subtraction_min_offset`` to the left or right of the assessed beam centres.
- ``dithering-offset``: Observations with different dithering positions are subtracted from each other. The two observations must be separated by ``sky_subtraction_min_offset``, otherwise the ``box_median`` method is used.

A gradient can remain in the sky-subtracted images. PIPPIN corrects for this with a linear fit to rows of pixels. If ``remove_horizontal_stripes = False`` (:ref:`config-file <Configuration file>`), 5 rows will be binned and the final gradient image will be smoothed and subtracted. In some datasets, horizontal stripes are still present after the sky-subtraction. These can be removed with a more aggressive fitting of each row, using ``remove_horizontal_stripes = True``.

Cropping and saving
^^^^^^^^^^^^^^^^^^^
The ordinary and extra-ordinary beams are cropped and saved as FITS-files, employing the ``size_to_crop`` parameter (:ref:`config-file <Configuration file>`). Any temporary data products ``*_reduced.fits`` and ``*_skysub.fits`` are removed if ``remove_data_products = True`` (:ref:`config-file <Configuration file>`). Open AO-loop observations are identified with an iterative sigma-clipping and the file-names are stored in ``open_loop_files.txt``. The :file:`plots/` directory stores a figure of this assessment in addition to figures of the reduction steps.

.. note::
   The ``size_to_crop`` parameter is extended to a shape with odd axes-lengths (if even lengths are given) so that the final images are centred on a single pixel. 

Polarimetric Differential Imaging
---------------------------------
The PDI part of PIPPIN begins by removing any incomplete HWP cycles and open AO-loop observations.

Instrumental polarisation
^^^^^^^^^^^^^^^^^^^^^^^^^
A number of instrumental polarisation (IP) corrections are performed. The ordinary and extra-ordinary beams are read into memory and their fluxes are equalised (per observation) using the method outlined by `Avenhaus et al. (2014) <https://ui.adsabs.harvard.edu/abs/2014ApJ...781...87A/abstract>`_ in the appendix. The stellar flux is assumed to be un-polarised and the annuli provided in the :ref:`config-file <Configuration file>` (``r_inner_IPS``, ``r_outer_IPS``) are employed to assess the stellar flux outside of the saturated core of the PSF.

.. note::
   If the rotator was used to record different Stokes parameters, the beams are de-rotated when read into memory.

Per observation, the intensity and Stokes parameter are obtained by summing and subtracting the (extra)-ordinary beams, respectively. Next, the double-difference method is applied with the redundant observations (i.e. :math:`Q^+`/:math:`Q^-` and :math:`U^+`/:math:`U^-`).

.. note::
   If the double-difference method cannot be applied (e.g. due to observations of :math:`Q^+` without :math:`Q^-`), PIPPIN simply uses the available observations as the IP-corrected observation (e.g. :math:`Q=Q^+` instead of :math:`Q=\frac{1}{2}(Q^+-Q^-)`)
   
Using the annuli described with the ``r_inner_IPS`` and ``r_outer_IPS`` parameters, PIPPIN corrects for the polarisation found near the stellar PSF. Any polarised signal found near the centre is believed to originate from IP because the stellar signal is assumed to be un-polarised. This correction is performed for each HWP-cycle, thus avoiding temporal differences in the instrument configuration and IP.

If ``crosstalk_correction = True`` (:ref:`config-file <Configuration file>`), PIPPIN evaluates and corrects for the reduced efficiency of the Stokes :math:`U` parameter which originates from crosstalk between the components of the Stokes vector. Following `Avenhaus et al. (2014) <https://ui.adsabs.harvard.edu/abs/2014ApJ...781...87A/abstract>`_, an annulus is used to reduce the number of pixels where a higher signal is found in :math:`Q` compared to :math:`U`. The ``r_crosstalk`` parameter in the :ref:`config-file <Configuration file>` provides the inner and outer radii of this annulus.

.. attention::
   The crosstalk correction is made to function for datasets with clear disk signal and a roughly axisymmetric distribution. If this is not the case, we do not recommend setting ``crosstalk_correction = True``. 

Finally, the :math:`U_\phi`-signal in the ``r_crosstalk`` annulus is minimised if requested (``minimise_U_phi = True``; :ref:`config-file <Configuration file>`). As described by `Avenhaus et al. (2014) <https://ui.adsabs.harvard.edu/abs/2014ApJ...781...87A/abstract>`_, an offset-angle can be estimated for the azimuthal Stokes parameters :math:`Q_\phi` and :math:`U_\phi`.

.. attention::
   Minimising the :math:`U_\phi`-signal should be done with caution as it can lead to the removal of real :math:`Q_\phi`-signal in high-inclination disks or in cases where crosstalk and IP have not been sufficiently corrected.

Final data products
-------------------
Depending on the observations, the :file:`PDI/` directory contains the following files:

Total intensities:

- :file:`I.fits`: Total intensity :math:`I = \frac{1}{2}(I_Q + I_U)`. [``CUBE_I,`` ``MEDIAN_I``, ``CUBE_I_CTC,`` ``MEDIAN_I_CTC``]
- :file:`I_Q.fits`: Intensity as :math:`I_Q = \frac{1}{2}(I_{Q^+} + I_{Q^-})`. [``CUBE_I_Q,`` ``MEDIAN_I_Q``]
- :file:`I_Q+.fits`: Intensity of :math:`Q^+` measurements. [``CUBE_I_Q+``, ``MEDIAN_I_Q+``]
- :file:`I_Q-.fits`: Intensity of :math:`Q^-` measurements. [``CUBE_I_Q-``, ``MEDIAN_I_Q-``]
- :file:`I_U.fits`: Intensity as :math:`I_U = \frac{1}{2}(I_{U^+} + I_{U^-})`. [``CUBE_I_U``, ``MEDIAN_I_U``, ``CUBE_I_U_CTC``, ``MEDIAN_I_U_CTC``]
- :file:`I_U+.fits`: Intensity of :math:`U^+` measurements. [``CUBE_I_U+``, ``MEDIAN_I_U+``, ``CUBE_I_U+_CTC``, ``MEDIAN_I_U+_CTC``]
- :file:`I_U-.fits`: Intensity of :math:`U^-` measurements. [``CUBE_I_U-``, ``MEDIAN_I_U-``, ``CUBE_I_U-_CTC``, ``MEDIAN_I_U-_CTC``]

Stokes parameters:

- :file:`Q.fits`: Stokes :math:`Q = \frac{1}{2}(Q^+ - Q^-)`. [``CUBE_Q``, ``MEDIAN_Q``, ``MEDIAN_Q_IPS``, ``MEDIAN_Q_CTC_IPS``]
- :file:`Q+.fits`: Stokes :math:`Q^+ = I_\mathrm{ord} - I_\mathrm{ext}`. [``CUBE_Q+``, ``MEDIAN_Q+``]
- :file:`Q-.fits`: Stokes :math:`Q^- = I_\mathrm{ord} - I_\mathrm{ext}`. [``CUBE_Q-``, ``MEDIAN_Q-``]
- :file:`U.fits`: Stokes :math:`U = \frac{1}{2}(U^+ - U^-)`. [``CUBE_U``, ``MEDIAN_U``, ``MEDIAN_U_IPS``, ``MEDIAN_U_CTC_IPS``]
- :file:`U+.fits`: Stokes :math:`U^+ = I_\mathrm{ord} - I_\mathrm{ext}`. [``CUBE_U+``, ``MEDIAN_U+``]
- :file:`U-.fits`: Stokes :math:`U^- = I_\mathrm{ord} - I_\mathrm{ext}`. [``CUBE_U-``, ``MEDIAN_U-``]

Final polarised light products:

- :file:`PI.fits`: Polarised intensity :math:`PI = \sqrt{Q^2 + U^2}`. [``PI``, ``PI_R2``, ``PI_IPS``, ``PI_IPS_R2``, ``PI_CTC_IPS``, ``PI_CTC_IPS_R2``]
- :file:`Q_phi.fits`: Azimuthal Stokes parameter :math:`Q_\phi = - Q \cos 2\phi - U \sin 2\phi`. [``Q_PHI``, ``Q_PHI_R2``, ``Q_PHI_IPS``, ``Q_PHI_IPS_R2``, ``Q_PHI_CTC_IPS``, ``Q_PHI_CTC_IPS_R2``, ``Q_PHI_UPC_CTC_IPS``, ``Q_PHI_UPC_CTC_IPS_R2``]
- :file:`U_phi.fits`: Azimuthal Stokes parameter :math:`U_\phi = + Q \sin 2\phi - U \cos 2\phi`. [``U_PHI``, ``U_PHI_R2``, ``U_PHI_IPS``, ``U_PHI_IPS_R2``, ``U_PHI_CTC_IPS``, ``U_PHI_CTC_IPS_R2``, ``U_PHI_UPC_CTC_IPS``, ``U_PHI_UPC_CTC_IPS_R2``]

The values given in brackets [] are the fits extensions. ``CUBE_`` extensions are given for each cycle, while ``MEDIAN_`` indicates the median-combination of the ``CUBE_`` images. The extension ``_IPS`` gives the instrumental-polarisation subtracted result, ``_CTC`` gives the crosstalk-corrected image, and ``_UPC`` gives the :math:`U_\phi`-minimised result. ``_R2``-products are scaled by the squared projected radius. 

.. note::
   ``_EXTENDED`` data products are generated if the position angle was rotated, rather than the half-wave plate, to obtain the linear Stokes parameters. The position-angle rotation permits a broader coverage of the sky, which results in images with the characteristic shape of an eight-pointed star. 
