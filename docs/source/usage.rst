
Usage instructions
==================

- :ref:`Running PIPPIN`
- :ref:`Pre-processing`
- :ref:`Polarimetric Differential Imaging`
- :ref:`Instrument configurations`

Running PIPPIN
--------------
PIPPIN requires the raw SCIENCE FITS-files to be uncompressed and stored in a directory (e.g. :file:`example_HD_135344B/`). In the same directory, PIPPIN will search for a configuration file named :file:`config.conf`. A :ref:`configuration file <Configuration file>` with default parameters can be created if PIPPIN cannot find it.

To calibrate the SCIENCE images, PIPPIN makes use of master FLATs, BPMs (Bad-Pixel Masks), and DARKs. On our `GitHub repository <https://github.com/samderegt/PIPPIN-NACO/tree/master/pippin/data>`_, we have made some calibration files available, covering different epochs. When running the pipeline from a terminal, one should add the paths to these master calibration files as:
::

   pippin --path_FLAT_dir /path/to/master_FLAT/ --path_DARK_dir /path/to/master_DARK/ --path_BPM_dir /path/to/master_BPM/

To create your own master calibration files, one should run:
::

   pippin --prepare_calib_files --path_FLAT_dir /path/to/FLATs/ --path_DARK_dir /path/to/DARKs/ --path_BPM_dir /path/to/master_BPM/

PIPPIN will median-combine the FLATs and DARKs of each observation type and save the master calibration files in a sub-directory (e.g. :file:`/path/to/FLATs/master_FLAT/`). Master bad-pixel masks (BPMs) are generated using the (non)-linear pixel response between FLAT observations with the FLAT-lamp on or off. The BPMs are stored in the :file:`/path/to/master_BPM/` directory which will be created if it does not exist yet.

To run the pipeline from the terminal, navigate to the directory where the raw SCIENCE images are stored (e.g. :file:`cd /example_HD_135344B/`) and run:
::

   pippin --run_pipeline --path_FLAT_dir /path/to/master_FLAT/ --path_DARK_dir /path/to/master_DARK/ --path_BPM_dir /path/to/master_BPM/

or

::

   pippin --run_pipeline --prepare_calib_files --path_FLAT_dir /path/to/FLATs/ --path_DARK_dir /path/to/DARKs/ --path_BPM_dir /path/to/master_BPM/

The data reduction is separated into a pre-processing and PDI part.

Pre-processing
--------------

Separating observation-types
^^^^^^^^^^^^^^^^^^^^^^^^^^^^
First, PIPPIN separates the raw SCIENCE data into different observation types with information from the headers. Observations are categorised by:

- HWP/rotator-usage
- Wollaston-usage (`Wollaston_00`, `Wollaston_45`, `wiregrid`)
- Detector (e.g. `S13`, `S27`, `L27`)
- Window shape
- Exposure time
- Filter
- Observing block ID (if ``split_observing_blocks = True``; :ref:`config-file <Configuration file>`).

The results of the data reduction are stored in sub-directories of the generated :file:`pipeline_output/` directory. A log-file ``log.txt`` is created, storing information on the used reduction methods.

The pipeline continues by DARK-subtracting and FLAT-normalising the observations. The bad pixels are replaced by the median of the surrounding box of 5x5 pixels, excluding any other bad pixels.

Beam-centre fitting
^^^^^^^^^^^^^^^^^^^
Next, PIPPIN locates the centres of the ordinary and extra-ordinary beams. PIPPIN provides several methods for fitting the beam-centres (``centering_method``; :ref:`config-file <Configuration file>`):

- ``maximum``: The maximum pixel in an image median-filtered with a 3x3 kernel.
- ``single-Moffat``: A single 2D Moffat function (for each beam).
- ``double-Moffat``: Two 2D Moffat functions (for each beam) subtracted from each other to replicate the flat top of a saturated beam.

The two Moffat fitting methods allow the beam-offset to be tied, based on the expected pixel-separation with the utilised detector (``tied_offset = True``; :ref:`config-file <Configuration file>`). The tied offset is useful when the stellar light does not form a point source (e.g. for embedded stars).

.. note::
   Only one beam is identified if the data consists of wiregrid-observations instead of Wollaston-observations.

Sky-subtraction
^^^^^^^^^^^^^^^
The sky-subtraction can be performed with one of the following methods (``sky_subtraction_method``; :ref:`config-file <Configuration file>`):

- ``box-median``: The sky-signal is estimated from the median signal of pixels which are at least ``sky_subtraction_min_offset`` to the left or right of the assessed beam centres.
- ``dithering-offset``: Observations with different dithering positions are subtracted from each other. The two observations must be separated by ``sky_subtraction_min_offset``, otherwise the ``box_median`` method is utilised.

A gradient can remain in the sky-subtracted images. PIPPIN corrects for this with a linear fit to rows of pixels. If ``remove_horizontal_stripes = False`` (:ref:`config-file <Configuration file>`), 5 rows will be binned and the final gradient image will be smoothed and subtracted. A read-out artefact can leave behind horizontal stripes which can be removed with a more aggressive fitting of each row, using ``remove_horizontal_stripes = True``.

Cropping and saving
^^^^^^^^^^^^^^^^^^^
The ordinary and extra-ordinary beams are cropped and saved as FITS-files, employing the ``size_to_crop`` parameter (:ref:`config-file <Configuration file>`). Any temporary data products ``*_reduced.fits`` and ``*_skysub.fits`` are removed if ``remove_data_products = True`` (:ref:`config-file <Configuration file>`). Open AO-loop observations are identified with an iterative sigma-clipping and the file-names are stored in ``open_loop_files.txt``. The :file:`plots/` directory stores a figure of this assessment in addition to figures of the reduction steps.


Polarimetric Differential Imaging
---------------------------------
The PDI part of PIPPIN begins by removing any incomplete HWP cycles and open AO-loop observations.

Instrumental polarisation
^^^^^^^^^^^^^^^^^^^^^^^^^
A number of instrumental polarisation (IP) corrections are performed. The ordinary and extra-ordinary beams are read into memory and their fluxes are equalised (per observation) using the method outlined by `Avenhaus et al. (2014) <https://ui.adsabs.harvard.edu/abs/2014ApJ...781...87A/abstract>`_ in the appendix. The stellar flux is assumed to be unpolarised and the annuli provided in the :ref:`config-file <Configuration file>` (``r_inner_IPS``, ``r_outer_IPS``) are employed to assess the stellar flux outside of the saturated core of the PSF.

.. note::
   If the rotator was used to record different Stokes parameters, the beams are de-rotated when read into memory.

Per observation, the intensity and Stokes parameter are obtained by summing and subtracting the (extra)-ordinary beams, respectively. Next, the double-difference method is applied with the redundant observations (i.e. :math:`Q^+`/:math:`Q^-` and :math:`U^+`/:math:`U^-`).

.. note::
   If the double-difference method cannot be applied (e.g. due to observations of :math:`Q^+` without :math:`Q^-`), PIPPIN simply uses the available observations as the IP-corrected observation (e.g. :math:`Q=Q^+` instead of :math:`Q=\frac{1}{2}(Q^+-Q^-)`)

If ``crosstalk_correction = True`` (:ref:`config-file <Configuration file>`), PIPPIN evaluates and correct for the reduced efficiency of the Stokes :math:`U` parameter which originates from crosstalk between the components of the Stokes vector. Following `Avenhaus et al. (2014) <https://ui.adsabs.harvard.edu/abs/2014ApJ...781...87A/abstract>`_, an annulus is used to minimise the number of pixels where a higher signal in :math:`Q` compared to :math:`U`. The ``r_crosstalk`` parameter in the :ref:`config-file <Configuration file>` gives the inner and outer radii of this annulus.

Using the annuli described with the ``r_inner_IPS`` and ``r_outer_IPS`` parameters, PIPPIN corrects for the polarisation that is measured near the image centre. Any polarised signal found near the stellar signal is believed to originate from IP, because the stellar signal is assumed to be un-polarised. This correction is performed for each HWP-cycle, thus avoiding temporal differences in the instrument configuration and IP.

Finally, the :math:`U_\phi`-signal in the ``r_crosstalk`` annulus is minimised if requested (``minimise_U_phi = True``; :ref:`config-file <Configuration file>`). As described by `Avenhaus et al. (2014) <https://ui.adsabs.harvard.edu/abs/2014ApJ...781...87A/abstract>`_, an offset-angle can be estimated for the azimuthal Stokes parameters :math:`Q_\phi` and :math:`U_\phi`.

.. attention::
   Minimising the :math:`U_\phi`-signal should be done with caution as it can lead to the removal of real :math:`Q_\phi`-signal in high-inclination disks or in cases where crosstalk and IP have not been sufficiently corrected.

PDI data products
^^^^^^^^^^^^^^^^^
Depending on the observations, the :file:`PDI/` directory can contain any of the following files:

Total intensities:

- :file:`cube_I.fits`: Total intensity per HWP-cycle.
- :file:`cube_I_Q+.fits`: Total intensity :math:`I_{Q^+} = I_\mathrm{ord} + I_\mathrm{ext}` per :math:`Q^+` frame.
- :file:`cube_I_Q-.fits`: Total intensity :math:`I_{Q^-} = I_\mathrm{ord} + I_\mathrm{ext}` per :math:`Q^-` frame.
- :file:`cube_I_Q.fits`: Total intensity :math:`I_Q = \frac{1}{2}(I_{Q^+} + I_{Q^-})` per combination of :math:`Q^\pm` frames.
- :file:`cube_I_U+.fits`: Total intensity :math:`I_{U^+} = I_\mathrm{ord} + I_\mathrm{ext}` per :math:`U^+` frame.
- :file:`cube_I_U-.fits`: Total intensity :math:`I_{U^-} = I_\mathrm{ord} + I_\mathrm{ext}` per :math:`U^-` frame.
- :file:`cube_I_U.fits`: Total intensity :math:`I_U = \frac{1}{2}(I_{U^+} + I_{U^-})` per combination of :math:`U^\pm` frames.

Median-combined total intensities:

- :file:`median_I.fits`: Median-combined over all HWP-cycles.
- :file:`median_I_Q+.fits`: Median-combined over all :math:`I_{Q^+}` frames.
- :file:`median_I_Q-.fits`: Median-combined over all :math:`I_{Q^-}` frames.
- :file:`median_I_Q.fits`: Median-combined over all :math:`I_{Q}` frames.
- :file:`median_I_U+.fits`: Median-combined over all :math:`I_{U^+}` frames.
- :file:`median_I_U-.fits`: Median-combined over all :math:`I_{U^-}` frames.
- :file:`median_I_U.fits`: Median-combined over all :math:`I_{U}` frames.

Stokes parameters:

- :file:`cube_Q+.fits`: Stokes :math:`Q^+ = I_\mathrm{ord} - I_\mathrm{ext}` parameter per frame.
- :file:`cube_Q-.fits`: Stokes :math:`Q^- = I_\mathrm{ord} - I_\mathrm{ext}` parameter per frame.
- :file:`cube_Q.fits`: Stokes :math:`Q = \frac{1}{2}(Q^+ + Q^-)` parameter per combination of :math:`Q^\pm` frames.
- :file:`cube_U+.fits`: Stokes :math:`U^+ = I_\mathrm{ord} - I_\mathrm{ext}` parameter per frame.
- :file:`cube_U-.fits`: Stokes :math:`U^- = I_\mathrm{ord} - I_\mathrm{ext}` parameter per frame.
- :file:`cube_U.fits`: Stokes :math:`U = \frac{1}{2}(U^+ + U^-)` parameter per combination of :math:`U^\pm` frames.

Median-combined Stokes parameters:

- :file:`median_Q+.fits`: Median-combined over all :math:`Q^+` observations.
- :file:`median_Q-.fits`: Median-combined over all :math:`Q^-` observations.
- :file:`median_Q.fits`: Median-combined over all :math:`Q` observations.
- :file:`median_Q_IPS.fits`: Median-combined + IP-corrected Stokes :math:`Q_\mathrm{IPS}` parameter.
- :file:`median_U+.fits`: Median-combined over all :math:`U^+` observations.
- :file:`median_U-.fits`: Median-combined over all :math:`U^-` observations.
- :file:`median_U.fits`: Median-combined over all :math:`U` observations.
- :file:`median_U_IPS.fits`: Median-combined + IP-corrected Stokes :math:`U_\mathrm{IPS}` parameter.

Final polarised light products:

- :file:`P_I.fits`: Polarised intensity :math:`PI = \sqrt{Q_\mathrm{IPS}^2 + U_\text{IPS}^2}`.
- :file:`P_I_r2.fits`: Polarised intensity :math:`PI` scaled by the de-projected squared separation :math:`r^2`.
- :file:`Q_phi.fits`: Azimuthal Stokes :math:`Q_\phi = - Q_\mathrm{IPS} \cos 2\phi - U_\mathrm{IPS} \sin 2\phi` parameter.
- :file:`Q_phi_r2.fits`: Azimuthal :math:`Q_\phi` parameter scaled by the de-projected squared separation :math:`r^2`.
- :file:`r.fits`: De-projected separation :math:`r`.
- :file:`U_phi.fits`: Azimuthal Stokes :math:`U_\phi = + Q_\mathrm{IPS} \sin 2\phi - U_\mathrm{IPS} \cos 2\phi` parameter.


Instrument configurations
-------------------------
#   HWP usage, wiregrid/Wollaston
#   Extended data products
#   IP removal
