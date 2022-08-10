
Usage instructions
==================

- :ref:`Running PIPPIN`
- :ref:`Pre-processing`
- :ref:`Polarimetric Differential Imaging`
- :ref:`Instrument configurations`

Running PIPPIN
--------------
PIPPIN requires the raw SCIENCE FITS-files to be uncompressed and stored in a directory (e.g. :file:`/example_HD_135344B/`). In the same directory, PIPPIN will search for a configuration file named :file:`config.conf`. A configuration file with default parameters can be created if PIPPIN cannot find it.

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
- Observing block ID (if ``split_observing_blocks = True`` in the config-file).

The results of the data reduction are stored in sub-directories of the generated :file:`pipeline_output/` directory. A log-file ``log.txt`` is created, storing information on the used reduction methods.

The pipeline continues by DARK-subtracting and FLAT-normalising the observations. The bad pixels are replaced by the median of the surrounding box of 5x5 pixels, excluding any other bad pixels.

Beam-centre fitting
^^^^^^^^^^^^^^^^^^^
Next, PIPPIN locates the centres of the ordinary and extra-ordinary beams. PIPPIN provides several methods for fitting the beam-centres (``centering_method`` in the config-file):

- ``maximum``: The maximum pixel in an image median-filtered with a 3x3 kernel.
- ``single-Moffat``: A single 2D Moffat function (for each beam).
- ``double-Moffat``: Two 2D Moffat functions (for each beam) subtracted from each other to replicate the flat top of a saturated beam.

The two Moffat fitting methods allow the beam-offset to be tied, based on the expected pixel-separation with the utilised detector (``tied_offset = True`` in the config-file). The tied offset is useful when the stellar light does not form a point source (e.g. for embedded stars).

.. note::
   Only one beam is identified if the data consists of wiregrid-observations instead of Wollaston-observations.

Sky-subtraction
^^^^^^^^^^^^^^^
The sky-subtraction can be performed with one of the following methods (``sky_subtraction_method`` in the config-file):

- ``box-median``: The sky-signal is estimated from the median signal of pixels which are at least ``sky_subtraction_min_offset`` to the left or right of the assessed beam centres.
- ``dithering-offset``: Observations with different dithering positions are subtracted from each other. The two observations must be separated by ``sky_subtraction_min_offset``, otherwise the ``box_median`` method is utilised.

A gradient can remain in the sky-subtracted images. PIPPIN corrects for this with a linear fit to rows of pixels. If ``remove_horizontal_stripes = False`` in the config-file, 5 rows will be binned and the final gradient image will be smoothed and subtracted. A read-out artefact can leave behind horizontal stripes which can be removed with a more aggressive fitting of each row, using ``remove_horizontal_stripes = True``.

Cropping and saving
^^^^^^^^^^^^^^^^^^^
#   Cropping and saving beams
The ordinary and extra-ordinary beams are cropped and saved as FITS-files, employing the ``size_to_crop`` parameter in the config-file. Any temporary data products ``_reduced.fits`` and ``_skysub.fits`` are removed if ``remove_data_products = True`` in the config-file. Open AO-loop observations are identified with an iterative sigma-clipping and the file-names are stored in ``open_loop_files.txt``. The :file:`plots/` directory stores a figure of this assessment in addition to figures of the reduction steps. 


Polarimetric Differential Imaging
---------------------------------
#   Ord./Ext. beam equalising
#   IP double-difference
#   IP crosstalk correction / Uphi minimisation

Different instrument configurations
-----------------------------------
#   HWP usage, wiregrid/Wollaston
#   Extended data products
#   IP removal
