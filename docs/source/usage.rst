
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

PIPPIN will median combine the FLATs and DARKs of each observation type and save the master calibration files in a sub-directory (e.g. :file:`/path/to/FLATs/master_FLAT/`). Master bad-pixel masks (BPMs) are generated using the (non)-linear pixel response between FLAT observations with the FLAT-lamp on or off. The BPMs are stored in the :file:`/path/to/master_BPM/` directory which will be created if it does not exist yet.

To run the pipeline from the terminal, navigate to the directory where the raw SCIENCE images are stored (e.g. :file:`cd /example_HD_135344B/`) and run:
::

   pippin --run_pipeline --path_FLAT_dir /path/to/master_FLAT/ --path_DARK_dir /path/to/master_DARK/ --path_BPM_dir /path/to/master_BPM/

or
::

   pippin --run_pipeline --prepare_calib_files --path_FLAT_dir /path/to/FLATs/ --path_DARK_dir /path/to/DARKs/ --path_BPM_dir /path/to/master_BPM/


Pre-processing
--------------
First, PIPPIN separates the raw SCIENCE data into different observation types with information from the headers. Observations are categorised by:

- HWP/rotator-usage
- Wollaston-usage (Wollaston_00, Wollaston_45, wiregrid)
- Detector (e.g. S13, S27, L27)
- Window shape
- Exposure time
- Filter
- Observing block ID (if ``split_observing_blocks = True`` in the config-file).

The pipeline continues by DARK-subtracting and FLAT-normalising the observations. The bad pixels are replaced by the median of the surrounding box of 5x5 pixels, excluding any other bad pixels.

Next, PIPPIN locates the centres of the ordinary and extra-ordinary beams. PIPPIN provides several methods for fitting the beam-centres (``centering_method`` in the config-file):

- ``maximum``: The maximum pixel in an image median-filtered with a 3x3 kernel.
- ``single-Moffat``: A single 2D Moffat function (for each beam).
- ``double-Moffat``: Two 2D Moffat functions (for each beam) subtracted from each other to replicate the flat top of a saturated beam.

The two Moffat fitting methods allow the beam-offset to be tied, based on the expected pixel-separation with the utilised detector (``tied_offset = True`` in the config-file). The tied offset is useful when the stellar light does not form a point source (e.g. for embedded stars).

.. note::
   If the data consists of wiregrid-observations instead of Wollaston-observations, only one beam is identified.


#   Sky-subtraction
#     box-median, dithering-offset
#     Removal of horizontal stripes
#   Cropping and saving beams

Polarimetric Differential Imaging
---------------------------------
#   Ord./Ext. beam equalising
#   IP double-difference
#   IP crosstalk correction / Uphi minimisation

Instrument configurations
-------------------------
#   HWP usage, wiregrid/Wollaston
#   Extended data products
#   IP removal
