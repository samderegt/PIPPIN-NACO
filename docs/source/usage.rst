
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

   pippin --path_FLAT_dir /path/to/FLAT/directory --path_DARK_dir /path/to/DARK/directory --path_BPM_dir /path/to/BPM/directory

In the terminal, navigate to the directory where the raw SCIENCE images are stored (e.g. :file:`cd /example_HD_135344B/`) and run:
::

   pippin --run_pipeline


#   Supplying your own FLATs/DARKs
#   Separating observation types (HWP, Wollaston/wiregrid, detector, window shape, exp time, filter)

Pre-processing
--------------
#   Calibrate image (DARK-subtract, FLAT-normalise)
#     Supplying your own FLATs/DARKs
#     Bad-pixel mask
#   Locating beam-centers
#     maximum, single-Moffat, double-Moffat
#     tied_offset
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
