
Example reduction
=================

To confirm that the PIPPIN installation was successful and as an introduction to the pipeline, we provide an example reduction of the Ks-band observations of HD 135344B, observed as part of ESO programme 089.C-0611(A). This dataset was previously published in `Garufi et al. (2013) <https://ui.adsabs.harvard.edu/abs/2013A%26A...560A.105G/abstract>`_.

To run the example reduction, navigate to any directory in the terminal and type:
::

   pippin --run_example

PIPPIN attempts to find the required data in your current directory. If these files do not exist, they can be downloaded (with an internet connection) from the `GitHub repository <https://github.com/samderegt/PIPPIN-NACO/tree/master/pippin/example_HD_135344B>`_ (48.4 MB). After successfully downloading the data, which includes SCIENCE, FLAT, and DARK observations, as well as a configuration-file with input parameters (`config.conf`), PIPPIN begins the reduction.

As the pipeline is running, information is printed in the terminal and stored in the `/example_HD_135344B/pipeline_output/log.txt` file.

::

   ds9 -tile Q_phi.fits -cube 2 -scale limits -20 50 U_phi.fits -cube 2 -scale limits -7 7 -lock frame wcs -lock colorbar yes -cmap cool
