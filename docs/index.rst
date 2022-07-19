
PIPPIN
======

**This documentation is under construction.**

PIPPIN (PdI PiPelIne for Naco data) is a pipeline capable of reducing the polarimetric observations made by the `VLT/NACO <https://www.eso.org/sci/facilities/paranal/decommissioned/naco.html>`_ instrument. It applies the Polarimetric Differential Imaging (PDI) technique to separate the polarised, scattered light from the (largely) un-polarised, stellar light. Thus, circumstellar dust can be uncovered. During its operation, NACO employed different instrument configurations (e.g. half-wave plate or de-rotator usage, Wollaston beam-splitter or wiregrid observations) which are appropriately handled by PIPPIN. Corrections can be made for instrument-induced polarisation or crosstalk.

To reduce polarimetric NACO data, the user can specify a limited number of input parameters and run PIPPIN from a terminal. Subsequently, PIPPIN applies a complete reduction from the raw SCIENCE observations to the final data products in a matter of minutes.

.. figure:: ./figures/figure_homepage.png
    :width: 750px

*Examples of detected polarised light in NACO observations reduced with PIPPIN.*

Contents
--------

.. toctree::

   Home <self>
   ./source/installation
   ./source/example
   ./source/usage
   ./source/configfile
   GitHub <https://github.com/samderegt/PIPPIN-NACO>
