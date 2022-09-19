
Configuration file
==================

The default configuration file is given below:

::

   [Pre-processing options]

   run_pre_processing     = True
   remove_data_products   = True
   split_observing_blocks = True
   y_pixel_range          = [0,1024]


   [Sky-subtraction]

   sky_subtraction_method     = dithering-offset
   sky_subtraction_min_offset = 100
   remove_horizontal_stripes  = True

   [Centering]

   centering_method = single-Moffat
   tied_offset      = False


   [PDI options]

   size_to_crop         = [120,120]
   r_inner_IPS          = [0,3,6,9,12]
   r_outer_IPS          = [3,6,9,12,15]
   crosstalk_correction = True
   minimise_U_phi       = True
   r_crosstalk          = [7,17]


   [Object information]

   object_name      =
   disk_pos_angle   = 0.0
   disk_inclination = 0.0

The input parameters are divided into five groups:

- :ref:`Pre-processing options`
- :ref:`Sky-subtraction`
- :ref:`Centering`
- :ref:`PDI options`
- :ref:`Object information`

Each parameter in the configuration file is explained below.

Pre-processing options
----------------------

.. py:function:: run_pre_processing:

   ``True``, ``False`` (default = ``True``)

   If ``True``, perform the :ref:`pre-processing <Pre-processing>` reduction on the SCIENCE data. `run_pre_processing` should bet ``True`` the first time PIPPIN is run, but can be changed to ``False`` if you wish to tweak the input parameters of the PDI.


.. py:function:: remove_data_products:

   ``True``, ``False`` (default = ``True``)

   If ``True``, remove the intermediate data products (:file:`_reduced.fits` and :file:`_skysub.fits` files) once the pre-processing reduction is finished.


.. py:function:: split_observing_blocks:

   ``True``, ``False`` (default = ``True``)

   If ``True``, split the observing-blocks by ID. `split_observing_blocks` can be set to ``False`` if concurrent observations have altering OBS IDs.


.. py:function:: y_pixel_range:

   `list` (default = ``[0, 1024]``)

   Vertical pixel indices to crop the intermediate data products between. Changing `y_pixel_range` can help to save time or memory with large data-cubes.


Sky-subtraction
---------------

.. py:function:: sky_subtraction_method:

   ``dithering-offset``, ``box-median`` (default = ``dithering-offset``)

   If ``dithering-offset``, PIPPIN uses observations at different dithering positions to subtract the :ref:`sky <Sky-subtraction>` background. The minimum separation of the dithering-positions is indicated with sky_subtraction_min_offset_. If `sky_subtraction_method` is set to ``box-median``, the sky contribution is estimated with the median in two boxes left and right of the beams. The boxes are drawn at a separation indicated by sky_subtraction_min_offset_ and continue to the edge of the detector.


.. _sky_subtraction_min_offset:

.. py:function:: sky_subtraction_min_offset:

   `integer` (default = 100)

   The minimum horizontal pixel-offset PIPPIN uses when carrying out the sky-subtraction via ``dithering-offset`` or ``box-median``.


.. py:function:: remove_horizontal_stripes:

   ``True``, ``False`` (default = ``False``)

   If ``True``, PIPPIN attempts to remove the horizontal stripe pattern found as a read-out artefact in certain observations by fitting polynomials to each row of pixels.


Centering
---------

.. _centering_method:

.. py:function:: centering_method:

   ``single-Moffat``, ``double-Moffat``, ``maximum`` (default = ``single-Moffat``)

   Method to use for fitting the :ref:`beam-centres <Beam-centre fitting>`. If `centering_method` is set to ``single-Moffat``, PIPPIN uses a Moffat function for each beam to locate the ordinary and extra-ordinary beam-centres. If `centering_method` is set to ``double-Moffat``, PIPPIN utilises two Moffat functions, one subtracted from the other, to replicate the flat top of saturated beams. If `centering_method` is set to ``maximum``, PIPPIN locates the beam-centres as the brightest pixel in a median-filtered image (using a 3:math:`\times`3 box). In general, the ``single-Moffat`` option should give a sufficiently accurate assessment of the beam-centres.


.. py:function:: tied_offset:

   ``True``, ``False`` (default = ``False``)

   If ``True``, PIPPIN will tie the offset between the two beams based on the used detector and fit for both beams at the same time. `tied_offset` is only supported by the ``single-Moffat`` and ``double-Moffat`` options of the centering_method_ parameter.


PDI options
-----------

.. py:function:: size_to_crop:

   `list` (default = ``[120,120]``)

   The image-size of the ordinary and extra-ordinary beams (:file:`_beams.fits` files). The final data products will inherit this image-size.


.. py:function:: r_inner_IPS:

   `list` or `integer` (default = ``[0,3,6,9,12]``)

   Inner radii of the annuli that PIPPIN uses to equalise the flux in the ordinary and extra-ordinary beams following `Avenhaus et al. (2014) <https://ui.adsabs.harvard.edu/abs/2014ApJ...781...87A/abstract>`_. These annuli are also used to perform the IP-subtraction under the assumption that the stellar light in the annulus is unpolarised (see:ref:`Instrumental polarisation <Instrumental polarisation>`).


.. py:function:: r_outer_IPS:

   `list` or `integer` (default = ``[3,6,9,12,15]``)

   Outer radii of the annuli that PIPPIN uses to equalise the flux in the ordinary and extra-ordinary beams. These annuli are also used to perform the IP-subtraction under the assumption that the stellar light in the annulus is unpolarised (see:ref:`Instrumental polarisation <Instrumental polarisation>`).


.. _crosstalk_correction:

.. py:function:: crosstalk_correction:

   ``True``, ``False`` (default = ``False``)

   If ``True``, PIPPIN corrects for the :ref:`instrumental crosstalk <Instrumental polarisation>` between the linear and circular Stokes parameters following `Avenhaus et al. (2014) <https://ui.adsabs.harvard.edu/abs/2014ApJ...781...87A/abstract>`_. The reduced efficiency of the Stokes :math:`U` parameter is assessed in the annulus provided by r_crosstalk_.

.. attention::
   The crosstalk correction is made to function with non-symmetric disks as it employs a counting method of pixels with higher :math:`Q`- than :math:`U`-signal (`Avenhaus et al. 2014 <https://ui.adsabs.harvard.edu/abs/2014ApJ...781...87A/abstract>`_). However, this method is not expected to work well with high-inclination disks due to the un-equal distribution of (any) signal-producing pixels over the :math:`Q`- and :math:`U`-quadrants.


.. _minimise_U_phi:

.. py:function:: minimise_U_phi:

   ``True``, ``False`` (default = ``False``)

   If ``True``, PIPPIN minimises the :math:`U_\phi`-signal in the r_crosstalk_ annulus following `Avenhaus et al. (2014) <https://ui.adsabs.harvard.edu/abs/2014ApJ...781...87A/abstract>`_. If ``True``, PIPPIN fits for an offset-angle when computing :math:`U_\phi` (and :math:`Q_\phi`).

.. attention::
   Minimising the :math:`U_\phi`-signal should be done with caution as it can lead to the removal of real :math:`Q_\phi`-signal in high-inclination disks or in cases where crosstalk and IP have not been sufficiently corrected.


.. _r_crosstalk:

.. py:function:: r_crosstalk:

   `list` (default = ``[7,17]``)

   Inner and outer radius of the annulus used in correcting for crosstalk with the crosstalk_correction_ and minimise_U_phi_ options.

.. attention::
   Performing a crosstalk correction via `crosstalk_correction = True` or `minimise_U_phi = True` requires sufficient :math:`Q`- and :math:`U`-signal to be present in the r_crosstalk_ annulus. In cases where the signal is found only in specific quadrants of the :math:`Q` or :math:`U` images (e.g. a high-inclination or asymmetric disk), we recommend **against** using the crosstalk_correction_ and minimise_U_phi_ corrections.


Object information
------------------

.. py:function:: object_name:

   `str` (default = ``file_path``)

   Object name to query the SIMBAD archive for target coordinates. These coordinates are subsequently used to set up a world-coordinate system. If this parameter is not provided, PIPPIN will attempt to infer the object name from the directory in which it is run. If the `object_name` is not recognised by SIMBAD, the pipeline will exit with an error.


.. py:function:: disk_pos_angle:

   `float` (default = 0.0)

   Disk position angle in degrees. This parameter is used to determine the de-projected radius.However, t

.. py:function:: disk_inclination:

   `float` (default = 0.0)

   Disk inclination in degrees. This parameter is used to determine the de-projected radius.
