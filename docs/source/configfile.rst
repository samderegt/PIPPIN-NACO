
Configuration file
==================

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

If ``True``, split the observing-blocks by ID. Can be set to ``False`` if concurrent observations have altering OBS IDs.


.. py:function:: y_pixel_range:

`list` (default = ``[0, 1024]``)

Vertical pixel indices to crop the intermediate data products between. This option can help to save time or memory with large data-cubes.


.. py:function:: sky_subtraction_method:

``dithering-offset``, ``box-median`` (default = ``dithering-offset``)

If ``dithering-offset``, PIPPIN uses observations at different dithering positions to subtract the sky background. The minimum separation of the dithering-positions is indicated with sky_subtraction_min_offset. If ``box-median``, the sky contribution is estimated with the median in two boxes left and right of the beams. The boxes are drawn at a separation indicated by sky_subtraction_min_offset.


Sky-subtraction
---------------

.. _sky_subtraction_min_offset:

.. py:function:: sky_subtraction_min_offset:

`integer` (default = 100)

The minimum horizontal pixel-offset PIPPIN uses when carrying out the sky-subtraction via ``dithering-offset`` or ``box-median``.


.. py:function:: remove_horizontal_stripes:

``True``, ``False`` (default = ``False``)

If ``True``, PIPPIN attempts to remove the horizontal stripe pattern found in several observations.
